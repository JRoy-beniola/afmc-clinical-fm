import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from afmc_fm.config import load_yaml
from afmc_fm.execution import manifest as _execution_manifest
from afmc_fm.execution.device import runtime_diagnostics
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark
from afmc_fm.experiments.runner import ExperimentConfig
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase05.protocol import build_protocol_lock
from afmc_fm.phase05.store import Phase05Store
from afmc_fm.schema.events import events_to_frame
from afmc_fm.simulator.cohort import SimulatedCohort, simulate_cohort
from afmc_fm.simulator.config import SimulatorConfig

_phase0_gate_summary = _execution_manifest._phase0_gate_summary
_plot_learning_curves = _execution_manifest._plot_learning_curves
_PHASE05_SPEC_COMMIT = "67c6662c64e69606bcbd3eca2bc8139548013051"
_PHASE0_EXECUTION_SHA = "d6f105eee73fcb8e9cc5987d292b1bb98a687382"


def _simulator_from_yaml(path: str | Path) -> tuple[SimulatorConfig, int]:
    raw = load_yaml(path)
    seed = int(raw.pop("seed", 0))
    return SimulatorConfig(**raw), seed


def _experiment_from_yaml(path: str | Path) -> ExperimentConfig:
    raw = load_yaml(path)
    for key in (
        "train_sizes",
        "cohort_seeds",
        "subset_seeds",
        "model_seeds",
        "worlds",
        "models",
        "ablations",
        "test_sites",
    ):
        if key in raw:
            raw[key] = tuple(raw[key])
    return ExperimentConfig(**raw)


def _write_simulation(cohort: SimulatedCohort, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    events = [event for patient in cohort.patients for event in patient.timeline.events]
    events_to_frame(events).to_csv(output / "events.csv", index=False)
    latent_arrays = {}
    for patient in cohort.patients:
        latent_arrays[f"{patient.patient_id}_times"] = patient.latent.times
        latent_arrays[f"{patient.patient_id}_states"] = patient.latent.states
    np.savez_compressed(output / "latent_truth.npz", **latent_arrays)
    manifest = {
        "synthetic": True,
        "seed": cohort.seed,
        "simulator_config": asdict(cohort.config),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (output / "simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _simulate(args: argparse.Namespace) -> int:
    config, seed = _simulator_from_yaml(args.config)
    _write_simulation(simulate_cohort(config, seed), Path(args.output))
    return 0


def _diagnostics(args: argparse.Namespace) -> int:
    diagnostics = runtime_diagnostics(args.device, args.workers)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("workers must be at least 1")
    return parsed


def _benchmark(args: argparse.Namespace) -> int:
    sim_config, seed = _simulator_from_yaml(args.sim_config)
    experiment = _experiment_from_yaml(args.exp_config)
    run_scheduled_benchmark(
        sim_config,
        experiment,
        Path(args.output),
        ExecutionOptions(
            device=args.device,
            workers=args.workers,
            resume=args.resume,
            fail_fast=args.fail_fast,
        ),
        template_seed=seed,
    )
    return 0


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _phase05_calibrate(args: argparse.Namespace) -> int:
    metrics_path = Path(args.phase0_metrics)
    metrics_bytes = metrics_path.read_bytes()
    metrics = pd.read_csv(metrics_path)
    config = load_phase05_config(args.exp_config)
    lock = build_protocol_lock(
        config,
        metrics,
        phase0_metrics_sha256=hashlib.sha256(metrics_bytes).hexdigest(),
        spec_commit=_PHASE05_SPEC_COMMIT,
        phase0_execution_sha=_PHASE0_EXECUTION_SHA,
    )
    protocol_hash = hashlib.sha256(_canonical_json_bytes(lock)).hexdigest()
    store = Phase05Store(
        Path(args.output),
        protocol_hash,
        config_hash=canonical_config_hash(config),
    )
    store.write_protocol_lock(lock)
    return 0


def _add_execution_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afmc-phase0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--output", required=True)
    simulate.set_defaults(handler=_simulate)
    diagnostics = subparsers.add_parser("diagnostics")
    diagnostics.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    diagnostics.add_argument("--workers", type=_positive_int, default=1)
    diagnostics.set_defaults(handler=_diagnostics)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--sim-config", required=True)
    benchmark.add_argument("--exp-config", required=True)
    benchmark.add_argument("--output", required=True)
    _add_execution_options(benchmark)
    benchmark.set_defaults(handler=_benchmark)

    phase05 = subparsers.add_parser("phase05")
    phase05_subparsers = phase05.add_subparsers(
        dest="phase05_command",
        required=True,
    )

    calibrate = phase05_subparsers.add_parser("calibrate")
    calibrate.add_argument("--phase0-metrics", required=True)
    calibrate.add_argument("--exp-config", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(handler=_phase05_calibrate)

    for name in ("develop", "confirm", "robustness"):
        stage = phase05_subparsers.add_parser(name)
        stage.add_argument("--sim-config", required=True)
        stage.add_argument("--exp-config", required=True)
        stage.add_argument("--output", required=True)
        _add_execution_options(stage)

    freeze = phase05_subparsers.add_parser("freeze")
    freeze.add_argument("--exp-config", required=True)
    freeze.add_argument("--output", required=True)

    report = phase05_subparsers.add_parser("report")
    report.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
