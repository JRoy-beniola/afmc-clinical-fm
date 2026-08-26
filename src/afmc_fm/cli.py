import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from afmc_fm.config import load_yaml
from afmc_fm.execution import manifest as _execution_manifest
from afmc_fm.execution.device import runtime_diagnostics
from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark
from afmc_fm.experiments.runner import ExperimentConfig
from afmc_fm.schema.events import events_to_frame
from afmc_fm.simulator.cohort import SimulatedCohort, simulate_cohort
from afmc_fm.simulator.config import SimulatorConfig

_phase0_gate_summary = _execution_manifest._phase0_gate_summary
_plot_learning_curves = _execution_manifest._plot_learning_curves


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
    benchmark.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    benchmark.add_argument("--workers", type=_positive_int, default=1)
    benchmark.add_argument("--resume", action="store_true")
    benchmark.add_argument("--fail-fast", action="store_true")
    benchmark.set_defaults(handler=_benchmark)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
