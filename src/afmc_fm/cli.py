import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from afmc_fm.config import load_yaml
from afmc_fm.execution import manifest as _execution_manifest
from afmc_fm.execution.device import runtime_diagnostics
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark
from afmc_fm.experiments.runner import ExperimentConfig
from afmc_fm.phase05.analysis import normalized_log_n_aulc
from afmc_fm.phase05.config import Phase05Config, load_phase05_config
from afmc_fm.phase05.development import (
    representation_timing_audit,
    select_flow,
    select_jump,
    select_uncertainty,
)
from afmc_fm.phase05.execution import (
    Phase05ExecutionOptions,
    Phase05Job,
    Phase05ShardSpec,
    plan_phase05_shards,
    run_phase05_jobs,
)
from afmc_fm.phase05.protocol import build_protocol_lock, freeze_candidate
from afmc_fm.phase05.runner import (
    PreparedPhase05Cohort,
    prepare_phase05_cohort,
    run_phase05_variant,
)
from afmc_fm.phase05.store import Phase05Store
from afmc_fm.schema.events import events_to_frame
from afmc_fm.simulator.cohort import SimulatedCohort, simulate_cohort, simulate_world
from afmc_fm.simulator.config import SimulatorConfig

_phase0_gate_summary = _execution_manifest._phase0_gate_summary
_plot_learning_curves = _execution_manifest._plot_learning_curves
_PHASE05_SPEC_COMMIT = "67c6662c64e69606bcbd3eca2bc8139548013051"
_PHASE0_EXECUTION_SHA = "d6f105eee73fcb8e9cc5987d292b1bb98a687382"
_BUNDLE_COLUMNS = ("cohort_seed", "subset_seed", "model_seed")


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


def _frame_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


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


def _phase05_development_jobs(
    config: Phase05Config,
    stage: str,
    *,
    selected_flow: str | None = None,
    selected_jump: str | None = None,
) -> tuple[Phase05Job, ...]:
    if stage == "flow":
        variants = (
            "none__none__deterministic",
            "gated__none__deterministic",
            "time_scaled__none__deterministic",
        )
    elif stage == "jump":
        if selected_flow is None:
            raise RuntimeError("jump planning requires a selected flow")
        variants = tuple(
            f"{selected_flow}__{jump}__deterministic"
            for jump in ("none", "gru", "residual")
        )
    elif stage == "uncertainty":
        if selected_flow is None or selected_jump is None:
            raise RuntimeError("uncertainty planning requires selected flow and jump")
        variants = tuple(
            f"{selected_flow}__{selected_jump}__{uncertainty}"
            for uncertainty in ("joint", "decoupled", "deterministic")
        )
    elif stage == "timing_audit":
        if selected_flow is None or selected_jump is None:
            raise RuntimeError("timing audit planning requires selected flow and jump")
        base = f"{selected_flow}__{selected_jump}__deterministic"
        variants = (
            f"{base}__strict_history",
            f"{base}__inclusive_history",
        )
    else:
        raise ValueError(f"unknown Phase-0.5 development stage: {stage}")

    return tuple(
        Phase05Job(
            shard=shard,
            n_train=n_train,
            model="phase05_flow_jump",
            variant=variant,
        )
        for shard in plan_phase05_shards(config, stage)
        for n_train in config.primary_train_sizes
        for variant in variants
    )


def _phase05_locked_store(
    output: Path,
    config: Phase05Config,
) -> tuple[dict[str, object], Phase05Store]:
    lock_path = output / "protocol_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("protocol_lock.json is required before Phase-0.5 development")
    data = lock_path.read_bytes()
    try:
        lock = json.loads(data)
    except json.JSONDecodeError as error:
        raise RuntimeError("protocol_lock.json is invalid") from error
    if not isinstance(lock, dict):
        raise TypeError("protocol_lock.json is invalid")
    config_hash = canonical_config_hash(config)
    if lock.get("phase05_config_sha256") != config_hash:
        raise RuntimeError("protocol_lock.json does not match the runtime Phase-0.5 config")
    return lock, Phase05Store(
        output,
        hashlib.sha256(data).hexdigest(),
        config_hash=config_hash,
    )


def _prepare_phase05_development_shard(
    shard: Phase05ShardSpec,
    *,
    simulator_config: SimulatorConfig,
) -> PreparedPhase05Cohort:
    cohort = simulate_world(
        shard.world,
        simulator_config,
        seed=shard.seed_bundle.cohort_seed,
    )
    return prepare_phase05_cohort(
        cohort,
        include_historical=shard.stage == "timing_audit",
    )


def _inclusive_phase05_prepared(
    prepared: PreparedPhase05Cohort,
) -> PreparedPhase05Cohort:
    historical = prepared.historical_sequences
    if historical is None:
        raise RuntimeError("inclusive timing audit requires historical representations")
    sequences = {}
    for patient_id, strict_sequence in prepared.sequences.items():
        inclusive = historical[patient_id]
        if inclusive.representations.shape != strict_sequence.representations.shape:
            raise RuntimeError("strict and inclusive representation shapes differ")
        sequences[patient_id] = replace(
            strict_sequence,
            representations=np.array(inclusive.representations, copy=True),
        )
    return replace(prepared, sequences=sequences)


def _run_phase05_development_job(
    job: Phase05Job,
    prepared: PreparedPhase05Cohort,
    device,
    *,
    config: Phase05Config,
) -> pd.DataFrame:
    parts = job.variant.split("__")
    run_prepared = prepared
    if len(parts) == 3:
        flow_mode, jump_mode, uncertainty_mode = parts
    elif len(parts) == 4 and job.shard.stage == "timing_audit":
        flow_mode, jump_mode, uncertainty_mode, history_mode = parts
        if uncertainty_mode != "deterministic":
            raise ValueError("timing audit must use deterministic uncertainty")
        if history_mode == "inclusive_history":
            run_prepared = _inclusive_phase05_prepared(prepared)
        elif history_mode != "strict_history":
            raise ValueError(f"unknown timing audit history mode: {history_mode}")
    else:
        raise ValueError(f"invalid Phase-0.5 development variant: {job.variant}")

    result = run_phase05_variant(
        run_prepared,
        config,
        world=job.shard.world,
        seed_bundle=job.shard.seed_bundle,
        n_train=job.n_train,
        flow_mode=flow_mode,
        jump_mode=jump_mode,
        uncertainty_mode=uncertainty_mode,
        device=device,
        stage=job.shard.stage,
    ).copy()
    result["stage"] = job.shard.stage
    result["model"] = job.model
    result["variant"] = job.variant
    return result


def _run_development_stage(
    jobs: tuple[Phase05Job, ...],
    *,
    store: Phase05Store,
    config: Phase05Config,
    options: Phase05ExecutionOptions,
    simulator_config: SimulatorConfig,
) -> pd.DataFrame:
    frame = run_phase05_jobs(
        jobs,
        store=store,
        config=config,
        options=options,
        prepare_shard=partial(
            _prepare_phase05_development_shard,
            simulator_config=simulator_config,
        ),
        run_job=partial(_run_phase05_development_job, config=config),
    )
    store.mark_stage_complete(
        jobs[0].shard.stage,
        frozenset(job.cell_id for job in jobs),
    )
    return frame


def _unique_variant_parameters(metrics: pd.DataFrame, variant: str) -> int:
    values = pd.unique(
        metrics.loc[metrics["variant"] == variant, "trainable_parameters"].dropna()
    )
    if len(values) != 1:
        raise RuntimeError(f"variant {variant} has invalid trainable parameter metadata")
    return int(values[0])


def _uncertainty_gate_table(
    metrics: pd.DataFrame,
    *,
    selected_flow: str,
    selected_jump: str,
    selection: dict[str, object],
) -> pd.DataFrame:
    selected = str(selection["selected"])
    point_admissible = dict(selection["point_admissible"])
    world_wins = dict(selection["probabilistic_world_wins"])
    rows = []
    for candidate in ("joint", "decoupled", "deterministic"):
        variant = f"{selected_flow}__{selected_jump}__{candidate}"
        rows.append(
            {
                "candidate": candidate,
                "selected": candidate == selected,
                "point_admissible": point_admissible.get(candidate),
                "probabilistic_world_wins": int(world_wins.get(candidate, 0)),
                "trainable_parameters": _unique_variant_parameters(metrics, variant),
            }
        )
    return pd.DataFrame(rows)


def _timing_bundle_naulc(
    metrics: pd.DataFrame,
    *,
    world: str,
    bundle,
    variant: str,
) -> float:
    subset = metrics.loc[
        (metrics["world"] == world)
        & (metrics["cohort_seed"] == bundle.cohort_seed)
        & (metrics["subset_seed"] == bundle.subset_seed)
        & (metrics["model_seed"] == bundle.model_seed)
        & (metrics["variant"] == variant)
        & (metrics["metric"] == "mae")
    ].copy()
    if "split" in subset.columns:
        subset = subset.loc[subset["split"] == "test"]
    if "site_or_shift" in subset.columns:
        subset = subset.loc[subset["site_or_shift"] == "all"]
    return normalized_log_n_aulc(subset)


def _timing_audit_table(
    metrics: pd.DataFrame,
    config: Phase05Config,
    *,
    selected_flow: str,
    selected_jump: str,
) -> pd.DataFrame:
    base = f"{selected_flow}__{selected_jump}__deterministic"
    strict_variant = f"{base}__strict_history"
    inclusive_variant = f"{base}__inclusive_history"
    rows = []
    for world in config.target_worlds:
        strict_values = []
        inclusive_values = []
        bundle_values = []
        for bundle in config.development_bundles:
            strict = _timing_bundle_naulc(
                metrics,
                world=world,
                bundle=bundle,
                variant=strict_variant,
            )
            inclusive = _timing_bundle_naulc(
                metrics,
                world=world,
                bundle=bundle,
                variant=inclusive_variant,
            )
            strict_values.append(strict)
            inclusive_values.append(inclusive)
            bundle_values.append((bundle, strict, inclusive))

        audit = representation_timing_audit(strict_values, inclusive_values)
        for bundle, strict, inclusive in bundle_values:
            rows.append(
                {
                    "world": world,
                    "cohort_seed": bundle.cohort_seed,
                    "subset_seed": bundle.subset_seed,
                    "model_seed": bundle.model_seed,
                    "strict_naulc": strict,
                    "inclusive_naulc": inclusive,
                    "delta_inclusive_minus_strict": inclusive - strict,
                    "strict_timing_required": audit["strict_timing_required"],
                    "world_mean_delta_inclusive_minus_strict": audit[
                        "mean_delta_inclusive_minus_strict"
                    ],
                    "world_median_delta_inclusive_minus_strict": audit[
                        "median_delta_inclusive_minus_strict"
                    ],
                }
            )
    return pd.DataFrame(rows)


def _validate_finalized_stage_cells(
    store: Phase05Store,
    jobs: tuple[Phase05Job, ...],
    *,
    artifact_name: str,
) -> Path | None:
    stage = jobs[0].shard.stage
    marker = store.output / "stages" / stage / "COMPLETE"
    artifact = store.output / "development" / artifact_name
    marker_exists = marker.is_file()
    artifact_exists = artifact.is_file()
    if not marker_exists and not artifact_exists:
        return None
    if marker_exists != artifact_exists:
        raise RuntimeError(f"{stage} resume state is incomplete")

    expected_cell_ids = frozenset(job.cell_id for job in jobs)
    expected_seed_bundles = frozenset(
        job.shard.seed_bundle.as_tuple() for job in jobs
    )
    observed = store.validate_resume(
        stage,
        expected_cell_ids=expected_cell_ids,
        expected_seed_bundles=expected_seed_bundles,
    )
    if observed != expected_cell_ids:
        raise RuntimeError(f"{stage} resume state does not contain the exact finalized cell set")
    return artifact


def _resume_selected_development_gate(
    store: Phase05Store,
    jobs: tuple[Phase05Job, ...],
    *,
    artifact_name: str,
    allowed_candidates: frozenset[str],
) -> str | None:
    stage = jobs[0].shard.stage
    artifact = _validate_finalized_stage_cells(
        store,
        jobs,
        artifact_name=artifact_name,
    )
    if artifact is None:
        return None

    try:
        gate = pd.read_csv(artifact)
    except Exception as error:
        raise RuntimeError(f"{stage} finalized gate artifact is unreadable") from error
    required = {"candidate", "passed", "selected"}
    if gate.empty or not required.issubset(gate.columns):
        raise RuntimeError(f"{stage} finalized gate artifact is invalid")
    for column in ("passed", "selected"):
        normalized = {
            str(value).strip().lower() for value in gate[column].dropna()
        }
        if not normalized or not normalized.issubset({"true", "false", "1", "0"}):
            raise RuntimeError(f"{stage} finalized gate artifact is invalid")

    selected = gate.loc[
        gate["selected"].astype(str).str.strip().str.lower().isin({"true", "1"})
    ]
    if len(selected) != 1:
        raise RuntimeError(f"{stage} finalized gate must contain exactly one selection")
    row = selected.iloc[0]
    if str(row["passed"]).strip().lower() not in {"true", "1"}:
        raise RuntimeError(f"{stage} finalized selection did not pass its gate")
    candidate = str(row["candidate"])
    if candidate not in allowed_candidates:
        raise RuntimeError(f"{stage} finalized gate selected an unknown candidate")
    return candidate


def _resume_selected_uncertainty(
    store: Phase05Store,
    jobs: tuple[Phase05Job, ...],
) -> str | None:
    stage = jobs[0].shard.stage
    artifact = _validate_finalized_stage_cells(
        store,
        jobs,
        artifact_name="uncertainty_gate.csv",
    )
    if artifact is None:
        return None
    try:
        gate = pd.read_csv(artifact)
    except Exception as error:
        raise RuntimeError("uncertainty finalized artifact is unreadable") from error
    required = {"candidate", "selected"}
    if gate.empty or not required.issubset(gate.columns):
        raise RuntimeError("uncertainty finalized artifact is invalid")
    normalized = {
        str(value).strip().lower() for value in gate["selected"].dropna()
    }
    if not normalized or not normalized.issubset({"true", "false", "1", "0"}):
        raise RuntimeError("uncertainty finalized artifact is invalid")
    selected = gate.loc[
        gate["selected"].astype(str).str.strip().str.lower().isin({"true", "1"})
    ]
    if len(selected) != 1:
        raise RuntimeError("uncertainty finalized artifact must contain exactly one selection")
    candidate = str(selected.iloc[0]["candidate"])
    if candidate not in {"joint", "decoupled", "deterministic"}:
        raise RuntimeError("uncertainty finalized artifact selected an unknown candidate")
    if stage != "uncertainty":
        raise RuntimeError("uncertainty resume validator received the wrong stage")
    return candidate


def _phase05_develop(args: argparse.Namespace) -> int:
    config = load_phase05_config(args.exp_config)
    simulator_config, _ = _simulator_from_yaml(args.sim_config)
    output = Path(args.output)
    lock, store = _phase05_locked_store(output, config)
    options = Phase05ExecutionOptions(
        device=args.device,
        workers=args.workers,
        resume=args.resume,
        fail_fast=args.fail_fast,
    )

    flow_jobs = _phase05_development_jobs(config, "flow")
    selected_flow = None
    if args.resume:
        selected_flow = _resume_selected_development_gate(
            store,
            flow_jobs,
            artifact_name="flow_gate.csv",
            allowed_candidates=frozenset({"gated", "time_scaled"}),
        )
    if selected_flow is None:
        flow_metrics = _run_development_stage(
            flow_jobs,
            store=store,
            config=config,
            options=options,
            simulator_config=simulator_config,
        )
        flow = select_flow(
            flow_metrics,
            locked_min_relative_effect=float(lock["locked_min_relative_effect"]),
        )
        flow_gate = flow["gate_table"]
        if not isinstance(flow_gate, pd.DataFrame):
            raise TypeError("flow selection did not produce a gate table")
        store.replace_development_artifact("flow_gate.csv", _frame_csv_bytes(flow_gate))
        if not bool(flow["passed"]):
            raise RuntimeError("flow development gate failed; development stopped")
        selected_flow = str(flow["selected"])

    jump_jobs = _phase05_development_jobs(
        config,
        "jump",
        selected_flow=selected_flow,
    )
    selected_jump = None
    if args.resume:
        selected_jump = _resume_selected_development_gate(
            store,
            jump_jobs,
            artifact_name="jump_gate.csv",
            allowed_candidates=frozenset({"gru", "residual"}),
        )
    if selected_jump is None:
        jump_metrics = _run_development_stage(
            jump_jobs,
            store=store,
            config=config,
            options=options,
            simulator_config=simulator_config,
        )
        jump = select_jump(
            jump_metrics,
            selected_flow=selected_flow,
            flow_passed=True,
            locked_min_relative_effect=float(lock["locked_min_relative_effect"]),
        )
        jump_gate = jump["gate_table"]
        if not isinstance(jump_gate, pd.DataFrame):
            raise TypeError("jump selection did not produce a gate table")
        store.replace_development_artifact("jump_gate.csv", _frame_csv_bytes(jump_gate))
        if not bool(jump["passed"]):
            raise RuntimeError("jump development gate failed; development stopped")
        selected_jump = str(jump["selected"])

    uncertainty_jobs = _phase05_development_jobs(
        config,
        "uncertainty",
        selected_flow=selected_flow,
        selected_jump=selected_jump,
    )
    selected_uncertainty = None
    if args.resume:
        selected_uncertainty = _resume_selected_uncertainty(store, uncertainty_jobs)
    if selected_uncertainty is None:
        uncertainty_metrics = _run_development_stage(
            uncertainty_jobs,
            store=store,
            config=config,
            options=options,
            simulator_config=simulator_config,
        )
        uncertainty = select_uncertainty(
            uncertainty_metrics,
            selected_flow=selected_flow,
            selected_jump=selected_jump,
            locked_uncertainty_mae_tolerance=float(
                lock["locked_uncertainty_mae_tolerance"]
            ),
        )
        uncertainty_gate = _uncertainty_gate_table(
            uncertainty_metrics,
            selected_flow=selected_flow,
            selected_jump=selected_jump,
            selection=uncertainty,
        )
        store.replace_development_artifact(
            "uncertainty_gate.csv",
            _frame_csv_bytes(uncertainty_gate),
        )

    timing_jobs = _phase05_development_jobs(
        config,
        "timing_audit",
        selected_flow=selected_flow,
        selected_jump=selected_jump,
    )
    timing_finalized = False
    if args.resume:
        timing_finalized = (
            _validate_finalized_stage_cells(
                store,
                timing_jobs,
                artifact_name="representation_timing_audit.csv",
            )
            is not None
        )
    if not timing_finalized:
        timing_metrics = _run_development_stage(
            timing_jobs,
            store=store,
            config=config,
            options=options,
            simulator_config=simulator_config,
        )
        timing_audit = _timing_audit_table(
            timing_metrics,
            config,
            selected_flow=selected_flow,
            selected_jump=selected_jump,
        )
        store.replace_development_artifact(
            "representation_timing_audit.csv",
            _frame_csv_bytes(timing_audit),
        )

    from afmc_fm.phase05.development_outputs import (
        persist_development_mechanism_metrics,
    )

    persist_development_mechanism_metrics(
        store,
        (flow_jobs, jump_jobs, uncertainty_jobs),
    )
    return 0


def _phase05_freeze(args: argparse.Namespace) -> int:
    config = load_phase05_config(args.exp_config)
    freeze_candidate(args.output, config)
    return 0


def _phase05_confirm(args: argparse.Namespace) -> int:
    from afmc_fm.phase05.cli_confirmation import run_phase05_confirmation_cli

    return run_phase05_confirmation_cli(
        args,
        run_jobs=run_phase05_jobs,
        locked_store=_phase05_locked_store,
        simulator_from_yaml=_simulator_from_yaml,
    )


def _phase05_robustness(args: argparse.Namespace) -> int:
    from afmc_fm.phase05.cli_robustness import run_phase05_robustness_cli

    return run_phase05_robustness_cli(
        args,
        locked_store=_phase05_locked_store,
    )


def _phase05_report(args: argparse.Namespace) -> int:
    from afmc_fm.phase05.reporting import write_phase05_report_artifacts

    write_phase05_report_artifacts(args.output)
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
        if name == "develop":
            stage.set_defaults(handler=_phase05_develop)
        elif name == "confirm":
            stage.set_defaults(handler=_phase05_confirm)
        elif name == "robustness":
            stage.set_defaults(handler=_phase05_robustness)

    freeze = phase05_subparsers.add_parser("freeze")
    freeze.add_argument("--exp-config", required=True)
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(handler=_phase05_freeze)

    report = phase05_subparsers.add_parser("report")
    report.add_argument("--output", required=True)
    report.set_defaults(handler=_phase05_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())