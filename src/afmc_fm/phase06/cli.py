from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from afmc_fm.config import load_yaml
from afmc_fm.execution.manifest import execution_commit_sha
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import Phase05Config, load_phase05_config
from afmc_fm.phase06.analysis import adjudicate_phase06, analyze_d1, analyze_d2a
from afmc_fm.phase06.config import Phase06Config, load_phase06_config
from afmc_fm.phase06.execution import run_phase06_stage
from afmc_fm.phase06.pipeline import (
    load_stage_summaries,
    load_stage_traces,
    open_bound_store,
    require_completed_stage,
    write_analysis_csv,
    write_analysis_json,
)
from afmc_fm.phase06.planning import plan_d1_cells, plan_d2a_cells
from afmc_fm.phase06.protocol import build_phase06_protocol_lock
from afmc_fm.phase06.store import Phase06Store
from afmc_fm.simulator.config import SimulatorConfig

_PHASE06_CONFIG_PATH = Path("configs/experiments/phase06.yaml")
_PHASE05_PROTOCOL_PATH = Path(
    "docs/results/phase05/raw/official_output/protocol_lock.json"
)


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


def _load_simulator_config(path: str | Path) -> SimulatorConfig:
    raw = dict(load_yaml(path))
    raw.pop("seed", None)
    return SimulatorConfig(**raw)


def _bind_store(
    config: Phase06Config,
    phase05_config: Phase05Config,
    output: str | Path,
) -> Phase06Store:
    execution_commit = execution_commit_sha()
    lock = build_phase06_protocol_lock(
        config,
        phase05_config,
        execution_commit=execution_commit,
        phase06_spec_path=config.phase06_spec,
        phase05_protocol_path=_PHASE05_PROTOCOL_PATH,
    )
    protocol_hash = hashlib.sha256(_canonical_json_bytes(lock)).hexdigest()
    store = Phase06Store(
        Path(output),
        protocol_hash=protocol_hash,
        config_hash=canonical_config_hash(config),
        execution_commit=execution_commit,
    )
    store.write_protocol_lock(lock)
    return store


def _load_bound_inputs(args: argparse.Namespace) -> tuple[
    Phase06Config,
    Phase05Config,
    Phase06Store,
]:
    config = load_phase06_config(args.config)
    phase05_config = load_phase05_config(config.phase05_config)
    store = _bind_store(config, phase05_config, args.output)
    return config, phase05_config, store


def _persist_d1_analysis(
    store: Phase06Store,
    cells,
) -> tuple[object, object]:
    require_completed_stage(store, cells)
    metrics = store.load_stage_metrics("d1")
    summaries = load_stage_summaries(store, cells)
    table, decision = analyze_d1(metrics, summaries)
    write_analysis_csv(store, "phase06_d1_reproduction.csv", table)
    write_analysis_json(store, "phase06_d1_classification.json", decision)
    return table, decision


def _persist_d2_analysis(
    store: Phase06Store,
    cells,
    config: Phase06Config,
):
    require_completed_stage(store, cells)
    result = analyze_d2a(store.load_stage_metrics("d2a"), config)
    write_analysis_csv(store, "phase06_d2_effects.csv", result.effect_rows)
    write_analysis_csv(
        store,
        "phase06_d2_factor_level_effects.csv",
        result.factor_level_effects,
    )
    write_analysis_csv(
        store,
        "phase06_d2_variance_components.csv",
        result.variance_components,
    )
    write_analysis_csv(
        store,
        "phase06_d2_bootstrap_diagnostics.csv",
        result.bootstrap_diagnostics,
    )
    write_analysis_csv(store, "phase06_d2_n_shift.csv", result.n_shift_rows)
    write_analysis_json(
        store,
        "phase06_d2_n_shift_summary.json",
        result.n_shift_summary,
    )
    return result


def _run_d1(args: argparse.Namespace) -> int:
    config, phase05_config, store = _load_bound_inputs(args)
    cells = plan_d1_cells(config, phase05_config)
    simulator_config = _load_simulator_config(config.simulator_config)
    run_phase06_stage(
        cells,
        store,
        phase05_config,
        simulator_config,
        device=args.device,
        resume=args.resume,
    )
    _persist_d1_analysis(store, cells)
    return 0


def _run_d2a(args: argparse.Namespace) -> int:
    config, phase05_config, store = _load_bound_inputs(args)
    cells = plan_d2a_cells(config)
    simulator_config = _load_simulator_config(config.simulator_config)
    run_phase06_stage(
        cells,
        store,
        phase05_config,
        simulator_config,
        device=args.device,
        resume=args.resume,
    )
    _persist_d2_analysis(store, cells, config)
    return 0


def _load_adjudication_protocol(store: Phase06Store) -> dict[str, object]:
    path = store.output / "protocol_lock.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Phase 0.6 protocol lock is not valid JSON") from error
    if not isinstance(payload, dict):
        raise TypeError("Phase 0.6 protocol lock must be a JSON object")
    return payload


def _adjudicate(args: argparse.Namespace) -> int:
    store = open_bound_store(args.output)
    if execution_commit_sha() != store.execution_commit:
        raise ValueError("adjudication checkout does not match the hash-bound store identity")

    config = load_phase06_config(_PHASE06_CONFIG_PATH)
    if canonical_config_hash(config) != store.config_hash:
        raise ValueError("adjudication config does not match the hash-bound store identity")
    phase05_config = load_phase05_config(config.phase05_config)
    protocol = _load_adjudication_protocol(store)
    if protocol.get("phase05_config_sha256") != canonical_config_hash(phase05_config):
        raise ValueError("adjudication Phase 0.5 config does not match the protocol lock")

    d1_cells = plan_d1_cells(config, phase05_config)
    d2_cells = plan_d2a_cells(config)
    require_completed_stage(store, d1_cells)
    require_completed_stage(store, d2_cells)

    d1_result = analyze_d1(
        store.load_stage_metrics("d1"),
        load_stage_summaries(store, d1_cells),
    )
    d2_result = analyze_d2a(store.load_stage_metrics("d2a"), config)
    traces = load_stage_traces(store, d1_cells)
    result = adjudicate_phase06(d1_result, d2_result, traces)
    write_analysis_json(store, "phase06_d3_adjudication.json", result)
    print(result["next_required_stage"])
    return 0


def _add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afmc-phase06")
    subparsers = parser.add_subparsers(dest="command", required=True)

    d1 = subparsers.add_parser("d1")
    _add_stage_arguments(d1)
    d1.set_defaults(handler=_run_d1)

    d2a = subparsers.add_parser("d2a")
    _add_stage_arguments(d2a)
    d2a.set_defaults(handler=_run_d2a)

    adjudicate = subparsers.add_parser("adjudicate")
    adjudicate.add_argument("--output", required=True)
    adjudicate.set_defaults(handler=_adjudicate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
