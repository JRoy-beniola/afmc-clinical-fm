from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from afmc_fm.config import load_yaml
from afmc_fm.execution.manifest import execution_commit_sha
from afmc_fm.phase05.config import Phase05Config, load_phase05_config
from afmc_fm.phase07.analysis import (
    adjudicate_phase07,
    build_phase07_pair_table,
    statistics_from_phase07_pairs,
)
from afmc_fm.phase07.config import Phase07Config, load_phase07_config
from afmc_fm.phase07.execution import (
    build_phase07_execution_manifest,
    load_completed_phase07_metrics,
    require_clean_phase07_checkout,
    require_phase07_official_authorization,
    run_phase07_device_smoke,
    run_phase07_official_cells,
)
from afmc_fm.phase07.planning import Phase07CellSpec, phase07_plan_sha256, plan_phase07_cells
from afmc_fm.phase07.protocol import build_phase07_protocol_lock
from afmc_fm.phase07.provenance import Phase07ExecutionProvenance
from afmc_fm.phase07.runner import prepare_phase07_cohort, run_phase07_cell
from afmc_fm.phase07.store import Phase07Store
from afmc_fm.simulator.config import SimulatorConfig

_CONFIG_PATH = Path("configs/experiments/phase07.yaml")
_STORE_IDENTITY_KEYS = (
    "execution_commit",
    "phase07_spec_sha256",
    "phase07_config_sha256",
    "phase05_config_sha256",
    "simulator_config_sha256",
    "protocol_lock_sha256",
    "phase07_plan_sha256",
    "expected_cell_count",
)


def _write_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cell_payload(cell: Phase07CellSpec) -> dict[str, object]:
    return {
        "world": cell.world,
        "cohort_seed": cell.cohort_seed,
        "subset_seed": cell.subset_seed,
        "model_seed": cell.model_seed,
        "n_train": cell.n_train,
        "flow_mode": cell.flow_mode,
        "optimization_policy": cell.optimization_policy,
        "cell_id": cell.cell_id,
    }


def _plan_payload(config: Phase07Config) -> dict[str, object]:
    cells = plan_phase07_cells(config)
    return {
        "schema_version": 1,
        "phase": "phase07",
        "phase07_plan_sha256": phase07_plan_sha256(cells),
        "expected_cell_count": len(cells),
        "cells": [_cell_payload(cell) for cell in cells],
    }


def _load_simulator_config(path: str | Path) -> SimulatorConfig:
    raw = dict(load_yaml(path))
    raw.pop("seed", None)
    return SimulatorConfig(**raw)


def _store_identity(manifest: dict[str, object]) -> dict[str, object]:
    missing = [key for key in _STORE_IDENTITY_KEYS if key not in manifest]
    if missing:
        raise ValueError(
            "Phase 0.7 execution manifest is missing store identity: "
            + ", ".join(missing)
        )
    return {key: manifest[key] for key in _STORE_IDENTITY_KEYS}


def _initialize_store(
    *,
    config: Phase07Config,
    manifest: dict[str, object],
    output: str | Path,
    resume: bool,
) -> tuple[Phase07Store, tuple[Phase07CellSpec, ...]]:
    cells = plan_phase07_cells(config)
    protocol_lock = build_phase07_protocol_lock(
        config,
        execution_commit=str(manifest["execution_commit"]),
        phase07_spec_path=config.phase07_spec,
    )
    store = Phase07Store(output, identity=_store_identity(manifest))
    store.initialize(
        manifest,
        protocol_lock,
        _plan_payload(config),
        resume=resume,
    )
    return store, cells


def _write_aggregate_metrics_atomic(output: str | Path, metrics) -> None:
    destination = Path(output) / "phase07_metrics.csv"
    temporary = destination.with_name(destination.name + ".tmp")
    metrics.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def execute_authorized_phase07(
    *,
    config: Phase07Config,
    phase05_config: Phase05Config,
    simulator_config: SimulatorConfig,
    manifest: dict[str, object],
    authorization_path: str | Path,
    output: str | Path,
    device: str,
    resume: bool = False,
) -> tuple[str, ...]:
    if not isinstance(phase05_config, Phase05Config):
        raise TypeError("phase05_config must be a Phase05Config")
    if not isinstance(simulator_config, SimulatorConfig):
        raise TypeError("simulator_config must be a SimulatorConfig")
    if type(resume) is not bool:
        raise TypeError("resume must be a bool")
    expected_manifest = build_phase07_execution_manifest(
        config,
        execution_commit=str(manifest.get("execution_commit", "")),
        phase07_spec_path=config.phase07_spec,
        phase05_config=phase05_config,
        simulator_config=simulator_config,
    )
    if manifest != expected_manifest:
        raise ValueError("Phase 0.7 supplied execution manifest does not match loaded configuration")

    require_clean_phase07_checkout()
    if execution_commit_sha() != manifest["execution_commit"]:
        raise ValueError("Phase 0.7 execution checkout does not match the authorized commit")
    require_phase07_official_authorization(authorization_path, manifest)
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")

    if phase05_config.max_epochs != config.max_epochs or phase05_config.patience != config.patience:
        raise ValueError("Phase 0.5 training constants do not match the frozen Phase 0.7 protocol")
    store, cells = _initialize_store(
        config=config,
        manifest=manifest,
        output=output,
        resume=resume,
    )
    completed_before = store.validate_resume(cells)
    provenance = Phase07ExecutionProvenance(
        output,
        identity=_store_identity(manifest),
        device=device,
        planned_cell_count=len(cells),
    )
    provenance.start(completed_before=len(completed_before))

    try:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for official Phase 0.7 execution")
        resolved_device = torch.device(device)
        prepared_by_cohort: dict[int, object] = {}

        def execute_cell(cell: Phase07CellSpec):
            provenance.attempt(cell.cell_id)
            prepared = prepared_by_cohort.get(cell.cohort_seed)
            if prepared is None:
                prepared = prepare_phase07_cohort(simulator_config, cell)
                prepared_by_cohort[cell.cohort_seed] = prepared
            return run_phase07_cell(prepared, phase05_config, cell, resolved_device)

        completed = run_phase07_official_cells(
            cells,
            config=config,
            phase05_config=phase05_config,
            simulator_config=simulator_config,
            execution_commit=str(manifest["execution_commit"]),
            phase07_spec_path=config.phase07_spec,
            authorization_path=authorization_path,
            execute_cell=execute_cell,
            store=store,
            resume=resume,
        )

        metrics = store.load_metrics(cells)
        build_phase07_pair_table(metrics)
        _write_aggregate_metrics_atomic(output, metrics)
        store.mark_complete(cells)
        load_completed_phase07_metrics(store, cells)
    except BaseException as error:
        try:
            completed_after = len(store.validate_resume(cells))
        except (OSError, TypeError, ValueError):
            completed_after = len(completed_before)
        provenance.fail(completed_after=completed_after, error=error)
        raise

    provenance.finish(completed_after=len(cells))
    return completed


def analyze_completed_phase07(
    *,
    config: Phase07Config,
    manifest: dict[str, object],
    output: str | Path,
) -> dict[str, object]:
    require_clean_phase07_checkout()
    if execution_commit_sha() != manifest["execution_commit"]:
        raise ValueError("Phase 0.7 analysis checkout does not match the execution commit")
    store, cells = _initialize_store(
        config=config,
        manifest=manifest,
        output=output,
        resume=True,
    )
    metrics = load_completed_phase07_metrics(store, cells)
    pairs = build_phase07_pair_table(metrics)
    statistics = statistics_from_phase07_pairs(pairs)
    return {
        "schema_version": 1,
        "phase": "phase07",
        "execution_commit": manifest["execution_commit"],
        "phase07_plan_sha256": manifest["phase07_plan_sha256"],
        "statistics": asdict(statistics),
        "adjudication": adjudicate_phase07(statistics),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afmc-phase07")
    subparsers = parser.add_subparsers(dest="command", required=True)

    protocol = subparsers.add_parser("protocol")
    protocol.add_argument("--config", default=str(_CONFIG_PATH))
    protocol.add_argument("--output", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", default=str(_CONFIG_PATH))
    plan.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--config", default=str(_CONFIG_PATH))
    verify.add_argument("--plan", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--config", default=str(_CONFIG_PATH))
    manifest.add_argument("--output", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    official = subparsers.add_parser("official")
    official.add_argument("--config", default=str(_CONFIG_PATH))
    official.add_argument("--authorization", required=True)
    official.add_argument("--output", required=True)
    official.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    official.add_argument("--resume", action="store_true")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--config", default=str(_CONFIG_PATH))
    analyze.add_argument("--output", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "smoke":
        print(json.dumps(run_phase07_device_smoke(args.device), sort_keys=True))
        return 0

    config = load_phase07_config(args.config)
    execution_commit = execution_commit_sha()

    if args.command == "protocol":
        lock = build_phase07_protocol_lock(
            config,
            execution_commit=execution_commit,
            phase07_spec_path=config.phase07_spec,
        )
        _write_json(args.output, lock)
        return 0

    if args.command == "plan":
        _write_json(args.output, _plan_payload(config))
        return 0

    if args.command == "verify":
        path = Path(args.plan)
        if not path.is_file():
            raise ValueError(f"Phase 0.7 plan is missing: {path}")
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Phase 0.7 plan is invalid JSON") from error
        expected = _plan_payload(config)
        if observed != expected:
            raise ValueError("Phase 0.7 plan does not match the frozen 200-cell plan")
        return 0

    phase05_config = load_phase05_config(config.phase05_config)
    simulator_config = _load_simulator_config(config.simulator_config)
    manifest = build_phase07_execution_manifest(
        config,
        execution_commit=execution_commit,
        phase07_spec_path=config.phase07_spec,
        phase05_config=phase05_config,
        simulator_config=simulator_config,
    )
    if args.command == "manifest":
        _write_json(args.output, manifest)
        return 0

    if args.command == "official":
        require_phase07_official_authorization(args.authorization, manifest)
        execute_authorized_phase07(
            config=config,
            phase05_config=phase05_config,
            simulator_config=simulator_config,
            manifest=manifest,
            authorization_path=args.authorization,
            output=args.output,
            device=args.device,
            resume=args.resume,
        )
        return 0

    if args.command == "analyze":
        print(
            json.dumps(
                analyze_completed_phase07(
                    config=config,
                    manifest=manifest,
                    output=args.output,
                ),
                sort_keys=True,
            )
        )
        return 0

    raise RuntimeError(f"unhandled Phase 0.7 command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
