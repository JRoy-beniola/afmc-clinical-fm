from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from afmc_fm.config import load_yaml
from afmc_fm.execution.manifest import execution_commit_sha
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase07.config import Phase07Config, load_phase07_config
from afmc_fm.phase07.execution import (
    build_phase07_execution_manifest,
    require_phase07_official_authorization,
    run_phase07_device_smoke,
    run_phase07_official_cells,
)
from afmc_fm.phase07.planning import Phase07CellSpec, phase07_plan_sha256, plan_phase07_cells
from afmc_fm.phase07.protocol import build_phase07_protocol_lock
from afmc_fm.phase07.runner import prepare_phase07_cohort, run_phase07_cell
from afmc_fm.simulator.config import SimulatorConfig

_CONFIG_PATH = Path("configs/experiments/phase07.yaml")


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


def _persist_official_cell(output: Path, cell: Phase07CellSpec, result) -> None:
    cell_dir = output / "cells" / cell.cell_id
    if cell_dir.exists():
        raise ValueError(f"official Phase 0.7 cell output already exists: {cell.cell_id}")
    cell_dir.mkdir(parents=True)
    result.metrics.to_csv(cell_dir / "metrics.csv", index=False)
    result.trace.to_csv(cell_dir / "training_trace.csv", index=False)
    _write_json(cell_dir / "summary.json", result.summary)
    _write_json(cell_dir / "cell.json", _cell_payload(cell))
    torch.save(result.production_state_dict, cell_dir / "production_checkpoint.pt")
    torch.save(result.shadow_state_dict, cell_dir / "shadow_mae_checkpoint.pt")


def execute_authorized_phase07(
    *,
    config: Phase07Config,
    manifest: dict[str, object],
    authorization_path: str | Path,
    output: str | Path,
    device: str,
) -> tuple[str, ...]:
    require_phase07_official_authorization(authorization_path, manifest)
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for official Phase 0.7 execution")

    output_root = Path(output)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("official Phase 0.7 output directory must be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "execution_manifest.json", manifest)
    protocol_lock = build_phase07_protocol_lock(
        config,
        execution_commit=str(manifest["execution_commit"]),
        phase07_spec_path=config.phase07_spec,
    )
    _write_json(output_root / "protocol_lock.json", protocol_lock)

    phase05_config = load_phase05_config(config.phase05_config)
    if phase05_config.max_epochs != config.max_epochs or phase05_config.patience != config.patience:
        raise ValueError("Phase 0.5 training constants do not match the frozen Phase 0.7 protocol")
    simulator_config = _load_simulator_config(config.simulator_config)
    cells = plan_phase07_cells(config)
    resolved_device = torch.device(device)
    prepared_by_cohort: dict[int, object] = {}

    def execute_cell(cell: Phase07CellSpec) -> str:
        prepared = prepared_by_cohort.get(cell.cohort_seed)
        if prepared is None:
            prepared = prepare_phase07_cohort(simulator_config, cell)
            prepared_by_cohort[cell.cohort_seed] = prepared
        result = run_phase07_cell(prepared, phase05_config, cell, resolved_device)
        _persist_official_cell(output_root, cell, result)
        return cell.cell_id

    completed = run_phase07_official_cells(
        cells,
        config=config,
        execution_commit=str(manifest["execution_commit"]),
        phase07_spec_path=config.phase07_spec,
        authorization_path=authorization_path,
        execute_cell=execute_cell,
    )
    metrics = [pd.read_csv(output_root / "cells" / cell_id / "metrics.csv") for cell_id in completed]
    pd.concat(metrics, ignore_index=True, sort=False).to_csv(
        output_root / "phase07_metrics.csv",
        index=False,
    )
    _write_json(
        output_root / "COMPLETE",
        {
            "schema_version": 1,
            "phase": "phase07",
            "execution_commit": manifest["execution_commit"],
            "phase07_plan_sha256": manifest["phase07_plan_sha256"],
            "completed_cell_count": len(completed),
            "cell_ids": list(completed),
        },
    )
    return completed


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

    manifest = build_phase07_execution_manifest(
        config,
        execution_commit=execution_commit,
        phase07_spec_path=config.phase07_spec,
    )
    if args.command == "manifest":
        _write_json(args.output, manifest)
        return 0

    if args.command == "official":
        require_phase07_official_authorization(args.authorization, manifest)
        execute_authorized_phase07(
            config=config,
            manifest=manifest,
            authorization_path=args.authorization,
            output=args.output,
            device=args.device,
        )
        return 0

    raise RuntimeError(f"unhandled Phase 0.7 command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
