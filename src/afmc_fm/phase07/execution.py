from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TypeVar

import torch

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.model import Phase05FlowJumpAdapter
from afmc_fm.phase05.training import fit_phase05_model
from afmc_fm.phase06.diagnostics import Phase06DiagnosticRecorder
from afmc_fm.phase07.config import Phase07Config
from afmc_fm.phase07.planning import (
    Phase07CellSpec,
    phase07_plan_sha256,
    plan_phase07_cells,
)
from afmc_fm.phase07.protocol import build_phase07_protocol_lock

_EXPECTED_CELL_COUNT = 200
_AUTHORIZATION_VALUE = "OFFICIAL_EXECUTION_AUTHORIZED"
_T = TypeVar("_T")


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


def build_phase07_execution_manifest(
    config: Phase07Config,
    *,
    execution_commit: str,
    phase07_spec_path: str | Path,
) -> dict[str, object]:
    if not isinstance(config, Phase07Config):
        raise TypeError("config must be a Phase07Config")

    cells = plan_phase07_cells(config)
    protocol_lock = build_phase07_protocol_lock(
        config,
        execution_commit=execution_commit,
        phase07_spec_path=phase07_spec_path,
    )
    protocol_hash = hashlib.sha256(_canonical_json_bytes(protocol_lock)).hexdigest()
    plan_hash = phase07_plan_sha256(cells)
    if len(cells) != _EXPECTED_CELL_COUNT:
        raise ValueError("Phase 0.7 execution manifest requires exactly 200 cells")

    return {
        "schema_version": 1,
        "phase": "phase07",
        "execution_commit": execution_commit,
        "phase07_spec_sha256": protocol_lock["phase07_spec_sha256"],
        "phase07_config_sha256": canonical_config_hash(config),
        "protocol_lock_sha256": protocol_hash,
        "phase07_plan_sha256": plan_hash,
        "expected_cell_count": len(cells),
        "forbidden_seed_sets": {
            "cohort": list(config.forbidden_cohort_seeds),
            "subset": list(config.forbidden_subset_seeds),
            "model": list(config.forbidden_model_seeds),
        },
    }


def _expected_authorization(manifest: dict[str, object]) -> dict[str, object]:
    required = (
        "execution_commit",
        "protocol_lock_sha256",
        "phase07_plan_sha256",
    )
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(
            "execution manifest is missing authorization identity: "
            + ", ".join(missing)
        )
    return {
        "schema_version": 1,
        "phase": "phase07",
        "authorization": _AUTHORIZATION_VALUE,
        "execution_commit": manifest["execution_commit"],
        "protocol_lock_sha256": manifest["protocol_lock_sha256"],
        "phase07_plan_sha256": manifest["phase07_plan_sha256"],
    }


def require_phase07_official_authorization(
    authorization_path: str | Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    path = Path(authorization_path)
    if not path.is_file():
        raise ValueError(f"Phase 0.7 official execution authorization is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Phase 0.7 official execution authorization is invalid") from error
    if not isinstance(payload, dict):
        raise TypeError("Phase 0.7 official execution authorization must be a JSON object")

    expected = _expected_authorization(manifest)
    if payload != expected:
        raise ValueError(
            "Phase 0.7 official execution authorization does not match the frozen identity"
        )
    return payload


def run_phase07_official_cells(
    cells: Sequence[Phase07CellSpec],
    *,
    config: Phase07Config,
    execution_commit: str,
    phase07_spec_path: str | Path,
    authorization_path: str | Path,
    execute_cell: Callable[[Phase07CellSpec], _T],
) -> tuple[_T, ...]:
    if not callable(execute_cell):
        raise TypeError("execute_cell must be callable")
    manifest = build_phase07_execution_manifest(
        config,
        execution_commit=execution_commit,
        phase07_spec_path=phase07_spec_path,
    )
    supplied_plan_hash = phase07_plan_sha256(cells)
    if supplied_plan_hash != manifest["phase07_plan_sha256"]:
        raise ValueError("Phase 0.7 official cell plan does not match the frozen plan")

    require_phase07_official_authorization(authorization_path, manifest)
    return tuple(execute_cell(cell) for cell in cells)


def _smoke_model() -> Phase05FlowJumpAdapter:
    torch.manual_seed(41)
    return Phase05FlowJumpAdapter(
        representation_dim=16,
        value_dim=3,
        event_dim=3,
        state_dim=24,
        flow_mode="time_scaled",
        jump_mode="none",
        uncertainty_mode="deterministic",
        time_scale_days=30.0,
    )


def _smoke_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(73)
    batch = 3
    steps = 4
    value_dim = 3
    return {
        "representations": torch.randn(batch, steps, 16),
        "values": torch.randn(batch, steps, value_dim),
        "masks": torch.ones(batch, steps, value_dim),
        "event_features": torch.randn(batch, steps, 3),
        "times": torch.tensor(
            [
                [0.0, 2.0, 8.0, 14.0],
                [0.0, 3.0, 7.0, 20.0],
                [0.0, 1.0, 9.0, 18.0],
            ]
        ),
        "update_mask": torch.ones(batch, steps),
        "jump_eligible_mask": torch.zeros(batch, steps),
        "target_values": torch.randn(batch, steps, value_dim),
        "target_masks": torch.ones(batch, steps, value_dim),
        "target_events": torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ),
        "event_valid": torch.ones(batch, steps),
        "valid": torch.ones(batch, steps, dtype=torch.bool),
    }


def run_phase07_device_smoke(device: str) -> dict[str, object]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the Phase 0.7 validation smoke")

    resolved = torch.device(device)
    config = Phase05Config(
        max_epochs=4,
        patience=1,
        learning_rate=1e-30,
        weight_decay=0.0,
    )
    batch = _smoke_batch()
    initial = _smoke_model()
    standard_model = deepcopy(initial)
    forced_model = deepcopy(initial)
    standard = Phase06DiagnosticRecorder()
    forced = Phase06DiagnosticRecorder()

    fit_phase05_model(
        standard_model,
        deepcopy(batch),
        deepcopy(batch),
        config,
        resolved,
        diagnostics=standard,
        stop_on_patience=True,
    )
    fit_phase05_model(
        forced_model,
        deepcopy(batch),
        deepcopy(batch),
        config,
        resolved,
        diagnostics=forced,
        stop_on_patience=False,
    )
    if device == "cuda":
        torch.cuda.synchronize(resolved)

    standard_trace = standard.trace_frame().reset_index(drop=True)
    forced_trace = forced.trace_frame().reset_index(drop=True)
    forced_prefix = forced_trace.iloc[: len(standard_trace)].reset_index(drop=True)
    prefix_identical = standard_trace.equals(forced_prefix)
    standard_summary = standard.summary_payload()
    forced_summary = forced.summary_payload()
    if not prefix_identical:
        raise RuntimeError("Phase 0.7 validation smoke detected paired-prefix divergence")
    if int(forced_summary["epochs_run"]) < int(standard_summary["epochs_run"]):
        raise RuntimeError("forced horizon terminated before the standard policy")

    return {
        "mode": "non_official_validation_smoke",
        "device": device,
        "official_cells_executed": 0,
        "standard_epochs_run": int(standard_summary["epochs_run"]),
        "forced_epochs_run": int(forced_summary["epochs_run"]),
        "standard_stop_reason": str(standard_summary["early_stop_reason"]),
        "forced_stop_reason": str(forced_summary["early_stop_reason"]),
        "prefix_identical": prefix_identical,
    }


__all__ = [
    "build_phase07_execution_manifest",
    "require_phase07_official_authorization",
    "run_phase07_device_smoke",
    "run_phase07_official_cells",
]
