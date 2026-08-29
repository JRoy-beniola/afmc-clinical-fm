from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

from afmc_fm.config import load_yaml
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import Phase05Config, load_phase05_config
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
from afmc_fm.phase07.runner import Phase07CellRun
from afmc_fm.phase07.store import Phase07Store
from afmc_fm.simulator.config import SimulatorConfig

_EXPECTED_CELL_COUNT = 200
_AUTHORIZATION_VALUE = "OFFICIAL_EXECUTION_AUTHORIZED"
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


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


def _resolve_dependency_configs(
    config: Phase07Config,
    *,
    phase05_config: Phase05Config | None,
    simulator_config: SimulatorConfig | None,
) -> tuple[Phase05Config, SimulatorConfig]:
    if phase05_config is None:
        phase05_config = load_phase05_config(config.phase05_config)
    elif not isinstance(phase05_config, Phase05Config):
        raise TypeError("phase05_config must be a Phase05Config")

    if simulator_config is None:
        simulator_config = _load_simulator_config(config.simulator_config)
    elif not isinstance(simulator_config, SimulatorConfig):
        raise TypeError("simulator_config must be a SimulatorConfig")

    return phase05_config, simulator_config


def _phase07_git_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _phase07_source_repository_root() -> Path:
    source = Path(__file__).resolve()
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=source.parent,
        env=_phase07_git_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    raw_root = result.stdout.strip()
    if not raw_root:
        raise RuntimeError("unable to resolve the Phase 0.7 source repository")
    root = Path(raw_root).resolve()
    if not source.is_relative_to(root):
        raise RuntimeError("Phase 0.7 source is not contained in its resolved Git repository")
    return root


def phase07_execution_commit_sha() -> str:
    """Return HEAD for the repository containing the imported Phase 0.7 source."""
    root = _phase07_source_repository_root()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        env=_phase07_git_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip().lower()
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("git rev-parse HEAD did not return a full commit SHA")
    return commit


def execution_commit_sha() -> str:
    """Compatibility wrapper for the Phase 0.7 source-anchored commit resolver."""
    return phase07_execution_commit_sha()


def require_clean_phase07_checkout() -> None:
    root = _phase07_source_repository_root()
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        env=_phase07_git_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ValueError("official Phase 0.7 execution requires a clean worktree")


def build_phase07_execution_manifest(
    config: Phase07Config,
    *,
    execution_commit: str,
    phase07_spec_path: str | Path,
    phase05_config: Phase05Config | None = None,
    simulator_config: SimulatorConfig | None = None,
) -> dict[str, object]:
    if not isinstance(config, Phase07Config):
        raise TypeError("config must be a Phase07Config")

    phase05_config, simulator_config = _resolve_dependency_configs(
        config,
        phase05_config=phase05_config,
        simulator_config=simulator_config,
    )
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
        "phase05_config_sha256": canonical_config_hash(phase05_config),
        "simulator_config_sha256": canonical_config_hash(simulator_config),
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
        "phase05_config_sha256",
        "simulator_config_sha256",
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
        "phase05_config_sha256": manifest["phase05_config_sha256"],
        "simulator_config_sha256": manifest["simulator_config_sha256"],
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


def _run_phase07_cells_with_store(
    cells: Sequence[Phase07CellSpec],
    *,
    store: Phase07Store,
    resume: bool,
    execute_cell: Callable[[Phase07CellSpec], Phase07CellRun],
) -> tuple[str, ...]:
    if not isinstance(store, Phase07Store):
        raise TypeError("store must be a Phase07Store")
    if type(resume) is not bool:
        raise TypeError("resume must be a bool")
    if not callable(execute_cell):
        raise TypeError("execute_cell must be callable")

    planned = tuple(cells)
    if not planned:
        raise ValueError("Phase 0.7 cell sequence must not be empty")
    if not all(isinstance(cell, Phase07CellSpec) for cell in planned):
        raise TypeError("Phase 0.7 cell sequence must contain Phase07CellSpec instances")
    if len({cell.cell_id for cell in planned}) != len(planned):
        raise ValueError("Phase 0.7 cell sequence contains duplicate cell IDs")

    completed_before = store.validate_resume(planned)
    if completed_before and not resume:
        raise ValueError("persisted Phase 0.7 cells exist; pass resume=True to continue")

    for cell in planned:
        if cell.cell_id in completed_before:
            continue
        result = execute_cell(cell)
        if not isinstance(result, Phase07CellRun):
            raise TypeError("execute_cell must return a Phase07CellRun")
        store.write_cell_bundle(
            cell,
            metrics=result.metrics,
            trace=result.trace,
            summary=result.summary,
            production_state_dict=result.production_state_dict,
            shadow_state_dict=result.shadow_state_dict,
        )

    completed_after = store.validate_resume(planned)
    expected_ids = frozenset(cell.cell_id for cell in planned)
    if completed_after != expected_ids:
        missing = sorted(expected_ids - completed_after)
        raise ValueError(
            "Phase 0.7 execution ended with missing expected cells: " + ", ".join(missing)
        )
    return tuple(cell.cell_id for cell in planned)


def load_completed_phase07_metrics(
    store: Phase07Store,
    expected_cells: Sequence[Phase07CellSpec],
) -> pd.DataFrame:
    if not isinstance(store, Phase07Store):
        raise TypeError("store must be a Phase07Store")
    planned = tuple(expected_cells)
    store.require_complete(planned)
    return store.load_metrics(planned)


def run_phase07_official_cells(
    cells: Sequence[Phase07CellSpec],
    *,
    config: Phase07Config,
    phase05_config: Phase05Config,
    simulator_config: SimulatorConfig,
    execution_commit: str,
    phase07_spec_path: str | Path,
    authorization_path: str | Path,
    execute_cell: Callable[[Phase07CellSpec], Phase07CellRun],
    store: Phase07Store | None = None,
    resume: bool = False,
) -> tuple[str, ...]:
    if not isinstance(phase05_config, Phase05Config):
        raise TypeError("phase05_config must be a Phase05Config")
    if not isinstance(simulator_config, SimulatorConfig):
        raise TypeError("simulator_config must be a SimulatorConfig")
    if not callable(execute_cell):
        raise TypeError("execute_cell must be callable")
    if type(resume) is not bool:
        raise TypeError("resume must be a bool")
    manifest = build_phase07_execution_manifest(
        config,
        execution_commit=execution_commit,
        phase07_spec_path=phase07_spec_path,
        phase05_config=phase05_config,
        simulator_config=simulator_config,
    )
    supplied_plan_hash = phase07_plan_sha256(cells)
    if supplied_plan_hash != manifest["phase07_plan_sha256"]:
        raise ValueError("Phase 0.7 official cell plan does not match the frozen plan")

    require_clean_phase07_checkout()
    if execution_commit_sha() != execution_commit:
        raise ValueError("Phase 0.7 execution checkout does not match the authorized commit")
    require_phase07_official_authorization(authorization_path, manifest)
    if store is None:
        raise ValueError("official Phase 0.7 execution requires a crash-resilient Phase07Store")
    return _run_phase07_cells_with_store(
        cells,
        store=store,
        resume=resume,
        execute_cell=execute_cell,
    )


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
    "execution_commit_sha",
    "load_completed_phase07_metrics",
    "phase07_execution_commit_sha",
    "require_clean_phase07_checkout",
    "require_phase07_official_authorization",
    "run_phase07_device_smoke",
    "run_phase07_official_cells",
]
