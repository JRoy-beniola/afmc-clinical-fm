from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from afmc_fm.execution.device import resolve_device
from afmc_fm.execution.manifest import collect_runtime_metadata, execution_commit_sha
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase06.planning import Phase06CellSpec
from afmc_fm.phase06.protocol import validate_development_seed_triplet
from afmc_fm.phase06.runner import (
    Phase06CellRun,
    prepare_phase06_cohort,
    run_phase06_cell,
)
from afmc_fm.phase06.store import Phase06Store
from afmc_fm.simulator.config import SimulatorConfig

PrepareCell = Callable[[SimulatorConfig, Phase06CellSpec], Any]
RunCell = Callable[[Any, Phase05Config, Phase06CellSpec, torch.device], Phase06CellRun]


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


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_json_bytes(payload)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_stage_cells(
    cells: Sequence[Phase06CellSpec],
) -> tuple[str, tuple[Phase06CellSpec, ...]]:
    planned = tuple(cells)
    if not planned:
        raise ValueError("Phase 0.6 stage plan must not be empty")
    if not all(isinstance(cell, Phase06CellSpec) for cell in planned):
        raise TypeError("cells must contain only Phase06CellSpec instances")

    stages = {cell.stage for cell in planned}
    if len(stages) != 1:
        raise ValueError("one run_phase06_stage call may execute only one stage")
    stage = next(iter(stages))

    cell_ids = [cell.cell_id for cell in planned]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Phase 0.6 stage plan contains duplicate cell IDs")

    for cell in planned:
        validate_development_seed_triplet(
            cell.cohort_seed,
            cell.subset_seed,
            cell.model_seed,
        )

    ordered = tuple(
        sorted(
            planned,
            key=lambda cell: (
                cell.cohort_seed,
                cell.subset_seed,
                cell.model_seed,
                cell.n_train,
                cell.flow_mode,
            ),
        )
    )
    return stage, ordered


def _execution_provenance(
    *,
    stage: str,
    planned_cells: int,
    completed_before: int,
    completed_after: int,
    failures: list[dict[str, str]],
    device: torch.device,
    started_at: datetime,
    ended_at: datetime,
    wall_seconds: float,
    store: Phase06Store,
) -> dict[str, object]:
    return {
        "stage": stage,
        "planned_cells": planned_cells,
        "completed_before": completed_before,
        "completed_after": completed_after,
        "failures": failures,
        "device": str(device),
        "workers": 1,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "wall_seconds": wall_seconds,
        "execution_commit": store.execution_commit,
        "protocol_lock_sha256": store.protocol_hash,
        "forbidden_seed_validation": "passed",
        "runtime": collect_runtime_metadata(str(device)),
    }


def _persist_provenance(
    store: Phase06Store,
    stage: str,
    payload: dict[str, object],
) -> None:
    _atomic_write_json(
        store.output / "stages" / stage / "execution_provenance.json",
        payload,
    )


def run_phase06_stage(
    cells: Sequence[Phase06CellSpec],
    store: Phase06Store,
    phase05_config: Phase05Config,
    simulator_config: SimulatorConfig,
    device: str = "cuda",
    resume: bool = False,
    *,
    prepare_cell: PrepareCell = prepare_phase06_cohort,
    run_cell: RunCell = run_phase06_cell,
) -> dict[str, object]:
    if not isinstance(store, Phase06Store):
        raise TypeError("store must be a Phase06Store")
    if not isinstance(phase05_config, Phase05Config):
        raise TypeError("phase05_config must be a Phase05Config")
    if not isinstance(simulator_config, SimulatorConfig):
        raise TypeError("simulator_config must be a SimulatorConfig")
    if type(resume) is not bool:
        raise TypeError("resume must be a boolean")
    if not callable(prepare_cell) or not callable(run_cell):
        raise TypeError("prepare_cell and run_cell must be callable")

    stage, ordered = _validate_stage_cells(cells)
    expected_cell_ids = frozenset(cell.cell_id for cell in ordered)
    current_commit = execution_commit_sha()
    if current_commit != store.execution_commit:
        raise ValueError("execution checkout does not match the hash-bound store identity")

    resolved_device = resolve_device(device)
    observed_before = store.validate_resume(
        stage,
        expected_cell_ids=expected_cell_ids,
    )
    if observed_before and not resume:
        raise ValueError("persisted Phase 0.6 cells require resume=True")

    completed_before = len(observed_before)
    failures: list[dict[str, str]] = []
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    caught: BaseException | None = None

    for cell in ordered:
        if cell.cell_id in observed_before:
            continue
        try:
            prepared = prepare_cell(simulator_config, cell)
            result = run_cell(
                prepared,
                phase05_config,
                cell,
                resolved_device,
            )
            if not isinstance(result, Phase06CellRun):
                raise TypeError("Phase 0.6 cell runner must return Phase06CellRun")
            store.write_cell_bundle(
                cell,
                metrics=result.metrics,
                trace=result.trace,
                summary=result.summary,
                production_state_dict=result.production_state_dict,
                shadow_state_dict=result.shadow_state_dict,
            )
        except Exception as error:
            failures.append(
                {
                    "cell_id": cell.cell_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            caught = error
            break

    observed_after = store.validate_resume(
        stage,
        expected_cell_ids=expected_cell_ids,
    )
    ended_at = datetime.now(UTC)
    wall_seconds = perf_counter() - started_clock
    provenance = _execution_provenance(
        stage=stage,
        planned_cells=len(ordered),
        completed_before=completed_before,
        completed_after=len(observed_after),
        failures=failures,
        device=resolved_device,
        started_at=started_at,
        ended_at=ended_at,
        wall_seconds=wall_seconds,
        store=store,
    )
    _persist_provenance(store, stage, provenance)

    if caught is not None:
        raise caught

    if observed_after != expected_cell_ids:
        raise RuntimeError("Phase 0.6 stage ended without the exact planned cell set")
    store.mark_stage_complete(stage, expected_cell_ids)
    return provenance


__all__ = ["run_phase06_stage"]
