from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from afmc_fm.phase07.planning import Phase07CellSpec
from afmc_fm.phase07.runner import Phase07CellRun
from afmc_fm.phase07.store import Phase07Store


def _execution_api():
    return importlib.import_module("afmc_fm.phase07.execution")


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _cells() -> tuple[Phase07CellSpec, ...]:
    return tuple(
        Phase07CellSpec(
            world="smooth",
            cohort_seed=406,
            subset_seed=506,
            model_seed=model_seed,
            n_train=40,
            flow_mode="time_scaled",
            optimization_policy=policy,
        )
        for model_seed in (1101, 1102)
        for policy in ("standard_early_stop", "forced_horizon")
    )


def _protocol_lock() -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase07_spec_sha256": "1" * 64,
        "phase07_config_sha256": "2" * 64,
        "execution_commit": "a" * 40,
        "world": "smooth",
        "n_train": 40,
        "expected_cell_count": 200,
    }


def _identity() -> dict[str, object]:
    return {
        "execution_commit": "a" * 40,
        "phase07_spec_sha256": "1" * 64,
        "phase07_config_sha256": "2" * 64,
        "phase05_config_sha256": "3" * 64,
        "simulator_config_sha256": "4" * 64,
        "protocol_lock_sha256": hashlib.sha256(
            _canonical_json_bytes(_protocol_lock())
        ).hexdigest(),
        "phase07_plan_sha256": "6" * 64,
        "expected_cell_count": 200,
    }


def _manifest() -> dict[str, object]:
    return {"schema_version": 1, "phase": "phase07", **_identity()}


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


def _plan_payload(cells: tuple[Phase07CellSpec, ...]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "phase07",
        "phase07_plan_sha256": "6" * 64,
        "expected_cell_count": 200,
        "cells": [_cell_payload(cell) for cell in cells],
    }


def _store(tmp_path: Path, cells: tuple[Phase07CellSpec, ...], *, resume: bool) -> Phase07Store:
    store = Phase07Store(tmp_path, identity=_identity())
    store.initialize(
        _manifest(),
        _protocol_lock(),
        _plan_payload(cells),
        resume=resume,
    )
    return store


def _result(cell: Phase07CellSpec) -> Phase07CellRun:
    value = 0.20 + (cell.model_seed - 1101) * 0.01
    metrics = pd.DataFrame(
        [
            {
                "stage": "phase07",
                "world": cell.world,
                "cohort_seed": cell.cohort_seed,
                "subset_seed": cell.subset_seed,
                "model_seed": cell.model_seed,
                "n_train": cell.n_train,
                "variant": f"{cell.flow_mode}__none__deterministic",
                "optimization_policy": cell.optimization_policy,
                "split": "test",
                "metric": "mae",
                "value": value,
            }
        ]
    )
    trace = pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_core_loss": 1.0,
                "validation_core_loss": 0.8,
                "validation_mae": 0.5,
                "stale_epochs": 0,
            }
        ]
    )
    summary = {
        "optimization_policy": cell.optimization_policy,
        "would_patience_exhaust_epoch": None,
        "epochs_run": 1,
        "stop_epoch": 1,
        "selected_checkpoint_epoch": 1,
        "selected_validation_core_loss": 0.8,
        "shadow_mae_checkpoint_epoch": 1,
        "shadow_validation_mae": 0.5,
        "early_stop_reason": "max_epochs_reached",
    }
    state = {"weight": torch.tensor([float(cell.model_seed)])}
    return Phase07CellRun(
        metrics=metrics,
        trace=trace,
        summary=summary,
        production_state_dict=state,
        shadow_state_dict={"weight": state["weight"].clone()},
        optimization_policy=cell.optimization_policy,
        would_patience_exhaust_epoch=None,
    )


def test_interruption_then_resume_executes_only_remaining_cells(tmp_path):
    module = _execution_api()
    cells = _cells()
    store = _store(tmp_path, cells, resume=False)
    first_calls: list[str] = []

    def fail_on_third(cell: Phase07CellSpec) -> Phase07CellRun:
        first_calls.append(cell.cell_id)
        if len(first_calls) == 3:
            raise RuntimeError("simulated interruption")
        return _result(cell)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        module._run_phase07_cells_with_store(
            cells,
            store=store,
            resume=False,
            execute_cell=fail_on_third,
        )

    assert first_calls == [cell.cell_id for cell in cells[:3]]
    assert store.validate_resume(cells) == frozenset(
        cell.cell_id for cell in cells[:2]
    )

    resumed_store = _store(tmp_path, cells, resume=True)
    resumed_calls: list[str] = []

    def finish(cell: Phase07CellSpec) -> Phase07CellRun:
        resumed_calls.append(cell.cell_id)
        return _result(cell)

    completed = module._run_phase07_cells_with_store(
        cells,
        store=resumed_store,
        resume=True,
        execute_cell=finish,
    )

    assert resumed_calls == [cell.cell_id for cell in cells[2:]]
    assert completed == tuple(cell.cell_id for cell in cells)
    assert resumed_store.validate_resume(cells) == frozenset(
        cell.cell_id for cell in cells
    )


def test_existing_cells_without_resume_fail_before_callback(tmp_path):
    module = _execution_api()
    cells = _cells()
    store = _store(tmp_path, cells, resume=False)
    store.write_cell_bundle(
        cells[0],
        metrics=_result(cells[0]).metrics,
        trace=_result(cells[0]).trace,
        summary=_result(cells[0]).summary,
        production_state_dict=_result(cells[0]).production_state_dict,
        shadow_state_dict=_result(cells[0]).shadow_state_dict,
    )
    called = False

    def forbidden(_cell: Phase07CellSpec) -> Phase07CellRun:
        nonlocal called
        called = True
        raise AssertionError("callback must not run")

    with pytest.raises(ValueError, match="resume"):
        module._run_phase07_cells_with_store(
            cells,
            store=store,
            resume=False,
            execute_cell=forbidden,
        )

    assert called is False


def test_completed_metrics_loader_requires_valid_complete_marker(tmp_path):
    module = _execution_api()
    cells = _cells()
    store = _store(tmp_path, cells, resume=False)
    for cell in cells:
        result = _result(cell)
        store.write_cell_bundle(
            cell,
            metrics=result.metrics,
            trace=result.trace,
            summary=result.summary,
            production_state_dict=result.production_state_dict,
            shadow_state_dict=result.shadow_state_dict,
        )

    with pytest.raises(ValueError, match="COMPLETE"):
        module.load_completed_phase07_metrics(store, cells)

    store.mark_complete(cells)
    (tmp_path / "phase07_metrics.csv").write_text(
        "stage,value\ncorrupt,999\n",
        encoding="utf-8",
    )
    loaded = module.load_completed_phase07_metrics(store, cells)
    assert len(loaded) == len(cells)
    assert set(loaded["model_seed"]) == {1101, 1102}
    assert "corrupt" not in set(loaded["stage"])


def test_interrupted_resume_matches_uninterrupted_semantic_evidence(tmp_path):
    module = _execution_api()
    cells = _cells()
    resumed_root = tmp_path / "resumed"
    uninterrupted_root = tmp_path / "uninterrupted"

    interrupted = _store(resumed_root, cells, resume=False)
    attempts = 0

    def fail_once(cell: Phase07CellSpec) -> Phase07CellRun:
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            raise RuntimeError("simulated interruption")
        return _result(cell)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        module._run_phase07_cells_with_store(
            cells,
            store=interrupted,
            resume=False,
            execute_cell=fail_once,
        )

    resumed = _store(resumed_root, cells, resume=True)
    module._run_phase07_cells_with_store(
        cells,
        store=resumed,
        resume=True,
        execute_cell=_result,
    )
    resumed.mark_complete(cells)

    uninterrupted = _store(uninterrupted_root, cells, resume=False)
    module._run_phase07_cells_with_store(
        cells,
        store=uninterrupted,
        resume=False,
        execute_cell=_result,
    )
    uninterrupted.mark_complete(cells)

    pd.testing.assert_frame_equal(
        resumed.load_metrics(cells).sort_values(
            ["model_seed", "optimization_policy"]
        ).reset_index(drop=True),
        uninterrupted.load_metrics(cells).sort_values(
            ["model_seed", "optimization_policy"]
        ).reset_index(drop=True),
        check_dtype=False,
    )

    for cell in cells:
        resumed_bundle = resumed.cells_dir / cell.cell_id
        uninterrupted_bundle = uninterrupted.cells_dir / cell.cell_id
        pd.testing.assert_frame_equal(
            pd.read_csv(resumed_bundle / "training_trace.csv"),
            pd.read_csv(uninterrupted_bundle / "training_trace.csv"),
            check_dtype=False,
        )
        assert json.loads((resumed_bundle / "summary.json").read_text(encoding="utf-8")) == json.loads(
            (uninterrupted_bundle / "summary.json").read_text(encoding="utf-8")
        )
        for checkpoint in (
            "production_checkpoint.pt",
            "shadow_mae_checkpoint.pt",
        ):
            resumed_state = torch.load(
                resumed_bundle / checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            uninterrupted_state = torch.load(
                uninterrupted_bundle / checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            assert resumed_state.keys() == uninterrupted_state.keys()
            for key in resumed_state:
                torch.testing.assert_close(
                    resumed_state[key],
                    uninterrupted_state[key],
                    rtol=0,
                    atol=0,
                )
