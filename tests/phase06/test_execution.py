import hashlib
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from afmc_fm.execution.manifest import execution_commit_sha
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase06.planning import Phase06CellSpec
from afmc_fm.phase06.runner import Phase06CellRun
from afmc_fm.phase06.store import Phase06Store
from afmc_fm.simulator.config import SimulatorConfig

execution = importlib.import_module("afmc_fm.phase06.execution")
run_phase06_stage = execution.run_phase06_stage


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _store(tmp_path: Path) -> Phase06Store:
    commit = execution_commit_sha()
    lock = {
        "schema_version": 1,
        "phase06_config_sha256": "b" * 64,
        "execution_commit": commit,
        "scope": "phase06-execution-test",
    }
    protocol_hash = hashlib.sha256(_canonical_json_bytes(lock)).hexdigest()
    store = Phase06Store(
        tmp_path,
        protocol_hash=protocol_hash,
        config_hash="b" * 64,
        execution_commit=commit,
    )
    store.write_protocol_lock(lock)
    return store


def _cell(*, flow_mode: str, n_train: int = 5) -> Phase06CellSpec:
    return Phase06CellSpec(
        stage="d1",
        world="smooth",
        cohort_seed=401,
        subset_seed=501,
        model_seed=601,
        n_train=n_train,
        flow_mode=flow_mode,
    )


def _run_result(cell: Phase06CellSpec) -> Phase06CellRun:
    variant = f"{cell.flow_mode}__none__deterministic"
    metrics = pd.DataFrame(
        [
            {
                "stage": cell.stage,
                "world": cell.world,
                "cohort_seed": cell.cohort_seed,
                "subset_seed": cell.subset_seed,
                "model_seed": cell.model_seed,
                "n_train": cell.n_train,
                "variant": variant,
                "split": "test",
                "metric": "mae",
                "value": 0.25,
            },
            {
                "stage": cell.stage,
                "world": cell.world,
                "cohort_seed": cell.cohort_seed,
                "subset_seed": cell.subset_seed,
                "model_seed": cell.model_seed,
                "n_train": cell.n_train,
                "variant": variant,
                "split": "test",
                "metric": "rmse",
                "value": 0.31,
            },
        ]
    )
    trace = pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_core_loss": 1.0,
                "validation_core_loss": 0.8,
                "validation_mae": 0.5,
            }
        ]
    )
    summary = {
        "epochs_run": 1,
        "stop_epoch": 1,
        "selected_checkpoint_epoch": 1,
        "selected_validation_core_loss": 0.8,
        "shadow_mae_checkpoint_epoch": 1,
        "shadow_validation_mae": 0.5,
        "early_stop_reason": "max_epochs_reached",
    }
    return Phase06CellRun(
        metrics=metrics,
        trace=trace,
        summary=summary,
        production_state_dict={"weight": torch.tensor([1.0])},
        shadow_state_dict={"weight": torch.tensor([2.0])},
    )


def _configs() -> tuple[Phase05Config, SimulatorConfig]:
    return (
        Phase05Config(max_epochs=1, patience=1),
        SimulatorConfig(cohort_size=16, followup_days=30.0),
    )


def test_stage_executes_in_locked_order_and_persists_exact_counts(tmp_path):
    store = _store(tmp_path)
    phase05_config, simulator_config = _configs()
    none = _cell(flow_mode="none")
    time_scaled = _cell(flow_mode="time_scaled")
    prepared: list[str] = []
    executed: list[str] = []

    def prepare_cell(_simulator_config, cell):
        prepared.append(cell.cell_id)
        return cell.cell_id

    def run_cell(prepared_cell, _phase05_config, cell, device):
        assert prepared_cell == cell.cell_id
        assert device == torch.device("cpu")
        executed.append(cell.cell_id)
        return _run_result(cell)

    result = run_phase06_stage(
        (time_scaled, none),
        store,
        phase05_config,
        simulator_config,
        device="cpu",
        resume=False,
        prepare_cell=prepare_cell,
        run_cell=run_cell,
    )

    expected_order = [none.cell_id, time_scaled.cell_id]
    assert prepared == expected_order
    assert executed == expected_order
    assert result["stage"] == "d1"
    assert result["planned_cells"] == 2
    assert result["completed_before"] == 0
    assert result["completed_after"] == 2
    assert result["failures"] == []
    assert result["workers"] == 1
    assert result["forbidden_seed_validation"] == "passed"
    assert (tmp_path / "stages" / "d1" / "COMPLETE").is_file()
    provenance = json.loads(
        (tmp_path / "stages" / "d1" / "execution_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance == result


def test_duplicate_cell_ids_are_rejected_before_any_work(tmp_path):
    store = _store(tmp_path)
    phase05_config, simulator_config = _configs()
    cell = _cell(flow_mode="none")
    prepared: list[str] = []

    def prepare_cell(_simulator_config, candidate):
        prepared.append(candidate.cell_id)
        return candidate

    with pytest.raises(ValueError, match="duplicate"):
        run_phase06_stage(
            (cell, cell),
            store,
            phase05_config,
            simulator_config,
            device="cpu",
            prepare_cell=prepare_cell,
            run_cell=lambda *_args: _run_result(cell),
        )

    assert prepared == []


def test_resume_skips_only_fully_hash_valid_cells(tmp_path):
    store = _store(tmp_path)
    phase05_config, simulator_config = _configs()
    first = _cell(flow_mode="none")
    second = _cell(flow_mode="time_scaled")
    first_run = _run_result(first)
    store.write_cell_bundle(
        first,
        metrics=first_run.metrics,
        trace=first_run.trace,
        summary=first_run.summary,
        production_state_dict=first_run.production_state_dict,
        shadow_state_dict=first_run.shadow_state_dict,
    )
    executed: list[str] = []

    result = run_phase06_stage(
        (first, second),
        store,
        phase05_config,
        simulator_config,
        device="cpu",
        resume=True,
        prepare_cell=lambda _config, cell: cell,
        run_cell=lambda _prepared, _config, cell, _device: (
            executed.append(cell.cell_id) or _run_result(cell)
        ),
    )

    assert executed == [second.cell_id]
    assert result["completed_before"] == 1
    assert result["completed_after"] == 2

    trace_path = tmp_path / "stages" / "d1" / "traces" / f"{first.cell_id}.csv"
    trace_path.write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        run_phase06_stage(
            (first, second),
            store,
            phase05_config,
            simulator_config,
            device="cpu",
            resume=True,
            prepare_cell=lambda _config, cell: cell,
            run_cell=lambda _prepared, _config, cell, _device: _run_result(cell),
        )


def test_cell_failure_is_fail_fast_and_records_execution_failure(tmp_path):
    store = _store(tmp_path)
    phase05_config, simulator_config = _configs()
    first = _cell(flow_mode="none")
    second = _cell(flow_mode="time_scaled")
    attempted: list[str] = []

    def run_cell(_prepared, _config, cell, _device):
        attempted.append(cell.cell_id)
        if cell.cell_id == first.cell_id:
            raise RuntimeError("boom")
        return _run_result(cell)

    with pytest.raises(RuntimeError, match="boom"):
        run_phase06_stage(
            (second, first),
            store,
            phase05_config,
            simulator_config,
            device="cpu",
            prepare_cell=lambda _config, cell: cell,
            run_cell=run_cell,
        )

    assert attempted == [first.cell_id]
    provenance = json.loads(
        (tmp_path / "stages" / "d1" / "execution_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["planned_cells"] == 2
    assert provenance["completed_before"] == 0
    assert provenance["completed_after"] == 0
    assert provenance["failures"] == [
        {
            "cell_id": first.cell_id,
            "error_type": "RuntimeError",
            "message": "boom",
        }
    ]
    assert not (tmp_path / "stages" / "d1" / "COMPLETE").exists()
