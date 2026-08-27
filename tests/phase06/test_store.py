import hashlib
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from afmc_fm.phase06.planning import Phase06CellSpec

store_module = importlib.import_module("afmc_fm.phase06.store")
Phase06Store = store_module.Phase06Store


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _protocol_lock() -> tuple[dict[str, object], str]:
    payload = {
        "schema_version": 1,
        "phase06_config_sha256": "b" * 64,
        "execution_commit": "a" * 40,
        "scope": "diagnostic-test",
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload, digest


def _cell() -> Phase06CellSpec:
    return Phase06CellSpec(
        stage="d1",
        world="smooth",
        cohort_seed=401,
        subset_seed=501,
        model_seed=601,
        n_train=5,
        flow_mode="none",
    )


def _metrics(cell: Phase06CellSpec) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": cell.stage,
                "world": cell.world,
                "cohort_seed": cell.cohort_seed,
                "subset_seed": cell.subset_seed,
                "model_seed": cell.model_seed,
                "n_train": cell.n_train,
                "variant": "none__none__deterministic",
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
                "variant": "none__none__deterministic",
                "split": "test",
                "metric": "rmse",
                "value": 0.31,
            },
        ]
    )


def _trace() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_core_loss": 1.0,
                "validation_core_loss": 0.8,
                "validation_mae": 0.5,
            },
            {
                "epoch": 2,
                "train_core_loss": 0.7,
                "validation_core_loss": 0.6,
                "validation_mae": 0.4,
            },
        ]
    )


def _summary() -> dict[str, object]:
    return {
        "epochs_run": 2,
        "stop_epoch": 2,
        "selected_checkpoint_epoch": 2,
        "selected_validation_core_loss": 0.6,
        "shadow_mae_checkpoint_epoch": 2,
        "shadow_validation_mae": 0.4,
        "early_stop_reason": "max_epochs_reached",
    }


def _production_state() -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}


def _shadow_state() -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor([[1.5, 2.5], [3.5, 4.5]])}


def _store(tmp_path: Path) -> tuple[object, dict[str, object]]:
    lock, protocol_hash = _protocol_lock()
    store = Phase06Store(
        tmp_path,
        protocol_hash=protocol_hash,
        config_hash="b" * 64,
        execution_commit="a" * 40,
    )
    store.write_protocol_lock(lock)
    return store, lock


def test_cell_bundle_is_written_to_exact_hash_bound_paths(tmp_path):
    store, _ = _store(tmp_path)
    cell = _cell()

    store.write_cell_bundle(
        cell,
        metrics=_metrics(cell),
        trace=_trace(),
        summary=_summary(),
        production_state_dict=_production_state(),
        shadow_state_dict=_shadow_state(),
    )

    stage = tmp_path / "stages" / "d1"
    cell_path = stage / "cells" / f"{cell.cell_id}.json"
    trace_path = stage / "traces" / f"{cell.cell_id}.csv"
    summary_path = stage / "summaries" / f"{cell.cell_id}.json"
    production_path = stage / "checkpoints" / f"{cell.cell_id}__production.pt"
    shadow_path = stage / "checkpoints" / f"{cell.cell_id}__shadow_mae.pt"

    for path in (cell_path, trace_path, summary_path, production_path, shadow_path):
        assert path.is_file(), path

    manifest = json.loads(summary_path.read_text(encoding="utf-8"))
    assert manifest["identity"] == {
        "protocol_lock_sha256": store.protocol_hash,
        "phase06_config_sha256": store.config_hash,
        "execution_commit": store.execution_commit,
    }
    assert manifest["artifact_sha256"] == {
        "cell_metrics": hashlib.sha256(cell_path.read_bytes()).hexdigest(),
        "training_trace": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "production_checkpoint": hashlib.sha256(production_path.read_bytes()).hexdigest(),
        "shadow_mae_checkpoint": hashlib.sha256(shadow_path.read_bytes()).hexdigest(),
    }
    assert manifest["selected_checkpoint_epoch"] == 2
    assert manifest["shadow_mae_checkpoint_epoch"] == 2

    persisted_production = torch.load(production_path, map_location="cpu", weights_only=True)
    persisted_shadow = torch.load(shadow_path, map_location="cpu", weights_only=True)
    assert torch.equal(persisted_production["weight"], _production_state()["weight"])
    assert torch.equal(persisted_shadow["weight"], _shadow_state()["weight"])


def test_identical_cell_bundle_rewrite_is_idempotent(tmp_path):
    store, _ = _store(tmp_path)
    cell = _cell()
    kwargs = {
        "metrics": _metrics(cell),
        "trace": _trace(),
        "summary": _summary(),
        "production_state_dict": _production_state(),
        "shadow_state_dict": _shadow_state(),
    }

    first = store.write_cell_bundle(cell, **kwargs)
    stage = tmp_path / "stages" / "d1"
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in stage.rglob("*")
        if path.is_file()
    }
    second = store.write_cell_bundle(cell, **kwargs)
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in stage.rglob("*")
        if path.is_file()
    }

    assert first == second == cell.cell_id
    assert after == before


def test_conflicting_cell_bundle_rewrite_fails_without_mutating_existing_bundle(tmp_path):
    store, _ = _store(tmp_path)
    cell = _cell()
    store.write_cell_bundle(
        cell,
        metrics=_metrics(cell),
        trace=_trace(),
        summary=_summary(),
        production_state_dict=_production_state(),
        shadow_state_dict=_shadow_state(),
    )
    stage = tmp_path / "stages" / "d1"
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in stage.rglob("*")
        if path.is_file()
    }
    changed = _metrics(cell)
    changed.loc[0, "value"] = 9.0

    with pytest.raises(ValueError, match="conflicting persisted cell bundle"):
        store.write_cell_bundle(
            cell,
            metrics=changed,
            trace=_trace(),
            summary=_summary(),
            production_state_dict=_production_state(),
            shadow_state_dict=_shadow_state(),
        )

    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in stage.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_protocol_lock_is_idempotent_but_conflicting_bytes_are_rejected(tmp_path):
    store, lock = _store(tmp_path)
    assert store.write_protocol_lock(lock) == store.protocol_hash

    conflicting = dict(lock)
    conflicting["scope"] = "changed"
    with pytest.raises(ValueError, match="protocol lock"):
        store.write_protocol_lock(conflicting)


def test_validate_resume_rejects_unexpected_persisted_cell_ids(tmp_path):
    store, _ = _store(tmp_path)
    cell = _cell()
    store.write_cell_bundle(
        cell,
        metrics=_metrics(cell),
        trace=_trace(),
        summary=_summary(),
        production_state_dict=_production_state(),
        shadow_state_dict=_shadow_state(),
    )

    with pytest.raises(ValueError, match="unexpected persisted cell IDs"):
        store.validate_resume("d1", expected_cell_ids={"different-cell"})


def test_resume_revalidates_hash_manifest_and_detects_tampering(tmp_path):
    store, _ = _store(tmp_path)
    cell = _cell()
    store.write_cell_bundle(
        cell,
        metrics=_metrics(cell),
        trace=_trace(),
        summary=_summary(),
        production_state_dict=_production_state(),
        shadow_state_dict=_shadow_state(),
    )
    trace_path = tmp_path / "stages" / "d1" / "traces" / f"{cell.cell_id}.csv"
    trace_path.write_text(trace_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        store.validate_resume("d1", expected_cell_ids={cell.cell_id})


def test_stage_completion_requires_exact_cell_set_and_metrics_round_trip(tmp_path):
    store, _ = _store(tmp_path)
    cell = _cell()
    store.write_cell_bundle(
        cell,
        metrics=_metrics(cell),
        trace=_trace(),
        summary=_summary(),
        production_state_dict=_production_state(),
        shadow_state_dict=_shadow_state(),
    )

    with pytest.raises(ValueError, match="missing expected cells"):
        store.mark_stage_complete("d1", {cell.cell_id, "missing-cell"})

    store.mark_stage_complete("d1", {cell.cell_id})
    marker = tmp_path / "stages" / "d1" / "COMPLETE"
    assert marker.is_file()
    observed = store.validate_resume("d1", expected_cell_ids={cell.cell_id})
    assert observed == frozenset({cell.cell_id})

    loaded = store.load_stage_metrics("d1")
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        _metrics(cell).reset_index(drop=True),
        check_dtype=False,
    )
