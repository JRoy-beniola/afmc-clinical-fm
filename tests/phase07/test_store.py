from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from afmc_fm.phase07.planning import Phase07CellSpec


def _store_api():
    try:
        return importlib.import_module("afmc_fm.phase07.store")
    except ModuleNotFoundError as error:
        raise AssertionError("Phase 0.7 crash-resilient store is not implemented") from error


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _cell(*, model_seed: int = 1101, policy: str = "standard_early_stop") -> Phase07CellSpec:
    return Phase07CellSpec(
        world="smooth",
        cohort_seed=406,
        subset_seed=506,
        model_seed=model_seed,
        n_train=40,
        flow_mode="time_scaled",
        optimization_policy=policy,
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
    protocol_hash = hashlib.sha256(_canonical_json_bytes(_protocol_lock())).hexdigest()
    return {
        "execution_commit": "a" * 40,
        "phase07_spec_sha256": "1" * 64,
        "phase07_config_sha256": "2" * 64,
        "phase05_config_sha256": "3" * 64,
        "simulator_config_sha256": "4" * 64,
        "protocol_lock_sha256": protocol_hash,
        "phase07_plan_sha256": "6" * 64,
        "expected_cell_count": 200,
    }


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "phase07",
        **_identity(),
    }


def _plan_payload(cells: tuple[Phase07CellSpec, ...]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "phase07",
        "phase07_plan_sha256": "6" * 64,
        "expected_cell_count": 200,
        "cells": [
            {
                "world": cell.world,
                "cohort_seed": cell.cohort_seed,
                "subset_seed": cell.subset_seed,
                "model_seed": cell.model_seed,
                "n_train": cell.n_train,
                "flow_mode": cell.flow_mode,
                "optimization_policy": cell.optimization_policy,
                "cell_id": cell.cell_id,
            }
            for cell in cells
        ],
    }


def _metrics(cell: Phase07CellSpec) -> pd.DataFrame:
    return pd.DataFrame(
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
                "value": 0.25,
            }
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
                "stale_epochs": 0,
            },
            {
                "epoch": 2,
                "train_core_loss": 0.7,
                "validation_core_loss": 0.6,
                "validation_mae": 0.4,
                "stale_epochs": 1,
            },
        ]
    )


def _summary(cell: Phase07CellSpec) -> dict[str, object]:
    return {
        "optimization_policy": cell.optimization_policy,
        "would_patience_exhaust_epoch": None,
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


def _initialized_store(tmp_path: Path, cells: tuple[Phase07CellSpec, ...]):
    module = _store_api()
    store = module.Phase07Store(tmp_path, identity=_identity())
    store.initialize(
        _manifest(),
        _protocol_lock(),
        _plan_payload(cells),
        resume=False,
    )
    return store


def _write(store, cell: Phase07CellSpec) -> str:
    return store.write_cell_bundle(
        cell,
        metrics=_metrics(cell),
        trace=_trace(),
        summary=_summary(cell),
        production_state_dict=_production_state(),
        shadow_state_dict=_shadow_state(),
    )


def test_cell_bundle_is_committed_atomically_and_validates(tmp_path):
    cell = _cell()
    store = _initialized_store(tmp_path, (cell,))

    assert _write(store, cell) == cell.cell_id

    final = tmp_path / "stages" / "phase07" / "cells" / cell.cell_id
    assert final.is_dir()
    expected = {
        "cell.json",
        "metrics.csv",
        "training_trace.csv",
        "summary.json",
        "production_checkpoint.pt",
        "shadow_mae_checkpoint.pt",
        "artifact_manifest.json",
        "COMPLETE",
    }
    assert {path.name for path in final.iterdir()} == expected
    assert store.validate_resume({cell}) == frozenset({cell.cell_id})

    manifest = json.loads((final / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["identity"] == _identity()
    hashes = manifest["artifact_sha256"]
    for name in (
        "cell.json",
        "metrics.csv",
        "training_trace.csv",
        "summary.json",
        "production_checkpoint.pt",
        "shadow_mae_checkpoint.pt",
    ):
        assert hashes[name] == hashlib.sha256((final / name).read_bytes()).hexdigest()


def test_abandoned_staging_directory_is_not_authoritative_and_is_removed_on_resume(tmp_path):
    cell = _cell()
    store = _initialized_store(tmp_path, (cell,))
    abandoned = tmp_path / "stages" / "phase07" / ".tmp" / f"{cell.cell_id}.deadbeef"
    abandoned.mkdir(parents=True)
    (abandoned / "cell.json").write_text("partial", encoding="utf-8")

    resumed = _store_api().Phase07Store(tmp_path, identity=_identity())
    resumed.initialize(_manifest(), _protocol_lock(), _plan_payload((cell,)), resume=True)

    assert resumed.validate_resume({cell}) == frozenset()
    assert not abandoned.exists()


def test_corrupt_authoritative_bundle_fails_closed(tmp_path):
    cell = _cell()
    store = _initialized_store(tmp_path, (cell,))
    _write(store, cell)
    trace = tmp_path / "stages" / "phase07" / "cells" / cell.cell_id / "training_trace.csv"
    trace.write_text(trace.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        store.validate_resume({cell})


def test_unexpected_authoritative_cell_fails_closed(tmp_path):
    expected = _cell(model_seed=1101)
    unexpected = _cell(model_seed=1102)
    store = _initialized_store(tmp_path, (expected, unexpected))
    _write(store, unexpected)

    with pytest.raises(ValueError, match="unexpected persisted cell"):
        store.validate_resume({expected})


def test_complete_requires_exact_expected_set_and_round_trips_metrics(tmp_path):
    first = _cell(model_seed=1101)
    second = _cell(model_seed=1102)
    store = _initialized_store(tmp_path, (first, second))
    _write(store, first)

    with pytest.raises(ValueError, match="missing expected cells"):
        store.mark_complete({first, second})

    _write(store, second)
    store.mark_complete({first, second})
    store.require_complete({first, second})

    marker = tmp_path / "stages" / "phase07" / "COMPLETE"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["completed_cell_count"] == 2
    assert payload["cell_ids"] == sorted([first.cell_id, second.cell_id])

    loaded = store.load_metrics({first, second})
    expected = pd.concat([_metrics(first), _metrics(second)], ignore_index=True, sort=False)
    pd.testing.assert_frame_equal(
        loaded.sort_values("model_seed").reset_index(drop=True),
        expected.sort_values("model_seed").reset_index(drop=True),
        check_dtype=False,
    )


def test_resume_rejects_root_identity_drift_before_cleaning_staging(tmp_path):
    cell = _cell()
    _initialized_store(tmp_path, (cell,))
    abandoned = tmp_path / "stages" / "phase07" / ".tmp" / f"{cell.cell_id}.deadbeef"
    abandoned.mkdir(parents=True)
    changed = dict(_identity())
    changed["phase07_plan_sha256"] = "f" * 64
    resumed = _store_api().Phase07Store(tmp_path, identity=changed)

    with pytest.raises(ValueError, match="identity"):
        resumed.initialize(_manifest(), _protocol_lock(), _plan_payload((cell,)), resume=True)

    assert abandoned.exists()
