from __future__ import annotations

import hashlib
import json

import pandas as pd
import torch

from afmc_fm.phase06.planning import Phase06CellSpec
from afmc_fm.phase06.store import Phase06Store


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def test_d4b_store_persists_validates_completes_and_reloads_hash_bound_cell(tmp_path):
    lock = {
        "schema_version": 3,
        "phase06_config_sha256": "b" * 64,
        "execution_commit": "a" * 40,
        "scope": "d4b-store-regression",
    }
    protocol_hash = hashlib.sha256(_canonical_json_bytes(lock)).hexdigest()
    store = Phase06Store(
        tmp_path,
        protocol_hash=protocol_hash,
        config_hash="b" * 64,
        execution_commit="a" * 40,
    )
    store.write_protocol_lock(lock)

    cell = Phase06CellSpec(
        stage="d4b",
        world="smooth",
        cohort_seed=401,
        subset_seed=501,
        model_seed=1001,
        n_train=40,
        flow_mode="none",
    )
    metrics = pd.DataFrame(
        [
            {
                "stage": "d4b",
                "world": "smooth",
                "cohort_seed": 401,
                "subset_seed": 501,
                "model_seed": 1001,
                "n_train": 40,
                "variant": "none__none__deterministic",
                "split": "test",
                "metric": "mae",
                "value": 0.25,
            }
        ]
    )
    trace = pd.DataFrame(
        [
            {
                "epoch": 1,
                "gradient_l2_norm": 1.0,
                "mean_flow_displacement": 0.0,
            }
        ]
    )
    summary = {
        "epochs_run": 1,
        "stop_epoch": 1,
        "selected_checkpoint_epoch": 1,
        "shadow_mae_checkpoint_epoch": 1,
        "early_stop_reason": "max_epochs_reached",
    }
    state = {"weight": torch.tensor([1.0])}

    store.write_cell_bundle(
        cell,
        metrics=metrics,
        trace=trace,
        summary=summary,
        production_state_dict=state,
        shadow_state_dict=state,
    )
    expected = frozenset({cell.cell_id})
    assert store.validate_resume("d4b", expected_cell_ids=expected) == expected
    store.mark_stage_complete("d4b", expected)

    assert (tmp_path / "stages" / "d4b" / "COMPLETE").is_file()
    reloaded = store.load_stage_metrics("d4b")
    assert len(reloaded) == 1
    assert reloaded.iloc[0]["stage"] == "d4b"
