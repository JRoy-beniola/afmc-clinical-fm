import hashlib
import json

import pandas as pd
import pytest

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import Phase05Config, SeedBundle
from afmc_fm.phase05.execution import (
    Phase05ExecutionOptions,
    Phase05Job,
    Phase05ShardSpec,
    run_phase05_jobs,
)
from afmc_fm.phase05.store import Phase05CellResult, Phase05Store


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _finalized_confirmation_store(tmp_path):
    config = Phase05Config(max_epochs=1, patience=1)
    config_hash = canonical_config_hash(config)
    lock = {
        "schema_version": 1,
        "phase05_config_sha256": config_hash,
    }
    protocol_hash = hashlib.sha256(_canonical_json_bytes(lock)).hexdigest()
    store = Phase05Store(
        tmp_path / "phase05",
        protocol_hash,
        spec_hash="1" * 64,
        config_hash=config_hash,
    )
    store.write_protocol_lock(lock)
    candidate_hash = store.write_frozen_candidate(
        {
            "flow_mode": "time_scaled",
            "jump_mode": "residual",
            "uncertainty_mode": "deterministic",
        }
    )
    store.mark_confirmation_started()
    confirmation_result = Phase05CellResult(
        stage="confirmation",
        world="smooth",
        cohort_seed=701,
        subset_seed=801,
        model_seed=901,
        n_train=5,
        model="phase05_candidate",
        variant="time_scaled__residual__deterministic",
        metrics=pd.DataFrame([{"metric": "mae", "value": 1.0}]),
        frozen_candidate_hash=candidate_hash,
    )
    store.write_cell(confirmation_result)
    store.mark_stage_complete("confirmation", {confirmation_result.cell_id})
    return store, config, candidate_hash


def test_robustness_rejects_empty_primary_gate_artifact(tmp_path):
    store, config, candidate_hash = _finalized_confirmation_store(tmp_path)
    gate_path = store.output / "confirmation" / "primary_gate_summary.csv"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_bytes(b"")
    job = Phase05Job(
        shard=Phase05ShardSpec(
            "robustness", "site_shift", SeedBundle(701, 801, 901)
        ),
        n_train=5,
        model="phase05_candidate",
        variant="time_scaled__residual__deterministic",
        frozen_candidate_hash=candidate_hash,
    )

    with pytest.raises(
        RuntimeError,
        match="primary confirmatory gate artifact is invalid",
    ):
        run_phase05_jobs(
            (job,),
            store=store,
            config=config,
            options=Phase05ExecutionOptions(device="cpu", workers=1),
            prepare_shard=lambda spec: object(),
            run_job=lambda job, prepared, device: pd.DataFrame(
                [{"metric": "mae", "value": 1.0}]
            ),
        )
