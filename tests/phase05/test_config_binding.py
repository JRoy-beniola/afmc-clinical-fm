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
from afmc_fm.phase05.store import Phase05Store


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def test_confirmation_rejects_runtime_config_drift_from_locked_identity(tmp_path):
    locked_config = Phase05Config(max_epochs=1, patience=1)
    config_hash = canonical_config_hash(locked_config)
    lock = {
        "schema_version": 1,
        "phase05_config_sha256": config_hash,
        "locked_min_relative_effect": 0.02,
        "locked_uncertainty_mae_tolerance": 0.02,
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

    job = Phase05Job(
        shard=Phase05ShardSpec(
            "confirmation",
            "smooth",
            SeedBundle(701, 801, 901),
        ),
        n_train=5,
        model="phase05_candidate",
        variant="time_scaled__residual__deterministic",
        frozen_candidate_hash=candidate_hash,
    )
    drifted_config = Phase05Config(
        max_epochs=1,
        patience=1,
        learning_rate=0.002,
    )
    calls = 0

    def runner(job, prepared, device):
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            [
                {
                    "stage": "confirmation",
                    "world": "smooth",
                    "cohort_seed": 701,
                    "subset_seed": 801,
                    "model_seed": 901,
                    "n_train": 5,
                    "model": "phase05_candidate",
                    "variant": "time_scaled__residual__deterministic",
                    "metric": "mae",
                    "value": 1.0,
                }
            ]
        )

    with pytest.raises(ValueError, match="config hash"):
        run_phase05_jobs(
            (job,),
            store=store,
            config=drifted_config,
            options=Phase05ExecutionOptions(device="cpu", workers=1),
            prepare_shard=lambda spec: object(),
            run_job=runner,
        )

    assert calls == 0
