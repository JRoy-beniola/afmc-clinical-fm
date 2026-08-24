import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

import pandas as pd
import pytest

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


def _protocol_lock(config_hash: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase05_config_sha256": config_hash,
        "locked_min_relative_effect": 0.02,
        "locked_uncertainty_mae_tolerance": 0.02,
    }


def _store(tmp_path: Path) -> Phase05Store:
    config_hash = "2" * 64
    lock = _protocol_lock(config_hash)
    protocol_hash = hashlib.sha256(_canonical_json_bytes(lock)).hexdigest()
    store = Phase05Store(
        tmp_path / "phase05",
        protocol_hash,
        spec_hash="1" * 64,
        config_hash=config_hash,
    )
    store.write_protocol_lock(lock)
    return store


def _metric_frame(job: Phase05Job, value: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": job.shard.stage,
                "world": job.shard.world,
                "cohort_seed": job.shard.seed_bundle.cohort_seed,
                "subset_seed": job.shard.seed_bundle.subset_seed,
                "model_seed": job.shard.seed_bundle.model_seed,
                "n_train": job.n_train,
                "model": job.model,
                "variant": job.variant,
                "metric": "mae",
                "value": value,
            }
        ]
    )


def _spawn_prepare(spec: Phase05ShardSpec) -> dict[str, object]:
    return {"shard": spec.shard_id, "prepared_pid": os.getpid()}


def _spawn_runner(job: Phase05Job, prepared: dict[str, object], device) -> pd.DataFrame:
    time.sleep(0.5)
    frame = _metric_frame(job, float(job.n_train))
    frame["worker_pid"] = os.getpid()
    frame["prepared_pid"] = prepared["prepared_pid"]
    frame["worker_start_method"] = multiprocessing.get_start_method()
    frame["device_type"] = device.type
    return frame


def test_resume_preserves_completed_cell_bytes_and_runs_only_missing_jobs(tmp_path):
    store = _store(tmp_path)
    shard = Phase05ShardSpec("flow", "smooth", SeedBundle(401, 501, 601))
    jobs = (
        Phase05Job(
            shard=shard,
            n_train=5,
            model="phase05_flow_jump",
            variant="time_scaled__none__deterministic",
        ),
        Phase05Job(
            shard=shard,
            n_train=10,
            model="phase05_flow_jump",
            variant="time_scaled__none__deterministic",
        ),
    )
    calls: list[int] = []

    def interrupting_runner(job, prepared, device):
        assert prepared is not None
        calls.append(job.n_train)
        if job.n_train == 10:
            raise RuntimeError("injected interruption")
        return _metric_frame(job, 2.0)

    with pytest.raises(RuntimeError, match="injected interruption"):
        run_phase05_jobs(
            jobs,
            store=store,
            config=Phase05Config(max_epochs=1, patience=1),
            options=Phase05ExecutionOptions(device="cpu", workers=1, fail_fast=True),
            prepare_shard=lambda spec: {"shard": spec.shard_id},
            run_job=interrupting_runner,
        )

    first_path = (
        tmp_path
        / "phase05"
        / "stages"
        / "flow"
        / "cells"
        / f"{jobs[0].cell_id}.json"
    )
    before_bytes = first_path.read_bytes()
    before_mtime = first_path.stat().st_mtime_ns
    time.sleep(0.01)
    resumed_calls: list[int] = []

    def resumed_runner(job, prepared, device):
        resumed_calls.append(job.n_train)
        return _metric_frame(job, 1.5)

    resumed = run_phase05_jobs(
        jobs,
        store=store,
        config=Phase05Config(max_epochs=1, patience=1),
        options=Phase05ExecutionOptions(
            device="cpu", workers=1, resume=True, fail_fast=True
        ),
        prepare_shard=lambda spec: {"shard": spec.shard_id},
        run_job=resumed_runner,
    )

    assert calls == [5, 10]
    assert resumed_calls == [10]
    assert first_path.read_bytes() == before_bytes
    assert first_path.stat().st_mtime_ns == before_mtime
    assert resumed["n_train"].tolist() == [5, 10]


def test_one_shard_preparation_is_reused_across_pending_jobs(tmp_path):
    store = _store(tmp_path)
    shard = Phase05ShardSpec("flow", "smooth", SeedBundle(401, 501, 601))
    jobs = tuple(
        Phase05Job(
            shard=shard,
            n_train=n_train,
            model="phase05_flow_jump",
            variant="time_scaled__none__deterministic",
        )
        for n_train in (5, 10, 20)
    )
    prepared_ids: list[int] = []
    prepare_calls = 0

    def prepare(spec):
        nonlocal prepare_calls
        prepare_calls += 1
        return object()

    def runner(job, prepared, device):
        prepared_ids.append(id(prepared))
        return _metric_frame(job, float(job.n_train))

    result = run_phase05_jobs(
        jobs,
        store=store,
        config=Phase05Config(max_epochs=1, patience=1),
        options=Phase05ExecutionOptions(device="cpu", workers=1),
        prepare_shard=prepare,
        run_job=runner,
    )

    assert prepare_calls == 1
    assert len(set(prepared_ids)) == 1
    assert result["n_train"].tolist() == [5, 10, 20]


def test_confirmatory_jobs_require_started_marker_and_exact_frozen_candidate(tmp_path):
    store = _store(tmp_path)
    candidate_hash = store.write_frozen_candidate(
        {
            "flow_mode": "time_scaled",
            "jump_mode": "residual",
            "uncertainty_mode": "deterministic",
        }
    )
    shard = Phase05ShardSpec(
        "confirmation", "smooth", SeedBundle(701, 801, 901)
    )
    job = Phase05Job(
        shard=shard,
        n_train=5,
        model="phase05_candidate",
        variant="time_scaled__residual__deterministic",
        frozen_candidate_hash=candidate_hash,
    )

    with pytest.raises(RuntimeError, match="confirmation must be started"):
        run_phase05_jobs(
            (job,),
            store=store,
            config=Phase05Config(max_epochs=1, patience=1),
            options=Phase05ExecutionOptions(device="cpu", workers=1),
            prepare_shard=lambda spec: object(),
            run_job=lambda job, prepared, device: _metric_frame(job, 1.0),
        )

    store.mark_confirmation_started()
    wrong = Phase05Job(
        shard=shard,
        n_train=5,
        model="phase05_candidate",
        variant="time_scaled__residual__deterministic",
        frozen_candidate_hash="f" * 64,
    )
    with pytest.raises(ValueError, match="frozen candidate hash"):
        run_phase05_jobs(
            (wrong,),
            store=store,
            config=Phase05Config(max_epochs=1, patience=1),
            options=Phase05ExecutionOptions(device="cpu", workers=1),
            prepare_shard=lambda spec: object(),
            run_job=lambda job, prepared, device: _metric_frame(job, 1.0),
        )


def test_multiworker_execution_uses_spawn_and_keeps_persistence_in_parent(tmp_path):
    store = _store(tmp_path)
    jobs = (
        Phase05Job(
            shard=Phase05ShardSpec(
                "flow", "smooth", SeedBundle(401, 501, 601)
            ),
            n_train=5,
            model="phase05_flow_jump",
            variant="time_scaled__none__deterministic",
        ),
        Phase05Job(
            shard=Phase05ShardSpec(
                "flow", "jumps", SeedBundle(402, 502, 602)
            ),
            n_train=5,
            model="phase05_flow_jump",
            variant="time_scaled__none__deterministic",
        ),
    )

    result = run_phase05_jobs(
        jobs,
        store=store,
        config=Phase05Config(max_epochs=1, patience=1),
        options=Phase05ExecutionOptions(device="cpu", workers=2, fail_fast=True),
        prepare_shard=_spawn_prepare,
        run_job=_spawn_runner,
    )

    assert set(result["worker_start_method"]) == {"spawn"}
    assert set(result["device_type"]) == {"cpu"}
    assert (result["worker_pid"] == result["prepared_pid"]).all()
    assert len(set(result["worker_pid"])) == 2
    cells = store.output / "stages" / "flow" / "cells"
    assert not list(cells.glob("*.tmp"))


def test_development_substages_are_locked_until_predecessor_is_complete(tmp_path):
    store = _store(tmp_path)
    job = Phase05Job(
        shard=Phase05ShardSpec("jump", "jumps", SeedBundle(401, 501, 601)),
        n_train=5,
        model="phase05_flow_jump",
        variant="time_scaled__residual__deterministic",
    )

    with pytest.raises(RuntimeError, match="flow stage must be complete before jump"):
        run_phase05_jobs(
            (job,),
            store=store,
            config=Phase05Config(max_epochs=1, patience=1),
            options=Phase05ExecutionOptions(device="cpu", workers=1),
            prepare_shard=lambda spec: object(),
            run_job=lambda job, prepared, device: _metric_frame(job, 1.0),
        )


def test_robustness_waits_for_completed_confirmation_stage(tmp_path):
    store = _store(tmp_path)
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
            "robustness", "site_shift", SeedBundle(701, 801, 901)
        ),
        n_train=5,
        model="phase05_candidate",
        variant="time_scaled__residual__deterministic",
        frozen_candidate_hash=candidate_hash,
    )

    with pytest.raises(
        RuntimeError,
        match="confirmation stage must be complete before robustness",
    ):
        run_phase05_jobs(
            (job,),
            store=store,
            config=Phase05Config(max_epochs=1, patience=1),
            options=Phase05ExecutionOptions(device="cpu", workers=1),
            prepare_shard=lambda spec: object(),
            run_job=lambda job, prepared, device: _metric_frame(job, 1.0),
        )


def test_duplicate_cell_ids_are_rejected_before_execution(tmp_path):
    store = _store(tmp_path)
    job = Phase05Job(
        shard=Phase05ShardSpec("flow", "smooth", SeedBundle(401, 501, 601)),
        n_train=5,
        model="phase05_flow_jump",
        variant="time_scaled__none__deterministic",
    )

    with pytest.raises(ValueError, match="cell IDs must be unique"):
        run_phase05_jobs(
            (job, job),
            store=store,
            config=Phase05Config(max_epochs=1, patience=1),
            options=Phase05ExecutionOptions(device="cpu", workers=2),
            prepare_shard=_spawn_prepare,
            run_job=_spawn_runner,
        )
