import json
from functools import partial
from pathlib import Path

import pandas as pd

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import Phase05Config, SeedBundle
from afmc_fm.phase05.execution import (
    Phase05ExecutionOptions,
    Phase05Job,
    Phase05ShardSpec,
    run_phase05_jobs,
)
from afmc_fm.simulator.config import SimulatorConfig

from .test_store import _store


def _job() -> Phase05Job:
    return Phase05Job(
        shard=Phase05ShardSpec(
            stage="flow",
            world="smooth",
            seed_bundle=SeedBundle(401, 501, 601),
        ),
        n_train=5,
        model="phase05_flow_jump",
        variant="time_scaled__none__deterministic",
    )


def _frame(job: Phase05Job) -> pd.DataFrame:
    bundle = job.shard.seed_bundle
    return pd.DataFrame(
        [
            {
                "stage": job.shard.stage,
                "world": job.shard.world,
                "cohort_seed": bundle.cohort_seed,
                "subset_seed": bundle.subset_seed,
                "model_seed": bundle.model_seed,
                "n_train": job.n_train,
                "model": job.model,
                "variant": job.variant,
                "split": "test",
                "site_or_shift": "all",
                "metric": "mae",
                "value": 0.75,
            }
        ]
    )


def _prepare_with_simulator(_shard, *, simulator_config: SimulatorConfig):
    assert isinstance(simulator_config, SimulatorConfig)
    return object()


def test_run_phase05_jobs_persists_identity_bound_execution_provenance(tmp_path: Path):
    store = _store(tmp_path)
    job = _job()
    simulator_hash = "a" * 64

    result = run_phase05_jobs(
        (job,),
        store=store,
        config=Phase05Config(),
        options=Phase05ExecutionOptions(
            device="cpu",
            workers=1,
            simulator_config_hash=simulator_hash,
        ),
        prepare_shard=lambda _shard: object(),
        run_job=lambda planned, _prepared, _device: _frame(planned),
    )

    assert len(result) == 1
    path = store.output / "execution_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["identity"] == store.identity
    assert payload["simulator_config_sha256"] == simulator_hash
    assert len(payload["implementation_sha"]) == 40

    invocations = payload["invocations"]
    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation["stage"] == "flow"
    assert invocation["invocation_number"] == 1
    assert invocation["requested_device"] == "cpu"
    assert invocation["resolved_device"] == "cpu"
    assert invocation["workers"] == 1
    assert invocation["resume"] is False
    assert invocation["fail_fast"] is False
    assert invocation["expected_cell_count"] == 1
    assert invocation["completed_before"] == 0
    assert invocation["completed_after"] == 1
    assert invocation["shard_count"] == 1
    assert invocation["failures"] == []
    assert invocation["wall_time_seconds"] >= 0.0
    assert invocation["started_at"] <= invocation["ended_at"]
    assert invocation["runtime_metadata"]["python_version"]
    assert "torch" in invocation["runtime_metadata"]["library_versions"]
    assert "cuda_available" in invocation["runtime_metadata"]


def test_run_phase05_jobs_derives_hash_from_bound_simulator_config(tmp_path: Path):
    store = _store(tmp_path)
    job = _job()
    simulator_config = SimulatorConfig(cohort_size=12)

    result = run_phase05_jobs(
        (job,),
        store=store,
        config=Phase05Config(),
        options=Phase05ExecutionOptions(device="cpu", workers=1),
        prepare_shard=partial(
            _prepare_with_simulator,
            simulator_config=simulator_config,
        ),
        run_job=lambda planned, _prepared, _device: _frame(planned),
    )

    assert len(result) == 1
    payload = json.loads(
        (store.output / "execution_provenance.json").read_text(encoding="utf-8")
    )
    assert payload["simulator_config_sha256"] == canonical_config_hash(
        simulator_config
    )


def test_run_phase05_jobs_records_failed_invocation_for_resume(tmp_path: Path):
    store = _store(tmp_path)
    job = _job()
    simulator_hash = "b" * 64

    first = run_phase05_jobs(
        (job,),
        store=store,
        config=Phase05Config(),
        options=Phase05ExecutionOptions(
            device="cpu",
            workers=1,
            simulator_config_hash=simulator_hash,
        ),
        prepare_shard=lambda _shard: object(),
        run_job=lambda _job, _prepared, _device: (_ for _ in ()).throw(
            RuntimeError("injected fit failure")
        ),
    )
    assert first.empty

    second = run_phase05_jobs(
        (job,),
        store=store,
        config=Phase05Config(),
        options=Phase05ExecutionOptions(
            device="cpu",
            workers=1,
            resume=True,
            simulator_config_hash=simulator_hash,
        ),
        prepare_shard=lambda _shard: object(),
        run_job=lambda planned, _prepared, _device: _frame(planned),
    )
    assert len(second) == 1

    payload = json.loads(
        (store.output / "execution_provenance.json").read_text(encoding="utf-8")
    )
    assert len(payload["invocations"]) == 2
    failed, resumed = payload["invocations"]
    assert failed["invocation_number"] == 1
    assert failed["completed_after"] == 0
    assert failed["failures"] == [
        {"type": "RuntimeError", "message": "injected fit failure"}
    ]
    assert resumed["invocation_number"] == 2
    assert resumed["resume"] is True
    assert resumed["completed_before"] == 0
    assert resumed["completed_after"] == 1
    assert resumed["failures"] == []
