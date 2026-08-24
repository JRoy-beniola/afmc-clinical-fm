from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from afmc_fm.execution.device import resolve_device
from afmc_fm.phase05.config import Phase05Config, SeedBundle
from afmc_fm.phase05.store import Phase05CellResult, Phase05Store

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_CONFIRMATORY_STAGES = frozenset({"confirmation", "robustness"})
_DEVELOPMENT_STAGES = frozenset({"flow", "jump", "uncertainty", "timing_audit"})


@dataclass(frozen=True, slots=True)
class Phase05ShardSpec:
    stage: str
    world: str
    seed_bundle: SeedBundle

    @property
    def shard_id(self) -> str:
        return (
            f"{self.stage}__{self.world}__cohort{self.seed_bundle.cohort_seed}__"
            f"subset{self.seed_bundle.subset_seed}__model{self.seed_bundle.model_seed}"
        )


@dataclass(frozen=True, slots=True)
class Phase05Job:
    shard: Phase05ShardSpec
    n_train: int
    model: str
    variant: str
    frozen_candidate_hash: str | None = None

    def __post_init__(self) -> None:
        if type(self.n_train) is not int or self.n_train <= 0:
            raise ValueError("n_train must be a positive integer")
        if not self.model or not self.variant:
            raise ValueError("model and variant must be non-empty")

    @property
    def cell_id(self) -> str:
        return phase05_cell_id(
            self.shard,
            n_train=self.n_train,
            model=self.model,
            variant=self.variant,
        )


@dataclass(frozen=True, slots=True)
class Phase05ExecutionOptions:
    device: str = "auto"
    workers: int = 1
    resume: bool = False
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


def phase05_cell_id(
    shard: Phase05ShardSpec,
    *,
    n_train: int,
    model: str,
    variant: str,
) -> str:
    return f"{shard.shard_id}__n{n_train}__{model}__{variant}"


def spawn_context():
    return multiprocessing.get_context("spawn")


def limit_worker_threads() -> dict[str, str]:
    for variable in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    return {variable: os.environ[variable] for variable in THREAD_ENVIRONMENT_VARIABLES}


@contextmanager
def worker_thread_limits() -> Iterator[dict[str, str]]:
    previous_environment = {
        variable: os.environ.get(variable) for variable in THREAD_ENVIRONMENT_VARIABLES
    }
    previous_torch_threads = torch.get_num_threads()
    try:
        environment = limit_worker_threads()
        with threadpool_limits(limits=1):
            yield environment
    finally:
        torch.set_num_threads(previous_torch_threads)
        for variable, value in previous_environment.items():
            if value is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = value


def resolve_phase05_device(requested: str) -> torch.device:
    return resolve_device(requested)


def plan_phase05_shards(
    config: Phase05Config,
    stage: str,
) -> tuple[Phase05ShardSpec, ...]:
    if stage in _DEVELOPMENT_STAGES:
        worlds = config.target_worlds
        bundles = config.development_bundles
    elif stage == "confirmation":
        worlds = config.target_worlds
        bundles = config.confirmatory_bundles
    elif stage == "robustness":
        worlds = config.robustness_worlds
        bundles = config.confirmatory_bundles
    else:
        raise ValueError(f"unknown Phase-0.5 execution stage: {stage}")
    return tuple(
        Phase05ShardSpec(stage=stage, world=world, seed_bundle=bundle)
        for world in worlds
        for bundle in bundles
    )


def aggregate_phase05_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame.copy() for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    combined = pd.concat(nonempty, ignore_index=True, sort=False)
    preferred_order = (
        "stage",
        "world",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "n_train",
        "model",
        "variant",
        "split",
        "site_or_shift",
        "metric",
    )
    sort_columns = [column for column in preferred_order if column in combined.columns]
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="mergesort")
    return combined.reset_index(drop=True)


def _validate_job_scope(jobs: tuple[Phase05Job, ...], config: Phase05Config) -> str:
    if not jobs:
        raise ValueError("Phase-0.5 job plan must not be empty")
    stages = {job.shard.stage for job in jobs}
    if len(stages) != 1:
        raise ValueError("one run_phase05_jobs call may execute only one stage")
    stage = next(iter(stages))
    allowed = {shard.shard_id for shard in plan_phase05_shards(config, stage)}
    unexpected = {job.shard.shard_id for job in jobs} - allowed
    if unexpected:
        raise ValueError("jobs fall outside the locked stage seed/world scope")
    cell_ids = [job.cell_id for job in jobs]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Phase-0.5 job cell IDs must be unique")
    return stage


def _persist_result(store: Phase05Store, job: Phase05Job, frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Phase-0.5 job runner must return a pandas DataFrame")
    if frame.empty:
        raise ValueError("Phase-0.5 job runner returned no metric rows")
    bundle = job.shard.seed_bundle
    store.write_cell(
        Phase05CellResult(
            stage=job.shard.stage,
            world=job.shard.world,
            cohort_seed=bundle.cohort_seed,
            subset_seed=bundle.subset_seed,
            model_seed=bundle.model_seed,
            n_train=job.n_train,
            model=job.model,
            variant=job.variant,
            metrics=frame,
            frozen_candidate_hash=job.frozen_candidate_hash,
        )
    )


def _load_persisted_frame(store: Phase05Store, job: Phase05Job) -> pd.DataFrame:
    path = (
        store.output
        / "stages"
        / job.shard.stage
        / "cells"
        / f"{job.cell_id}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("metric_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"persisted Phase-0.5 cell has no metric rows: {job.cell_id}")
    return pd.DataFrame(rows)


def _candidate_hash(store: Phase05Store) -> str:
    path = store.output / "frozen_candidate.json"
    if not path.is_file():
        raise ValueError("frozen candidate hash unavailable: frozen_candidate.json missing")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_confirmation_binding(
    store: Phase05Store,
    jobs: tuple[Phase05Job, ...],
) -> str:
    marker = store.output / "confirmation" / "STARTED"
    if not marker.is_file():
        raise RuntimeError("confirmation must be started before confirmatory execution")
    candidate_hash = _candidate_hash(store)
    if any(job.frozen_candidate_hash != candidate_hash for job in jobs):
        raise ValueError("frozen candidate hash does not match persisted candidate")
    return candidate_hash


def run_phase05_jobs(
    jobs: Sequence[Phase05Job],
    *,
    store: Phase05Store,
    config: Phase05Config,
    options: Phase05ExecutionOptions,
    prepare_shard: Callable[[Phase05ShardSpec], Any],
    run_job: Callable[[Phase05Job, Any, torch.device], pd.DataFrame],
) -> pd.DataFrame:
    planned = tuple(jobs)
    stage = _validate_job_scope(planned, config)
    if options.workers != 1:
        raise NotImplementedError("parallel Phase-0.5 workers are not implemented yet")

    device = resolve_phase05_device(options.device)
    expected_cell_ids = frozenset(job.cell_id for job in planned)
    expected_seed_bundles = frozenset(job.shard.seed_bundle.as_tuple() for job in planned)
    candidate_hash = (
        _require_confirmation_binding(store, planned)
        if stage in _CONFIRMATORY_STAGES
        else None
    )
    completed = store.validate_resume(
        stage,
        expected_cell_ids=expected_cell_ids,
        expected_seed_bundles=expected_seed_bundles,
        frozen_candidate_hash=candidate_hash,
    )
    if completed and not options.resume:
        raise ValueError("persisted Phase-0.5 cells exist; enable resume to reuse them")

    by_shard: dict[str, list[Phase05Job]] = {}
    shard_specs: dict[str, Phase05ShardSpec] = {}
    for job in planned:
        by_shard.setdefault(job.shard.shard_id, []).append(job)
        shard_specs[job.shard.shard_id] = job.shard

    failures: list[BaseException] = []
    for shard_id, shard_jobs in by_shard.items():
        pending = [job for job in shard_jobs if job.cell_id not in completed]
        if not pending:
            continue
        with worker_thread_limits():
            prepared = prepare_shard(shard_specs[shard_id])
            for job in pending:
                try:
                    frame = run_job(job, prepared, device)
                    _persist_result(store, job, frame)
                except Exception as error:
                    failures.append(error)
                    if options.fail_fast:
                        raise

    persisted = store.validate_resume(
        stage,
        expected_cell_ids=expected_cell_ids,
        expected_seed_bundles=expected_seed_bundles,
        frozen_candidate_hash=candidate_hash,
    )
    if not failures and persisted == expected_cell_ids:
        store.mark_stage_complete(stage, expected_cell_ids)

    available_jobs = [job for job in planned if job.cell_id in persisted]
    frames = [_load_persisted_frame(store, job) for job in available_jobs]
    return aggregate_phase05_frames(frames)


__all__ = [
    "THREAD_ENVIRONMENT_VARIABLES",
    "Phase05ExecutionOptions",
    "Phase05Job",
    "Phase05ShardSpec",
    "aggregate_phase05_frames",
    "limit_worker_threads",
    "phase05_cell_id",
    "plan_phase05_shards",
    "resolve_phase05_device",
    "run_phase05_jobs",
    "spawn_context",
    "worker_thread_limits",
]
