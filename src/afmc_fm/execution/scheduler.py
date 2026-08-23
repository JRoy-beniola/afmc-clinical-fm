"""Spawn-based scheduling and persisted finalization for benchmark shards."""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from afmc_fm.execution.device import resolve_device
from afmc_fm.execution.jobs import (
    ShardSpec,
    benchmark_cell_id,
    effective_world_name,
    plan_shards,
    run_shard,
)
from afmc_fm.execution.manifest import (
    aggregate_persisted_metrics,
    baseline_backend_ids,
    build_run_manifest,
    execution_commit_sha,
    write_derived_artifacts,
    write_run_manifest,
)
from afmc_fm.execution.persistence import (
    PROTOCOL_ANCHOR,
    RunIdentity,
    RunStore,
    canonical_config_hash,
)
from afmc_fm.experiments.runner import ABLATION_IDS, MODEL_NAMES, ExperimentConfig
from afmc_fm.simulator.config import SimulatorConfig

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
PROGRESS_POLL_SECONDS = 0.25
KNOWN_RUN_ARTIFACTS = (
    "run_record.json",
    "run_manifest.json",
    "run.log",
    "shards",
    "metrics.csv",
    "ablation_metrics.csv",
    "gate_summary.csv",
    "learning_curves.png",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    device: str = "auto"
    workers: int = 1
    resume: bool = False
    fail_fast: bool = True

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


@dataclass(frozen=True, slots=True)
class ShardFailure:
    shard_id: str
    world: str
    cohort_seed: int
    subset_seed: int
    model_seed: int
    exception_type: str
    exception_message: str
    device: str


class ShardExecutionError(Exception):
    """Pickleable worker exception containing the failed shard identity."""

    def __init__(self, failure: ShardFailure) -> None:
        self.failure = failure
        super().__init__(failure)

    def __str__(self) -> str:
        return (
            f"shard {self.failure.shard_id} failed on {self.failure.device}: "
            f"{self.failure.exception_type}: {self.failure.exception_message}"
        )


class ScheduledBenchmarkError(RuntimeError):
    """Fail-fast scheduler error with explicit cancellation accounting."""

    def __init__(
        self,
        failure: ShardFailure,
        *,
        cancelled_pending: int,
        running_not_terminated: int,
    ) -> None:
        self.failure = failure
        self.cancelled_pending = cancelled_pending
        self.running_not_terminated = running_not_terminated
        self.running_at_cancellation = running_not_terminated
        super().__init__(
            f"{ShardExecutionError(failure)}; cancelled {cancelled_pending} pending "
            f"future(s); {running_not_terminated} running future(s) reached a "
            "quiescent boundary before finalization"
        )


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    shards_complete: int
    shards_total: int
    cells_complete: int
    cells_expected: int
    failures: int
    elapsed_seconds: float
    active_workers: int
    device: str
    completion_percentage: float


@dataclass(frozen=True, slots=True)
class _WorkerRequest:
    shard: ShardSpec
    simulator: SimulatorConfig
    experiment: ExperimentConfig
    output: Path
    identity: RunIdentity
    device: str
    completed_cell_ids: frozenset[str]
    expected_cell_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    shard: ShardSpec
    process_id: int
    start_method: str
    torch_threads: int
    thread_environment: dict[str, str]
    inherited_thread_environment: dict[str, str]
    numeric_thread_pools: tuple[dict[str, str | int], ...]
    returned_rows: int
    returned_columns: tuple[str, ...]


def _run_identity(
    simulator: SimulatorConfig,
    experiment: ExperimentConfig,
) -> RunIdentity:
    return RunIdentity(
        protocol_anchor=PROTOCOL_ANCHOR,
        simulator_config_hash=canonical_config_hash(simulator),
        experiment_config_hash=canonical_config_hash(experiment),
    )


def _model_variants(
    experiment: ExperimentConfig,
    *,
    observation_shift: bool,
) -> tuple[tuple[str, str], ...]:
    variants: list[tuple[str, str]] = []
    for model in experiment.models:
        if observation_shift and not model.startswith("flow_jump"):
            continue
        ablations = experiment.ablations if model.startswith("flow_jump") else ("none",)
        for ablation in ablations:
            if ablation == "no_observation_head" and model != "flow_jump_observation":
                continue
            variants.append((model, ablation))
    return tuple(variants)


def _expected_cell_ids(
    shard: ShardSpec,
    sim_config: SimulatorConfig,
    experiment: ExperimentConfig,
) -> frozenset[str]:
    expected = {
        benchmark_cell_id("low_n", shard, n_train, model, ablation)
        for n_train in experiment.train_sizes
        for model, ablation in _model_variants(experiment, observation_shift=False)
    }
    if effective_world_name(shard, sim_config) == "site_shift":
        expected.update(
            benchmark_cell_id(
                "observation_shift", shard, n_train, model, ablation
            )
            for n_train in experiment.train_sizes
            for model, ablation in _model_variants(
                experiment, observation_shift=True
            )
        )
    if not expected:
        raise ValueError(f"shard has no expected benchmark cells: {shard.shard_id}")
    return frozenset(expected)


def _limit_worker_threads() -> dict[str, str]:
    for variable in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    return {variable: os.environ[variable] for variable in THREAD_ENVIRONMENT_VARIABLES}


@contextmanager
def _inherited_thread_limits():
    previous = {
        variable: os.environ.get(variable)
        for variable in THREAD_ENVIRONMENT_VARIABLES
    }
    try:
        for variable in THREAD_ENVIRONMENT_VARIABLES:
            os.environ[variable] = "1"
        yield
    finally:
        for variable, value in previous.items():
            if value is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = value


def _numeric_thread_pools() -> tuple[dict[str, str | int], ...]:
    return tuple(
        {
            "user_api": str(pool["user_api"]),
            "internal_api": str(pool["internal_api"]),
            "prefix": str(pool["prefix"]),
            "num_threads": int(pool["num_threads"]),
        }
        for pool in threadpool_info()
    )


def _failure_for(
    shard: ShardSpec,
    device: str,
    error: BaseException,
) -> ShardFailure:
    return ShardFailure(
        shard_id=shard.shard_id,
        world=shard.world,
        cohort_seed=shard.cohort_seed,
        subset_seed=shard.subset_seed,
        model_seed=shard.model_seed,
        exception_type=type(error).__name__,
        exception_message=str(error),
        device=device,
    )


def _run_shard_worker(request: _WorkerRequest) -> _WorkerResult:
    """Run exactly one complete shard; all arguments and results are pickleable."""
    inherited_thread_environment = {
        variable: os.environ.get(variable, "")
        for variable in THREAD_ENVIRONMENT_VARIABLES
    }
    thread_environment = _limit_worker_threads()
    try:
        with threadpool_limits(limits=1):
            store = RunStore(request.output, request.identity)
            frame = run_shard(
                request.shard,
                request.simulator,
                request.experiment,
                torch.device(request.device),
                completed_cell_ids=request.completed_cell_ids,
                on_cell_complete=store.write_cell,
            )
            if frame.empty:
                frame = pd.DataFrame()
            persisted = store.load_completed_cell_ids(request.shard.shard_id)
            if persisted != request.expected_cell_ids:
                missing = sorted(request.expected_cell_ids - persisted)
                unexpected = sorted(persisted - request.expected_cell_ids)
                raise ValueError(
                    "persisted shard cells do not match expectation "
                    f"(missing={missing}, unexpected={unexpected})"
                )
            store.mark_shard_complete(request.shard.shard_id, request.expected_cell_ids)
            return _WorkerResult(
                shard=request.shard,
                process_id=os.getpid(),
                start_method=multiprocessing.get_start_method(),
                torch_threads=torch.get_num_threads(),
                thread_environment=thread_environment,
                inherited_thread_environment=inherited_thread_environment,
                numeric_thread_pools=_numeric_thread_pools(),
                returned_rows=len(frame),
                returned_columns=tuple(str(column) for column in frame.columns),
            )
    except Exception as error:
        if isinstance(error, ShardExecutionError):
            raise
        raise ShardExecutionError(
            _failure_for(request.shard, request.device, error)
        ) from error


def _persisted_progress(
    store: RunStore,
    expected_by_shard: dict[str, frozenset[str]],
) -> tuple[int, int]:
    completed_by_shard = store.load_completed_cell_ids_by_shard()
    cells_complete = sum(
        len(expected.intersection(completed_by_shard.get(shard_id, frozenset())))
        for shard_id, expected in expected_by_shard.items()
    )
    shards_complete = sum(
        completed_by_shard.get(shard_id, frozenset()) == expected
        for shard_id, expected in expected_by_shard.items()
    )
    return shards_complete, cells_complete


def _validate_experiment_axes(experiment: ExperimentConfig) -> None:
    for name in ("train_sizes", "models", "ablations"):
        values = getattr(experiment, name)
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {name} are not allowed")
    unknown_models = set(experiment.models) - set(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"unknown models: {sorted(unknown_models)}")
    unknown_ablations = set(experiment.ablations) - set(ABLATION_IDS)
    if unknown_ablations:
        raise ValueError(f"unknown ablations: {sorted(unknown_ablations)}")


def _validate_persisted_scope(
    completed_by_shard: dict[str, frozenset[str]],
    expected_cell_ids: frozenset[str],
) -> frozenset[str]:
    completed = (
        frozenset().union(*completed_by_shard.values())
        if completed_by_shard
        else frozenset()
    )
    unexpected = completed - expected_cell_ids
    if unexpected:
        raise ValueError(
            "unplanned persisted cells are not allowed: "
            + ", ".join(sorted(unexpected))
        )
    return completed


def _has_run_artifacts(output: Path) -> bool:
    return any(os.path.lexists(output / name) for name in KNOWN_RUN_ARTIFACTS)


def _progress_event(
    *,
    store: RunStore,
    expected_by_shard: dict[str, frozenset[str]],
    shards_total: int,
    cells_expected: int,
    failures: int,
    started: float,
    active_workers: int,
    device: str,
) -> ProgressEvent:
    shards_complete, cells_complete = _persisted_progress(store, expected_by_shard)
    percentage = 100.0 * cells_complete / cells_expected
    return ProgressEvent(
        shards_complete=shards_complete,
        shards_total=shards_total,
        cells_complete=cells_complete,
        cells_expected=cells_expected,
        failures=failures,
        elapsed_seconds=time.monotonic() - started,
        active_workers=active_workers,
        device=device,
        completion_percentage=percentage,
    )


def _emit_progress(event: ProgressEvent) -> dict[str, Any]:
    payload = asdict(event)
    logger.info("benchmark progress: %s", payload, extra={"progress": payload})
    return payload


def _active_workers(
    futures: dict[Future[_WorkerResult], ShardSpec],
    worker_limit: int,
) -> int:
    running_or_queued = sum(
        future.running() and not future.done() for future in futures
    )
    return min(worker_limit, running_or_queued)


def _cancel_pending_futures(
    futures: list[Future[_WorkerResult]],
    worker_limit: int,
) -> tuple[int, int]:
    if worker_limit < 1:
        raise ValueError("worker_limit must be at least 1")
    cancelled = sum(future.cancel() for future in futures)
    running_not_terminated = sum(
        future.running() and not future.cancelled() for future in futures
    )
    return cancelled, running_not_terminated


def _worker_metadata(result: _WorkerResult) -> dict[str, Any]:
    return {
        **asdict(result.shard),
        "shard_id": result.shard.shard_id,
        "process_id": result.process_id,
        "start_method": result.start_method,
        "torch_threads": result.torch_threads,
        "thread_environment": result.thread_environment,
        "inherited_thread_environment": result.inherited_thread_environment,
        "numeric_thread_pools": [dict(pool) for pool in result.numeric_thread_pools],
        "returned_rows": result.returned_rows,
        "returned_columns": list(result.returned_columns),
    }


def run_scheduled_benchmark(
    sim_config: SimulatorConfig,
    experiment: ExperimentConfig,
    output: Path,
    options: ExecutionOptions,
    *,
    template_seed: int = 0,
) -> pd.DataFrame:
    """Run deterministic whole-shard futures and rebuild rows from persistence."""
    output = Path(output).resolve()
    _validate_experiment_axes(experiment)
    shards = plan_shards(experiment, supplied_seed=template_seed)
    if not shards:
        raise ValueError("benchmark plan must contain at least one shard")
    if len({shard.shard_id for shard in shards}) != len(shards):
        raise ValueError("planned benchmark shards must have unique identities")
    expected_by_shard = {
        shard.shard_id: _expected_cell_ids(shard, sim_config, experiment)
        for shard in shards
    }
    expected_cell_ids = frozenset().union(*expected_by_shard.values())
    cells_expected = sum(len(expected) for expected in expected_by_shard.values())
    existing_artifacts = _has_run_artifacts(output)
    if not options.resume and existing_artifacts:
        raise ValueError(
            "output contains existing run artifacts; choose a fresh output or enable resume"
        )
    device = str(resolve_device(options.device))
    identity = _run_identity(sim_config, experiment)
    store = RunStore(output, identity)
    captured_execution_commit = execution_commit_sha()
    invocation_started_at = datetime.now(UTC)
    started = time.monotonic()
    store.initialize_run(
        expected_by_shard,
        execution_commit_sha=captured_execution_commit,
        original_started_at=invocation_started_at,
        resume=options.resume and existing_artifacts,
    )
    store.validate_resume()
    initial_snapshot = store.load_completed_cell_ids_by_shard()
    _validate_persisted_scope(initial_snapshot, expected_cell_ids)
    failures: list[ShardFailure] = []
    cancelled_shard_ids: set[str] = set()
    progress_events: list[dict[str, Any]] = []
    worker_results: list[_WorkerResult] = []
    fail_fast_error: ScheduledBenchmarkError | None = None
    fail_fast_cause: BaseException | None = None

    requests = [
        _WorkerRequest(
            shard=shard,
            simulator=sim_config,
            experiment=experiment,
            output=output,
            identity=identity,
            device=device,
            completed_cell_ids=(
                initial_snapshot.get(shard.shard_id, frozenset())
                if options.resume
                else frozenset()
            ),
            expected_cell_ids=expected_by_shard[shard.shard_id],
        )
        for shard in shards
    ]

    ctx = multiprocessing.get_context("spawn")
    executor: ProcessPoolExecutor | None = None
    futures: dict[Future[_WorkerResult], ShardSpec] = {}
    last_progress: tuple[int, ...] | None = None

    def record_progress(*, force: bool = False, active_workers: int | None = None) -> None:
        nonlocal last_progress
        event = _progress_event(
            store=store,
            expected_by_shard=expected_by_shard,
            shards_total=len(shards),
            cells_expected=cells_expected,
            failures=len(failures),
            started=started,
            active_workers=(
                _active_workers(futures, options.workers)
                if active_workers is None
                else min(options.workers, max(0, active_workers))
            ),
            device=device,
        )
        signature = (
            event.shards_complete,
            event.cells_complete,
            event.failures,
            event.active_workers,
        )
        if force or signature != last_progress:
            progress_events.append(_emit_progress(event))
            last_progress = signature

    try:
        with _inherited_thread_limits():
            executor = ProcessPoolExecutor(
                max_workers=options.workers,
                mp_context=ctx,
            )
            for request in requests:
                future = executor.submit(_run_shard_worker, request)
                futures[future] = request.shard

        pending = set(futures)
        record_progress(force=True)
        while pending:
            done, _ = wait(
                pending,
                timeout=PROGRESS_POLL_SECONDS,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                record_progress()
                continue
            for future in done:
                pending.remove(future)
                shard = futures[future]
                if future.cancelled():
                    cancelled_shard_ids.add(shard.shard_id)
                    record_progress(force=True)
                    continue
                try:
                    worker_results.append(future.result())
                except Exception as error:  # noqa: BLE001 - worker errors are data
                    failure = (
                        error.failure
                        if isinstance(error, ShardExecutionError)
                        else _failure_for(shard, device, error)
                    )
                    failures.append(failure)
                    logger.error("benchmark shard failure: %s", asdict(failure))
                    if options.fail_fast and fail_fast_error is None:
                        candidates = [
                            candidate
                            for candidate in futures
                            if candidate is not future and not candidate.done()
                        ]
                        cancelled, running = _cancel_pending_futures(
                            candidates,
                            options.workers,
                        )
                        cancelled_shard_ids.update(
                            futures[candidate].shard_id
                            for candidate in candidates
                            if candidate.cancelled()
                        )
                        record_progress(force=True, active_workers=running)
                        fail_fast_error = ScheduledBenchmarkError(
                            failure,
                            cancelled_pending=cancelled,
                            running_not_terminated=running,
                        )
                        fail_fast_cause = error
                record_progress(force=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    store.validate_resume()
    final_snapshot = store.load_completed_cell_ids_by_shard()
    _validate_persisted_scope(final_snapshot, expected_cell_ids)
    frame = aggregate_persisted_metrics(store, expected_by_shard)
    frame.attrs["failures"] = [asdict(failure) for failure in failures]
    frame.attrs["cancelled_shard_ids"] = sorted(cancelled_shard_ids)
    frame.attrs["progress_events"] = progress_events
    frame.attrs["worker_metadata"] = [
        _worker_metadata(result)
        for result in sorted(worker_results, key=lambda item: item.shard.shard_id)
    ]
    write_derived_artifacts(frame, output)
    ended_at = datetime.now(UTC)
    invocation_wall_time = time.monotonic() - started
    root_record = store.complete_invocation(
        expected_by_shard,
        execution_commit_sha=captured_execution_commit,
        invocation_started_at=invocation_started_at,
        invocation_ended_at=ended_at,
        invocation_wall_time_seconds=invocation_wall_time,
        terminal_state=(
            "fail_fast"
            if fail_fast_error is not None
            else "completed_with_failures" if failures else "completed"
        ),
    )
    original_started_at = datetime.fromisoformat(root_record["original_started_at"])
    manifest = build_run_manifest(
        simulator=sim_config,
        experiment=experiment,
        resolved_device=device,
        worker_count=options.workers,
        backend_ids=baseline_backend_ids(experiment),
        started_at=original_started_at,
        ended_at=ended_at,
        wall_time_seconds=float(root_record["cumulative_wall_time_seconds"]),
        expected_by_shard=expected_by_shard,
        completed_by_shard=final_snapshot,
        failures=failures,
        cancelled_shard_ids=frozenset(cancelled_shard_ids),
        execution_commit_sha=captured_execution_commit,
        template_seed=template_seed,
        original_started_at=original_started_at,
        invocation_started_at=invocation_started_at,
        invocation_number=int(root_record["completed_invocation_count"]),
        invocation_wall_time_seconds=invocation_wall_time,
        cumulative_wall_time_seconds=float(
            root_record["cumulative_wall_time_seconds"]
        ),
    )
    write_run_manifest(manifest, output)
    if fail_fast_error is not None:
        raise fail_fast_error from fail_fast_cause
    return frame


__all__ = [
    "ExecutionOptions",
    "ProgressEvent",
    "ScheduledBenchmarkError",
    "ShardFailure",
    "run_scheduled_benchmark",
]
