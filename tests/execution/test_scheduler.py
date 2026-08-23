import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from afmc_fm.experiments.runner import ExperimentConfig
from afmc_fm.simulator.config import SimulatorConfig

SCIENTIFIC_KEY = [
    "benchmark",
    "world",
    "cohort_seed",
    "subset_seed",
    "model_seed",
    "n_train",
    "model",
    "ablation",
    "split",
    "site_or_shift",
    "metric",
]


def _persisted_cell_ids(output: Path) -> set[str]:
    return {
        json.loads(path.read_text(encoding="utf-8"))["cell"]["cell_id"]
        for path in output.glob("shards/*/cells/*.json")
    }


def test_one_and_two_spawn_workers_are_scientifically_equivalent(tmp_path):
    from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark

    simulator = SimulatorConfig(cohort_size=30, followup_days=45.0)
    experiment = ExperimentConfig(
        train_sizes=(5,),
        worlds=("smooth",),
        cohort_seeds=(17, 19),
        subset_seeds=(23, 29),
        model_seeds=(31, 37),
        models=("engineered_linear",),
        ablations=("none",),
        max_epochs=1,
        patience=1,
    )
    serial_output = tmp_path / "serial"
    parallel_output = tmp_path / "parallel"

    serial = run_scheduled_benchmark(
        simulator,
        experiment,
        serial_output,
        ExecutionOptions(device="cpu", workers=1, resume=False, fail_fast=True),
    ).sort_values(SCIENTIFIC_KEY, ignore_index=True)
    parallel = run_scheduled_benchmark(
        simulator,
        experiment,
        parallel_output,
        ExecutionOptions(device="cpu", workers=2, resume=False, fail_fast=True),
    ).sort_values(SCIENTIFIC_KEY, ignore_index=True)
    serial.attrs.clear()
    parallel.attrs.clear()

    assert _persisted_cell_ids(serial_output) == _persisted_cell_ids(parallel_output)
    pd.testing.assert_frame_equal(
        serial.drop(columns="value"),
        parallel.drop(columns="value"),
        check_like=False,
    )
    max_delta = float(
        np.max(np.abs(serial["value"].to_numpy() - parallel["value"].to_numpy()))
    )
    np.testing.assert_allclose(
        serial["value"].to_numpy(),
        parallel["value"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert max_delta <= 1e-12
    print(f"serial_parallel_max_delta={max_delta:.17g}")


def _tiny_experiment(*, worlds=("smooth",), seed_bundles=1) -> ExperimentConfig:
    return ExperimentConfig(
        train_sizes=(5,),
        worlds=worlds,
        cohort_seeds=tuple(range(17, 17 + seed_bundles)),
        subset_seeds=tuple(range(23, 23 + seed_bundles)),
        model_seeds=tuple(range(31, 31 + seed_bundles)),
        models=("engineered_linear",),
        ablations=("none",),
        max_epochs=1,
        patience=1,
    )


def test_workers_use_spawn_one_thread_and_one_matched_shard_per_future(
    tmp_path, monkeypatch
):
    from concurrent.futures import wait as real_wait

    from afmc_fm.execution import scheduler

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    parent_values_during_poll: list[str | None] = []

    def observed_wait(*args, **kwargs):
        parent_values_during_poll.append(os.environ.get("OPENBLAS_NUM_THREADS"))
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(scheduler, "wait", observed_wait)
    results = scheduler.run_scheduled_benchmark(
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        _tiny_experiment(seed_bundles=2),
        tmp_path / "run",
        scheduler.ExecutionOptions(
            device="cpu", workers=2, resume=False, fail_fast=True
        ),
    )

    worker_metadata = results.attrs["worker_metadata"]
    assert len(worker_metadata) == 2
    assert {metadata["start_method"] for metadata in worker_metadata} == {"spawn"}
    assert {metadata["torch_threads"] for metadata in worker_metadata} == {1}
    assert all(
        set(metadata["thread_environment"].values()) == {"1"}
        for metadata in worker_metadata
    )
    assert all(
        set(metadata["inherited_thread_environment"].values()) == {"1"}
        for metadata in worker_metadata
    )
    assert all(metadata["numeric_thread_pools"] for metadata in worker_metadata)
    assert all(
        pool["num_threads"] == 1
        for metadata in worker_metadata
        for pool in metadata["numeric_thread_pools"]
    )
    assert {
        (
            metadata["world"],
            metadata["cohort_seed"],
            metadata["subset_seed"],
            metadata["model_seed"],
        )
        for metadata in worker_metadata
    } == {("smooth", 17, 23, 31), ("smooth", 18, 24, 32)}
    assert len({metadata["shard_id"] for metadata in worker_metadata}) == 2
    assert set(parent_values_during_poll) == {"7"}
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"


def test_progress_uses_persisted_cells_and_reports_every_required_field(tmp_path):
    from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark

    results = run_scheduled_benchmark(
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        _tiny_experiment(),
        tmp_path / "run",
        ExecutionOptions(device="cpu", workers=1, resume=False, fail_fast=True),
    )

    final = results.attrs["progress_events"][-1]
    assert final.keys() == {
        "active_workers",
        "cells_complete",
        "cells_expected",
        "completion_percentage",
        "device",
        "elapsed_seconds",
        "failures",
        "shards_complete",
        "shards_total",
    }
    assert final["shards_complete"] == final["shards_total"] == 1
    assert final["cells_complete"] == final["cells_expected"] == 1
    assert final["failures"] == 0
    assert final["active_workers"] == 0
    assert final["device"] == "cpu"
    assert final["completion_percentage"] == 100.0
    assert final["elapsed_seconds"] >= 0.0
    assert "eta" not in final


def test_fail_fast_false_records_failure_and_independent_shard_persists(tmp_path):
    from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark

    output = tmp_path / "run"
    results = run_scheduled_benchmark(
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        _tiny_experiment(worlds=("not_a_world", "smooth")),
        output,
        ExecutionOptions(device="cpu", workers=2, resume=False, fail_fast=False),
    )

    assert set(results["world"]) == {"smooth"}
    assert len(_persisted_cell_ids(output)) == 1
    assert (output / "shards" / "smooth__cohort17__subset23__model31" / "COMPLETE").exists()
    failure = results.attrs["failures"][0]
    assert failure == {
        "cohort_seed": 17,
        "device": "cpu",
        "exception_message": "unknown simulation world: not_a_world",
        "exception_type": "ValueError",
        "model_seed": 31,
        "shard_id": "not_a_world__cohort17__subset23__model31",
        "subset_seed": 23,
        "world": "not_a_world",
    }


def test_fail_fast_true_cancels_pending_futures_and_propagates_structured_failure(
    tmp_path,
):
    import multiprocessing

    from afmc_fm.execution.scheduler import (
        ExecutionOptions,
        ScheduledBenchmarkError,
        run_scheduled_benchmark,
    )

    experiment = _tiny_experiment(
        worlds=tuple(f"not_a_world_{index}" for index in range(6))
    )
    output = tmp_path / "run"
    children_before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(ScheduledBenchmarkError) as caught:
        run_scheduled_benchmark(
            SimulatorConfig(cohort_size=30, followup_days=45.0),
            experiment,
            output,
            ExecutionOptions(device="cpu", workers=1, resume=False, fail_fast=True),
        )

    error = caught.value
    assert error.failure.exception_type == "ValueError"
    assert error.failure.device == "cpu"
    assert error.failure.shard_id.startswith("not_a_world_")
    assert error.cancelled_pending >= 0
    assert error.running_not_terminated >= 0
    assert {child.pid for child in multiprocessing.active_children()} <= children_before
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    root = json.loads((output / "run_record.json").read_text(encoding="utf-8"))
    assert (output / "metrics.csv").exists()
    assert (output / "ablation_metrics.csv").exists()
    assert (output / "gate_summary.csv").exists()
    assert (output / "learning_curves.png").exists()
    assert manifest["failures"][0]["shard_id"] == error.failure.shard_id
    assert manifest["expected_shard_count"] == 6
    assert (
        manifest["completed_shard_count"]
        + manifest["failed_shard_count"]
        + manifest["cancelled_shard_count"]
        + manifest["incomplete_shard_count"]
        == 6
    )
    assert (
        manifest["completed_cell_count"]
        + manifest["failed_cell_count"]
        + manifest["cancelled_cell_count"]
        + manifest["incomplete_cell_count"]
        == manifest["expected_cell_count"]
    )
    assert root["last_invocation"]["terminal_state"] == "fail_fast"


def test_fail_fast_cancellation_distinguishes_pending_from_running_futures():
    from concurrent.futures import Future

    from afmc_fm.execution.scheduler import _cancel_pending_futures

    pending = Future()
    first_running = Future()
    second_running = Future()
    assert first_running.set_running_or_notify_cancel()
    assert second_running.set_running_or_notify_cancel()

    cancelled, not_terminated = _cancel_pending_futures(
        [pending, first_running, second_running], worker_limit=1
    )

    assert cancelled == 1
    assert pending.cancelled()
    assert not_terminated == 2
    assert first_running.running()
    assert second_running.running()


def test_progress_cycle_uses_one_validated_snapshot_for_all_shards():
    from afmc_fm.execution.scheduler import _persisted_progress

    class InstrumentedStore:
        def __init__(self):
            self.snapshot_calls = 0

        def load_completed_cell_ids_by_shard(self):
            self.snapshot_calls += 1
            return {
                "shard-1": frozenset({"cell-1"}),
                "shard-2": frozenset({"cell-2", "unexpected-cell"}),
            }

    store = InstrumentedStore()
    expected = {
        "shard-1": frozenset({"cell-1"}),
        "shard-2": frozenset({"cell-2"}),
        "shard-3": frozenset({"cell-3"}),
    }

    shards_complete, cells_complete = _persisted_progress(store, expected)

    assert shards_complete == 1
    assert cells_complete == 2
    assert store.snapshot_calls == 1


def test_fully_resumed_shard_succeeds_when_worker_returns_columnless_empty_frame(
    tmp_path,
):
    from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark

    simulator = SimulatorConfig(cohort_size=30, followup_days=45.0)
    experiment = _tiny_experiment()
    output = tmp_path / "run"
    options = ExecutionOptions(device="cpu", workers=1, resume=True, fail_fast=True)
    first = run_scheduled_benchmark(simulator, experiment, output, options)

    resumed = run_scheduled_benchmark(simulator, experiment, output, options)

    assert not resumed.empty
    assert resumed.attrs["worker_metadata"][0]["returned_rows"] == 0
    assert resumed.attrs["worker_metadata"][0]["returned_columns"] == []
    assert resumed.attrs["progress_events"][-1]["cells_complete"] == 1
    assert resumed.attrs["progress_events"][-1]["shards_complete"] == 1
    first.attrs.clear()
    resumed.attrs.clear()
    pd.testing.assert_frame_equal(
        first.sort_values(SCIENTIFIC_KEY, ignore_index=True),
        resumed.sort_values(SCIENTIFIC_KEY, ignore_index=True),
    )


def test_custom_site_shift_world_plans_exactly_the_cells_the_runner_executes(tmp_path):
    from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark

    simulator = SimulatorConfig(
        cohort_size=30,
        followup_days=45.0,
        world_name="site_shift",
    )
    experiment = ExperimentConfig(
        train_sizes=(5,),
        worlds=("custom",),
        cohort_seeds=(17,),
        subset_seeds=(23,),
        model_seeds=(31,),
        models=("flow_jump",),
        ablations=("none",),
        max_epochs=1,
        patience=1,
    )
    output = tmp_path / "run"

    results = run_scheduled_benchmark(
        simulator,
        experiment,
        output,
        ExecutionOptions(device="cpu", workers=1, resume=False, fail_fast=True),
    )

    assert _persisted_cell_ids(output) == {
        "low_n__custom__cohort17__subset23__model31__n5__flow_jump__none",
        "observation_shift__custom__cohort17__subset23__model31__n5__flow_jump__none",
    }
    assert set(results["benchmark"]) == {"low_n", "observation_shift"}
    assert set(results["world"]) == {"custom"}


def test_fresh_run_rejects_existing_run_artifacts_without_modifying_them(tmp_path):
    from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark

    simulator = SimulatorConfig(cohort_size=30, followup_days=45.0)
    experiment = _tiny_experiment()
    output = tmp_path / "run"
    fresh = ExecutionOptions(device="cpu", workers=1, resume=False, fail_fast=True)
    run_scheduled_benchmark(simulator, experiment, output, fresh)
    cell_path = next(output.glob("shards/*/cells/*.json"))
    original = cell_path.read_bytes()

    with pytest.raises(ValueError, match="existing run artifacts.*resume"):
        run_scheduled_benchmark(simulator, experiment, output, fresh)

    assert cell_path.read_bytes() == original


def test_resume_rejects_valid_same_identity_cell_from_unplanned_shard(tmp_path):
    from afmc_fm.execution.jobs import CellResult, ShardSpec
    from afmc_fm.execution.persistence import RunStore
    from afmc_fm.execution.scheduler import (
        ExecutionOptions,
        _run_identity,
        run_scheduled_benchmark,
    )

    simulator = SimulatorConfig(cohort_size=30, followup_days=45.0)
    experiment = _tiny_experiment()
    output = tmp_path / "run"
    planned = run_scheduled_benchmark(
        simulator,
        experiment,
        output,
        ExecutionOptions(device="cpu", workers=1, resume=False, fail_fast=True),
    )
    unplanned_shard = ShardSpec("jumps", 17, 23, 31)
    unplanned_metrics = planned.copy()
    unplanned_metrics.attrs.clear()
    unplanned_metrics["world"] = "jumps"
    RunStore(output, _run_identity(simulator, experiment)).write_cell(
        CellResult(
            shard=unplanned_shard,
            benchmark="low_n",
            n_train=5,
            model="engineered_linear",
            ablation="none",
            metrics=unplanned_metrics,
        )
    )

    with pytest.raises(ValueError, match="unplanned persisted cells.*jumps"):
        run_scheduled_benchmark(
            simulator,
            experiment,
            output,
            ExecutionOptions(device="cpu", workers=1, resume=True, fail_fast=True),
        )


def test_progress_observes_checkpoint_before_multi_shard_future_completion(tmp_path):
    from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark

    experiment = ExperimentConfig(
        train_sizes=(5,),
        worlds=("smooth",),
        cohort_seeds=(17, 19),
        subset_seeds=(23, 29),
        model_seeds=(31, 37),
        models=("engineered_linear", "flow_jump"),
        ablations=("none",),
        max_epochs=100,
        patience=100,
    )

    results = run_scheduled_benchmark(
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        experiment,
        tmp_path / "run",
        ExecutionOptions(device="cpu", workers=1, resume=False, fail_fast=True),
    )

    events = results.attrs["progress_events"]
    assert any(
        event["cells_complete"] in {1, 3}
        for event in events
    )
    assert any(event["active_workers"] == 1 for event in events[:-1])
    assert all(0 <= event["active_workers"] <= 1 for event in events)


@pytest.mark.parametrize(
    ("axis", "experiment"),
    [
        ("train_sizes", replace(_tiny_experiment(), train_sizes=(5, 5))),
        (
            "models",
            replace(
                _tiny_experiment(),
                models=("engineered_linear", "engineered_linear"),
            ),
        ),
        ("ablations", replace(_tiny_experiment(), ablations=("none", "none"))),
    ],
)
def test_duplicate_experiment_axes_are_rejected_before_submission(
    tmp_path, axis, experiment
):
    from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark

    output = tmp_path / "run"
    with pytest.raises(ValueError, match=f"duplicate {axis}"):
        run_scheduled_benchmark(
            SimulatorConfig(cohort_size=30, followup_days=45.0),
            experiment,
            output,
            ExecutionOptions(device="cpu", workers=1, resume=False, fail_fast=True),
        )

    assert not output.exists()


def test_empty_shard_plan_is_rejected_before_submission(tmp_path):
    from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark

    with pytest.raises(ValueError, match="at least one shard"):
        run_scheduled_benchmark(
            SimulatorConfig(cohort_size=30, followup_days=45.0),
            replace(_tiny_experiment(), worlds=()),
            tmp_path / "run",
            ExecutionOptions(device="cpu", workers=1, resume=False, fail_fast=True),
        )


def test_partial_submission_failure_shuts_down_executor(tmp_path, monkeypatch):
    from concurrent.futures import Future

    from afmc_fm.execution import scheduler

    shutdown_calls: list[tuple[bool, bool]] = []

    class FailingExecutor:
        def __init__(self, **kwargs):
            self.submissions = 0

        def submit(self, function, request):
            self.submissions += 1
            if self.submissions == 2:
                raise RuntimeError("synchronous submission failure")
            return Future()

        def shutdown(self, *, wait, cancel_futures):
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", FailingExecutor)

    with pytest.raises(RuntimeError, match="synchronous submission failure"):
        scheduler.run_scheduled_benchmark(
            SimulatorConfig(cohort_size=30, followup_days=45.0),
            _tiny_experiment(seed_bundles=2),
            tmp_path / "run",
            scheduler.ExecutionOptions(
                device="cpu", workers=1, resume=False, fail_fast=True
            ),
        )

    assert shutdown_calls == [(True, True)]


@pytest.mark.parametrize(
    "artifact",
    [
        "run_record.json",
        "run_manifest.json",
        "metrics.csv",
        "ablation_metrics.csv",
        "gate_summary.csv",
        "learning_curves.png",
        "run.log",
        "shards",
    ],
)
def test_fresh_run_detects_every_known_run_artifact(tmp_path, artifact):
    from afmc_fm.execution.scheduler import _has_run_artifacts

    output = tmp_path / "run"
    path = output / artifact
    if artifact == "shards":
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("existing", encoding="utf-8")

    assert _has_run_artifacts(output)


def test_scheduler_captures_commit_and_persists_root_before_executor(
    tmp_path,
    monkeypatch,
):
    from concurrent.futures import ProcessPoolExecutor as RealExecutor

    from afmc_fm.execution import manifest, scheduler

    output = tmp_path / "run"
    executor_started = False
    commit_calls = 0

    def captured_commit():
        nonlocal commit_calls
        commit_calls += 1
        assert not executor_started
        return "a" * 40

    def observed_executor(**kwargs):
        nonlocal executor_started
        executor_started = True
        assert (output / "run_record.json").is_file()
        return RealExecutor(**kwargs)

    monkeypatch.setattr(scheduler, "execution_commit_sha", captured_commit, raising=False)
    monkeypatch.setattr(manifest, "execution_commit_sha", captured_commit)
    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", observed_executor)

    scheduler.run_scheduled_benchmark(
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        _tiny_experiment(),
        output,
        scheduler.ExecutionOptions(
            device="cpu", workers=1, resume=False, fail_fast=True
        ),
    )

    root = json.loads((output / "run_record.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert commit_calls == 1
    assert root["execution_commit_sha"] == manifest_payload["execution_commit_sha"] == "a" * 40


def test_resume_preserves_original_execution_sha_and_accumulates_timing(
    tmp_path,
    monkeypatch,
):
    from afmc_fm.execution import scheduler
    from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark

    simulator = SimulatorConfig(cohort_size=30, followup_days=45.0)
    experiment = _tiny_experiment()
    output = tmp_path / "run"
    options = ExecutionOptions(device="cpu", workers=1, resume=True, fail_fast=True)

    monkeypatch.setattr(scheduler, "execution_commit_sha", lambda: "a" * 40)
    run_scheduled_benchmark(simulator, experiment, output, options)
    first = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(scheduler, "execution_commit_sha", lambda: "b" * 40)
    run_scheduled_benchmark(simulator, experiment, output, options)
    second = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    root = json.loads((output / "run_record.json").read_text(encoding="utf-8"))

    assert second["original_started_at"] == first["original_started_at"]
    assert second["started_at"] == first["started_at"]
    first_entry = {"invocation_number": 1, "execution_commit_sha": "a" * 40}
    second_entry = {"invocation_number": 2, "execution_commit_sha": "b" * 40}
    assert first["execution_commit_sha"] == "a" * 40
    assert first["invocation_execution_commit_sha"] == "a" * 40
    assert first["execution_commit_history"] == [first_entry]
    assert second["execution_commit_sha"] == "a" * 40
    assert second["invocation_execution_commit_sha"] == "b" * 40
    assert second["execution_commit_history"] == [first_entry, second_entry]
    assert first["invocation_number"] == 1
    assert second["invocation_number"] == 2
    assert second["cumulative_wall_time_seconds"] >= (
        first["cumulative_wall_time_seconds"]
        + second["invocation_wall_time_seconds"]
    )
    assert second["wall_time_seconds"] == second["cumulative_wall_time_seconds"]
    assert root["completed_invocation_count"] == 2
    assert root["execution_commit_sha"] == "a" * 40
    assert root["execution_commit_history"] == [first_entry, second_entry]
    assert root["original_started_at"] == second["original_started_at"]
    assert root["cumulative_wall_time_seconds"] == second["cumulative_wall_time_seconds"]
    assert root["last_invocation"]["started_at"] == second["invocation_started_at"]
    assert root["last_invocation"]["ended_at"] == second["invocation_ended_at"]
    assert root["last_invocation"]["execution_commit_sha"] == "b" * 40


def test_invalid_backend_identifiers_fail_before_output_or_workers(tmp_path):
    from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark

    experiment = replace(_tiny_experiment(), models=("unknown_model",))
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="unknown models.*unknown_model"):
        run_scheduled_benchmark(
            SimulatorConfig(cohort_size=30, followup_days=45.0),
            experiment,
            output,
            ExecutionOptions(device="cpu", workers=1, resume=False, fail_fast=False),
        )

    assert not output.exists()
