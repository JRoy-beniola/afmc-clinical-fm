import json
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


def test_workers_use_spawn_one_thread_and_one_matched_shard_per_future(tmp_path):
    from afmc_fm.execution.scheduler import ExecutionOptions, run_scheduled_benchmark

    results = run_scheduled_benchmark(
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        _tiny_experiment(seed_bundles=2),
        tmp_path / "run",
        ExecutionOptions(device="cpu", workers=2, resume=False, fail_fast=True),
    )

    worker_metadata = results.attrs["worker_metadata"]
    assert len(worker_metadata) == 2
    assert {metadata["start_method"] for metadata in worker_metadata} == {"spawn"}
    assert {metadata["torch_threads"] for metadata in worker_metadata} == {1}
    assert all(
        set(metadata["thread_environment"].values()) == {"1"}
        for metadata in worker_metadata
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
    assert len({metadata["process_id"] for metadata in worker_metadata}) == 2


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
    from afmc_fm.execution.scheduler import (
        ExecutionOptions,
        ScheduledBenchmarkError,
        run_scheduled_benchmark,
    )

    experiment = _tiny_experiment(
        worlds=tuple(f"not_a_world_{index}" for index in range(6))
    )
    with pytest.raises(ScheduledBenchmarkError) as caught:
        run_scheduled_benchmark(
            SimulatorConfig(cohort_size=30, followup_days=45.0),
            experiment,
            tmp_path / "run",
            ExecutionOptions(device="cpu", workers=1, resume=False, fail_fast=True),
        )

    error = caught.value
    assert error.failure.exception_type == "ValueError"
    assert error.failure.device == "cpu"
    assert error.failure.shard_id.startswith("not_a_world_")
    assert error.cancelled_pending >= 1
    assert error.running_not_terminated >= 0


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
