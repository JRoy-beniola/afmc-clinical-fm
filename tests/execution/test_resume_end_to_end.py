from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from afmc_fm.execution import scheduler
from afmc_fm.execution.persistence import RunStore
from afmc_fm.experiments import runner
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


class _InlineExecutor:
    """Synchronous executor retaining the scheduler's public control flow."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def submit(self, function, /, *args, **kwargs) -> Future:
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as error:  # noqa: BLE001 - worker exceptions are scheduler data
            future.set_exception(error)
        return future

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        del wait, cancel_futures


def _tiny_cpu_config() -> tuple[SimulatorConfig, ExperimentConfig]:
    return (
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        ExperimentConfig(
            train_sizes=(5,),
            worlds=("smooth",),
            cohort_seeds=(17,),
            subset_seeds=(23,),
            model_seeds=(31,),
            models=("engineered_linear", "flow_jump"),
            ablations=("none",),
            max_epochs=1,
            patience=1,
        ),
    )


def _sorted_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(SCIENTIFIC_KEY, ignore_index=True)


def _only_persisted_cell(output: Path) -> Path:
    cells = tuple(output.glob("shards/*/cells/*.json"))
    assert len(cells) == 1
    return cells[0]


def test_public_scheduler_resume_preserves_completed_cell_and_skips_its_fits(
    tmp_path, monkeypatch
):
    simulator, experiment = _tiny_cpu_config()
    options = scheduler.ExecutionOptions(
        device="cpu", workers=1, resume=False, fail_fast=True
    )
    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", _InlineExecutor)

    execution_calls: list[str] = []
    real_ridge_fit = runner.TorchRidgeRegressor.fit
    real_fit_neural = runner._fit_neural

    def tracked_ridge_fit(estimator, features, targets):
        execution_calls.append("engineered_linear")
        return real_ridge_fit(estimator, features, targets)

    def tracked_fit_neural(*args, **kwargs):
        execution_calls.append("flow_jump")
        return real_fit_neural(*args, **kwargs)

    monkeypatch.setattr(runner.TorchRidgeRegressor, "fit", tracked_ridge_fit)
    monkeypatch.setattr(runner, "_fit_neural", tracked_fit_neural)

    reference = _sorted_metrics(
        scheduler.run_scheduled_benchmark(
            simulator,
            experiment,
            tmp_path / "reference",
            options,
        )
    )
    assert execution_calls == ["engineered_linear", "flow_jump"]

    execution_calls.clear()
    real_write_cell = RunStore.write_cell
    successful_writes = 0

    def interrupt_after_first_durable_write(store, result):
        nonlocal successful_writes
        cell_id = real_write_cell(store, result)
        successful_writes += 1
        if successful_writes == 1:
            raise RuntimeError("simulated interruption")
        return cell_id

    monkeypatch.setattr(RunStore, "write_cell", interrupt_after_first_durable_write)
    interrupted_output = tmp_path / "interrupted"
    with pytest.raises(scheduler.ScheduledBenchmarkError, match="simulated interruption"):
        scheduler.run_scheduled_benchmark(
            simulator,
            experiment,
            interrupted_output,
            options,
        )

    assert execution_calls == ["engineered_linear"]
    assert successful_writes == 1
    completed_cell = _only_persisted_cell(interrupted_output)
    assert completed_cell.name == (
        "low_n__smooth__cohort17__subset23__model31__n5__engineered_linear__none.json"
    )
    completed_bytes = completed_cell.read_bytes()
    completed_mtime_ns = completed_cell.stat().st_mtime_ns

    execution_calls.clear()
    monkeypatch.setattr(RunStore, "write_cell", real_write_cell)
    resumed = _sorted_metrics(
        scheduler.run_scheduled_benchmark(
            simulator,
            experiment,
            interrupted_output,
            scheduler.ExecutionOptions(
                device="cpu", workers=1, resume=True, fail_fast=True
            ),
        )
    )

    assert execution_calls == ["flow_jump"]
    assert completed_cell.read_bytes() == completed_bytes
    assert completed_cell.stat().st_mtime_ns == completed_mtime_ns
    pd.testing.assert_frame_equal(
        reference.drop(columns="value"),
        resumed.drop(columns="value"),
        check_like=False,
    )
    np.testing.assert_allclose(
        reference["value"].to_numpy(),
        resumed["value"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
