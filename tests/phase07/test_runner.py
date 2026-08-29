from __future__ import annotations

import importlib

import pandas as pd
import torch

from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase07.planning import Phase07CellSpec
from afmc_fm.simulator.config import SimulatorConfig


def _runner_api():
    try:
        module = importlib.import_module("afmc_fm.phase07.runner")
    except ModuleNotFoundError as error:
        raise AssertionError("Phase 0.7 runner module is not implemented") from error
    return module.Phase07CellRun, module.prepare_phase07_cohort, module.run_phase07_cell


def _diagnostic_api():
    try:
        module = importlib.import_module("afmc_fm.phase07.diagnostics")
    except ModuleNotFoundError as error:
        raise AssertionError("Phase 0.7 diagnostics module is not implemented") from error
    return module.first_patience_exhaustion_epoch


def _phase05_config() -> Phase05Config:
    return Phase05Config(max_epochs=4, patience=2)


def _simulator_config() -> SimulatorConfig:
    return SimulatorConfig(cohort_size=60, followup_days=45.0, intervention_rate=0.2)


def _cell(policy: str, flow_mode: str = "time_scaled") -> Phase07CellSpec:
    return Phase07CellSpec(
        world="smooth",
        cohort_seed=406,
        subset_seed=506,
        model_seed=1101,
        n_train=40,
        flow_mode=flow_mode,
        optimization_policy=policy,
    )


def test_first_patience_exhaustion_epoch_uses_first_ordinary_patience_event():
    first_patience_exhaustion_epoch = _diagnostic_api()
    trace = pd.DataFrame(
        {
            "epoch": [1, 2, 3, 4, 5],
            "stale_epochs": [0, 1, 2, 3, 0],
        }
    )

    assert first_patience_exhaustion_epoch(trace, patience=2) == 3
    assert first_patience_exhaustion_epoch(trace, patience=4) is None


def test_phase07_runner_preserves_exact_paired_prefix_and_policy_metadata():
    Phase07CellRun, prepare_phase07_cohort, run_phase07_cell = _runner_api()
    config = _phase05_config()
    simulator_config = _simulator_config()
    standard_cell = _cell("standard_early_stop")
    forced_cell = _cell("forced_horizon")

    prepared_standard = prepare_phase07_cohort(simulator_config, standard_cell)
    prepared_forced = prepare_phase07_cohort(simulator_config, forced_cell)

    assert tuple(prepared_standard.patient_by_id) == tuple(prepared_forced.patient_by_id)

    standard = run_phase07_cell(
        prepared_standard,
        config,
        standard_cell,
        torch.device("cpu"),
    )
    forced = run_phase07_cell(
        prepared_forced,
        config,
        forced_cell,
        torch.device("cpu"),
    )

    assert isinstance(standard, Phase07CellRun)
    assert isinstance(forced, Phase07CellRun)
    assert standard.optimization_policy == "standard_early_stop"
    assert forced.optimization_policy == "forced_horizon"
    assert forced.summary["epochs_run"] == config.max_epochs
    assert forced.summary["early_stop_reason"] == "max_epochs_reached"

    prefix_length = len(standard.trace)
    pd.testing.assert_frame_equal(
        standard.trace.reset_index(drop=True),
        forced.trace.iloc[:prefix_length].reset_index(drop=True),
        check_exact=True,
    )

    first_patience_exhaustion_epoch = _diagnostic_api()
    expected_forced_event = first_patience_exhaustion_epoch(
        forced.trace,
        patience=config.patience,
    )
    assert forced.would_patience_exhaust_epoch == expected_forced_event
    if standard.summary["early_stop_reason"] == "patience_exhausted":
        assert standard.would_patience_exhaust_epoch == standard.summary["stop_epoch"]
        assert forced.would_patience_exhaust_epoch == standard.summary["stop_epoch"]
    else:
        assert standard.would_patience_exhaust_epoch is None

    assert set(standard.metrics["optimization_policy"]) == {"standard_early_stop"}
    assert set(forced.metrics["optimization_policy"]) == {"forced_horizon"}
    assert set(standard.metrics["stage"]) == {"phase07"}
    assert set(forced.metrics["stage"]) == {"phase07"}
    assert set(standard.metrics["variant"]) == {"time_scaled__none__deterministic"}
    assert set(forced.metrics["variant"]) == {"time_scaled__none__deterministic"}

    for result in (standard, forced):
        assert result.production_state_dict
        assert result.shadow_state_dict
        assert all(tensor.device.type == "cpu" for tensor in result.production_state_dict.values())
        assert all(tensor.device.type == "cpu" for tensor in result.shadow_state_dict.values())
