import importlib

import numpy as np
import pandas as pd
import pytest
import torch

from afmc_fm.data.splits import split_patient_ids
from afmc_fm.experiments.runner import select_low_n_budget
from afmc_fm.phase05.config import Phase05Config, SeedBundle
from afmc_fm.phase06.planning import Phase06CellSpec
from afmc_fm.simulator.config import SimulatorConfig

runner = importlib.import_module("afmc_fm.phase06.runner")
Phase06CellRun = runner.Phase06CellRun
prepare_phase06_cohort = runner.prepare_phase06_cohort
run_phase06_cell = runner.run_phase06_cell


def _phase05_config() -> Phase05Config:
    return Phase05Config(
        train_sizes=(5, 10, 20, 40, 80, 100),
        development_bundles=(
            SeedBundle(401, 501, 601),
            SeedBundle(402, 502, 602),
            SeedBundle(403, 503, 603),
            SeedBundle(404, 504, 604),
            SeedBundle(405, 505, 605),
        ),
        confirmatory_bundles=tuple(
            SeedBundle(700 + index, 800 + index, 900 + index)
            for index in range(1, 11)
        ),
        max_epochs=2,
        patience=2,
    )


def _simulator_config() -> SimulatorConfig:
    return SimulatorConfig(
        cohort_size=36,
        followup_days=45.0,
        intervention_rate=0.2,
    )


def _cell(flow_mode: str = "none") -> Phase06CellSpec:
    return Phase06CellSpec(
        stage="d1",
        world="smooth",
        cohort_seed=401,
        subset_seed=501,
        model_seed=601,
        n_train=5,
        flow_mode=flow_mode,
    )


def test_prepare_phase06_cohort_reuses_strict_phase05_preparation():
    cell = _cell()

    prepared = prepare_phase06_cohort(_simulator_config(), cell)

    assert len(prepared.patient_by_id) == 36
    assert set(prepared.sequences) == set(prepared.patient_by_id)
    assert prepared.historical_sequences is None
    assert all(
        sequence.jump_eligible_mask.shape == sequence.update_mask.shape
        for sequence in prepared.sequences.values()
    )


@pytest.mark.parametrize("flow_mode", ["none", "time_scaled"])
def test_run_phase06_cell_returns_locked_metrics_and_diagnostics(flow_mode):
    cell = _cell(flow_mode)
    prepared = prepare_phase06_cohort(_simulator_config(), cell)

    result = run_phase06_cell(
        prepared,
        _phase05_config(),
        cell,
        torch.device("cpu"),
    )

    assert isinstance(result, Phase06CellRun)
    assert isinstance(result.metrics, pd.DataFrame)
    assert isinstance(result.trace, pd.DataFrame)
    assert result.metrics["variant"].unique().tolist() == [
        f"{flow_mode}__none__deterministic"
    ]
    assert result.metrics["stage"].unique().tolist() == ["d1"]
    assert result.metrics["world"].unique().tolist() == ["smooth"]
    assert result.metrics["model"].unique().tolist() == ["phase05_flow_jump"]
    assert result.metrics["backend"].unique().tolist() == ["torch"]
    assert {"mae", "rmse", "latent_aligned_r2"} <= set(result.metrics["metric"])
    defined = result.metrics[result.metrics["metric"] != "event_roc_auc"]
    assert np.isfinite(defined["value"]).all()

    assert not result.trace.empty
    assert result.trace["epoch"].tolist() == list(
        range(1, int(result.summary["epochs_run"]) + 1)
    )
    assert 1 <= int(result.summary["selected_checkpoint_epoch"]) <= len(result.trace)
    assert 1 <= int(result.summary["shadow_mae_checkpoint_epoch"]) <= len(result.trace)
    assert int(result.summary["stop_epoch"]) == int(result.trace.iloc[-1]["epoch"])
    assert result.production_state_dict
    assert result.shadow_state_dict
    assert all(tensor.device.type == "cpu" for tensor in result.production_state_dict.values())
    assert all(tensor.device.type == "cpu" for tensor in result.shadow_state_dict.values())


def test_run_phase06_cell_preserves_phase05_low_n_split_semantics():
    cell = _cell()
    prepared = prepare_phase06_cohort(_simulator_config(), cell)
    all_ids = list(prepared.patient_by_id)
    development_pool, _, test_ids = split_patient_ids(
        all_ids,
        seed=0,
        train_fraction=0.8,
        val_fraction=0.0,
    )
    expected = select_low_n_budget(development_pool, cell.n_train, cell.subset_seed)

    result = run_phase06_cell(
        prepared,
        _phase05_config(),
        cell,
        torch.device("cpu"),
    )

    assert set(result.metrics["n_fit"]) == {len(expected.fit_ids)}
    assert set(result.metrics["n_validation"]) == {len(expected.validation_ids)}
    assert set(result.metrics["n_train"]) == {cell.n_train}
    assert len(test_ids) > 0


def test_production_checkpoint_matches_returned_model_metrics_not_shadow_state():
    cell = _cell("time_scaled")
    prepared = prepare_phase06_cohort(_simulator_config(), cell)

    result = run_phase06_cell(
        prepared,
        _phase05_config(),
        cell,
        torch.device("cpu"),
    )

    assert result.summary["selected_checkpoint_epoch"] is not None
    assert result.summary["shadow_mae_checkpoint_epoch"] is not None
    assert "shadow_test_metrics" not in result.summary
    assert "shadow_test_metrics" not in result.metrics.columns
