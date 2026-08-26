from __future__ import annotations

import importlib

import pandas as pd
import torch

phase06_cli = importlib.import_module("afmc_fm.phase06.cli")


def _fake_metrics(cell) -> pd.DataFrame:
    if cell.stage == "d1":
        delta_mae = {5: -0.01, 10: -0.01, 20: -0.005, 40: 0.03}[cell.n_train]
    else:
        delta_mae = -0.01 if cell.n_train == 5 else 0.03

    control_mae = 0.50
    mae = control_mae if cell.flow_mode == "none" else control_mae - delta_mae
    control_r2 = 0.40
    delta_r2 = -0.01 if cell.n_train == 40 else 0.0
    latent_r2 = (
        control_r2
        if cell.flow_mode == "none"
        else control_r2 + delta_r2
    )
    variant = f"{cell.flow_mode}__none__deterministic"
    common = {
        "stage": cell.stage,
        "world": cell.world,
        "cohort_seed": cell.cohort_seed,
        "subset_seed": cell.subset_seed,
        "model_seed": cell.model_seed,
        "n_train": cell.n_train,
        "variant": variant,
        "split": "test",
    }
    return pd.DataFrame(
        [
            {**common, "metric": "mae", "value": mae},
            {**common, "metric": "latent_aligned_r2", "value": latent_r2},
        ]
    )


def _fake_trace() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"epoch": 1, "validation_mae": 0.40},
            {"epoch": 2, "validation_mae": 0.50},
            {"epoch": 3, "validation_mae": 0.50},
        ]
    )


def _fake_summary() -> dict[str, object]:
    return {
        "epochs_run": 3,
        "stop_epoch": 3,
        "selected_checkpoint_epoch": 2,
        "selected_validation_core_loss": 0.60,
        "shadow_mae_checkpoint_epoch": 1,
        "shadow_validation_mae": 0.40,
        "early_stop_reason": "max_epochs_reached",
    }


def _fake_stage_run(cells, store, *_args, **_kwargs):
    planned = tuple(cells)
    state = {"weight": torch.tensor([1.0])}
    for cell in planned:
        store.write_cell_bundle(
            cell,
            metrics=_fake_metrics(cell),
            trace=_fake_trace(),
            summary=_fake_summary(),
            production_state_dict=state,
            shadow_state_dict=state,
        )
    expected = frozenset(cell.cell_id for cell in planned)
    store.mark_stage_complete(planned[0].stage, expected)
    return {
        "stage": planned[0].stage,
        "planned_cells": len(planned),
        "completed_after": len(planned),
    }


def test_fake_complete_d1_d2_pipeline_persists_analysis_and_adjudicates(
    tmp_path, monkeypatch, capsys
):
    calls: list[str] = []

    def capture_stage_run(cells, store, *args, **kwargs):
        calls.append(tuple(cells)[0].stage)
        return _fake_stage_run(cells, store, *args, **kwargs)

    monkeypatch.setattr(phase06_cli, "run_phase06_stage", capture_stage_run)

    config = "configs/experiments/phase06.yaml"
    assert (
        phase06_cli.main(
            ["d1", "--config", config, "--output", str(tmp_path), "--device", "cpu"]
        )
        == 0
    )
    assert (tmp_path / "stages" / "d1" / "COMPLETE").is_file()
    assert (tmp_path / "analysis" / "phase06_d1_reproduction.csv").is_file()
    assert (tmp_path / "analysis" / "phase06_d1_classification.json").is_file()

    assert (
        phase06_cli.main(
            ["d2a", "--config", config, "--output", str(tmp_path), "--device", "cpu"]
        )
        == 0
    )
    assert (tmp_path / "stages" / "d2a" / "COMPLETE").is_file()
    for name in (
        "phase06_d2_effects.csv",
        "phase06_d2_factor_level_effects.csv",
        "phase06_d2_variance_components.csv",
        "phase06_d2_bootstrap_diagnostics.csv",
        "phase06_d2_n_shift.csv",
        "phase06_d2_n_shift_summary.json",
    ):
        assert (tmp_path / "analysis" / name).is_file(), name

    before_adjudicate = list(calls)
    assert phase06_cli.main(["adjudicate", "--output", str(tmp_path)]) == 0
    assert calls == before_adjudicate
    assert (tmp_path / "analysis" / "phase06_d3_adjudication.json").is_file()
    assert capsys.readouterr().out.strip() == "D4_CAPACITY_TIME"
