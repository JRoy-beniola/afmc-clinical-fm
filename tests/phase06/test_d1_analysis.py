import importlib

import pandas as pd
import pytest

analysis = importlib.import_module("afmc_fm.phase06.analysis")
analyze_d1 = analysis.analyze_d1

_BUNDLES = tuple((400 + i, 500 + i, 600 + i) for i in range(1, 6))
_NS = (5, 10, 20, 40)
_CONTROL = "none__none__deterministic"
_CANDIDATE = "time_scaled__none__deterministic"


def _fixture(
    deltas_by_n: dict[int, tuple[float, ...]],
    *,
    latent_delta: float = -0.02,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for bundle_index, (cohort, subset, model) in enumerate(_BUNDLES):
        for n_train in _NS:
            delta_mae = deltas_by_n[n_train][bundle_index]
            control_mae = 0.30 + 0.001 * bundle_index
            candidate_mae = control_mae - delta_mae
            control_r2 = 0.25 + 0.01 * bundle_index
            candidate_r2 = control_r2 + latent_delta
            for variant, mae, latent_r2, epoch_offset in (
                (_CONTROL, control_mae, control_r2, 0),
                (_CANDIDATE, candidate_mae, candidate_r2, 1),
            ):
                identity = {
                    "stage": "d1",
                    "world": "smooth",
                    "cohort_seed": cohort,
                    "subset_seed": subset,
                    "model_seed": model,
                    "n_train": n_train,
                    "variant": variant,
                }
                metric_rows.extend(
                    [
                        {
                            **identity,
                            "split": "test",
                            "metric": "mae",
                            "value": mae,
                        },
                        {
                            **identity,
                            "split": "test",
                            "metric": "latent_aligned_r2",
                            "value": latent_r2,
                        },
                    ]
                )
                summary_rows.append(
                    {
                        **identity,
                        "selected_checkpoint_epoch": 6 + epoch_offset,
                        "shadow_mae_checkpoint_epoch": 4 + epoch_offset,
                        "stop_epoch": 9 + epoch_offset,
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(summary_rows)


def _reproduced_deltas() -> dict[int, tuple[float, ...]]:
    return {
        5: (-0.03, -0.02, -0.01, -0.02, -0.01),
        10: (-0.02, -0.01, -0.01, -0.005, -0.015),
        20: (-0.01, 0.0, -0.005, 0.0, -0.005),
        40: (0.04, 0.03, 0.02, 0.01, -0.005),
    }


def test_analyze_d1_reproduced_builds_complete_table_and_orients_effects():
    metrics, summaries = _fixture(_reproduced_deltas())

    table, decision = analyze_d1(metrics, summaries)

    assert len(table) == 20
    assert set(table["n_train"]) == set(_NS)
    assert table[["cohort_seed", "subset_seed", "model_seed", "n_train"]].duplicated().sum() == 0
    required = {
        "bundle",
        "n_train",
        "control_mae",
        "time_scaled_mae",
        "Delta_MAE",
        "winner",
        "control_latent_aligned_r2",
        "time_scaled_latent_aligned_r2",
        "Delta_R2",
        "control_selected_epoch",
        "time_scaled_selected_epoch",
        "control_shadow_mae_epoch",
        "time_scaled_shadow_mae_epoch",
        "control_stop_epoch",
        "time_scaled_stop_epoch",
    }
    assert required <= set(table.columns)

    n40_first = table[(table["cohort_seed"] == 401) & (table["n_train"] == 40)].iloc[0]
    assert n40_first["Delta_MAE"] == pytest.approx(0.04)
    assert n40_first["Delta_R2"] == pytest.approx(-0.02)
    assert n40_first["winner"] == "time_scaled"
    assert n40_first["control_selected_epoch"] == 6
    assert n40_first["time_scaled_selected_epoch"] == 7

    assert decision["classification"] == "reproduced"
    assert decision["mean_delta_mae_by_n"]["5"] <= 0
    assert decision["mean_delta_mae_by_n"]["10"] <= 0
    assert decision["mean_delta_mae_by_n"]["20"] <= 0
    assert decision["mean_delta_mae_by_n"]["40"] > 0
    assert decision["n40_time_scaled_wins"] == 4
    assert decision["conditions"] == {
        "mean_delta_mae_n5_le_zero": True,
        "mean_delta_mae_n10_le_zero": True,
        "mean_delta_mae_n20_le_zero": True,
        "mean_delta_mae_n40_gt_zero": True,
        "n40_time_scaled_wins_ge_4_of_5": True,
    }
    assert decision["effect_orientation"] == {
        "mae": "none - time_scaled",
        "latent_aligned_r2": "time_scaled - none",
    }


def test_analyze_d1_not_reproduced_when_n40_mean_is_nonpositive():
    deltas = _reproduced_deltas()
    deltas[40] = (-0.03, -0.02, -0.01, 0.01, 0.01)
    metrics, summaries = _fixture(deltas)

    _, decision = analyze_d1(metrics, summaries)

    assert decision["mean_delta_mae_by_n"]["40"] < 0
    assert decision["n40_time_scaled_wins"] == 2
    assert decision["classification"] == "not_reproduced"


def test_analyze_d1_ambiguous_when_n40_mean_positive_but_only_three_wins():
    deltas = _reproduced_deltas()
    deltas[40] = (0.04, 0.03, 0.02, -0.005, -0.005)
    metrics, summaries = _fixture(deltas)

    _, decision = analyze_d1(metrics, summaries)

    assert decision["mean_delta_mae_by_n"]["40"] > 0
    assert decision["n40_time_scaled_wins"] == 3
    assert decision["classification"] == "ambiguous"


def test_analyze_d1_rejects_incomplete_summary_matrix():
    metrics, summaries = _fixture(_reproduced_deltas())
    summaries = summaries.iloc[:-1].copy()

    with pytest.raises(ValueError, match="summary matrix"):
        analyze_d1(metrics, summaries)
