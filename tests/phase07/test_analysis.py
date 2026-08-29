from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

_CONTEXTS = tuple((406 + index, 506 + index) for index in range(5))
_MODEL_SEEDS = tuple(range(1101, 1111))


def _analysis_api():
    try:
        module = importlib.import_module("afmc_fm.phase07.analysis")
    except ModuleNotFoundError as error:
        raise AssertionError("Phase 0.7 analysis module is not implemented") from error
    return module


def _metric_matrix(*, forced_shift: float = 0.20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for context_index, (cohort_seed, subset_seed) in enumerate(_CONTEXTS):
        for model_index, model_seed in enumerate(_MODEL_SEEDS):
            interaction = 0.015 * ((context_index - 2) * (model_index - 4.5))
            delta_standard = 0.05 + interaction
            delta_forced = delta_standard + forced_shift + 0.005 * context_index
            control_standard = 1.0 + 0.01 * context_index + 0.002 * model_index
            time_standard = control_standard - delta_standard
            control_forced = control_standard - 0.03
            time_forced = control_forced - delta_forced
            values = {
                ("none__none__deterministic", "standard_early_stop"): control_standard,
                ("time_scaled__none__deterministic", "standard_early_stop"): time_standard,
                ("none__none__deterministic", "forced_horizon"): control_forced,
                ("time_scaled__none__deterministic", "forced_horizon"): time_forced,
            }
            for (variant, policy), value in values.items():
                rows.append(
                    {
                        "stage": "phase07",
                        "world": "smooth",
                        "cohort_seed": cohort_seed,
                        "subset_seed": subset_seed,
                        "model_seed": model_seed,
                        "n_train": 40,
                        "variant": variant,
                        "optimization_policy": policy,
                        "split": "test",
                        "metric": "mae",
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def test_pair_table_matches_frozen_estimand_algebra():
    module = _analysis_api()
    pairs = module.build_phase07_pair_table(_metric_matrix())

    assert len(pairs) == 50
    row = pairs.iloc[0]
    assert row["Delta_standard"] == pytest.approx(
        row["mae_control_standard"] - row["mae_time_scaled_standard"]
    )
    assert row["Delta_forced"] == pytest.approx(
        row["mae_control_forced"] - row["mae_time_scaled_forced"]
    )
    assert row["G"] == pytest.approx(row["Delta_forced"] - row["Delta_standard"])
    assert row["H_time_scaled"] == pytest.approx(
        row["mae_time_scaled_standard"] - row["mae_time_scaled_forced"]
    )
    assert row["H_control"] == pytest.approx(
        row["mae_control_standard"] - row["mae_control_forced"]
    )
    assert row["G"] == pytest.approx(row["H_time_scaled"] - row["H_control"])


def test_two_way_residuals_match_exact_frozen_formula_and_ddof_one():
    module = _analysis_api()
    matrix = np.arange(50, dtype=float).reshape(5, 10)
    matrix[2, 7] += 3.5
    expected = (
        matrix
        - matrix.mean(axis=1, keepdims=True)
        - matrix.mean(axis=0, keepdims=True)
        + matrix.mean()
    )

    observed = module.two_way_residuals(matrix)

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    assert module.residual_sample_sd(matrix) == pytest.approx(
        float(np.std(expected.ravel(), ddof=1))
    )


def test_r_sd_is_undefined_without_epsilon_rescue_when_standard_residual_sd_is_zero():
    module = _analysis_api()
    additive = np.add.outer(np.arange(5, dtype=float), np.arange(10, dtype=float))
    forced = additive.copy()
    forced[0, 0] += 1.0

    value = module.residual_sd_ratio(additive, forced)

    assert np.isnan(value)


def test_crossed_bootstrap_is_deterministic_and_recomputes_residualization():
    module = _analysis_api()
    pairs = module.build_phase07_pair_table(_metric_matrix())

    first = module.crossed_phase07_bootstrap(pairs)
    second = module.crossed_phase07_bootstrap(pairs)

    for key in (
        "bootstrap_resamples",
        "bootstrap_seed",
        "mean_G_ci_lower",
        "mean_G_ci_upper",
        "R_SD_valid_replicates",
    ):
        assert first[key] == second[key]
    assert np.isnan(first["R_SD_ci_lower"])
    assert np.isnan(second["R_SD_ci_lower"])
    assert np.isnan(first["R_SD_ci_upper"])
    assert np.isnan(second["R_SD_ci_upper"])
    assert first["bootstrap_resamples"] == 10_000
    assert first["bootstrap_seed"] == 20260827
    assert first["mean_G_ci_lower"] > 0
    assert first["mean_G_ci_upper"] > first["mean_G_ci_lower"]
    assert first["R_SD_valid_replicates"] < first["bootstrap_resamples"]
    assert first["R_SD_valid_replicates"] > 0

    changed_pairs = module.build_phase07_pair_table(_metric_matrix(forced_shift=0.35))
    changed = module.crossed_phase07_bootstrap(changed_pairs)
    assert changed["mean_G_ci_lower"] != first["mean_G_ci_lower"]


def _statistics(module, *, causal: bool, heterogeneity: bool, valid_replicates: int = 10_000):
    return module.Phase07Statistics(
        mean_G=0.2 if causal else -0.01,
        mean_G_ci_lower=0.05 if causal else -0.05,
        mean_G_ci_upper=0.3,
        R_SD=0.7 if heterogeneity else 1.1,
        R_SD_ci_lower=0.5 if heterogeneity else 0.9,
        R_SD_ci_upper=0.9 if heterogeneity else 1.3,
        positive_pair_G=45,
        positive_context_mean_G=5 if causal else 3,
        positive_model_mean_G=9 if causal else 7,
        bootstrap_resamples=10_000,
        bootstrap_seed=20260827,
        R_SD_valid_replicates=valid_replicates,
    )


@pytest.mark.parametrize(
    ("causal", "heterogeneity", "expected"),
    [
        (
            True,
            True,
            "P07_OPTIMIZATION_HORIZON_EFFECT_AND_HETEROGENEITY_SUPPORTED",
        ),
        (True, False, "P07_OPTIMIZATION_HORIZON_EFFECT_ONLY"),
        (False, True, "P07_OPTIMIZATION_HORIZON_NOT_ESTABLISHED"),
        (False, False, "P07_OPTIMIZATION_HORIZON_NOT_ESTABLISHED"),
    ],
)
def test_frozen_adjudication_has_exact_three_level_outcome(causal, heterogeneity, expected):
    module = _analysis_api()
    assert module.adjudicate_phase07(
        _statistics(module, causal=causal, heterogeneity=heterogeneity)
    ) == expected


def test_incomplete_r_sd_bootstrap_invalidates_heterogeneity_gate():
    module = _analysis_api()
    statistics = _statistics(
        module,
        causal=True,
        heterogeneity=True,
        valid_replicates=9_999,
    )

    assert (
        module.adjudicate_phase07(statistics)
        == "P07_OPTIMIZATION_HORIZON_EFFECT_ONLY"
    )


def test_analysis_rejects_any_matrix_other_than_exact_frozen_200_cells():
    module = _analysis_api()
    metrics = _metric_matrix().iloc[:-1].copy()

    with pytest.raises(ValueError, match="200"):
        module.build_phase07_pair_table(metrics)