from __future__ import annotations

import importlib
import importlib.util

import pandas as pd
import pytest

from afmc_fm.phase06.analysis import D2AnalysisResult, analyze_d2a
from afmc_fm.phase06.config import Phase06Config

_CONTROL_VARIANT = "none__none__deterministic"
_CANDIDATE_VARIANT = "time_scaled__none__deterministic"


def _d2_fixture(*, stage: str = "d2a") -> pd.DataFrame:
    cohorts = (401, 402, 403, 404, 405)
    subsets = (501, 502, 503, 504, 505)
    models = (601, 602, 603, 604, 605)
    multiplier = {"d2a": 1, "d2b": 2}[stage]
    cohort_effect = {401: 0.00, 402: 0.02, 403: -0.02, 404: 0.00, 405: 0.00}
    subset_effect = {501: 0.00, 502: 0.005, 503: -0.005, 504: 0.00, 505: 0.00}
    model_effect = {601: -0.20, 602: -0.10, 603: 0.00, 604: 0.10, 605: 0.20}

    rows: list[dict[str, object]] = []
    for i, cohort_seed in enumerate(cohorts):
        for j, subset_seed in enumerate(subsets):
            model_seed = models[(i + multiplier * j) % 5]
            base_delta = (
                cohort_effect[cohort_seed]
                + subset_effect[subset_seed]
                + model_effect[model_seed]
            )
            for n_train, intercept in ((5, -0.04), (40, 0.04)):
                delta_mae = base_delta + intercept
                control_mae = 1.0
                candidate_mae = control_mae - delta_mae
                for variant, value in (
                    (_CONTROL_VARIANT, control_mae),
                    (_CANDIDATE_VARIANT, candidate_mae),
                ):
                    rows.append(
                        {
                            "stage": stage,
                            "world": "smooth",
                            "cohort_seed": cohort_seed,
                            "subset_seed": subset_seed,
                            "model_seed": model_seed,
                            "n_train": n_train,
                            "variant": variant,
                            "split": "test",
                            "metric": "mae",
                            "value": value,
                        }
                    )
    return pd.DataFrame(rows)


def _analyze_d2b():
    spec = importlib.util.find_spec("afmc_fm.phase06.d2b")
    assert spec is not None, "afmc_fm.phase06.d2b must exist"
    module = importlib.import_module("afmc_fm.phase06.d2b")
    analyze = getattr(module, "analyze_d2b", None)
    assert callable(analyze), "analyze_d2b must exist"
    return analyze


def test_d2_analysis_recovers_dominant_model_seed_and_weak_subset() -> None:
    result = analyze_d2a(_d2_fixture(), Phase06Config())

    assert isinstance(result, D2AnalysisResult)
    assert len(result.effect_rows) == 50
    assert set(result.effect_rows["n_train"]) == {5, 40}
    assert set(result.effect_rows["stage"]) == {"d2a"}
    assert result.effect_rows["Delta_MAE"].notna().all()

    for n_train in (5, 40):
        rows = result.variance_components[
            result.variance_components["n_train"] == n_train
        ].set_index("component")
        assert rows.loc["model", "variance_share"] > rows.loc["cohort", "variance_share"]
        assert rows.loc["cohort", "variance_share"] > rows.loc["subset", "variance_share"]
        assert rows.loc["model", "factor_classification"] == "dominant"
        assert rows.loc["subset", "factor_classification"] == "weak"
        assert rows.loc["residual", "factor_classification"] == "not_applicable"

        bootstrap = result.bootstrap_diagnostics[
            result.bootstrap_diagnostics["n_train"] == n_train
        ].set_index("factor")
        assert bootstrap.loc["model", "largest_frequency"] >= 0.80
        assert bootstrap.loc["subset", "largest_frequency"] <= 0.20
        assert bootstrap["largest_count"].sum() == Phase06Config().bootstrap_resamples


def test_d2_analysis_computes_locked_paired_n_shift() -> None:
    result = analyze_d2a(_d2_fixture(), Phase06Config())

    assert len(result.n_shift_rows) == 25
    assert result.n_shift_rows["T"].tolist() == pytest.approx([0.08] * 25)
    assert result.n_shift_summary["mean_T"] == pytest.approx(0.08)
    assert result.n_shift_summary["positive_T_count"] == 25
    assert result.n_shift_summary["pair_count"] == 25
    assert result.n_shift_summary["mean_delta_mae_n5"] == pytest.approx(-0.04)
    assert result.n_shift_summary["mean_delta_mae_n40"] == pytest.approx(0.04)
    assert result.n_shift_summary["bootstrap_seed"] == 20260826
    assert result.n_shift_summary["bootstrap_resamples"] == 10_000
    assert result.n_shift_summary["bootstrap_ci_lower"] > 0
    assert result.n_shift_summary["bootstrap_ci_upper"] > 0


def test_d2_analysis_rejects_incomplete_orthogonal_array() -> None:
    metrics = _d2_fixture().iloc[:-1].copy()

    with pytest.raises(ValueError, match="complete 25-combination orthogonal array"):
        analyze_d2a(metrics, Phase06Config())


def test_d2b_analysis_accepts_only_complementary_array_and_preserves_stage() -> None:
    analyze_d2b = _analyze_d2b()
    result = analyze_d2b(_d2_fixture(stage="d2b"), Phase06Config())

    assert isinstance(result, D2AnalysisResult)
    assert len(result.effect_rows) == 50
    assert set(result.effect_rows["stage"]) == {"d2b"}
    observed = {
        (int(row.cohort_seed), int(row.subset_seed), int(row.model_seed))
        for row in result.effect_rows.itertuples()
    }
    expected = {
        (401 + i, 501 + j, 601 + ((i + 2 * j) % 5))
        for i in range(5)
        for j in range(5)
    }
    assert observed == expected
    assert result.n_shift_summary["bootstrap_seed"] == 20260826
    assert result.n_shift_summary["bootstrap_resamples"] == 10_000


def test_d2a_and_d2b_reject_each_others_array() -> None:
    analyze_d2b = _analyze_d2b()

    with pytest.raises(ValueError, match="complete 25-combination orthogonal array"):
        analyze_d2a(_d2_fixture(stage="d2b"), Phase06Config())
    with pytest.raises(ValueError, match="complete 25-combination orthogonal array"):
        analyze_d2b(_d2_fixture(stage="d2a"), Phase06Config())
