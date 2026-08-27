from __future__ import annotations

import importlib
import importlib.util

import pandas as pd
import pytest

from afmc_fm.phase06.analysis import D2AnalysisResult

_FACTORS = ("cohort", "subset", "model")


def _effect_rows(stage: str, *, offset: float = 0.0) -> pd.DataFrame:
    multiplier = {"d2a": 1, "d2b": 2}[stage]
    rows: list[dict[str, object]] = []
    for i in range(5):
        for j in range(5):
            model_index = (i + multiplier * j) % 5
            for n_train in (5, 40):
                delta = 0.01 * i + 0.002 * j + (0.05 if n_train == 40 else 0.0) + offset
                rows.append(
                    {
                        "stage": stage,
                        "world": "smooth",
                        "cohort_seed": 401 + i,
                        "subset_seed": 501 + j,
                        "model_seed": 601 + model_index,
                        "n_train": n_train,
                        "control_mae": 1.0,
                        "time_scaled_mae": 1.0 - delta,
                        "Delta_MAE": delta,
                    }
                )
    return pd.DataFrame(rows)


def _result(
    stage: str,
    *,
    n5_dominant: str | None,
    n40_dominant: str | None,
    offset: float = 0.0,
) -> D2AnalysisResult:
    component_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for n_train, dominant in ((5, n5_dominant), (40, n40_dominant)):
        if dominant is None:
            shares = {"cohort": 0.20, "subset": 0.18, "model": 0.22}
            residual_share = 0.40
            frequencies = {"cohort": 0.33, "subset": 0.32, "model": 0.35}
        else:
            shares = {
                factor: (0.60 if factor == dominant else 0.10)
                for factor in _FACTORS
            }
            residual_share = 0.20
            frequencies = {
                factor: (0.90 if factor == dominant else 0.05)
                for factor in _FACTORS
            }
        for factor in _FACTORS:
            component_rows.append(
                {
                    "n_train": n_train,
                    "component": factor,
                    "sum_squares": shares[factor],
                    "variance_share": shares[factor],
                    "factor_classification": (
                        "dominant" if factor == dominant else "unresolved"
                    ),
                }
            )
            bootstrap_rows.append(
                {
                    "n_train": n_train,
                    "factor": factor,
                    "largest_count": int(frequencies[factor] * 10_000),
                    "largest_frequency": frequencies[factor],
                    "bootstrap_resamples": 10_000,
                    "bootstrap_seed": 20260826,
                }
            )
        component_rows.append(
            {
                "n_train": n_train,
                "component": "residual",
                "sum_squares": residual_share,
                "variance_share": residual_share,
                "factor_classification": "not_applicable",
            }
        )

    multiplier = {"d2a": 1, "d2b": 2}[stage]
    shifts = []
    for i in range(5):
        for j in range(5):
            model_index = (i + multiplier * j) % 5
            shifts.append(
                {
                    "world": "smooth",
                    "cohort_seed": 401 + i,
                    "subset_seed": 501 + j,
                    "model_seed": 601 + model_index,
                    "Delta_MAE_N5": offset + 0.01 * i + 0.002 * j,
                    "Delta_MAE_N40": offset + 0.05 + 0.01 * i + 0.002 * j,
                    "T": 0.05,
                }
            )
    return D2AnalysisResult(
        effect_rows=_effect_rows(stage, offset=offset),
        factor_level_effects=pd.DataFrame(
            [
                {
                    "n_train": 5,
                    "factor": "model",
                    "level": 601,
                    "mean_delta_mae": 0.0,
                    "centered_effect": 0.0,
                }
            ]
        ),
        variance_components=pd.DataFrame(component_rows),
        bootstrap_diagnostics=pd.DataFrame(bootstrap_rows),
        n_shift_rows=pd.DataFrame(shifts),
        n_shift_summary={
            "mean_T": 0.05,
            "positive_T_count": 25,
            "pair_count": 25,
            "mean_delta_mae_n5": 0.01,
            "mean_delta_mae_n40": 0.06,
            "bootstrap_ci_lower": 0.04,
            "bootstrap_ci_upper": 0.06,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 20260826,
        },
    )


def _parent_d3() -> dict[str, object]:
    return {
        "interaction_ambiguity": True,
        "next_required_stage": "D2B",
        "triggered_escalations": ["D2B", "D4_OPTIMIZATION", "D4_CAPACITY_TIME"],
    }


def _adjudicator():
    spec = importlib.util.find_spec("afmc_fm.phase06.d2b")
    assert spec is not None, "afmc_fm.phase06.d2b must exist"
    module = importlib.import_module("afmc_fm.phase06.d2b")
    adjudicate = getattr(module, "adjudicate_d2b", None)
    assert callable(adjudicate), "adjudicate_d2b must exist"
    return adjudicate


def _adjudicate(d2a: D2AnalysisResult, d2b: D2AnalysisResult) -> dict[str, object]:
    return _adjudicator()(_parent_d3(), d2a, d2b)


def test_d2b_adjudication_accepts_same_n40_dominance_and_no_n5_dominance() -> None:
    result = _adjudicate(
        _result("d2a", n5_dominant=None, n40_dominant="model"),
        _result("d2b", n5_dominant=None, n40_dominant="model", offset=0.01),
    )

    assert result["complementary_array_evidence"] == "sufficient"
    assert result["dominant_factors"] == {
        "d2a": {"5": "none", "40": "model"},
        "d2b": {"5": "none", "40": "model"},
    }
    assert result["remaining_parent_escalations"] == [
        "D4_OPTIMIZATION",
        "D4_CAPACITY_TIME",
    ]
    assert result["next_required_stage"] == "D4_OPTIMIZATION"

    overlap = result["overlap_rerun_diagnostics"]
    for n_train in ("5", "40"):
        assert overlap[n_train]["overlap_pair_count"] == 5
        assert overlap[n_train]["mean_absolute_delta_mae_difference"] == pytest.approx(0.01)
        assert overlap[n_train]["maximum_absolute_delta_mae_difference"] == pytest.approx(0.01)
        assert overlap[n_train]["pearson_correlation"] == pytest.approx(1.0)


def test_d2b_adjudication_accepts_same_named_dominance_at_both_n() -> None:
    result = _adjudicate(
        _result("d2a", n5_dominant="model", n40_dominant="model"),
        _result("d2b", n5_dominant="model", n40_dominant="model"),
    )

    assert result["complementary_array_evidence"] == "sufficient"
    assert result["next_required_stage"] == "D4_OPTIMIZATION"


@pytest.mark.parametrize(
    ("a_n5", "a_n40", "b_n5", "b_n40"),
    [
        (None, "model", None, "cohort"),
        (None, "model", None, None),
        ("model", "model", None, "model"),
        ("model", "model", "cohort", "model"),
    ],
)
def test_d2b_adjudication_stops_for_full_factorial_addendum_when_arrays_disagree(
    a_n5: str | None,
    a_n40: str | None,
    b_n5: str | None,
    b_n40: str | None,
) -> None:
    result = _adjudicate(
        _result("d2a", n5_dominant=a_n5, n40_dominant=a_n40),
        _result("d2b", n5_dominant=b_n5, n40_dominant=b_n40),
    )

    assert result["complementary_array_evidence"] == "insufficient"
    assert result["next_required_stage"] == "FULL_FACTORIAL_ADDENDUM"
    assert result["remaining_parent_escalations"] == [
        "D4_OPTIMIZATION",
        "D4_CAPACITY_TIME",
    ]


def test_d2b_adjudication_rejects_parent_that_did_not_authorize_d2b() -> None:
    adjudicate = _adjudicator()
    parent = _parent_d3()
    parent["next_required_stage"] = "D4_OPTIMIZATION"

    with pytest.raises(ValueError, match="parent D3.*D2B"):
        adjudicate(
            parent,
            _result("d2a", n5_dominant=None, n40_dominant="model"),
            _result("d2b", n5_dominant=None, n40_dominant="model"),
        )
