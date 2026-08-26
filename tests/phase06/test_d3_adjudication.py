from __future__ import annotations

import pandas as pd
import pytest

from afmc_fm.phase06.analysis import D2AnalysisResult, adjudicate_phase06

_CONTROL_VARIANT = "none__none__deterministic"
_CANDIDATE_VARIANT = "time_scaled__none__deterministic"
_BUNDLES = tuple((401 + i, 501 + i, 601 + i) for i in range(5))


def _d1_result(
    classification: str = "reproduced",
    *,
    n40_mae: tuple[float, ...] = (0.03, 0.04, 0.02, 0.05, 0.01),
    n40_r2: tuple[float, ...] = (0.02, 0.03, 0.01, 0.04, 0.02),
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bundle_index, (cohort_seed, subset_seed, model_seed) in enumerate(_BUNDLES):
        for n_train in (5, 10, 20, 40):
            rows.append(
                {
                    "stage": "d1",
                    "world": "smooth",
                    "bundle": f"{cohort_seed}-{subset_seed}-{model_seed}",
                    "cohort_seed": cohort_seed,
                    "subset_seed": subset_seed,
                    "model_seed": model_seed,
                    "n_train": n_train,
                    "Delta_MAE": (
                        n40_mae[bundle_index] if n_train == 40 else -0.01
                    ),
                    "Delta_R2": n40_r2[bundle_index] if n_train == 40 else 0.0,
                    "control_selected_epoch": 2,
                    "control_shadow_mae_epoch": 1,
                    "control_stop_epoch": 3,
                    "time_scaled_selected_epoch": 2,
                    "time_scaled_shadow_mae_epoch": 1,
                    "time_scaled_stop_epoch": 3,
                }
            )
    return pd.DataFrame(rows), {"classification": classification}


def _traces(mode: str = "weakened") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, (cohort_seed, subset_seed, model_seed) in enumerate(_BUNDLES):
        control_selected = 0.50
        if mode == "strengthened":
            candidate_selected = 0.60
        elif mode == "weakened":
            candidate_selected = 0.45
        elif mode == "unresolved":
            candidate_selected = 0.60 if index < 3 else 0.45
        else:
            raise ValueError(mode)
        for variant, selected_mae in (
            (_CONTROL_VARIANT, control_selected),
            (_CANDIDATE_VARIANT, candidate_selected),
        ):
            for epoch, validation_mae in ((1, 0.40), (2, selected_mae), (3, selected_mae)):
                rows.append(
                    {
                        "stage": "d1",
                        "world": "smooth",
                        "cohort_seed": cohort_seed,
                        "subset_seed": subset_seed,
                        "model_seed": model_seed,
                        "n_train": 40,
                        "variant": variant,
                        "epoch": epoch,
                        "validation_mae": validation_mae,
                    }
                )
    return pd.DataFrame(rows)


def _d2_result(
    *,
    factor_status: dict[str, str] | None = None,
    interaction_ambiguous: bool = False,
    mean_n5: float = -0.02,
    mean_n40: float = 0.03,
    mean_t: float = 0.05,
    positive_t_count: int = 22,
    ci_lower: float = 0.01,
    ci_upper: float = 0.08,
) -> D2AnalysisResult:
    statuses = factor_status or {
        "cohort": "unresolved",
        "subset": "unresolved",
        "model": "unresolved",
    }
    if interaction_ambiguous:
        shares = {"cohort": 0.15, "subset": 0.10, "model": 0.15, "residual": 0.60}
        frequencies = {"cohort": 0.35, "subset": 0.25, "model": 0.40}
    elif statuses.get("model") == "dominant":
        shares = {"cohort": 0.10, "subset": 0.05, "model": 0.80, "residual": 0.05}
        frequencies = {"cohort": 0.05, "subset": 0.03, "model": 0.92}
    elif statuses.get("subset") == "dominant":
        shares = {"cohort": 0.10, "subset": 0.80, "model": 0.05, "residual": 0.05}
        frequencies = {"cohort": 0.05, "subset": 0.92, "model": 0.03}
    elif statuses.get("cohort") == "dominant":
        shares = {"cohort": 0.80, "subset": 0.05, "model": 0.10, "residual": 0.05}
        frequencies = {"cohort": 0.92, "subset": 0.03, "model": 0.05}
    else:
        shares = {"cohort": 0.35, "subset": 0.20, "model": 0.40, "residual": 0.05}
        frequencies = {"cohort": 0.10, "subset": 0.05, "model": 0.85}

    component_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for n_train in (5, 40):
        for component in ("cohort", "subset", "model", "residual"):
            component_rows.append(
                {
                    "n_train": n_train,
                    "component": component,
                    "sum_squares": shares[component],
                    "variance_share": shares[component],
                    "factor_classification": (
                        statuses[component]
                        if component in statuses
                        else "not_applicable"
                    ),
                }
            )
        for factor in ("cohort", "subset", "model"):
            bootstrap_rows.append(
                {
                    "n_train": n_train,
                    "factor": factor,
                    "largest_count": int(round(frequencies[factor] * 10_000)),
                    "largest_frequency": frequencies[factor],
                    "bootstrap_resamples": 10_000,
                    "bootstrap_seed": 20260826,
                }
            )

    shift_rows = pd.DataFrame(
        {
            "cohort_seed": range(401, 426),
            "T": [mean_t] * 25,
        }
    )
    return D2AnalysisResult(
        effect_rows=pd.DataFrame(),
        factor_level_effects=pd.DataFrame(),
        variance_components=pd.DataFrame(component_rows),
        bootstrap_diagnostics=pd.DataFrame(bootstrap_rows),
        n_shift_rows=shift_rows,
        n_shift_summary={
            "mean_T": mean_t,
            "positive_T_count": positive_t_count,
            "pair_count": 25,
            "mean_delta_mae_n5": mean_n5,
            "mean_delta_mae_n40": mean_n40,
            "bootstrap_ci_lower": ci_lower,
            "bootstrap_ci_upper": ci_upper,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 20260826,
        },
    )


@pytest.mark.parametrize(
    ("trace_mode", "expected"),
    (("strengthened", "strengthened"), ("weakened", "weakened"), ("unresolved", "unresolved")),
)
def test_h5_uses_validation_checkpoint_gap_only(trace_mode: str, expected: str) -> None:
    result = adjudicate_phase06(
        _d1_result(),
        _d2_result(mean_t=-0.01, positive_t_count=5, ci_lower=-0.03, ci_upper=0.01),
        _traces(trace_mode),
    )
    assert result["H5_checkpoint_objective"] == expected


def test_h1_h2_h3_map_directly_from_d2_factor_classifications() -> None:
    result = adjudicate_phase06(
        _d1_result(),
        _d2_result(
            factor_status={"cohort": "unresolved", "subset": "dominant", "model": "weak"},
            mean_t=-0.01,
            positive_t_count=4,
            ci_lower=-0.03,
            ci_upper=0.01,
        ),
        _traces("weakened"),
    )
    assert result["H1_initialization"] == "weakened"
    assert result["H2_subset_composition"] == "strengthened"
    assert result["H3_cohort_heterogeneity"] == "unresolved"
    assert result["next_required_stage"] == "D4_DATA_REGIME"

    result = adjudicate_phase06(
        _d1_result(),
        _d2_result(
            factor_status={"cohort": "dominant", "subset": "weak", "model": "unresolved"},
            mean_t=-0.01,
            positive_t_count=4,
            ci_lower=-0.03,
            ci_upper=0.01,
        ),
        _traces("weakened"),
    )
    assert result["H1_initialization"] == "unresolved"
    assert result["H2_subset_composition"] == "weakened"
    assert result["H3_cohort_heterogeneity"] == "strengthened"


@pytest.mark.parametrize(
    ("d1_classification", "d2_kwargs", "expected"),
    (
        ("reproduced", {}, "strong"),
        ("reproduced", {"positive_t_count": 18, "ci_lower": -0.01}, "partial"),
        ("not_reproduced", {}, "absent"),
        ("ambiguous", {}, "unresolved"),
    ),
)
def test_h4_predeclared_classification(
    d1_classification: str,
    d2_kwargs: dict[str, object],
    expected: str,
) -> None:
    result = adjudicate_phase06(
        _d1_result(d1_classification),
        _d2_result(**d2_kwargs),
        _traces("weakened"),
    )
    assert result["H4_sample_complexity"] == expected


@pytest.mark.parametrize(
    ("mae", "r2", "expected"),
    (
        ((0.03, 0.04, 0.02, 0.05, 0.01), (-0.02, -0.01, 0.0, -0.03, 0.01), "strengthened"),
        ((0.03, 0.04, 0.02, 0.05, 0.01), (0.02, 0.03, 0.01, 0.04, 0.02), "weakened"),
        ((0.03, 0.04, -0.01, 0.05, 0.01), (0.02, -0.03, 0.01, -0.04, 0.02), "unresolved"),
    ),
)
def test_h7_predictive_mechanistic_disconnect(
    mae: tuple[float, ...],
    r2: tuple[float, ...],
    expected: str,
) -> None:
    result = adjudicate_phase06(
        _d1_result(n40_mae=mae, n40_r2=r2),
        _d2_result(mean_t=-0.01, positive_t_count=4, ci_lower=-0.03, ci_upper=0.01),
        _traces("weakened"),
    )
    assert result["H7_predictive_mechanistic_disconnect"] == expected
    assert result["H6_capacity"] == "not_yet_tested"


def test_routing_priority_and_all_triggered_escalations_are_recorded() -> None:
    result = adjudicate_phase06(
        _d1_result(n40_r2=(-0.02, -0.01, 0.0, -0.03, -0.01)),
        _d2_result(
            factor_status={"cohort": "weak", "subset": "weak", "model": "dominant"},
            interaction_ambiguous=True,
        ),
        _traces("strengthened"),
    )
    assert result["next_required_stage"] == "D2B"
    assert result["triggered_escalations"] == [
        "D2B",
        "D4_CHECKPOINT",
        "D4_OPTIMIZATION",
        "D4_CAPACITY_TIME",
    ]


def test_routing_checkpoint_then_optimization_then_capacity_time_then_stop() -> None:
    checkpoint = adjudicate_phase06(
        _d1_result(),
        _d2_result(),
        _traces("strengthened"),
    )
    assert checkpoint["next_required_stage"] == "D4_CHECKPOINT"

    optimization = adjudicate_phase06(
        _d1_result(),
        _d2_result(
            factor_status={"cohort": "weak", "subset": "weak", "model": "dominant"},
            mean_t=-0.01,
            positive_t_count=4,
            ci_lower=-0.03,
            ci_upper=0.01,
        ),
        _traces("weakened"),
    )
    assert optimization["next_required_stage"] == "D4_OPTIMIZATION"

    capacity_time = adjudicate_phase06(
        _d1_result(),
        _d2_result(),
        _traces("weakened"),
    )
    assert capacity_time["next_required_stage"] == "D4_CAPACITY_TIME"

    stop = adjudicate_phase06(
        _d1_result(
            "not_reproduced",
            n40_mae=(-0.03, -0.02, 0.01, -0.04, 0.0),
            n40_r2=(0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        _d2_result(
            mean_t=-0.01,
            positive_t_count=4,
            ci_lower=-0.03,
            ci_upper=0.01,
        ),
        _traces("weakened"),
    )
    assert stop["next_required_stage"] == "STOP"
    assert stop["triggered_escalations"] == []
