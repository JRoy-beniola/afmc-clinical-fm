from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from afmc_fm.phase06.d4b import D4BAnalysisResult, adjudicate_d4b, analyze_d4b

_CONTEXTS = ((401, 501), (402, 502), (403, 503), (404, 504), (405, 505))
_MODELS = tuple(range(1001, 1011))
_VARIANTS = ("none__none__deterministic", "time_scaled__none__deterministic")


def _inputs(delta_by_model: dict[int, float] | None = None):
    if delta_by_model is None:
        delta_by_model = {seed: 0.01 + (seed - 1001) * 0.001 for seed in _MODELS}

    metric_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for context_index, (cohort, subset) in enumerate(_CONTEXTS):
        for model in _MODELS:
            delta = float(delta_by_model[model]) + context_index * 0.0002
            for variant in _VARIANTS:
                flow = variant.split("__", 1)[0]
                mae = 1.0 + delta if flow == "none" else 1.0
                identity = {
                    "stage": "d4b",
                    "world": "smooth",
                    "cohort_seed": cohort,
                    "subset_seed": subset,
                    "model_seed": model,
                    "n_train": 40,
                    "variant": variant,
                }
                metric_rows.append(
                    {
                        **identity,
                        "split": "test",
                        "metric": "mae",
                        "value": mae,
                    }
                )
                selected = 4 + ((model + context_index) % 3)
                stop = selected + 2
                summary_rows.append(
                    {
                        **identity,
                        "selected_checkpoint_epoch": selected,
                        "shadow_mae_checkpoint_epoch": selected + 1,
                        "stop_epoch": stop,
                        "early_stop_reason": "patience_exhausted",
                    }
                )
                for epoch in (1, 2):
                    trace_rows.append(
                        {
                            **identity,
                            "epoch": epoch,
                            "gradient_l2_norm": 1.0 + model / 10_000 + epoch / 100,
                            "parameter_l2_norm": 2.0 + epoch / 100,
                            "mean_flow_displacement": (
                                0.0 if flow == "none" else 0.1 + model / 100_000
                            ),
                            "median_flow_displacement": 0.0 if flow == "none" else 0.09,
                            "p95_flow_displacement": 0.0 if flow == "none" else 0.12,
                            "validation_core_loss": 0.5,
                            "validation_mae": 0.4,
                        }
                    )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(trace_rows),
    )


def test_d4b_analysis_builds_exact_frozen_surfaces_and_cluster_bootstrap() -> None:
    metrics, summaries, traces = _inputs()

    result = analyze_d4b(metrics, summaries, traces)

    assert isinstance(result, D4BAnalysisResult)
    assert len(result.effect_rows) == 50
    assert len(result.model_seed_summary) == 10
    assert len(result.context_summary) == 5
    assert set(result.effect_rows["n_train"]) == {40}
    assert set(result.effect_rows["model_seed"]) == set(_MODELS)
    assert {
        (int(row.cohort_seed), int(row.subset_seed))
        for row in result.context_summary.itertuples()
    } == set(_CONTEXTS)
    assert {
        "control_mae",
        "time_scaled_mae",
        "Delta_MAE",
        "control_selected_epoch",
        "time_scaled_selected_epoch",
        "control_stop_epoch",
        "time_scaled_stop_epoch",
    } <= set(result.effect_rows)

    bootstrap = result.bootstrap_diagnostics
    assert bootstrap["bootstrap_resamples"] == 10_000
    assert bootstrap["bootstrap_seed"] == 20260827
    assert bootstrap["resampling_unit"] == "model_seed_cluster"
    assert bootstrap["confidence_interval"] == "percentile_95"
    assert bootstrap["bootstrap_ci_lower"] > 0
    assert bootstrap["bootstrap_ci_upper"] > bootstrap["bootstrap_ci_lower"]

    # Deterministic fixed-seed bootstrap.
    repeated = analyze_d4b(metrics, summaries, traces)
    assert repeated.bootstrap_diagnostics == bootstrap

    dispersion = result.optimization_dispersion
    assert len(dispersion) == 10  # five contexts x two flow modes
    assert {
        "selected_checkpoint_epoch_mean",
        "selected_checkpoint_epoch_std",
        "selected_checkpoint_epoch_iqr",
        "stop_epoch_mean",
        "stop_epoch_std",
        "stop_epoch_iqr",
        "mean_gradient_l2_norm_mean",
        "mean_gradient_l2_norm_std",
        "mean_gradient_l2_norm_iqr",
        "mean_flow_displacement_mean",
        "mean_flow_displacement_std",
        "mean_flow_displacement_iqr",
        "patience_exhausted_count",
        "max_epochs_reached_count",
    } <= set(dispersion)


def test_d4b_adjudication_stable_routes_only_to_capacity_time() -> None:
    result = analyze_d4b(*_inputs())

    decision = adjudicate_d4b(result)

    assert decision["classification"] == "stable"
    assert decision["next_required_stage"] == "D4_CAPACITY_TIME"
    assert decision["positive_model_seed_count"] == 10
    assert decision["positive_context_count"] == 5
    assert decision["Delta_overall"] > 0
    assert decision["bootstrap_ci_lower"] > 0
    assert set(decision["input_artifact_hashes"]) == {
        "effect_rows",
        "model_seed_summary",
        "context_summary",
        "bootstrap_diagnostics",
        "optimization_dispersion",
    }


def _decision_fixture(
    *,
    overall: float,
    model_values: list[float],
    context_values: list[float],
    ci_lower: float,
) -> D4BAnalysisResult:
    base = analyze_d4b(*_inputs())
    model_summary = pd.DataFrame(
        {
            "model_seed": list(_MODELS),
            "mean_Delta_MAE": model_values,
        }
    )
    context_summary = pd.DataFrame(
        {
            "cohort_seed": [c for c, _ in _CONTEXTS],
            "subset_seed": [s for _, s in _CONTEXTS],
            "mean_Delta_MAE": context_values,
        }
    )
    bootstrap = dict(base.bootstrap_diagnostics)
    bootstrap.update(
        {
            "mean_Delta_MAE": overall,
            "bootstrap_ci_lower": ci_lower,
            "bootstrap_ci_upper": max(ci_lower + 0.01, 0.01),
        }
    )
    return replace(
        base,
        model_seed_summary=model_summary,
        context_summary=context_summary,
        bootstrap_diagnostics=bootstrap,
    )


@pytest.mark.parametrize(
    ("overall", "models", "contexts"),
    [
        (0.0, [0.1] * 10, [0.1] * 5),
        (0.1, [0.1] * 5 + [-0.1] * 5, [0.1] * 5),
        (0.1, [0.1] * 10, [0.1] * 2 + [-0.1] * 3),
    ],
)
def test_d4b_adjudication_fragile_thresholds_stop(
    overall: float,
    models: list[float],
    contexts: list[float],
) -> None:
    result = _decision_fixture(
        overall=overall,
        model_values=models,
        context_values=contexts,
        ci_lower=0.01,
    )

    decision = adjudicate_d4b(result)

    assert decision["classification"] == "fragile"
    assert decision["next_required_stage"] == "STOP"


def test_d4b_adjudication_ambiguous_stops_without_expansion() -> None:
    result = _decision_fixture(
        overall=0.1,
        model_values=[0.1] * 7 + [-0.1] * 3,
        context_values=[0.1] * 4 + [-0.1],
        ci_lower=0.01,
    )

    decision = adjudicate_d4b(result)

    assert decision["classification"] == "ambiguous"
    assert decision["next_required_stage"] == "STOP"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_n", "wrong_seed", "nan"])
def test_d4b_analysis_rejects_invalid_metric_matrix(mutation: str) -> None:
    metrics, summaries, traces = _inputs()
    if mutation == "missing":
        metrics = metrics.iloc[:-1].copy()
    elif mutation == "duplicate":
        metrics = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    elif mutation == "wrong_n":
        metrics.loc[0, "n_train"] = 5
    elif mutation == "wrong_seed":
        metrics.loc[0, "model_seed"] = 601
    else:
        metrics.loc[0, "value"] = np.nan

    with pytest.raises(ValueError):
        analyze_d4b(metrics, summaries, traces)
