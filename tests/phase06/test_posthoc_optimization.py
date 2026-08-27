from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from afmc_fm.phase06.posthoc_optimization import (
    PosthocOptimizationResult,
    analyze_posthoc_optimization,
    load_d4b_archive,
    run_posthoc_archive_analysis,
)

_CONTEXTS = ((401, 501), (402, 502), (403, 503), (404, 504), (405, 505))
_MODELS = tuple(range(1001, 1011))
_CONTROL = "none__none__deterministic"
_CANDIDATE = "time_scaled__none__deterministic"


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effect_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []

    for context_index, (cohort, subset) in enumerate(_CONTEXTS):
        for model_index, model in enumerate(_MODELS):
            interaction = (context_index - 2.0) * (model_index - 4.5)
            delta_selected = -20.0 + 1.75 * interaction
            delta_stop = delta_selected + 2.0
            delta_shadow = -12.0 + 1.1 * interaction
            delta_mae = (
                0.0025 * context_index
                + 0.001 * model_index
                - 0.003 * delta_selected
            )
            effect_rows.append(
                {
                    "stage": "d4b",
                    "world": "smooth",
                    "cohort_seed": cohort,
                    "subset_seed": subset,
                    "model_seed": model,
                    "n_train": 40,
                    "control_mae": 0.35,
                    "time_scaled_mae": 0.35 - delta_mae,
                    "control_selected_epoch": 100,
                    "control_stop_epoch": 100,
                    "time_scaled_selected_epoch": 100 + delta_selected,
                    "time_scaled_stop_epoch": 100 + delta_stop,
                    "Delta_MAE": delta_mae,
                }
            )

            for variant in (_CONTROL, _CANDIDATE):
                candidate = variant == _CANDIDATE
                selected_epoch = 100 + delta_selected if candidate else 100.0
                stop_epoch = 100 + delta_stop if candidate else 100.0
                shadow_epoch = 100 + delta_shadow if candidate else 100.0
                summary_rows.append(
                    {
                        "stage": "d4b",
                        "world": "smooth",
                        "cohort_seed": cohort,
                        "subset_seed": subset,
                        "model_seed": model,
                        "n_train": 40,
                        "variant": variant,
                        "selected_checkpoint_epoch": selected_epoch,
                        "shadow_mae_checkpoint_epoch": shadow_epoch,
                        "stop_epoch": stop_epoch,
                        "early_stop_reason": (
                            "patience_exhausted" if candidate and stop_epoch < 100 else "max_epochs_reached"
                        ),
                    }
                )
                for epoch in range(1, 13):
                    base_gradient = 0.5 + 0.01 * context_index + 0.005 * model_index
                    base_parameter = 7.0 + 0.02 * model_index
                    trace_rows.append(
                        {
                            "stage": "d4b",
                            "world": "smooth",
                            "cohort_seed": cohort,
                            "subset_seed": subset,
                            "model_seed": model,
                            "n_train": 40,
                            "variant": variant,
                            "epoch": epoch,
                            "gradient_l2_norm": base_gradient
                            + (0.03 * interaction if candidate else 0.0),
                            "parameter_l2_norm": base_parameter
                            + (0.01 * interaction if candidate else 0.0),
                            "mean_flow_displacement": (
                                0.2 + 0.002 * interaction if candidate else 0.0
                            ),
                        }
                    )

    return (
        pd.DataFrame(effect_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(trace_rows),
    )


def test_posthoc_builds_exact_pair_level_mechanistic_surface() -> None:
    effects, summaries, traces = _synthetic_inputs()

    result = analyze_posthoc_optimization(
        effects,
        summaries,
        traces,
        permutation_resamples=499,
        permutation_seed=20260827,
    )

    assert isinstance(result, PosthocOptimizationResult)
    assert len(result.pair_table) == 50
    assert {
        "Delta_MAE",
        "delta_selected_epoch",
        "delta_stop_epoch",
        "delta_shadow_mae_epoch",
        "delta_selection_shadow_gap",
        "delta_patience_exhausted",
        "delta_prefix10_gradient_l2_mean",
        "delta_prefix10_parameter_l2_mean",
        "candidate_prefix10_flow_displacement_mean",
    } <= set(result.pair_table)
    assert set(result.pair_table["model_seed"]) == set(_MODELS)
    assert {
        (int(row.cohort_seed), int(row.subset_seed))
        for row in result.pair_table.itertuples()
    } == set(_CONTEXTS)

    selected = result.associations.set_index("mechanism").loc["delta_selected_epoch"]
    assert selected["family"] == "primary"
    assert selected["two_way_pearson"] < -0.99
    assert selected["two_way_spearman"] < -0.95
    assert 0.0 < selected["permutation_pvalue"] <= 1.0
    assert selected["permutation_resamples"] == 499
    assert selected["permutation_seed"] == 20260827

    assert len(result.leave_one_out) == len(result.associations) * 15
    assert set(result.leave_one_out["omitted_axis"]) == {"context", "model_seed"}
    assert result.screening["effect_orientation"] == "Delta_MAE = control - time_scaled"
    assert result.screening["mechanism_orientation"] == "candidate - control"
    assert result.screening["phase06_terminal_decision"] == "AMBIGUOUS -> STOP"


def test_posthoc_permutation_and_screening_are_deterministic() -> None:
    inputs = _synthetic_inputs()

    first = analyze_posthoc_optimization(
        *inputs,
        permutation_resamples=999,
        permutation_seed=17,
    )
    second = analyze_posthoc_optimization(
        *inputs,
        permutation_resamples=999,
        permutation_seed=17,
    )

    pd.testing.assert_frame_equal(first.associations, second.associations)
    pd.testing.assert_frame_equal(first.leave_one_out, second.leave_one_out)
    assert first.screening == second.screening
    assert first.screening["classification"] == (
        "structured optimization-conditioned heterogeneity worth prospective testing"
    )
    assert "delta_selected_epoch" in first.screening["passing_primary_mechanisms"]


def test_posthoc_rejects_incomplete_crossed_pair_matrix() -> None:
    effects, summaries, traces = _synthetic_inputs()

    with pytest.raises(ValueError, match="50 paired"):
        analyze_posthoc_optimization(
            effects.iloc[:-1].copy(),
            summaries,
            traces,
            permutation_resamples=99,
        )


def test_frozen_archive_loader_and_runner_keep_outputs_separate(tmp_path: Path) -> None:
    archive = Path("docs/results/phase06/evidence/d4b")
    effects, summaries, traces = load_d4b_archive(archive)

    assert len(effects) == 50
    assert len(summaries) == 100
    assert set(summaries["early_stop_reason"]) <= {
        "patience_exhausted",
        "max_epochs_reached",
    }
    assert not traces.empty

    output_dir = tmp_path / "posthoc"
    result = run_posthoc_archive_analysis(
        archive,
        output_dir,
        permutation_resamples=99,
        permutation_seed=20260827,
    )

    assert len(result.pair_table) == 50
    assert {path.name for path in output_dir.iterdir()} == {
        "phase06_posthoc_pair_mechanisms.csv",
        "phase06_posthoc_associations.csv",
        "phase06_posthoc_leave_one_out.csv",
        "phase06_posthoc_screening.json",
    }

    with pytest.raises(ValueError, match="frozen Phase 0.6 archive"):
        run_posthoc_archive_analysis(
            archive,
            archive / "posthoc",
            permutation_resamples=9,
        )
