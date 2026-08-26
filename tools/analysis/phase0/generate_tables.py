from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE0 = ROOT / "docs" / "results" / "phase0"
RAW = PHASE0 / "raw"
TABLES = PHASE0 / "tables"

METRICS_PATH = RAW / "metrics.csv"
ABLATIONS_PATH = RAW / "ablation_metrics.csv"
GATE_PATH = RAW / "gate_summary.csv"

PRIMARY_MODELS = (
    "flow_jump",
    "gru_from_scratch",
    "representation_linear",
)

LOW_N = (5, 10, 20, 40)

LOWER_IS_BETTER = {
    "mae",
    "rmse",
    "nll",
    "event_brier",
    "event_log_loss",
}

HIGHER_IS_BETTER = {
    "event_roc_auc",
    "latent_aligned_r2",
}


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    *,
    name: str,
) -> None:
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{name} is missing required columns: {sorted(missing)}"
        )


def write_csv(df: pd.DataFrame, name: str) -> None:
    path = TABLES / name
    df.to_csv(path, index=False)
    print(f"{name:<48} {len(df):>6} rows")


def sample_sd(series: pd.Series) -> float:
    if len(series) <= 1:
        return float("nan")
    return float(series.std(ddof=1))


def summarize(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)["value"]
        .agg(
            mean="mean",
            median="median",
            sd=sample_sd,
            minimum="min",
            maximum="max",
            n="size",
        )
        .reset_index()
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(METRICS_PATH)
    ablations = pd.read_csv(ABLATIONS_PATH)
    gate = pd.read_csv(GATE_PATH)

    required = {
        "model",
        "ablation",
        "n_train",
        "n_fit",
        "n_validation",
        "seed",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "split",
        "site_or_shift",
        "metric",
        "value",
        "trainable_parameters",
        "backend",
        "world",
        "benchmark",
    }

    require_columns(metrics, required, name="metrics.csv")
    require_columns(ablations, required, name="ablation_metrics.csv")

    if len(metrics) != 23520:
        raise RuntimeError(
            f"Expected 23,520 Phase-0 metric rows, found {len(metrics):,}"
        )

    if len(ablations) != 17280:
        raise RuntimeError(
            f"Expected 17,280 ablation rows, found {len(ablations):,}"
        )

    # ------------------------------------------------------------------
    # Canonical full-model result surface
    # ------------------------------------------------------------------

    full = metrics[
        (metrics["ablation"] == "none")
        & (metrics["split"] == "test")
    ].copy()

    # Preserve every official non-ablation test result.
    write_csv(
        full.sort_values(
            [
                "world",
                "n_train",
                "cohort_seed",
                "subset_seed",
                "model_seed",
                "model",
                "site_or_shift",
                "metric",
            ]
        ),
        "all_full_model_test_results.csv",
    )

    # ------------------------------------------------------------------
    # 1. Complete metric summary
    # ------------------------------------------------------------------

    all_summary = summarize(
        full,
        [
            "world",
            "n_train",
            "model",
            "site_or_shift",
            "metric",
        ],
    )

    write_csv(
        all_summary.sort_values(
            ["metric", "world", "n_train", "model", "site_or_shift"]
        ),
        "all_metric_summary.csv",
    )

    # ------------------------------------------------------------------
    # 2. Primary MAE table used for learning-curve comparisons
    # ------------------------------------------------------------------

    primary_mae_rows = full[
        (full["metric"] == "mae")
        & (full["site_or_shift"] == "all")
    ].copy()

    primary_mae_summary = summarize(
        primary_mae_rows,
        ["world", "n_train", "model"],
    )

    write_csv(
        primary_mae_summary.sort_values(
            ["world", "n_train", "mean", "model"]
        ),
        "primary_mae_by_world_n_model.csv",
    )

    # ------------------------------------------------------------------
    # 3. Actual seed-level MAE observations
    # ------------------------------------------------------------------

    seed_level_mae = primary_mae_rows[
        [
            "world",
            "n_train",
            "cohort_seed",
            "subset_seed",
            "model_seed",
            "model",
            "value",
            "trainable_parameters",
        ]
    ].rename(columns={"value": "mae"})

    write_csv(
        seed_level_mae.sort_values(
            [
                "world",
                "n_train",
                "cohort_seed",
                "subset_seed",
                "model_seed",
                "model",
            ]
        ),
        "seed_level_mae.csv",
    )

    # ------------------------------------------------------------------
    # 4. Flow-Jump paired MAE effects against primary comparators
    # ------------------------------------------------------------------

    comparison_source = seed_level_mae[
        seed_level_mae["model"].isin(PRIMARY_MODELS)
    ].copy()

    wide = comparison_source.pivot(
        index=[
            "world",
            "n_train",
            "cohort_seed",
            "subset_seed",
            "model_seed",
        ],
        columns="model",
        values="mae",
    ).reset_index()

    expected_cols = set(PRIMARY_MODELS)
    missing = expected_cols - set(wide.columns)
    if missing:
        raise RuntimeError(
            f"Primary comparator MAE matrix missing: {sorted(missing)}"
        )

    wide["delta_vs_gru"] = (
        wide["gru_from_scratch"] - wide["flow_jump"]
    )
    wide["delta_vs_representation_linear"] = (
        wide["representation_linear"] - wide["flow_jump"]
    )

    wide["relative_vs_gru"] = (
        wide["delta_vs_gru"] / wide["gru_from_scratch"]
    )
    wide["relative_vs_representation_linear"] = (
        wide["delta_vs_representation_linear"]
        / wide["representation_linear"]
    )

    wide["beats_gru"] = wide["delta_vs_gru"] > 0
    wide["beats_representation_linear"] = (
        wide["delta_vs_representation_linear"] > 0
    )
    wide["beats_both"] = (
        wide["beats_gru"]
        & wide["beats_representation_linear"]
    )

    write_csv(
        wide.sort_values(
            [
                "world",
                "n_train",
                "cohort_seed",
                "subset_seed",
                "model_seed",
            ]
        ),
        "primary_comparator_seed_effects.csv",
    )

    # ------------------------------------------------------------------
    # 5. Per-world/N win matrix
    # ------------------------------------------------------------------

    win_matrix = (
        wide.groupby(["world", "n_train"], dropna=False)
        .agg(
            flow_jump_mean_mae=("flow_jump", "mean"),
            gru_mean_mae=("gru_from_scratch", "mean"),
            representation_linear_mean_mae=(
                "representation_linear",
                "mean",
            ),
            flow_jump_vs_gru_seed_wins=("beats_gru", "sum"),
            flow_jump_vs_rep_seed_wins=(
                "beats_representation_linear",
                "sum",
            ),
            flow_jump_simultaneous_seed_wins=("beats_both", "sum"),
            n_seed_bundles=("beats_both", "size"),
            mean_delta_vs_gru=("delta_vs_gru", "mean"),
            mean_delta_vs_representation_linear=(
                "delta_vs_representation_linear",
                "mean",
            ),
        )
        .reset_index()
    )

    win_matrix["flow_jump_beats_gru_on_mean"] = (
        win_matrix["flow_jump_mean_mae"]
        < win_matrix["gru_mean_mae"]
    )
    win_matrix["flow_jump_beats_rep_on_mean"] = (
        win_matrix["flow_jump_mean_mae"]
        < win_matrix["representation_linear_mean_mae"]
    )
    win_matrix["flow_jump_beats_both_on_mean"] = (
        win_matrix["flow_jump_beats_gru_on_mean"]
        & win_matrix["flow_jump_beats_rep_on_mean"]
    )

    write_csv(
        win_matrix.sort_values(["world", "n_train"]),
        "low_n_and_full_win_matrix.csv",
    )

    write_csv(
        win_matrix[
            win_matrix["n_train"].isin(LOW_N)
        ].sort_values(["world", "n_train"]),
        "low_n_win_matrix.csv",
    )

    # ------------------------------------------------------------------
    # 6. Official gate summary plus independently recomputed flag
    # ------------------------------------------------------------------

    gate_check = gate.merge(
        win_matrix[
            [
                "world",
                "n_train",
                "flow_jump_beats_both_on_mean",
            ]
        ],
        on=["world", "n_train"],
        how="left",
        validate="one_to_one",
    )

    gate_check["gate_flag_matches_recomputation"] = (
        gate_check["flow_jump_beats_both_primary_comparators"]
        == gate_check["flow_jump_beats_both_on_mean"]
    )

    if not gate_check["gate_flag_matches_recomputation"].all():
        bad = gate_check[
            ~gate_check["gate_flag_matches_recomputation"]
        ]
        raise RuntimeError(
            "Recomputed simultaneous-win flags disagree with "
            f"gate_summary.csv:\n{bad}"
        )

    write_csv(
        gate_check.sort_values(["world", "n_train"]),
        "gate_summary_verified.csv",
    )

    # ------------------------------------------------------------------
    # 7. Learning-curve summary for the primary three models
    # ------------------------------------------------------------------

    learning = primary_mae_summary[
        primary_mae_summary["model"].isin(PRIMARY_MODELS)
    ].copy()

    write_csv(
        learning.sort_values(["world", "model", "n_train"]),
        "learning_curve_summary.csv",
    )

    # ------------------------------------------------------------------
    # 8. Parameter-count table
    # ------------------------------------------------------------------

    parameter_counts = (
        full.groupby(["model"], dropna=False)["trainable_parameters"]
        .agg(
            minimum_parameters="min",
            maximum_parameters="max",
            median_parameters="median",
            n_rows="size",
        )
        .reset_index()
    )

    parameter_counts["constant_parameter_count"] = (
        parameter_counts["minimum_parameters"]
        == parameter_counts["maximum_parameters"]
    )

    write_csv(
        parameter_counts.sort_values("median_parameters"),
        "parameter_counts.csv",
    )

    # ------------------------------------------------------------------
    # 9. Training/validation budget table
    # ------------------------------------------------------------------

    budgets = (
        full[
            [
                "n_train",
                "n_fit",
                "n_validation",
            ]
        ]
        .drop_duplicates()
        .sort_values(["n_train", "n_fit", "n_validation"])
    )

    write_csv(
        budgets,
        "training_validation_budget.csv",
    )

    # ------------------------------------------------------------------
    # 10. Calibration / uncertainty
    # ------------------------------------------------------------------

    calibration = all_summary[
        all_summary["metric"].isin(["nll", "coverage_90"])
    ].copy()

    if not calibration.empty:
        calibration["coverage_abs_error_from_0_90"] = np.where(
            calibration["metric"].eq("coverage_90"),
            np.abs(calibration["mean"] - 0.90),
            np.nan,
        )

    write_csv(
        calibration.sort_values(
            ["metric", "world", "n_train", "model", "site_or_shift"]
        ),
        "calibration_summary.csv",
    )

    # ------------------------------------------------------------------
    # 11. Latent recovery
    # ------------------------------------------------------------------

    latent = all_summary[
        all_summary["metric"] == "latent_aligned_r2"
    ].copy()

    write_csv(
        latent.sort_values(
            ["world", "n_train", "model", "site_or_shift"]
        ),
        "latent_recovery_summary.csv",
    )

    # ------------------------------------------------------------------
    # 12. Event prediction
    # ------------------------------------------------------------------

    event = all_summary[
        all_summary["metric"].isin(
            [
                "event_brier",
                "event_log_loss",
                "event_roc_auc",
            ]
        )
    ].copy()

    write_csv(
        event.sort_values(
            ["metric", "world", "n_train", "model", "site_or_shift"]
        ),
        "event_metrics_summary.csv",
    )

    # ------------------------------------------------------------------
    # 13. Site-shift evidence
    # ------------------------------------------------------------------

    site_shift = all_summary[
        all_summary["world"] == "site_shift"
    ].copy()

    write_csv(
        site_shift.sort_values(
            ["metric", "n_train", "model", "site_or_shift"]
        ),
        "site_shift_summary.csv",
    )

    # ------------------------------------------------------------------
    # 14. Misspecification evidence
    # ------------------------------------------------------------------

    misspecified = all_summary[
        all_summary["world"] == "misspecified"
    ].copy()

    write_csv(
        misspecified.sort_values(
            ["metric", "n_train", "model", "site_or_shift"]
        ),
        "misspecification_summary.csv",
    )

    # ------------------------------------------------------------------
    # 15. Full ablation pairing against corresponding full model
    # ------------------------------------------------------------------

    pair_keys = [
        "model",
        "n_train",
        "n_fit",
        "n_validation",
        "seed",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "split",
        "site_or_shift",
        "metric",
        "world",
        "benchmark",
    ]

    ablation_values = ablations[
        pair_keys
        + [
            "ablation",
            "value",
            "trainable_parameters",
        ]
    ].rename(
        columns={
            "value": "ablated_value",
            "trainable_parameters": "ablated_parameters",
        }
    )

    full_values = full[
        pair_keys
        + [
            "value",
            "trainable_parameters",
        ]
    ].rename(
        columns={
            "value": "full_value",
            "trainable_parameters": "full_parameters",
        }
    )

    paired_ablation = ablation_values.merge(
        full_values,
        on=pair_keys,
        how="left",
        validate="many_to_one",
    )

    if paired_ablation["full_value"].isna().any():
        missing_count = int(
            paired_ablation["full_value"].isna().sum()
        )
        raise RuntimeError(
            f"{missing_count} ablation rows could not be paired "
            "with their full-model result."
        )

    # Historical Phase-0 convention:
    # positive delta = removing component worsened the metric
    # for lower-is-better metrics.
    paired_ablation["raw_delta_ablated_minus_full"] = (
        paired_ablation["ablated_value"]
        - paired_ablation["full_value"]
    )

    paired_ablation["oriented_component_benefit"] = np.nan

    lower_mask = paired_ablation["metric"].isin(LOWER_IS_BETTER)
    higher_mask = paired_ablation["metric"].isin(HIGHER_IS_BETTER)

    paired_ablation.loc[
        lower_mask,
        "oriented_component_benefit",
    ] = paired_ablation.loc[
        lower_mask,
        "ablated_value",
    ] - paired_ablation.loc[
        lower_mask,
        "full_value",
    ]

    paired_ablation.loc[
        higher_mask,
        "oriented_component_benefit",
    ] = paired_ablation.loc[
        higher_mask,
        "full_value",
    ] - paired_ablation.loc[
        higher_mask,
        "ablated_value",
    ]

    # Coverage is not monotonic; closer to 0.90 is preferable.
    coverage_mask = paired_ablation["metric"].eq("coverage_90")
    paired_ablation.loc[
        coverage_mask,
        "oriented_component_benefit",
    ] = (
        np.abs(
            paired_ablation.loc[
                coverage_mask,
                "ablated_value",
            ]
            - 0.90
        )
        - np.abs(
            paired_ablation.loc[
                coverage_mask,
                "full_value",
            ]
            - 0.90
        )
    )

    write_csv(
        paired_ablation.sort_values(
            [
                "ablation",
                "metric",
                "world",
                "n_train",
                "site_or_shift",
                "cohort_seed",
                "subset_seed",
                "model_seed",
            ]
        ),
        "ablation_seed_level_effects.csv",
    )

    ablation_summary = (
        paired_ablation.groupby(
            [
                "model",
                "ablation",
                "metric",
                "world",
                "n_train",
                "site_or_shift",
            ],
            dropna=False,
        )
        .agg(
            mean_full=("full_value", "mean"),
            mean_ablated=("ablated_value", "mean"),
            mean_raw_delta=(
                "raw_delta_ablated_minus_full",
                "mean",
            ),
            median_raw_delta=(
                "raw_delta_ablated_minus_full",
                "median",
            ),
            sd_raw_delta=(
                "raw_delta_ablated_minus_full",
                sample_sd,
            ),
            mean_oriented_component_benefit=(
                "oriented_component_benefit",
                "mean",
            ),
            median_oriented_component_benefit=(
                "oriented_component_benefit",
                "median",
            ),
            n=("raw_delta_ablated_minus_full", "size"),
        )
        .reset_index()
    )

    write_csv(
        ablation_summary.sort_values(
            [
                "ablation",
                "metric",
                "world",
                "n_train",
                "site_or_shift",
            ]
        ),
        "ablation_summary.csv",
    )

    # MAE-only ablation table corresponding most directly to the report.
    ablation_mae = ablation_summary[
        (ablation_summary["metric"] == "mae")
        & (ablation_summary["site_or_shift"] == "all")
    ].copy()

    write_csv(
        ablation_mae.sort_values(
            ["ablation", "world", "n_train"]
        ),
        "ablation_mae_summary.csv",
    )

    # ------------------------------------------------------------------
    # 16. Low-N-only comprehensive metric table
    # ------------------------------------------------------------------

    low_n_all_metrics = all_summary[
        all_summary["n_train"].isin(LOW_N)
    ].copy()

    write_csv(
        low_n_all_metrics.sort_values(
            [
                "metric",
                "world",
                "n_train",
                "model",
                "site_or_shift",
            ]
        ),
        "low_n_all_metric_summary.csv",
    )

    # ------------------------------------------------------------------
    # 17. Compact Phase-0 headline table
    # ------------------------------------------------------------------

    headline = win_matrix[
        [
            "world",
            "n_train",
            "flow_jump_mean_mae",
            "gru_mean_mae",
            "representation_linear_mean_mae",
            "flow_jump_vs_gru_seed_wins",
            "flow_jump_vs_rep_seed_wins",
            "flow_jump_simultaneous_seed_wins",
            "n_seed_bundles",
            "flow_jump_beats_both_on_mean",
        ]
    ].copy()

    write_csv(
        headline.sort_values(["world", "n_train"]),
        "phase0_headline_mae_results.csv",
    )

    # ------------------------------------------------------------------
    # Assertions for the known Phase-0 scientific result
    # ------------------------------------------------------------------

    low_n_gate = gate_check[
        gate_check["n_train"].isin(LOW_N)
    ]

    low_n_simultaneous_wins = int(
        low_n_gate[
            "flow_jump_beats_both_primary_comparators"
        ].sum()
    )

    if len(low_n_gate) != 20:
        # Five worlds x four low-N budgets.
        raise RuntimeError(
            "Expected 20 world/N rows in the complete low-N matrix, "
            f"found {len(low_n_gate)}."
        )

    # The report's 0/12 statement concerns the three dynamics worlds
    # in the primary extreme-low-N comparison, not all five worlds.
    report_primary_worlds = {
        "smooth",
        "jumps",
        "informative_observation",
    }

    report_low_n = low_n_gate[
        low_n_gate["world"].isin(report_primary_worlds)
    ]

    if len(report_low_n) != 12:
        raise RuntimeError(
            "Expected 12 primary dynamics-world/N low-N settings, "
            f"found {len(report_low_n)}."
        )

    report_wins = int(
        report_low_n[
            "flow_jump_beats_both_primary_comparators"
        ].sum()
    )

    if report_wins != 0:
        raise RuntimeError(
            "Official Phase-0 headline result changed unexpectedly: "
            f"found {report_wins}/12 simultaneous low-N wins."
        )

    print()
    print("=" * 72)
    print("PHASE-0 TABLE GENERATION COMPLETE")
    print("=" * 72)
    print(f"Official metrics rows:       {len(metrics):,}")
    print(f"Ablation rows:               {len(ablations):,}")
    print(f"Primary low-N settings:      {len(report_low_n)}")
    print(f"Simultaneous low-N wins:     {report_wins}/12")
    print(f"All-five-world low-N wins:   {low_n_simultaneous_wins}/20")
    print(f"Tables written to:           {TABLES}")
    print("=" * 72)


if __name__ == "__main__":
    main()
