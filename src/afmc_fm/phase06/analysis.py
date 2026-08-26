from __future__ import annotations

import numpy as np
import pandas as pd

_CONTROL_VARIANT = "none__none__deterministic"
_CANDIDATE_VARIANT = "time_scaled__none__deterministic"
_D1_TRAIN_SIZES = (5, 10, 20, 40)
_D1_TARGET_METRICS = ("mae", "latent_aligned_r2")
_IDENTITY_COLUMNS = (
    "stage",
    "world",
    "cohort_seed",
    "subset_seed",
    "model_seed",
    "n_train",
)
_PAIR_COLUMNS = (
    "world",
    "cohort_seed",
    "subset_seed",
    "model_seed",
    "n_train",
)
_SUMMARY_COLUMNS = (
    "selected_checkpoint_epoch",
    "shadow_mae_checkpoint_epoch",
    "stop_epoch",
)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")


def _validate_d1_metric_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(metrics, pd.DataFrame):
        raise TypeError("metrics must be a pandas DataFrame")
    required = set(_IDENTITY_COLUMNS) | {"variant", "split", "metric", "value"}
    _require_columns(metrics, required, "D1 metric matrix")

    d1 = metrics[(metrics["stage"] == "d1") & (metrics["split"] == "test")].copy()
    if d1.empty:
        raise ValueError("D1 metric matrix contains no test rows")

    variants = set(d1["variant"].astype(str))
    if variants != {_CONTROL_VARIANT, _CANDIDATE_VARIANT}:
        raise ValueError("D1 metric matrix has an unexpected variant set")

    target = d1[d1["metric"].isin(_D1_TARGET_METRICS)].copy()
    duplicate_key = [*_IDENTITY_COLUMNS, "variant", "metric"]
    if target.duplicated(duplicate_key).any():
        raise ValueError("D1 metric matrix contains duplicate target metric rows")

    bundles = target[["cohort_seed", "subset_seed", "model_seed"]].drop_duplicates()
    if len(bundles) != 5:
        raise ValueError("D1 metric matrix must contain exactly five seed bundles")
    if set(target["n_train"]) != set(_D1_TRAIN_SIZES):
        raise ValueError("D1 metric matrix has an unexpected training-size set")
    expected_rows = 5 * len(_D1_TRAIN_SIZES) * 2 * len(_D1_TARGET_METRICS)
    if len(target) != expected_rows:
        raise ValueError("D1 metric matrix is incomplete")

    values = pd.to_numeric(target["value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("D1 metric matrix contains non-finite target metrics")
    return target


def _metric_pairs(target: pd.DataFrame, metric: str) -> pd.DataFrame:
    selected = target[target["metric"] == metric]
    pivoted = selected.pivot(
        index=list(_PAIR_COLUMNS),
        columns="variant",
        values="value",
    ).reset_index()
    if set(pivoted.columns) != {*_PAIR_COLUMNS, _CONTROL_VARIANT, _CANDIDATE_VARIANT}:
        raise ValueError(f"D1 {metric} matrix is incomplete")
    return pivoted


def _validate_d1_summary_matrix(
    summaries: pd.DataFrame,
    pair_keys: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(summaries, pd.DataFrame):
        raise TypeError("summaries must be a pandas DataFrame")
    required = set(_IDENTITY_COLUMNS) | {"variant", *_SUMMARY_COLUMNS}
    _require_columns(summaries, required, "D1 summary matrix")

    d1 = summaries[summaries["stage"] == "d1"].copy()
    variants = set(d1["variant"].astype(str))
    duplicate_key = [*_IDENTITY_COLUMNS, "variant"]
    expected_pairs = {
        tuple(row)
        for row in pair_keys[list(_PAIR_COLUMNS)].itertuples(index=False, name=None)
    }
    observed_pairs = {
        tuple(row)
        for row in d1[list(_PAIR_COLUMNS)].itertuples(index=False, name=None)
    }
    if (
        len(d1) != 40
        or variants != {_CONTROL_VARIANT, _CANDIDATE_VARIANT}
        or d1.duplicated(duplicate_key).any()
        or observed_pairs != expected_pairs
    ):
        raise ValueError("D1 summary matrix must contain exactly one row per cell")

    for column in _SUMMARY_COLUMNS:
        numeric = pd.to_numeric(d1[column], errors="coerce")
        if numeric.isna().any() or (numeric <= 0).any() or (numeric % 1 != 0).any():
            raise ValueError("D1 summary matrix contains invalid epoch indices")
    return d1


def _summary_side(summaries: pd.DataFrame, variant: str, prefix: str) -> pd.DataFrame:
    columns = [*_PAIR_COLUMNS, *_SUMMARY_COLUMNS]
    selected = summaries[summaries["variant"] == variant][columns].copy()
    return selected.rename(
        columns={
            "selected_checkpoint_epoch": f"{prefix}_selected_epoch",
            "shadow_mae_checkpoint_epoch": f"{prefix}_shadow_mae_epoch",
            "stop_epoch": f"{prefix}_stop_epoch",
        }
    )


def _classify_d1(table: pd.DataFrame) -> dict[str, object]:
    mean_by_n = {
        str(n_train): float(
            table.loc[table["n_train"] == n_train, "Delta_MAE"].mean()
        )
        for n_train in _D1_TRAIN_SIZES
    }
    n40 = table[table["n_train"] == 40]
    n40_wins = int((n40["Delta_MAE"] > 0).sum())
    conditions = {
        "mean_delta_mae_n5_le_zero": mean_by_n["5"] <= 0,
        "mean_delta_mae_n10_le_zero": mean_by_n["10"] <= 0,
        "mean_delta_mae_n20_le_zero": mean_by_n["20"] <= 0,
        "mean_delta_mae_n40_gt_zero": mean_by_n["40"] > 0,
        "n40_time_scaled_wins_ge_4_of_5": n40_wins >= 4,
    }
    if all(conditions.values()):
        classification = "reproduced"
    elif mean_by_n["40"] <= 0 or n40_wins <= 2:
        classification = "not_reproduced"
    else:
        classification = "ambiguous"

    return {
        "classification": classification,
        "mean_delta_mae_by_n": mean_by_n,
        "n40_time_scaled_wins": n40_wins,
        "n40_bundle_count": len(n40),
        "conditions": conditions,
        "effect_orientation": {
            "mae": "none - time_scaled",
            "latent_aligned_r2": "time_scaled - none",
        },
    }


def analyze_d1(
    metrics: pd.DataFrame,
    summaries: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    target = _validate_d1_metric_matrix(metrics)
    mae = _metric_pairs(target, "mae").rename(
        columns={
            _CONTROL_VARIANT: "control_mae",
            _CANDIDATE_VARIANT: "time_scaled_mae",
        }
    )
    latent = _metric_pairs(target, "latent_aligned_r2").rename(
        columns={
            _CONTROL_VARIANT: "control_latent_aligned_r2",
            _CANDIDATE_VARIANT: "time_scaled_latent_aligned_r2",
        }
    )
    paired = mae.merge(latent, on=list(_PAIR_COLUMNS), validate="one_to_one")
    if len(paired) != 20:
        raise ValueError("D1 metric matrix does not form exactly 20 paired cells")

    paired["Delta_MAE"] = paired["control_mae"] - paired["time_scaled_mae"]
    paired["Delta_R2"] = (
        paired["time_scaled_latent_aligned_r2"]
        - paired["control_latent_aligned_r2"]
    )
    paired["winner"] = np.select(
        [paired["Delta_MAE"] > 0, paired["Delta_MAE"] < 0],
        ["time_scaled", "none"],
        default="tie",
    )
    paired["bundle"] = paired.apply(
        lambda row: (
            f"{int(row['cohort_seed'])}-{int(row['subset_seed'])}-"
            f"{int(row['model_seed'])}"
        ),
        axis=1,
    )

    validated_summaries = _validate_d1_summary_matrix(summaries, paired)
    control_summary = _summary_side(
        validated_summaries,
        _CONTROL_VARIANT,
        "control",
    )
    candidate_summary = _summary_side(
        validated_summaries,
        _CANDIDATE_VARIANT,
        "time_scaled",
    )
    table = paired.merge(
        control_summary,
        on=list(_PAIR_COLUMNS),
        validate="one_to_one",
    ).merge(
        candidate_summary,
        on=list(_PAIR_COLUMNS),
        validate="one_to_one",
    )
    if len(table) != 20:
        raise ValueError("D1 summary matrix does not match the paired metric matrix")

    table.insert(0, "stage", "d1")
    table = table.sort_values(
        ["cohort_seed", "subset_seed", "model_seed", "n_train"],
        kind="mergesort",
    ).reset_index(drop=True)
    decision = _classify_d1(table)
    return table, decision


__all__ = ["analyze_d1"]
