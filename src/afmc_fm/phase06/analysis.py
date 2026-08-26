from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from afmc_fm.phase06.config import Phase06Config

_CONTROL_VARIANT = "none__none__deterministic"
_CANDIDATE_VARIANT = "time_scaled__none__deterministic"
_D1_TRAIN_SIZES = (5, 10, 20, 40)
_D1_TARGET_METRICS = ("mae", "latent_aligned_r2")
_D2_TRAIN_SIZES = (5, 40)
_D2_COHORTS = (401, 402, 403, 404, 405)
_D2_SUBSETS = (501, 502, 503, 504, 505)
_D2_MODELS = (601, 602, 603, 604, 605)
_D2_FACTORS = ("cohort", "subset", "model")
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


@dataclass(frozen=True, slots=True)
class D2AnalysisResult:
    effect_rows: pd.DataFrame
    factor_level_effects: pd.DataFrame
    variance_components: pd.DataFrame
    bootstrap_diagnostics: pd.DataFrame
    n_shift_rows: pd.DataFrame
    n_shift_summary: dict[str, object]


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


def _expected_d2_triples() -> set[tuple[int, int, int]]:
    return {
        (cohort, subset, _D2_MODELS[(i + j) % len(_D2_MODELS)])
        for i, cohort in enumerate(_D2_COHORTS)
        for j, subset in enumerate(_D2_SUBSETS)
    }


def _validate_d2_metric_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(metrics, pd.DataFrame):
        raise TypeError("metrics must be a pandas DataFrame")
    required = set(_IDENTITY_COLUMNS) | {"variant", "split", "metric", "value"}
    _require_columns(metrics, required, "D2 metric matrix")

    target = metrics[
        (metrics["stage"] == "d2a")
        & (metrics["world"] == "smooth")
        & (metrics["split"] == "test")
        & (metrics["metric"] == "mae")
    ].copy()
    if target.empty:
        raise ValueError("D2 metric matrix contains no test MAE rows")

    duplicate_key = [*_IDENTITY_COLUMNS, "variant", "metric"]
    observed_triples = {
        (int(cohort), int(subset), int(model))
        for cohort, subset, model in target[
            ["cohort_seed", "subset_seed", "model_seed"]
        ].itertuples(index=False, name=None)
    }
    complete = (
        len(target) == 25 * 2 * 2
        and set(target["n_train"]) == set(_D2_TRAIN_SIZES)
        and set(target["variant"].astype(str))
        == {_CONTROL_VARIANT, _CANDIDATE_VARIANT}
        and observed_triples == _expected_d2_triples()
        and not target.duplicated(duplicate_key).any()
    )
    if not complete:
        raise ValueError("D2 metric matrix must be a complete 25-combination orthogonal array")

    values = pd.to_numeric(target["value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("D2 metric matrix contains invalid MAE values")
    return target


def _d2_effect_rows(target: pd.DataFrame) -> pd.DataFrame:
    pivoted = target.pivot(
        index=list(_PAIR_COLUMNS),
        columns="variant",
        values="value",
    ).reset_index()
    if len(pivoted) != 50:
        raise ValueError("D2 metric matrix does not form exactly 50 paired effects")
    paired = pivoted.rename(
        columns={
            _CONTROL_VARIANT: "control_mae",
            _CANDIDATE_VARIANT: "time_scaled_mae",
        }
    )
    paired["Delta_MAE"] = paired["control_mae"] - paired["time_scaled_mae"]
    paired.insert(0, "stage", "d2a")
    return paired.sort_values(
        ["n_train", "cohort_seed", "subset_seed", "model_seed"],
        kind="mergesort",
    ).reset_index(drop=True)


def _design_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    columns: list[np.ndarray] = [np.ones((len(frame), 1), dtype=float)]
    factor_indices: dict[str, np.ndarray] = {}
    start = 1
    factor_columns = {
        "cohort": ("cohort_seed", _D2_COHORTS),
        "subset": ("subset_seed", _D2_SUBSETS),
        "model": ("model_seed", _D2_MODELS),
    }
    for factor in _D2_FACTORS:
        column, levels = factor_columns[factor]
        encoded = np.column_stack(
            [(frame[column].to_numpy() == level).astype(float) for level in levels[1:]]
        )
        columns.append(encoded)
        stop = start + encoded.shape[1]
        factor_indices[factor] = np.arange(start, stop)
        start = stop
    return np.column_stack(columns), factor_indices


def _least_squares_sse(design: np.ndarray, values: np.ndarray) -> float:
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ coefficients
    return float(residual @ residual)


def _main_effect_sums_of_squares(
    design: np.ndarray,
    values: np.ndarray,
    factor_indices: dict[str, np.ndarray],
) -> tuple[dict[str, float], float, float]:
    full_sse = _least_squares_sse(design, values)
    main_ss: dict[str, float] = {}
    all_indices = np.arange(design.shape[1])
    for factor in _D2_FACTORS:
        keep = np.setdiff1d(all_indices, factor_indices[factor], assume_unique=True)
        reduced_sse = _least_squares_sse(design[:, keep], values)
        main_ss[factor] = max(0.0, reduced_sse - full_sse)
    centered = values - values.mean()
    total_ss = float(centered @ centered)
    return main_ss, full_sse, total_ss


def _factor_level_effect_rows(frame: pd.DataFrame, n_train: int) -> list[dict[str, object]]:
    grand_mean = float(frame["Delta_MAE"].mean())
    factor_columns = {
        "cohort": "cohort_seed",
        "subset": "subset_seed",
        "model": "model_seed",
    }
    rows: list[dict[str, object]] = []
    for factor in _D2_FACTORS:
        column = factor_columns[factor]
        for level, mean_value in frame.groupby(column, sort=True)["Delta_MAE"].mean().items():
            rows.append(
                {
                    "n_train": n_train,
                    "factor": factor,
                    "level": int(level),
                    "mean_delta_mae": float(mean_value),
                    "centered_effect": float(mean_value - grand_mean),
                }
            )
    return rows


def _bootstrap_largest_counts(
    frame: pd.DataFrame,
    config: Phase06Config,
) -> dict[str, int]:
    design, factor_indices = _design_matrix(frame)
    values = frame["Delta_MAE"].to_numpy(dtype=float)
    rng = np.random.default_rng(config.bootstrap_seed)
    counts = {factor: 0 for factor in _D2_FACTORS}
    for _ in range(config.bootstrap_resamples):
        indices = rng.integers(0, len(frame), size=len(frame))
        sampled_design = design[indices]
        sampled_values = values[indices]
        main_ss, _, _ = _main_effect_sums_of_squares(
            sampled_design,
            sampled_values,
            factor_indices,
        )
        winner = _D2_FACTORS[
            int(np.argmax([main_ss[factor] for factor in _D2_FACTORS]))
        ]
        counts[winner] += 1
    return counts


def _factor_classifications(
    shares: dict[str, float],
    frequencies: dict[str, float],
    total_ss: float,
) -> dict[str, str]:
    if total_ss <= np.finfo(float).eps:
        return {factor: "unresolved" for factor in _D2_FACTORS}

    ordered = sorted(shares.items(), key=lambda item: item[1], reverse=True)
    largest_factor, largest_share = ordered[0]
    next_largest_share = ordered[1][1]
    smallest_share = ordered[-1][1]
    classifications = {factor: "unresolved" for factor in _D2_FACTORS}
    if (
        largest_share > 0
        and largest_share >= 2.0 * next_largest_share
        and frequencies[largest_factor] >= 0.80
    ):
        classifications[largest_factor] = "dominant"
    for factor in _D2_FACTORS:
        if np.isclose(shares[factor], smallest_share) and frequencies[factor] <= 0.20:
            classifications[factor] = "weak"
    return classifications


def _d2_decomposition(
    effect_rows: pd.DataFrame,
    config: Phase06Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    level_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []

    for n_train in _D2_TRAIN_SIZES:
        frame = effect_rows[effect_rows["n_train"] == n_train].copy()
        design, factor_indices = _design_matrix(frame)
        values = frame["Delta_MAE"].to_numpy(dtype=float)
        main_ss, residual_ss, total_ss = _main_effect_sums_of_squares(
            design,
            values,
            factor_indices,
        )
        shares = {
            factor: (main_ss[factor] / total_ss if total_ss > 0 else 0.0)
            for factor in _D2_FACTORS
        }
        counts = _bootstrap_largest_counts(frame, config)
        frequencies = {
            factor: counts[factor] / config.bootstrap_resamples for factor in _D2_FACTORS
        }
        classifications = _factor_classifications(shares, frequencies, total_ss)

        level_rows.extend(_factor_level_effect_rows(frame, n_train))
        for factor in _D2_FACTORS:
            component_rows.append(
                {
                    "n_train": n_train,
                    "component": factor,
                    "sum_squares": main_ss[factor],
                    "variance_share": shares[factor],
                    "factor_classification": classifications[factor],
                }
            )
            bootstrap_rows.append(
                {
                    "n_train": n_train,
                    "factor": factor,
                    "largest_count": counts[factor],
                    "largest_frequency": frequencies[factor],
                    "bootstrap_resamples": config.bootstrap_resamples,
                    "bootstrap_seed": config.bootstrap_seed,
                }
            )
        component_rows.append(
            {
                "n_train": n_train,
                "component": "residual",
                "sum_squares": residual_ss,
                "variance_share": residual_ss / total_ss if total_ss > 0 else 0.0,
                "factor_classification": "not_applicable",
            }
        )

    return (
        pd.DataFrame(level_rows),
        pd.DataFrame(component_rows),
        pd.DataFrame(bootstrap_rows),
    )


def _d2_n_shift(
    effect_rows: pd.DataFrame,
    config: Phase06Config,
) -> tuple[pd.DataFrame, dict[str, object]]:
    keys = ["world", "cohort_seed", "subset_seed", "model_seed"]
    shifted = effect_rows.pivot(
        index=keys,
        columns="n_train",
        values="Delta_MAE",
    ).reset_index()
    if len(shifted) != 25 or set(shifted.columns) != {*keys, 5, 40}:
        raise ValueError("D2 effects do not form 25 complete N5-to-N40 pairs")
    shifted = shifted.rename(columns={5: "Delta_MAE_N5", 40: "Delta_MAE_N40"})
    shifted["T"] = shifted["Delta_MAE_N40"] - shifted["Delta_MAE_N5"]
    shifted = shifted.sort_values(
        ["cohort_seed", "subset_seed", "model_seed"],
        kind="mergesort",
    ).reset_index(drop=True)

    values = shifted["T"].to_numpy(dtype=float)
    rng = np.random.default_rng(config.bootstrap_seed)
    indices = rng.integers(
        0,
        len(values),
        size=(config.bootstrap_resamples, len(values)),
    )
    bootstrap_means = values[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    summary: dict[str, object] = {
        "mean_T": float(values.mean()),
        "positive_T_count": int((values > 0).sum()),
        "pair_count": len(values),
        "mean_delta_mae_n5": float(shifted["Delta_MAE_N5"].mean()),
        "mean_delta_mae_n40": float(shifted["Delta_MAE_N40"].mean()),
        "bootstrap_ci_lower": float(lower),
        "bootstrap_ci_upper": float(upper),
        "bootstrap_resamples": config.bootstrap_resamples,
        "bootstrap_seed": config.bootstrap_seed,
        "effect_orientation": "T = Delta_MAE_N40 - Delta_MAE_N5",
    }
    return shifted, summary


def analyze_d2a(metrics: pd.DataFrame, config: Phase06Config) -> D2AnalysisResult:
    if not isinstance(config, Phase06Config):
        raise TypeError("config must be a Phase06Config")
    target = _validate_d2_metric_matrix(metrics)
    effect_rows = _d2_effect_rows(target)
    level_effects, components, bootstrap = _d2_decomposition(effect_rows, config)
    n_shift_rows, n_shift_summary = _d2_n_shift(effect_rows, config)
    return D2AnalysisResult(
        effect_rows=effect_rows,
        factor_level_effects=level_effects,
        variance_components=components,
        bootstrap_diagnostics=bootstrap,
        n_shift_rows=n_shift_rows,
        n_shift_summary=n_shift_summary,
    )


__all__ = ["D2AnalysisResult", "analyze_d1", "analyze_d2a"]
