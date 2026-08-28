from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_CONTEXTS = tuple((406 + index, 506 + index) for index in range(5))
_MODEL_SEEDS = tuple(range(1101, 1111))
_CONTROL_VARIANT = "none__none__deterministic"
_CANDIDATE_VARIANT = "time_scaled__none__deterministic"
_STANDARD_POLICY = "standard_early_stop"
_FORCED_POLICY = "forced_horizon"
_BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_SEED = 20260827
_PAIR_COLUMNS = ("cohort_seed", "subset_seed", "model_seed")
_CELL_COLUMNS = (*_PAIR_COLUMNS, "variant", "optimization_policy")

_CLASS_BOTH = "P07_OPTIMIZATION_HORIZON_EFFECT_AND_HETEROGENEITY_SUPPORTED"
_CLASS_EFFECT_ONLY = "P07_OPTIMIZATION_HORIZON_EFFECT_ONLY"
_CLASS_NOT_ESTABLISHED = "P07_OPTIMIZATION_HORIZON_NOT_ESTABLISHED"


@dataclass(frozen=True, slots=True)
class Phase07Statistics:
    mean_G: float
    mean_G_ci_lower: float
    mean_G_ci_upper: float
    R_SD: float
    R_SD_ci_lower: float
    R_SD_ci_upper: float
    positive_pair_G: int
    positive_context_mean_G: int
    positive_model_mean_G: int
    bootstrap_resamples: int
    bootstrap_seed: int
    R_SD_valid_replicates: int


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")


def _expected_cell_keys() -> set[tuple[object, ...]]:
    return {
        (cohort, subset, model, variant, policy)
        for cohort, subset in _CONTEXTS
        for model in _MODEL_SEEDS
        for variant in (_CONTROL_VARIANT, _CANDIDATE_VARIANT)
        for policy in (_STANDARD_POLICY, _FORCED_POLICY)
    }


def _validate_metric_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "stage",
        "world",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "n_train",
        "variant",
        "optimization_policy",
        "split",
        "metric",
        "value",
    }
    _require_columns(metrics, required, "Phase 0.7 metric matrix")
    target = metrics[
        (metrics["stage"] == "phase07")
        & (metrics["world"] == "smooth")
        & (metrics["n_train"] == 40)
        & (metrics["split"] == "test")
        & (metrics["metric"] == "mae")
    ].copy()
    if len(target) != 200:
        raise ValueError("Phase 0.7 metric matrix must contain exactly 200 test MAE rows")
    if target.duplicated(list(_CELL_COLUMNS)).any():
        raise ValueError("Phase 0.7 metric matrix contains duplicate cells")

    observed = set(target[list(_CELL_COLUMNS)].itertuples(index=False, name=None))
    if observed != _expected_cell_keys():
        raise ValueError("Phase 0.7 metric matrix does not match the frozen 200-cell design")

    values = pd.to_numeric(target["value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Phase 0.7 metric matrix contains invalid MAE values")
    return target


def _cell_mae(
    metrics: pd.DataFrame,
    *,
    variant: str,
    policy: str,
    name: str,
) -> pd.DataFrame:
    return metrics[
        (metrics["variant"] == variant)
        & (metrics["optimization_policy"] == policy)
    ][[*_PAIR_COLUMNS, "value"]].rename(columns={"value": name})


def build_phase07_pair_table(metrics: pd.DataFrame) -> pd.DataFrame:
    target = _validate_metric_matrix(metrics)
    parts = (
        _cell_mae(
            target,
            variant=_CONTROL_VARIANT,
            policy=_STANDARD_POLICY,
            name="mae_control_standard",
        ),
        _cell_mae(
            target,
            variant=_CANDIDATE_VARIANT,
            policy=_STANDARD_POLICY,
            name="mae_time_scaled_standard",
        ),
        _cell_mae(
            target,
            variant=_CONTROL_VARIANT,
            policy=_FORCED_POLICY,
            name="mae_control_forced",
        ),
        _cell_mae(
            target,
            variant=_CANDIDATE_VARIANT,
            policy=_FORCED_POLICY,
            name="mae_time_scaled_forced",
        ),
    )
    paired = parts[0]
    for part in parts[1:]:
        paired = paired.merge(part, on=list(_PAIR_COLUMNS), validate="one_to_one")
    if len(paired) != 50:
        raise ValueError("Phase 0.7 metrics must form exactly 50 paired effects")

    paired["Delta_standard"] = (
        paired["mae_control_standard"] - paired["mae_time_scaled_standard"]
    )
    paired["Delta_forced"] = (
        paired["mae_control_forced"] - paired["mae_time_scaled_forced"]
    )
    paired["G"] = paired["Delta_forced"] - paired["Delta_standard"]
    paired["H_time_scaled"] = (
        paired["mae_time_scaled_standard"] - paired["mae_time_scaled_forced"]
    )
    paired["H_control"] = paired["mae_control_standard"] - paired["mae_control_forced"]
    return paired.sort_values(list(_PAIR_COLUMNS), kind="mergesort").reset_index(drop=True)


def _numeric_matrix(matrix: object) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.size < 2:
        raise ValueError("matrix must be a non-empty two-dimensional numeric array")
    if not np.isfinite(values).all():
        raise ValueError("matrix values must be finite")
    return values


def two_way_residuals(matrix: object) -> np.ndarray:
    values = _numeric_matrix(matrix)
    return (
        values
        - values.mean(axis=1, keepdims=True)
        - values.mean(axis=0, keepdims=True)
        + values.mean()
    )


def residual_sample_sd(matrix: object) -> float:
    residuals = two_way_residuals(matrix)
    return float(np.std(residuals.ravel(), ddof=1))


def residual_sd_ratio(standard_matrix: object, forced_matrix: object) -> float:
    standard = _numeric_matrix(standard_matrix)
    forced = _numeric_matrix(forced_matrix)
    if standard.shape != forced.shape:
        raise ValueError("standard and forced matrices must have identical shapes")
    standard_sd = residual_sample_sd(standard)
    forced_sd = residual_sample_sd(forced)
    if not np.isfinite(standard_sd) or standard_sd <= 0:
        return float("nan")
    if not np.isfinite(forced_sd):
        return float("nan")
    return float(forced_sd / standard_sd)


def _pair_matrix(pairs: pd.DataFrame, column: str) -> np.ndarray:
    _require_columns(
        pairs,
        {"cohort_seed", "subset_seed", "model_seed", column},
        "Phase 0.7 pair table",
    )
    if len(pairs) != 50 or pairs.duplicated(list(_PAIR_COLUMNS)).any():
        raise ValueError("Phase 0.7 pair table must contain exactly 50 unique pairs")
    expected_pairs = {
        (cohort, subset, model)
        for cohort, subset in _CONTEXTS
        for model in _MODEL_SEEDS
    }
    observed_pairs = set(
        pairs[list(_PAIR_COLUMNS)].itertuples(index=False, name=None)
    )
    if observed_pairs != expected_pairs:
        raise ValueError("Phase 0.7 pair table does not match the frozen 5x10 design")

    indexed = pairs.set_index(list(_PAIR_COLUMNS))
    rows: list[list[float]] = []
    for cohort, subset in _CONTEXTS:
        row = [float(indexed.loc[(cohort, subset, model), column]) for model in _MODEL_SEEDS]
        rows.append(row)
    matrix = np.asarray(rows, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Phase 0.7 pair table contains invalid {column} values")
    return matrix


def crossed_phase07_bootstrap(pairs: pd.DataFrame) -> dict[str, int | float]:
    standard = _pair_matrix(pairs, "Delta_standard")
    forced = _pair_matrix(pairs, "Delta_forced")
    interaction = _pair_matrix(pairs, "G")

    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    mean_g = np.empty(_BOOTSTRAP_RESAMPLES, dtype=float)
    ratios = np.full(_BOOTSTRAP_RESAMPLES, np.nan, dtype=float)

    for index in range(_BOOTSTRAP_RESAMPLES):
        row_indices = rng.integers(0, len(_CONTEXTS), size=len(_CONTEXTS))
        column_indices = rng.integers(0, len(_MODEL_SEEDS), size=len(_MODEL_SEEDS))
        selection = np.ix_(row_indices, column_indices)
        sampled_standard = standard[selection]
        sampled_forced = forced[selection]
        mean_g[index] = float(interaction[selection].mean())
        ratios[index] = residual_sd_ratio(sampled_standard, sampled_forced)

    mean_lower, mean_upper = np.quantile(mean_g, [0.025, 0.975])
    finite_ratios = ratios[np.isfinite(ratios)]
    if finite_ratios.size:
        ratio_lower, ratio_upper = np.quantile(finite_ratios, [0.025, 0.975])
    else:
        ratio_lower = ratio_upper = float("nan")

    return {
        "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": _BOOTSTRAP_SEED,
        "mean_G_ci_lower": float(mean_lower),
        "mean_G_ci_upper": float(mean_upper),
        "R_SD_ci_lower": float(ratio_lower),
        "R_SD_ci_upper": float(ratio_upper),
        "R_SD_valid_replicates": int(finite_ratios.size),
    }


def statistics_from_phase07_pairs(pairs: pd.DataFrame) -> Phase07Statistics:
    standard = _pair_matrix(pairs, "Delta_standard")
    forced = _pair_matrix(pairs, "Delta_forced")
    interaction = _pair_matrix(pairs, "G")
    bootstrap = crossed_phase07_bootstrap(pairs)

    context_means = interaction.mean(axis=1)
    model_means = interaction.mean(axis=0)
    return Phase07Statistics(
        mean_G=float(interaction.mean()),
        mean_G_ci_lower=float(bootstrap["mean_G_ci_lower"]),
        mean_G_ci_upper=float(bootstrap["mean_G_ci_upper"]),
        R_SD=residual_sd_ratio(standard, forced),
        R_SD_ci_lower=float(bootstrap["R_SD_ci_lower"]),
        R_SD_ci_upper=float(bootstrap["R_SD_ci_upper"]),
        positive_pair_G=int(np.count_nonzero(interaction > 0)),
        positive_context_mean_G=int(np.count_nonzero(context_means > 0)),
        positive_model_mean_G=int(np.count_nonzero(model_means > 0)),
        bootstrap_resamples=int(bootstrap["bootstrap_resamples"]),
        bootstrap_seed=int(bootstrap["bootstrap_seed"]),
        R_SD_valid_replicates=int(bootstrap["R_SD_valid_replicates"]),
    )


def adjudicate_phase07(statistics: Phase07Statistics) -> str:
    if not isinstance(statistics, Phase07Statistics):
        raise TypeError("statistics must be a Phase07Statistics")
    causal_effect = (
        statistics.mean_G > 0
        and statistics.mean_G_ci_lower > 0
        and statistics.positive_context_mean_G >= 4
        and statistics.positive_model_mean_G >= 8
    )
    heterogeneity = (
        np.isfinite(statistics.R_SD)
        and statistics.R_SD < 1
        and np.isfinite(statistics.R_SD_ci_upper)
        and statistics.R_SD_ci_upper < 1
    )
    if not causal_effect:
        return _CLASS_NOT_ESTABLISHED
    if heterogeneity:
        return _CLASS_BOTH
    return _CLASS_EFFECT_ONLY


__all__ = [
    "Phase07Statistics",
    "adjudicate_phase07",
    "build_phase07_pair_table",
    "crossed_phase07_bootstrap",
    "residual_sample_sd",
    "residual_sd_ratio",
    "statistics_from_phase07_pairs",
    "two_way_residuals",
]
