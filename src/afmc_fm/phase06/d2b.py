from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from afmc_fm.phase06.analysis import (
    D2AnalysisResult,
    _d2_decomposition,
    _d2_n_shift,
    _validate_d2_adjudication_input,
)
from afmc_fm.phase06.config import Phase06Config

_CONTROL_VARIANT = "none__none__deterministic"
_CANDIDATE_VARIANT = "time_scaled__none__deterministic"
_D2_TRAIN_SIZES = (5, 40)
_D2_COHORTS = (401, 402, 403, 404, 405)
_D2_SUBSETS = (501, 502, 503, 504, 505)
_D2_MODELS = (601, 602, 603, 604, 605)
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
_EFFECT_COLUMNS = {
    "stage",
    "world",
    "cohort_seed",
    "subset_seed",
    "model_seed",
    "n_train",
    "Delta_MAE",
}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")


def _expected_triples(multiplier: int) -> set[tuple[int, int, int]]:
    return {
        (cohort, subset, _D2_MODELS[(i + multiplier * j) % len(_D2_MODELS)])
        for i, cohort in enumerate(_D2_COHORTS)
        for j, subset in enumerate(_D2_SUBSETS)
    }


def _validate_d2b_metric_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(metrics, pd.DataFrame):
        raise TypeError("metrics must be a pandas DataFrame")
    required = set(_IDENTITY_COLUMNS) | {"variant", "split", "metric", "value"}
    _require_columns(metrics, required, "D2-B metric matrix")

    target = metrics[
        (metrics["stage"] == "d2b")
        & (metrics["world"] == "smooth")
        & (metrics["split"] == "test")
        & (metrics["metric"] == "mae")
    ].copy()
    if target.empty:
        raise ValueError("D2-B metric matrix contains no test MAE rows")

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
        and observed_triples == _expected_triples(2)
        and not target.duplicated(duplicate_key).any()
    )
    if not complete:
        raise ValueError(
            "D2-B metric matrix must be a complete 25-combination orthogonal array"
        )

    values = pd.to_numeric(target["value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("D2-B metric matrix contains invalid MAE values")
    return target


def _d2b_effect_rows(target: pd.DataFrame) -> pd.DataFrame:
    pivoted = target.pivot(
        index=list(_PAIR_COLUMNS),
        columns="variant",
        values="value",
    ).reset_index()
    if (
        len(pivoted) != 50
        or _CONTROL_VARIANT not in pivoted.columns
        or _CANDIDATE_VARIANT not in pivoted.columns
    ):
        raise ValueError("D2-B metric matrix does not form exactly 50 paired effects")
    paired = pivoted.rename(
        columns={
            _CONTROL_VARIANT: "control_mae",
            _CANDIDATE_VARIANT: "time_scaled_mae",
        }
    )
    paired["Delta_MAE"] = paired["control_mae"] - paired["time_scaled_mae"]
    paired.insert(0, "stage", "d2b")
    return paired.sort_values(
        ["n_train", "cohort_seed", "subset_seed", "model_seed"],
        kind="mergesort",
    ).reset_index(drop=True)


def analyze_d2b(metrics: pd.DataFrame, config: Phase06Config) -> D2AnalysisResult:
    """Analyze the predeclared D2-B complementary orthogonal array."""
    if not isinstance(config, Phase06Config):
        raise TypeError("config must be a Phase06Config")
    target = _validate_d2b_metric_matrix(metrics)
    effect_rows = _d2b_effect_rows(target)
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


def _validate_effect_grid(result: D2AnalysisResult, stage: str, multiplier: int) -> None:
    frame = result.effect_rows
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{stage} effect rows must be a pandas DataFrame")
    _require_columns(frame, _EFFECT_COLUMNS, f"{stage} effect rows")
    triples = {
        (int(cohort), int(subset), int(model))
        for cohort, subset, model in frame[
            ["cohort_seed", "subset_seed", "model_seed"]
        ].itertuples(index=False, name=None)
    }
    duplicate_key = [*_PAIR_COLUMNS]
    valid = (
        len(frame) == 50
        and set(frame["stage"].astype(str)) == {stage}
        and set(frame["world"].astype(str)) == {"smooth"}
        and set(frame["n_train"]) == set(_D2_TRAIN_SIZES)
        and triples == _expected_triples(multiplier)
        and not frame.duplicated(duplicate_key).any()
    )
    if not valid:
        raise ValueError(f"{stage} effects do not match the frozen orthogonal array")
    values = pd.to_numeric(frame["Delta_MAE"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{stage} effects contain non-finite Delta_MAE")


def _dominant_factor(result: D2AnalysisResult, n_train: int) -> str:
    selected = result.variance_components[
        (result.variance_components["n_train"] == n_train)
        & (result.variance_components["component"].isin(("cohort", "subset", "model")))
        & (result.variance_components["factor_classification"] == "dominant")
    ]
    factors = selected["component"].astype(str).tolist()
    if len(factors) > 1:
        raise ValueError("D2 evidence contains multiple named dominant factors at one N")
    return factors[0] if factors else "none"


def _pearson_or_none(left: np.ndarray, right: np.ndarray) -> float | None:
    eps = np.finfo(float).eps
    if len(left) < 2 or float(np.ptp(left)) <= eps or float(np.ptp(right)) <= eps:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _overlap_rerun_diagnostics(
    d2a_result: D2AnalysisResult,
    d2b_result: D2AnalysisResult,
) -> dict[str, dict[str, object]]:
    keys = [*_PAIR_COLUMNS]
    left = d2a_result.effect_rows[keys + ["Delta_MAE"]].rename(
        columns={"Delta_MAE": "Delta_MAE_D2A"}
    )
    right = d2b_result.effect_rows[keys + ["Delta_MAE"]].rename(
        columns={"Delta_MAE": "Delta_MAE_D2B"}
    )
    overlap = left.merge(right, on=keys, validate="one_to_one")
    if len(overlap) != 10:
        raise ValueError("D2-A/D2-B overlap must contain exactly ten paired N-specific effects")

    output: dict[str, dict[str, object]] = {}
    for n_train in _D2_TRAIN_SIZES:
        frame = overlap[overlap["n_train"] == n_train]
        if len(frame) != 5:
            raise ValueError("D2-A/D2-B overlap must contain five seed triples per N")
        left_values = frame["Delta_MAE_D2A"].to_numpy(dtype=float)
        right_values = frame["Delta_MAE_D2B"].to_numpy(dtype=float)
        differences = np.abs(left_values - right_values)
        output[str(n_train)] = {
            "overlap_pair_count": len(frame),
            "mean_absolute_delta_mae_difference": float(differences.mean()),
            "maximum_absolute_delta_mae_difference": float(differences.max()),
            "pearson_correlation": _pearson_or_none(left_values, right_values),
        }
    return output


def _canonical_json_hash(payload: object) -> str:
    data = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def adjudicate_d2b(
    parent_d3: dict[str, object],
    d2a_result: D2AnalysisResult,
    d2b_result: D2AnalysisResult,
) -> dict[str, object]:
    """Apply the frozen D2-A/D2-B cross-array sufficiency rule."""
    if not isinstance(parent_d3, dict):
        raise TypeError("parent_d3 must be a dictionary")
    if parent_d3.get("next_required_stage") != "D2B":
        raise ValueError("parent D3 must authorize D2B as next_required_stage")
    triggered = parent_d3.get("triggered_escalations")
    if not isinstance(triggered, list) or not triggered or triggered[0] != "D2B":
        raise ValueError("parent D3 escalation queue must begin with D2B")
    if not all(isinstance(route, str) for route in triggered):
        raise TypeError("parent D3 escalation queue must contain strings")

    _validate_d2_adjudication_input(d2a_result)
    _validate_d2_adjudication_input(d2b_result)
    _validate_effect_grid(d2a_result, "d2a", 1)
    _validate_effect_grid(d2b_result, "d2b", 2)

    dominant = {
        "d2a": {
            "5": _dominant_factor(d2a_result, 5),
            "40": _dominant_factor(d2a_result, 40),
        },
        "d2b": {
            "5": _dominant_factor(d2b_result, 5),
            "40": _dominant_factor(d2b_result, 40),
        },
    }
    n40_consistent = (
        dominant["d2a"]["40"] != "none"
        and dominant["d2a"]["40"] == dominant["d2b"]["40"]
    )
    n5_consistent = dominant["d2a"]["5"] == dominant["d2b"]["5"]
    sufficient = n40_consistent and n5_consistent

    remaining = [route for route in triggered if route != "D2B"]
    next_required_stage = (
        (remaining[0] if remaining else "STOP")
        if sufficient
        else "FULL_FACTORIAL_ADDENDUM"
    )
    status = "sufficient" if sufficient else "insufficient"
    overlap = _overlap_rerun_diagnostics(d2a_result, d2b_result)
    return {
        "complementary_array_evidence": status,
        "dominant_factors": dominant,
        "overlap_rerun_diagnostics": overlap,
        "remaining_parent_escalations": remaining,
        "next_required_stage": next_required_stage,
        "rationale": [
            f"D2-A/D2-B N40 dominant factors: {dominant['d2a']['40']}/{dominant['d2b']['40']}.",
            f"D2-A/D2-B N5 dominant factors: {dominant['d2a']['5']}/{dominant['d2b']['5']}.",
            f"Complementary-array evidence: {status}.",
            f"Next required stage: {next_required_stage}.",
        ],
        "input_artifact_hashes": {
            "parent_d3": _canonical_json_hash(parent_d3),
            "d2a_effect_rows": _frame_hash(d2a_result.effect_rows),
            "d2a_variance_components": _frame_hash(d2a_result.variance_components),
            "d2b_effect_rows": _frame_hash(d2b_result.effect_rows),
            "d2b_variance_components": _frame_hash(d2b_result.variance_components),
        },
    }


__all__ = ["adjudicate_d2b", "analyze_d2b"]
