from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

_CONTROL_VARIANT = "none__none__deterministic"
_CANDIDATE_VARIANT = "time_scaled__none__deterministic"
_CONTEXTS = (
    (401, 501),
    (402, 502),
    (403, 503),
    (404, 504),
    (405, 505),
)
_MODEL_SEEDS = tuple(range(1001, 1011))
_BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_SEED = 20260827
_PAIR_COLUMNS = (
    "world",
    "cohort_seed",
    "subset_seed",
    "model_seed",
    "n_train",
)
_CELL_COLUMNS = (*_PAIR_COLUMNS, "variant")
_ALLOWED_STOP_REASONS = frozenset({"patience_exhausted", "max_epochs_reached"})


@dataclass(frozen=True, slots=True)
class D4BAnalysisResult:
    effect_rows: pd.DataFrame
    model_seed_summary: pd.DataFrame
    context_summary: pd.DataFrame
    bootstrap_diagnostics: dict[str, object]
    optimization_dispersion: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")


def _expected_cell_keys() -> set[tuple[object, ...]]:
    return {
        ("smooth", cohort, subset, model, 40, variant)
        for cohort, subset in _CONTEXTS
        for model in _MODEL_SEEDS
        for variant in (_CONTROL_VARIANT, _CANDIDATE_VARIANT)
    }


def _observed_cell_keys(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    return set(frame[list(_CELL_COLUMNS)].itertuples(index=False, name=None))


def _validate_metric_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "stage",
        *_CELL_COLUMNS,
        "split",
        "metric",
        "value",
    }
    _require_columns(metrics, required, "D4-B metric matrix")
    target = metrics[
        (metrics["stage"] == "d4b")
        & (metrics["world"] == "smooth")
        & (metrics["split"] == "test")
        & (metrics["metric"] == "mae")
    ].copy()
    if target.empty:
        raise ValueError("D4-B metric matrix contains no test MAE rows")
    if len(target) != 100:
        raise ValueError("D4-B metric matrix must contain exactly 100 test MAE rows")
    if target.duplicated(list(_CELL_COLUMNS)).any():
        raise ValueError("D4-B metric matrix contains duplicate cells")
    if _observed_cell_keys(target) != _expected_cell_keys():
        raise ValueError("D4-B metric matrix does not match the frozen 100-cell design")
    values = pd.to_numeric(target["value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("D4-B metric matrix contains invalid MAE values")
    return target


def _validate_summaries(summaries: pd.DataFrame) -> pd.DataFrame:
    required = {
        "stage",
        *_CELL_COLUMNS,
        "selected_checkpoint_epoch",
        "stop_epoch",
        "early_stop_reason",
    }
    _require_columns(summaries, required, "D4-B summaries")
    target = summaries[
        (summaries["stage"] == "d4b") & (summaries["world"] == "smooth")
    ].copy()
    if len(target) != 100 or target.duplicated(list(_CELL_COLUMNS)).any():
        raise ValueError("D4-B summaries must contain exactly one row per frozen cell")
    if _observed_cell_keys(target) != _expected_cell_keys():
        raise ValueError("D4-B summaries do not match the frozen 100-cell design")
    for column in ("selected_checkpoint_epoch", "stop_epoch"):
        values = pd.to_numeric(target[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"D4-B summaries contain invalid {column}")
    if not set(target["early_stop_reason"].astype(str)).issubset(_ALLOWED_STOP_REASONS):
        raise ValueError("D4-B summaries contain an invalid early-stop reason")
    return target


def _validate_traces(traces: pd.DataFrame) -> pd.DataFrame:
    required = {
        "stage",
        *_CELL_COLUMNS,
        "epoch",
        "gradient_l2_norm",
        "mean_flow_displacement",
    }
    _require_columns(traces, required, "D4-B traces")
    target = traces[
        (traces["stage"] == "d4b") & (traces["world"] == "smooth")
    ].copy()
    if target.empty:
        raise ValueError("D4-B traces are empty")
    if _observed_cell_keys(target) != _expected_cell_keys():
        raise ValueError("D4-B traces do not cover every frozen cell")
    if target.duplicated([*_CELL_COLUMNS, "epoch"]).any():
        raise ValueError("D4-B traces contain duplicate epochs")
    for column in ("epoch", "gradient_l2_norm", "mean_flow_displacement"):
        values = pd.to_numeric(target[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"D4-B traces contain non-finite {column}")
    if (pd.to_numeric(target["epoch"], errors="coerce") <= 0).any():
        raise ValueError("D4-B traces contain invalid epochs")
    return target


def _effect_rows(metrics: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    control = metrics[metrics["variant"] == _CONTROL_VARIANT][
        [*_PAIR_COLUMNS, "value"]
    ].rename(columns={"value": "control_mae"})
    candidate = metrics[metrics["variant"] == _CANDIDATE_VARIANT][
        [*_PAIR_COLUMNS, "value"]
    ].rename(columns={"value": "time_scaled_mae"})
    paired = control.merge(candidate, on=list(_PAIR_COLUMNS), validate="one_to_one")
    if len(paired) != 50:
        raise ValueError("D4-B metrics do not form exactly 50 paired effects")

    control_summary = summaries[summaries["variant"] == _CONTROL_VARIANT][
        [*_PAIR_COLUMNS, "selected_checkpoint_epoch", "stop_epoch"]
    ].rename(
        columns={
            "selected_checkpoint_epoch": "control_selected_epoch",
            "stop_epoch": "control_stop_epoch",
        }
    )
    candidate_summary = summaries[summaries["variant"] == _CANDIDATE_VARIANT][
        [*_PAIR_COLUMNS, "selected_checkpoint_epoch", "stop_epoch"]
    ].rename(
        columns={
            "selected_checkpoint_epoch": "time_scaled_selected_epoch",
            "stop_epoch": "time_scaled_stop_epoch",
        }
    )
    paired = paired.merge(control_summary, on=list(_PAIR_COLUMNS), validate="one_to_one")
    paired = paired.merge(candidate_summary, on=list(_PAIR_COLUMNS), validate="one_to_one")
    paired["Delta_MAE"] = paired["control_mae"] - paired["time_scaled_mae"]
    paired.insert(0, "stage", "d4b")
    return paired.sort_values(
        ["cohort_seed", "subset_seed", "model_seed"],
        kind="mergesort",
    ).reset_index(drop=True)


def _model_seed_summary(effects: pd.DataFrame) -> pd.DataFrame:
    grouped = effects.groupby("model_seed", sort=True)["Delta_MAE"]
    frame = grouped.agg(["mean", "median", "std"]).reset_index()
    frame = frame.rename(
        columns={
            "mean": "mean_Delta_MAE",
            "median": "median_Delta_MAE",
            "std": "std_Delta_MAE",
        }
    )
    frame["context_count"] = grouped.size().to_numpy(dtype=int)
    if len(frame) != 10 or set(frame["model_seed"]) != set(_MODEL_SEEDS):
        raise ValueError("D4-B model-seed summary is incomplete")
    if set(frame["context_count"]) != {5}:
        raise ValueError("D4-B model-seed summary must contain five contexts per seed")
    return frame


def _context_summary(effects: pd.DataFrame) -> pd.DataFrame:
    grouped = effects.groupby(["cohort_seed", "subset_seed"], sort=True)["Delta_MAE"]
    frame = grouped.agg(["mean", "median", "std"]).reset_index()
    frame = frame.rename(
        columns={
            "mean": "mean_Delta_MAE",
            "median": "median_Delta_MAE",
            "std": "std_Delta_MAE",
        }
    )
    frame["model_seed_count"] = grouped.size().to_numpy(dtype=int)
    observed = set(
        frame[["cohort_seed", "subset_seed"]].itertuples(index=False, name=None)
    )
    if len(frame) != 5 or observed != set(_CONTEXTS):
        raise ValueError("D4-B context summary is incomplete")
    if set(frame["model_seed_count"]) != {10}:
        raise ValueError("D4-B context summary must contain ten model seeds per context")
    return frame


def _cluster_bootstrap(
    effects: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> dict[str, object]:
    means = model_summary.sort_values("model_seed", kind="mergesort")[
        "mean_Delta_MAE"
    ].to_numpy(dtype=float)
    if len(means) != 10 or not np.isfinite(means).all():
        raise ValueError("D4-B model-seed means are invalid")
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    indices = rng.integers(0, len(means), size=(_BOOTSTRAP_RESAMPLES, len(means)))
    bootstrap_means = means[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "mean_Delta_MAE": float(effects["Delta_MAE"].mean()),
        "bootstrap_ci_lower": float(lower),
        "bootstrap_ci_upper": float(upper),
        "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": _BOOTSTRAP_SEED,
        "resampling_unit": "model_seed_cluster",
        "confidence_interval": "percentile_95",
        "pair_count": 50,
        "model_seed_count": 10,
        "contexts_per_model_seed": 5,
    }


def _iqr(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not len(numeric):
        raise ValueError("D4-B dispersion statistic contains invalid values")
    q25, q75 = np.quantile(numeric, [0.25, 0.75])
    return float(q75 - q25)


def _mean(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="raise").mean())


def _std(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="raise").std(ddof=1))


def _optimization_dispersion(
    summaries: pd.DataFrame,
    traces: pd.DataFrame,
) -> pd.DataFrame:
    trace_cell = (
        traces.groupby(list(_CELL_COLUMNS), sort=True)
        .agg(
            mean_gradient_l2_norm=("gradient_l2_norm", "mean"),
            mean_flow_displacement=("mean_flow_displacement", "mean"),
        )
        .reset_index()
    )
    if len(trace_cell) != 100:
        raise ValueError("D4-B trace aggregation must contain exactly 100 cells")
    joined = summaries.merge(trace_cell, on=list(_CELL_COLUMNS), validate="one_to_one")
    rows: list[dict[str, object]] = []
    for (cohort, subset, variant), frame in joined.groupby(
        ["cohort_seed", "subset_seed", "variant"], sort=True
    ):
        if len(frame) != 10 or set(frame["model_seed"]) != set(_MODEL_SEEDS):
            raise ValueError("D4-B dispersion group must contain ten model seeds")
        reason_counts = frame["early_stop_reason"].astype(str).value_counts().to_dict()
        rows.append(
            {
                "cohort_seed": int(cohort),
                "subset_seed": int(subset),
                "variant": str(variant),
                "model_seed_count": 10,
                "selected_checkpoint_epoch_mean": _mean(
                    frame["selected_checkpoint_epoch"]
                ),
                "selected_checkpoint_epoch_std": _std(
                    frame["selected_checkpoint_epoch"]
                ),
                "selected_checkpoint_epoch_iqr": _iqr(
                    frame["selected_checkpoint_epoch"]
                ),
                "stop_epoch_mean": _mean(frame["stop_epoch"]),
                "stop_epoch_std": _std(frame["stop_epoch"]),
                "stop_epoch_iqr": _iqr(frame["stop_epoch"]),
                "mean_gradient_l2_norm_mean": _mean(frame["mean_gradient_l2_norm"]),
                "mean_gradient_l2_norm_std": _std(frame["mean_gradient_l2_norm"]),
                "mean_gradient_l2_norm_iqr": _iqr(frame["mean_gradient_l2_norm"]),
                "mean_flow_displacement_mean": _mean(frame["mean_flow_displacement"]),
                "mean_flow_displacement_std": _std(frame["mean_flow_displacement"]),
                "mean_flow_displacement_iqr": _iqr(frame["mean_flow_displacement"]),
                "patience_exhausted_count": int(
                    reason_counts.get("patience_exhausted", 0)
                ),
                "max_epochs_reached_count": int(
                    reason_counts.get("max_epochs_reached", 0)
                ),
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != 10:
        raise ValueError("D4-B optimization dispersion must contain ten groups")
    return output


def analyze_d4b(
    metrics: pd.DataFrame,
    summaries: pd.DataFrame,
    traces: pd.DataFrame,
) -> D4BAnalysisResult:
    """Analyze the frozen D4-B initialization/optimization stability matrix."""
    target_metrics = _validate_metric_matrix(metrics)
    target_summaries = _validate_summaries(summaries)
    target_traces = _validate_traces(traces)
    effects = _effect_rows(target_metrics, target_summaries)
    model_summary = _model_seed_summary(effects)
    context_summary = _context_summary(effects)
    bootstrap = _cluster_bootstrap(effects, model_summary)
    dispersion = _optimization_dispersion(target_summaries, target_traces)
    return D4BAnalysisResult(
        effect_rows=effects,
        model_seed_summary=model_summary,
        context_summary=context_summary,
        bootstrap_diagnostics=bootstrap,
        optimization_dispersion=dispersion,
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("D4-B hash input must be a pandas DataFrame")
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _json_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _validate_adjudication_result(result: D4BAnalysisResult) -> tuple[float, int, int]:
    if not isinstance(result, D4BAnalysisResult):
        raise TypeError("result must be a D4BAnalysisResult")
    effects = result.effect_rows
    models = result.model_seed_summary
    contexts = result.context_summary
    bootstrap = result.bootstrap_diagnostics
    _require_columns(
        effects,
        {"cohort_seed", "subset_seed", "model_seed", "Delta_MAE"},
        "D4-B effect rows",
    )
    _require_columns(models, {"model_seed", "mean_Delta_MAE"}, "D4-B model summary")
    _require_columns(
        contexts,
        {"cohort_seed", "subset_seed", "mean_Delta_MAE"},
        "D4-B context summary",
    )
    if len(effects) != 50 or len(models) != 10 or len(contexts) != 5:
        raise ValueError("D4-B adjudication evidence has incorrect cardinality")
    if set(models["model_seed"]) != set(_MODEL_SEEDS):
        raise ValueError("D4-B adjudication model-seed summary is incomplete")
    observed_contexts = set(
        contexts[["cohort_seed", "subset_seed"]].itertuples(index=False, name=None)
    )
    if observed_contexts != set(_CONTEXTS):
        raise ValueError("D4-B adjudication context summary is incomplete")

    recomputed_models = (
        effects.groupby("model_seed", sort=True)["Delta_MAE"].mean().sort_index()
    )
    recorded_models = models.set_index("model_seed")["mean_Delta_MAE"].sort_index()
    if not np.allclose(
        recomputed_models.to_numpy(dtype=float),
        recorded_models.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("D4-B model-seed summary does not match paired effects")
    recomputed_contexts = (
        effects.groupby(["cohort_seed", "subset_seed"], sort=True)["Delta_MAE"]
        .mean()
        .sort_index()
    )
    recorded_contexts = (
        contexts.set_index(["cohort_seed", "subset_seed"])["mean_Delta_MAE"].sort_index()
    )
    if not np.allclose(
        recomputed_contexts.to_numpy(dtype=float),
        recorded_contexts.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("D4-B context summary does not match paired effects")

    required_bootstrap = {
        "mean_Delta_MAE",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "bootstrap_resamples",
        "bootstrap_seed",
        "resampling_unit",
        "confidence_interval",
    }
    missing = required_bootstrap - set(bootstrap)
    if missing:
        raise ValueError("D4-B bootstrap diagnostics are incomplete")
    if bootstrap["bootstrap_resamples"] != _BOOTSTRAP_RESAMPLES:
        raise ValueError("D4-B bootstrap resample count drift")
    if bootstrap["bootstrap_seed"] != _BOOTSTRAP_SEED:
        raise ValueError("D4-B bootstrap seed drift")
    if bootstrap["resampling_unit"] != "model_seed_cluster":
        raise ValueError("D4-B bootstrap resampling unit drift")
    if bootstrap["confidence_interval"] != "percentile_95":
        raise ValueError("D4-B bootstrap confidence interval drift")

    overall = float(effects["Delta_MAE"].mean())
    recorded_overall = float(bootstrap["mean_Delta_MAE"])
    if not np.isclose(overall, recorded_overall, rtol=1e-12, atol=1e-12):
        raise ValueError("D4-B overall effect does not match paired effects")
    model_values = recorded_models.to_numpy(dtype=float)
    context_values = recorded_contexts.to_numpy(dtype=float)
    if not np.isfinite(model_values).all() or not np.isfinite(context_values).all():
        raise ValueError("D4-B adjudication summaries contain non-finite effects")
    return (
        overall,
        int((model_values > 0).sum()),
        int((context_values > 0).sum()),
    )


def adjudicate_d4b(result: D4BAnalysisResult) -> dict[str, object]:
    """Apply the frozen D4-B stability/fragility decision rule."""
    overall, positive_models, positive_contexts = _validate_adjudication_result(result)
    lower = float(result.bootstrap_diagnostics["bootstrap_ci_lower"])
    upper = float(result.bootstrap_diagnostics["bootstrap_ci_upper"])
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("D4-B bootstrap confidence interval is invalid")

    stable = (
        overall > 0
        and positive_models >= 8
        and positive_contexts >= 4
        and lower > 0
    )
    fragile = (
        overall <= 0
        or positive_models <= 5
        or positive_contexts <= 2
    )
    if stable:
        classification = "stable"
        next_required_stage = "D4_CAPACITY_TIME"
    elif fragile:
        classification = "fragile"
        next_required_stage = "STOP"
    else:
        classification = "ambiguous"
        next_required_stage = "STOP"

    return {
        "classification": classification,
        "Delta_overall": overall,
        "positive_model_seed_count": positive_models,
        "positive_context_count": positive_contexts,
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "next_required_stage": next_required_stage,
        "rationale": [
            f"Overall paired Delta_MAE: {overall:.12g}.",
            f"Positive model-seed means: {positive_models}/10.",
            f"Positive fixed-context means: {positive_contexts}/5.",
            f"Model-seed-cluster bootstrap 95% CI: [{lower:.12g}, {upper:.12g}].",
            f"D4-B classification: {classification}.",
            f"Next required stage: {next_required_stage}.",
        ],
        "input_artifact_hashes": {
            "effect_rows": _frame_hash(result.effect_rows),
            "model_seed_summary": _frame_hash(result.model_seed_summary),
            "context_summary": _frame_hash(result.context_summary),
            "bootstrap_diagnostics": _json_hash(result.bootstrap_diagnostics),
            "optimization_dispersion": _frame_hash(result.optimization_dispersion),
        },
    }


__all__ = ["D4BAnalysisResult", "adjudicate_d4b", "analyze_d4b"]
