from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

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
_PAIR_COLUMNS = (
    "world",
    "cohort_seed",
    "subset_seed",
    "model_seed",
    "n_train",
)
_CELL_COLUMNS = (*_PAIR_COLUMNS, "variant")
_PREFIX_EPOCHS = tuple(range(1, 11))
_ALLOWED_STOP_REASONS = frozenset({"patience_exhausted", "max_epochs_reached"})
_PRIMARY_MECHANISMS = (
    "delta_selected_epoch",
    "delta_stop_epoch",
    "delta_shadow_mae_epoch",
    "delta_selection_shadow_gap",
    "delta_patience_exhausted",
)
_SECONDARY_MECHANISMS = (
    "delta_prefix10_gradient_l2_mean",
    "delta_prefix10_parameter_l2_mean",
    "candidate_prefix10_flow_displacement_mean",
)


@dataclass(frozen=True, slots=True)
class PosthocOptimizationResult:
    pair_table: pd.DataFrame
    associations: pd.DataFrame
    leave_one_out: pd.DataFrame
    screening: dict[str, object]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")


def _expected_pair_keys() -> set[tuple[object, ...]]:
    return {
        ("smooth", cohort, subset, model, 40)
        for cohort, subset in _CONTEXTS
        for model in _MODEL_SEEDS
    }


def _expected_cell_keys() -> set[tuple[object, ...]]:
    return {
        (*pair, variant)
        for pair in _expected_pair_keys()
        for variant in (_CONTROL_VARIANT, _CANDIDATE_VARIANT)
    }


def _validate_effects(effects: pd.DataFrame) -> pd.DataFrame:
    required = {"stage", *_PAIR_COLUMNS, "Delta_MAE"}
    _require_columns(effects, required, "D4-B paired effects")
    target = effects[
        (effects["stage"] == "d4b") & (effects["world"] == "smooth")
    ].copy()
    observed = set(target[list(_PAIR_COLUMNS)].itertuples(index=False, name=None))
    if len(target) != 50 or observed != _expected_pair_keys():
        raise ValueError("D4-B post-hoc analysis requires exactly the frozen 50 paired effects")
    if target.duplicated(list(_PAIR_COLUMNS)).any():
        raise ValueError("D4-B paired effects contain duplicate cells")
    values = pd.to_numeric(target["Delta_MAE"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("D4-B paired effects contain non-finite Delta_MAE values")
    return target


def _validate_summaries(summaries: pd.DataFrame) -> pd.DataFrame:
    required = {
        "stage",
        *_CELL_COLUMNS,
        "selected_checkpoint_epoch",
        "shadow_mae_checkpoint_epoch",
        "stop_epoch",
        "early_stop_reason",
    }
    _require_columns(summaries, required, "D4-B post-hoc summaries")
    target = summaries[
        (summaries["stage"] == "d4b") & (summaries["world"] == "smooth")
    ].copy()
    observed = set(target[list(_CELL_COLUMNS)].itertuples(index=False, name=None))
    if len(target) != 100 or observed != _expected_cell_keys():
        raise ValueError("D4-B post-hoc summaries must contain the frozen 100 cells")
    if target.duplicated(list(_CELL_COLUMNS)).any():
        raise ValueError("D4-B post-hoc summaries contain duplicate cells")
    for column in (
        "selected_checkpoint_epoch",
        "shadow_mae_checkpoint_epoch",
        "stop_epoch",
    ):
        values = pd.to_numeric(target[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"D4-B post-hoc summaries contain invalid {column}")
    reasons = set(target["early_stop_reason"].astype(str))
    if not reasons.issubset(_ALLOWED_STOP_REASONS):
        raise ValueError("D4-B post-hoc summaries contain invalid early-stop reasons")
    return target


def _validate_traces(traces: pd.DataFrame) -> pd.DataFrame:
    required = {
        "stage",
        *_CELL_COLUMNS,
        "epoch",
        "gradient_l2_norm",
        "parameter_l2_norm",
        "mean_flow_displacement",
    }
    _require_columns(traces, required, "D4-B post-hoc traces")
    target = traces[
        (traces["stage"] == "d4b") & (traces["world"] == "smooth")
    ].copy()
    observed = set(target[list(_CELL_COLUMNS)].itertuples(index=False, name=None))
    if observed != _expected_cell_keys():
        raise ValueError("D4-B post-hoc traces do not cover the frozen 100 cells")
    if target.duplicated([*_CELL_COLUMNS, "epoch"]).any():
        raise ValueError("D4-B post-hoc traces contain duplicate epochs")
    for column in (
        "epoch",
        "gradient_l2_norm",
        "parameter_l2_norm",
        "mean_flow_displacement",
    ):
        values = pd.to_numeric(target[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"D4-B post-hoc traces contain non-finite {column}")

    prefix = target[target["epoch"].isin(_PREFIX_EPOCHS)]
    counts = prefix.groupby(list(_CELL_COLUMNS), sort=False)["epoch"].nunique()
    if len(counts) != 100 or set(counts.to_numpy(dtype=int)) != {10}:
        raise ValueError("D4-B post-hoc traces require epochs 1..10 for every frozen cell")
    return target


def _summary_side(summaries: pd.DataFrame, variant: str, prefix: str) -> pd.DataFrame:
    selected = summaries[summaries["variant"] == variant][
        [
            *_PAIR_COLUMNS,
            "selected_checkpoint_epoch",
            "shadow_mae_checkpoint_epoch",
            "stop_epoch",
            "early_stop_reason",
        ]
    ].copy()
    return selected.rename(
        columns={
            "selected_checkpoint_epoch": f"{prefix}_selected_epoch",
            "shadow_mae_checkpoint_epoch": f"{prefix}_shadow_mae_epoch",
            "stop_epoch": f"{prefix}_stop_epoch",
            "early_stop_reason": f"{prefix}_early_stop_reason",
        }
    )


def _prefix_trace_cells(traces: pd.DataFrame) -> pd.DataFrame:
    prefix = traces[traces["epoch"].isin(_PREFIX_EPOCHS)].copy()
    aggregated = (
        prefix.groupby(list(_CELL_COLUMNS), sort=True)
        .agg(
            prefix10_gradient_l2_mean=("gradient_l2_norm", "mean"),
            prefix10_parameter_l2_mean=("parameter_l2_norm", "mean"),
            prefix10_flow_displacement_mean=("mean_flow_displacement", "mean"),
        )
        .reset_index()
    )
    if len(aggregated) != 100:
        raise ValueError("D4-B prefix-trace aggregation must contain 100 frozen cells")
    return aggregated


def _trace_side(trace_cells: pd.DataFrame, variant: str, prefix: str) -> pd.DataFrame:
    selected = trace_cells[trace_cells["variant"] == variant][
        [
            *_PAIR_COLUMNS,
            "prefix10_gradient_l2_mean",
            "prefix10_parameter_l2_mean",
            "prefix10_flow_displacement_mean",
        ]
    ].copy()
    return selected.rename(
        columns={
            "prefix10_gradient_l2_mean": f"{prefix}_prefix10_gradient_l2_mean",
            "prefix10_parameter_l2_mean": f"{prefix}_prefix10_parameter_l2_mean",
            "prefix10_flow_displacement_mean": f"{prefix}_prefix10_flow_displacement_mean",
        }
    )


def _build_pair_table(
    effects: pd.DataFrame,
    summaries: pd.DataFrame,
    traces: pd.DataFrame,
) -> pd.DataFrame:
    effect_columns = ["stage", *_PAIR_COLUMNS, "Delta_MAE"]
    effect_columns.extend(
        column
        for column in ("control_mae", "time_scaled_mae")
        if column in effects.columns
    )
    pair = effects[effect_columns].copy()
    control_summary = _summary_side(summaries, _CONTROL_VARIANT, "control")
    candidate_summary = _summary_side(summaries, _CANDIDATE_VARIANT, "candidate")
    pair = pair.merge(control_summary, on=list(_PAIR_COLUMNS), validate="one_to_one")
    pair = pair.merge(candidate_summary, on=list(_PAIR_COLUMNS), validate="one_to_one")

    trace_cells = _prefix_trace_cells(traces)
    control_trace = _trace_side(trace_cells, _CONTROL_VARIANT, "control")
    candidate_trace = _trace_side(trace_cells, _CANDIDATE_VARIANT, "candidate")
    pair = pair.merge(control_trace, on=list(_PAIR_COLUMNS), validate="one_to_one")
    pair = pair.merge(candidate_trace, on=list(_PAIR_COLUMNS), validate="one_to_one")
    if len(pair) != 50:
        raise ValueError("D4-B post-hoc joins did not preserve all 50 paired effects")

    pair["delta_selected_epoch"] = (
        pair["candidate_selected_epoch"] - pair["control_selected_epoch"]
    )
    pair["delta_stop_epoch"] = pair["candidate_stop_epoch"] - pair["control_stop_epoch"]
    pair["delta_shadow_mae_epoch"] = (
        pair["candidate_shadow_mae_epoch"] - pair["control_shadow_mae_epoch"]
    )
    pair["control_selection_shadow_gap"] = (
        pair["control_selected_epoch"] - pair["control_shadow_mae_epoch"]
    )
    pair["candidate_selection_shadow_gap"] = (
        pair["candidate_selected_epoch"] - pair["candidate_shadow_mae_epoch"]
    )
    pair["delta_selection_shadow_gap"] = (
        pair["candidate_selection_shadow_gap"] - pair["control_selection_shadow_gap"]
    )
    pair["control_patience_exhausted"] = (
        pair["control_early_stop_reason"] == "patience_exhausted"
    ).astype(int)
    pair["candidate_patience_exhausted"] = (
        pair["candidate_early_stop_reason"] == "patience_exhausted"
    ).astype(int)
    pair["delta_patience_exhausted"] = (
        pair["candidate_patience_exhausted"] - pair["control_patience_exhausted"]
    )
    pair["delta_prefix10_gradient_l2_mean"] = (
        pair["candidate_prefix10_gradient_l2_mean"]
        - pair["control_prefix10_gradient_l2_mean"]
    )
    pair["delta_prefix10_parameter_l2_mean"] = (
        pair["candidate_prefix10_parameter_l2_mean"]
        - pair["control_prefix10_parameter_l2_mean"]
    )
    return pair.sort_values(
        ["cohort_seed", "subset_seed", "model_seed"],
        kind="mergesort",
    ).reset_index(drop=True)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(pearsonr(x, y).statistic)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _context_demean(frame: pd.DataFrame, column: str) -> np.ndarray:
    context = ["cohort_seed", "subset_seed"]
    values = pd.to_numeric(frame[column], errors="raise").astype(float)
    return (values - values.groupby([frame[c] for c in context]).transform("mean")).to_numpy()


def _two_way_residual(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").astype(float)
    context_mean = values.groupby(
        [frame["cohort_seed"], frame["subset_seed"]]
    ).transform("mean")
    model_mean = values.groupby(frame["model_seed"]).transform("mean")
    return (values - context_mean - model_mean + values.mean()).to_numpy(dtype=float)


def _two_way_correlations(frame: pd.DataFrame, mechanism: str) -> tuple[float, float]:
    x = _two_way_residual(frame, mechanism)
    y = _two_way_residual(frame, "Delta_MAE")
    return _safe_pearson(x, y), _safe_spearman(x, y)


def _crossed_permutation_pvalue(
    frame: pd.DataFrame,
    mechanism: str,
    *,
    resamples: int,
    seed: int,
) -> float:
    if resamples <= 0:
        raise ValueError("permutation_resamples must be positive")
    contexts = list(_CONTEXTS)
    models = list(_MODEL_SEEDS)
    x = _two_way_residual(frame, mechanism)
    y = _two_way_residual(frame, "Delta_MAE")
    residual_frame = frame[["cohort_seed", "subset_seed", "model_seed"]].copy()
    residual_frame["x"] = x
    residual_frame["y"] = y
    residual = residual_frame.set_index(
        ["cohort_seed", "subset_seed", "model_seed"]
    ).sort_index()
    x_matrix = np.asarray(
        [
            [residual.loc[(cohort, subset, model), "x"] for model in models]
            for cohort, subset in contexts
        ],
        dtype=float,
    )
    y_matrix = np.asarray(
        [
            [residual.loc[(cohort, subset, model), "y"] for model in models]
            for cohort, subset in contexts
        ],
        dtype=float,
    )
    observed = _safe_pearson(x_matrix.ravel(), y_matrix.ravel())
    if not np.isfinite(observed):
        return 1.0

    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(resamples):
        row_order = rng.permutation(len(contexts))
        column_order = rng.permutation(len(models))
        permuted = x_matrix[row_order][:, column_order]
        statistic = _safe_pearson(permuted.ravel(), y_matrix.ravel())
        if np.isfinite(statistic) and abs(statistic) >= abs(observed):
            exceedances += 1
    return float((exceedances + 1) / (resamples + 1))


def _mechanism_family(mechanism: str) -> str:
    if mechanism in _PRIMARY_MECHANISMS:
        return "primary"
    if mechanism in _SECONDARY_MECHANISMS:
        return "secondary"
    raise ValueError(f"Unknown post-hoc mechanism: {mechanism}")


def _holm_adjust_primary(associations: pd.DataFrame) -> pd.DataFrame:
    output = associations.copy()
    output["permutation_pvalue_holm"] = np.nan
    primary = output[output["family"] == "primary"].copy()
    order = primary.sort_values("permutation_pvalue", kind="mergesort").index.tolist()
    count = len(order)
    running = 0.0
    for rank, index in enumerate(order):
        raw = float(output.loc[index, "permutation_pvalue"])
        adjusted = min(1.0, raw * (count - rank))
        running = max(running, adjusted)
        output.loc[index, "permutation_pvalue_holm"] = running
    return output


def _association_table(
    pair: pd.DataFrame,
    *,
    permutation_resamples: int,
    permutation_seed: int,
) -> pd.DataFrame:
    y = pd.to_numeric(pair["Delta_MAE"], errors="raise").to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    mechanisms = (*_PRIMARY_MECHANISMS, *_SECONDARY_MECHANISMS)
    for mechanism_index, mechanism in enumerate(mechanisms):
        x = pd.to_numeric(pair[mechanism], errors="raise").to_numpy(dtype=float)
        context_x = _context_demean(pair, mechanism)
        context_y = _context_demean(pair, "Delta_MAE")
        two_way_pearson, two_way_spearman = _two_way_correlations(pair, mechanism)
        rows.append(
            {
                "mechanism": mechanism,
                "family": _mechanism_family(mechanism),
                "raw_pearson": _safe_pearson(x, y),
                "raw_spearman": _safe_spearman(x, y),
                "context_demeaned_pearson": _safe_pearson(context_x, context_y),
                "context_demeaned_spearman": _safe_spearman(context_x, context_y),
                "two_way_pearson": two_way_pearson,
                "two_way_spearman": two_way_spearman,
                "permutation_pvalue": _crossed_permutation_pvalue(
                    pair,
                    mechanism,
                    resamples=permutation_resamples,
                    seed=permutation_seed + mechanism_index,
                ),
                "permutation_resamples": permutation_resamples,
                "permutation_seed": permutation_seed,
            }
        )
    return _holm_adjust_primary(pd.DataFrame(rows))


def _leave_one_out_table(pair: pd.DataFrame, associations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mechanism in associations["mechanism"].astype(str):
        for cohort, subset in _CONTEXTS:
            reduced = pair[
                ~(
                    (pair["cohort_seed"] == cohort)
                    & (pair["subset_seed"] == subset)
                )
            ]
            pearson, spearman = _two_way_correlations(reduced, mechanism)
            rows.append(
                {
                    "mechanism": mechanism,
                    "omitted_axis": "context",
                    "omitted_level": f"{cohort}-{subset}",
                    "pair_count": len(reduced),
                    "two_way_pearson": pearson,
                    "two_way_spearman": spearman,
                }
            )
        for model in _MODEL_SEEDS:
            reduced = pair[pair["model_seed"] != model]
            pearson, spearman = _two_way_correlations(reduced, mechanism)
            rows.append(
                {
                    "mechanism": mechanism,
                    "omitted_axis": "model_seed",
                    "omitted_level": str(model),
                    "pair_count": len(reduced),
                    "two_way_pearson": pearson,
                    "two_way_spearman": spearman,
                }
            )
    return pd.DataFrame(rows)


def _same_sign_fraction(values: pd.Series, reference: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if not np.isfinite(reference) or len(finite) == 0 or reference == 0:
        return 0.0
    return float(np.mean(np.sign(finite) == np.sign(reference)))


def _screening_summary(
    associations: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> dict[str, object]:
    passing: list[str] = []
    diagnostics: dict[str, dict[str, object]] = {}
    for row in associations[associations["family"] == "primary"].itertuples():
        omitted = leave_one_out[leave_one_out["mechanism"] == row.mechanism]
        context_fraction = _same_sign_fraction(
            omitted.loc[omitted["omitted_axis"] == "context", "two_way_pearson"],
            float(row.two_way_pearson),
        )
        model_fraction = _same_sign_fraction(
            omitted.loc[omitted["omitted_axis"] == "model_seed", "two_way_pearson"],
            float(row.two_way_pearson),
        )
        same_direction = (
            np.isfinite(row.two_way_pearson)
            and np.isfinite(row.two_way_spearman)
            and float(row.two_way_pearson) * float(row.two_way_spearman) > 0
        )
        criteria = {
            "abs_two_way_pearson_ge_0_35": bool(
                np.isfinite(row.two_way_pearson) and abs(float(row.two_way_pearson)) >= 0.35
            ),
            "abs_two_way_spearman_ge_0_30": bool(
                np.isfinite(row.two_way_spearman)
                and abs(float(row.two_way_spearman)) >= 0.30
            ),
            "pearson_spearman_same_direction": bool(same_direction),
            "holm_permutation_p_le_0_05": bool(
                np.isfinite(row.permutation_pvalue_holm)
                and float(row.permutation_pvalue_holm) <= 0.05
            ),
            "leave_one_context_out_same_sign_ge_0_80": context_fraction >= 0.80,
            "leave_one_model_out_same_sign_ge_0_80": model_fraction >= 0.80,
        }
        diagnostics[str(row.mechanism)] = {
            "criteria": criteria,
            "leave_one_context_out_same_sign_fraction": context_fraction,
            "leave_one_model_out_same_sign_fraction": model_fraction,
        }
        if all(criteria.values()):
            passing.append(str(row.mechanism))

    if passing:
        classification = (
            "structured optimization-conditioned heterogeneity worth prospective testing"
        )
    else:
        classification = "no sufficiently coherent mechanism identified"
    return {
        "classification": classification,
        "passing_primary_mechanisms": passing,
        "primary_mechanisms": list(_PRIMARY_MECHANISMS),
        "secondary_mechanisms": list(_SECONDARY_MECHANISMS),
        "mechanism_diagnostics": diagnostics,
        "effect_orientation": "Delta_MAE = control - time_scaled",
        "mechanism_orientation": "candidate - control",
        "trace_prefix_epochs": list(_PREFIX_EPOCHS),
        "permutation_scheme": "two-axis context/model label permutation after two-way demeaning",
        "multiple_testing": "Holm correction across the primary exploratory family",
        "exploratory_not_confirmatory": True,
        "phase06_terminal_decision": "AMBIGUOUS -> STOP",
    }


def analyze_posthoc_optimization(
    effects: pd.DataFrame,
    summaries: pd.DataFrame,
    traces: pd.DataFrame,
    *,
    permutation_resamples: int = 10_000,
    permutation_seed: int = 20260827,
) -> PosthocOptimizationResult:
    """Probe optimization-conditioned heterogeneity in the frozen D4-B 50-pair matrix."""
    if permutation_resamples <= 0:
        raise ValueError("permutation_resamples must be positive")
    target_effects = _validate_effects(effects)
    target_summaries = _validate_summaries(summaries)
    target_traces = _validate_traces(traces)
    pair = _build_pair_table(target_effects, target_summaries, target_traces)
    associations = _association_table(
        pair,
        permutation_resamples=permutation_resamples,
        permutation_seed=permutation_seed,
    )
    leave_one_out = _leave_one_out_table(pair, associations)
    screening = _screening_summary(associations, leave_one_out)
    return PosthocOptimizationResult(
        pair_table=pair,
        associations=associations,
        leave_one_out=leave_one_out,
        screening=screening,
    )


def _parse_cell_filename(path: Path) -> dict[str, object]:
    parts = path.stem.split("__")
    if len(parts) < 9 or parts[0] != "d4b" or parts[1] != "smooth":
        raise ValueError(f"Unexpected D4-B archived cell filename: {path.name}")
    try:
        cohort = int(parts[2].removeprefix("cohort"))
        subset = int(parts[3].removeprefix("subset"))
        model = int(parts[4].removeprefix("model"))
        n_train = int(parts[5].removeprefix("n"))
    except ValueError as error:
        raise ValueError(f"Invalid D4-B archived cell filename: {path.name}") from error
    variant = "__".join(parts[6:])
    return {
        "stage": "d4b",
        "world": "smooth",
        "cohort_seed": cohort,
        "subset_seed": subset,
        "model_seed": model,
        "n_train": n_train,
        "variant": variant,
    }


def load_d4b_archive(
    archive_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the immutable repository copy of D4-B effects, summaries, and traces."""
    root = Path(archive_root)
    effects_path = root / "analysis" / "phase06_d4b_effects.csv"
    summaries_dir = root / "stages" / "d4b" / "summaries"
    traces_dir = root / "stages" / "d4b" / "traces"
    if not effects_path.is_file() or not summaries_dir.is_dir() or not traces_dir.is_dir():
        raise ValueError("D4-B archive is missing effects, summaries, or traces")

    effects = pd.read_csv(effects_path)
    summary_rows: list[dict[str, object]] = []
    for path in sorted(summaries_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = _parse_cell_filename(path)
        summary_rows.append(
            {
                **identity,
                "selected_checkpoint_epoch": payload["selected_checkpoint_epoch"],
                "shadow_mae_checkpoint_epoch": payload["shadow_mae_checkpoint_epoch"],
                "stop_epoch": payload["stop_epoch"],
                "early_stop_reason": payload["early_stop_reason"],
            }
        )
    summaries = pd.DataFrame(summary_rows)

    trace_frames: list[pd.DataFrame] = []
    for path in sorted(traces_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        identity = _parse_cell_filename(path)
        for column, value in identity.items():
            frame[column] = value
        trace_frames.append(frame)
    if not trace_frames:
        raise ValueError("D4-B archive contains no trace files")
    traces = pd.concat(trace_frames, ignore_index=True)
    return effects, summaries, traces


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def run_posthoc_archive_analysis(
    archive_root: str | Path,
    output_dir: str | Path,
    *,
    permutation_resamples: int = 10_000,
    permutation_seed: int = 20260827,
) -> PosthocOptimizationResult:
    """Run the deterministic exploratory analysis without modifying the frozen archive."""
    archive = Path(archive_root).resolve()
    output = Path(output_dir).resolve()
    if _is_within(output, archive):
        raise ValueError("post-hoc outputs may not be written inside the frozen Phase 0.6 archive")
    effects, summaries, traces = load_d4b_archive(archive)
    result = analyze_posthoc_optimization(
        effects,
        summaries,
        traces,
        permutation_resamples=permutation_resamples,
        permutation_seed=permutation_seed,
    )
    output.mkdir(parents=True, exist_ok=True)
    result.pair_table.to_csv(output / "phase06_posthoc_pair_mechanisms.csv", index=False)
    result.associations.to_csv(output / "phase06_posthoc_associations.csv", index=False)
    result.leave_one_out.to_csv(output / "phase06_posthoc_leave_one_out.csv", index=False)
    (output / "phase06_posthoc_screening.json").write_text(
        json.dumps(result.screening, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "PosthocOptimizationResult",
    "analyze_posthoc_optimization",
    "load_d4b_archive",
    "run_posthoc_archive_analysis",
]
