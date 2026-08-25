import numpy as np
import pandas as pd

_PRIMARY_TRAIN_SIZES = (5, 10, 20, 40)
_BUNDLE_COLUMNS = ["cohort_seed", "subset_seed", "model_seed"]


def normalized_log_n_aulc(frame: pd.DataFrame) -> float:
    required = {"n_train", "value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing nAULC columns: {sorted(missing)}")

    primary = frame.loc[frame["n_train"].isin(_PRIMARY_TRAIN_SIZES), ["n_train", "value"]]
    counts = primary["n_train"].value_counts().to_dict()
    if any(counts.get(size, 0) != 1 for size in _PRIMARY_TRAIN_SIZES):
        raise ValueError("nAULC requires exactly one value at each primary train size")

    ordered = primary.set_index("n_train").loc[list(_PRIMARY_TRAIN_SIZES)]
    values = ordered["value"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nAULC primary values must be finite")

    x = np.log(np.asarray(_PRIMARY_TRAIN_SIZES, dtype=float))
    return float(np.trapezoid(values, x) / (x[-1] - x[0]))


def paired_development_gate(
    candidate_aulc,
    control_aulc,
    *,
    min_relative_effect: float,
) -> dict[str, object]:
    candidate = np.asarray(candidate_aulc, dtype=float)
    control = np.asarray(control_aulc, dtype=float)
    if candidate.ndim != 1 or control.ndim != 1 or candidate.shape != control.shape:
        raise ValueError("development gate requires matched one-dimensional paired arrays")
    if candidate.size != 5:
        raise ValueError("development gate requires exactly five paired development bundles")
    if not np.isfinite(candidate).all() or not np.isfinite(control).all():
        raise ValueError("development gate inputs must be finite")
    if np.any(control <= 0):
        raise ValueError("development gate control losses must be positive")
    if not np.isfinite(min_relative_effect) or min_relative_effect < 0:
        raise ValueError("min_relative_effect must be finite and non-negative")

    paired = control - candidate
    relative = paired / control
    mean_improvement = float(np.mean(paired))
    mean_relative_improvement = float(np.mean(relative))
    wins = int(np.sum(paired > 0))
    passed = bool(
        mean_improvement > 0
        and wins >= 4
        and mean_relative_improvement >= min_relative_effect
    )
    return {
        "passed": passed,
        "wins": wins,
        "win_fraction": wins / 5.0,
        "mean_improvement": mean_improvement,
        "median_improvement": float(np.median(paired)),
        "sd_improvement": float(np.std(paired, ddof=1)),
        "mean_relative_improvement": mean_relative_improvement,
        "candidate_mean_aulc": float(np.mean(candidate)),
        "control_mean_aulc": float(np.mean(control)),
        "paired_improvements": paired,
    }


def _confirmatory_model_aulcs(
    metrics: pd.DataFrame,
    *,
    model: str,
    world: str,
) -> pd.DataFrame:
    required = {
        "world",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "n_train",
        "model",
        "metric",
        "value",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"missing confirmatory metric columns: {sorted(missing)}")
    frame = metrics.loc[
        (metrics["world"] == world)
        & (metrics["model"] == model)
        & (metrics["metric"] == "mae")
    ].copy()
    if "split" in frame.columns:
        frame = frame.loc[frame["split"] == "test"]
    if "site_or_shift" in frame.columns:
        frame = frame.loc[frame["site_or_shift"] == "all"]
    rows: list[dict[str, object]] = []
    for bundle, group in frame.groupby(_BUNDLE_COLUMNS, sort=True):
        rows.append(
            {
                "cohort_seed": int(bundle[0]),
                "subset_seed": int(bundle[1]),
                "model_seed": int(bundle[2]),
                "naulc": normalized_log_n_aulc(group),
            }
        )
    return pd.DataFrame(rows)


def paired_confirmatory_naulc_effects(
    metrics: pd.DataFrame,
    candidate: str,
    comparator: str,
) -> pd.DataFrame:
    if not candidate or not comparator or candidate == comparator:
        raise ValueError("candidate and comparator must be distinct non-empty model names")
    worlds = tuple(sorted(pd.unique(metrics.get("world", pd.Series(dtype=str)))))
    if not worlds:
        raise ValueError("confirmatory metrics contain no worlds")

    rows: list[pd.DataFrame] = []
    for world in worlds:
        candidate_aulc = _confirmatory_model_aulcs(metrics, model=candidate, world=world)
        comparator_aulc = _confirmatory_model_aulcs(metrics, model=comparator, world=world)
        candidate_bundles = set(
            candidate_aulc[_BUNDLE_COLUMNS].itertuples(index=False, name=None)
        )
        comparator_bundles = set(
            comparator_aulc[_BUNDLE_COLUMNS].itertuples(index=False, name=None)
        )
        if (
            len(candidate_bundles) != 10
            or len(comparator_bundles) != 10
            or candidate_bundles != comparator_bundles
        ):
            raise ValueError("nAULC comparison requires exactly ten matched confirmatory bundles")
        merged = candidate_aulc.merge(
            comparator_aulc,
            on=_BUNDLE_COLUMNS,
            suffixes=("_candidate", "_comparator"),
            validate="one_to_one",
        )
        merged.insert(0, "world", world)
        merged["candidate"] = candidate
        merged["comparator"] = comparator
        merged["effect"] = merged["naulc_comparator"] - merged["naulc_candidate"]
        rows.append(merged)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["world", *_BUNDLE_COLUMNS]).reset_index(drop=True)


__all__ = [
    "normalized_log_n_aulc",
    "paired_confirmatory_naulc_effects",
    "paired_development_gate",
]
