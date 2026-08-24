import numpy as np
import pandas as pd

_PRIMARY_TRAIN_SIZES = (5, 10, 20, 40)


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


__all__ = ["normalized_log_n_aulc", "paired_development_gate"]
