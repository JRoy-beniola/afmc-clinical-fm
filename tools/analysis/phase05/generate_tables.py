from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from afmc_fm.phase05.analysis import (
    normalized_log_n_aulc,
    paired_development_gate,
)

BUNDLE_COLUMNS = ["cohort_seed", "subset_seed", "model_seed"]
PRIMARY_TRAIN_SIZES = (5, 10, 20, 40)


def _load_metric_rows(input_dir: Path) -> pd.DataFrame:
    cell_dir = input_dir / "stages" / "flow" / "cells"
    files = sorted(cell_dir.glob("*.json"))

    if not files:
        raise ValueError(f"no Phase-0.5 flow cells found under {cell_dir}")

    rows: list[dict[str, object]] = []

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))

        metric_rows = payload.get("metric_rows")
        if not isinstance(metric_rows, list):
            raise TypeError(f"{path} does not contain metric_rows")

        rows.extend(metric_rows)

    return pd.DataFrame(rows)


def _bundle_aulcs(
    metrics: pd.DataFrame,
    *,
    variant: str,
) -> pd.DataFrame:
    frame = metrics.loc[
        (metrics["stage"] == "flow")
        & (metrics["world"] == "smooth")
        & (metrics["split"] == "test")
        & (metrics["site_or_shift"] == "all")
        & (metrics["metric"] == "mae")
        & (metrics["variant"] == variant)
        & (metrics["n_train"].isin(PRIMARY_TRAIN_SIZES))
    ].copy()

    rows: list[dict[str, object]] = []

    for bundle, group in frame.groupby(BUNDLE_COLUMNS, sort=True):
        rows.append(
            {
                "cohort_seed": int(bundle[0]),
                "subset_seed": int(bundle[1]),
                "model_seed": int(bundle[2]),
                "aulc": normalized_log_n_aulc(group[["n_train", "value"]]),
            }
        )

    result = pd.DataFrame(rows)

    if len(result) != 5:
        raise ValueError(
            f"expected five matched development bundles for {variant}, "
            f"found {len(result)}"
        )

    return result.sort_values(BUNDLE_COLUMNS).reset_index(drop=True)


def _recompute_candidate_gate(
    metrics: pd.DataFrame,
    *,
    candidate: str,
    locked_min_relative_effect: float,
) -> dict[str, object]:
    candidate_variant = f"{candidate}__none__deterministic"
    control_variant = "none__none__deterministic"

    candidate_aulc = _bundle_aulcs(
        metrics,
        variant=candidate_variant,
    )

    control_aulc = _bundle_aulcs(
        metrics,
        variant=control_variant,
    )

    paired = candidate_aulc.merge(
        control_aulc,
        on=BUNDLE_COLUMNS,
        suffixes=("_candidate", "_control"),
        validate="one_to_one",
    )

    gate = paired_development_gate(
        paired["aulc_candidate"].to_numpy(dtype=float),
        paired["aulc_control"].to_numpy(dtype=float),
        min_relative_effect=locked_min_relative_effect,
    )

    return {
        key: value
        for key, value in gate.items()
        if key != "paired_improvements"
    }


def _verify_flow_gate(input_dir: Path) -> pd.DataFrame:
    metrics = _load_metric_rows(input_dir)

    protocol_lock = json.loads(
        (input_dir / "protocol_lock.json").read_text(encoding="utf-8")
    )
    locked_min_relative_effect = float(
        protocol_lock["locked_min_relative_effect"]
    )

    official = pd.read_csv(
        input_dir / "development" / "flow_gate.csv",
        float_precision="round_trip",
    )

    rows: list[dict[str, object]] = []

    for candidate in ("gated", "time_scaled"):
        official_rows = official.loc[official["candidate"] == candidate]

        if len(official_rows) != 1:
            raise ValueError(
                f"official flow gate must contain exactly one row for {candidate}"
            )

        official_row = official_rows.iloc[0]
        recomputed = _recompute_candidate_gate(
            metrics,
            candidate=candidate,
            locked_min_relative_effect=locked_min_relative_effect,
        )

        checks = [
            bool(official_row["passed"]) == bool(recomputed["passed"]),
            int(official_row["wins"]) == int(recomputed["wins"]),
            np.isclose(
                float(official_row["win_fraction"]),
                float(recomputed["win_fraction"]),
                rtol=0.0,
                atol=1e-12,
            ),
            np.isclose(
                float(official_row["mean_improvement"]),
                float(recomputed["mean_improvement"]),
                rtol=0.0,
                atol=1e-12,
            ),
            np.isclose(
                float(official_row["median_improvement"]),
                float(recomputed["median_improvement"]),
                rtol=0.0,
                atol=1e-12,
            ),
            np.isclose(
                float(official_row["sd_improvement"]),
                float(recomputed["sd_improvement"]),
                rtol=0.0,
                atol=1e-12,
            ),
            np.isclose(
                float(official_row["mean_relative_improvement"]),
                float(recomputed["mean_relative_improvement"]),
                rtol=0.0,
                atol=1e-12,
            ),
            np.isclose(
                float(official_row["candidate_mean_aulc"]),
                float(recomputed["candidate_mean_aulc"]),
                rtol=0.0,
                atol=1e-12,
            ),
            np.isclose(
                float(official_row["control_mean_aulc"]),
                float(recomputed["control_mean_aulc"]),
                rtol=0.0,
                atol=1e-12,
            ),
        ]

        rows.append(
            {
                "candidate": candidate,
                "official_passed": bool(official_row["passed"]),
                "recomputed_passed": bool(recomputed["passed"]),
                "official_wins": int(official_row["wins"]),
                "recomputed_wins": int(recomputed["wins"]),
                "official_mean_improvement": float(
                    official_row["mean_improvement"]
                ),
                "recomputed_mean_improvement": float(
                    recomputed["mean_improvement"]
                ),
                "official_mean_relative_improvement": float(
                    official_row["mean_relative_improvement"]
                ),
                "recomputed_mean_relative_improvement": float(
                    recomputed["mean_relative_improvement"]
                ),
                "official_candidate_mean_aulc": float(
                    official_row["candidate_mean_aulc"]
                ),
                "recomputed_candidate_mean_aulc": float(
                    recomputed["candidate_mean_aulc"]
                ),
                "official_control_mean_aulc": float(
                    official_row["control_mean_aulc"]
                ),
                "recomputed_control_mean_aulc": float(
                    recomputed["control_mean_aulc"]
                ),
                "verification_passed": bool(all(checks)),
            }
        )

    result = pd.DataFrame(rows)

    if not result["verification_passed"].all():
        raise RuntimeError(
            "recomputed Phase-0.5 flow gate does not reproduce "
            "the archived official gate"
        )

    return result



LOWER_IS_BETTER = {
    "mae",
    "rmse",
    "event_brier",
    "event_log_loss",
}

HIGHER_IS_BETTER = {
    "event_roc_auc",
    "latent_aligned_r2",
}


def _flow_bundle_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    control_variant = "none__none__deterministic"
    rows: list[dict[str, object]] = []

    control = _bundle_aulcs(metrics, variant=control_variant)

    for candidate in ("gated", "time_scaled"):
        candidate_variant = f"{candidate}__none__deterministic"
        candidate_aulc = _bundle_aulcs(
            metrics,
            variant=candidate_variant,
        )

        paired = candidate_aulc.merge(
            control,
            on=BUNDLE_COLUMNS,
            suffixes=("_candidate", "_control"),
            validate="one_to_one",
        )

        for row in paired.itertuples(index=False):
            effect = float(row.aulc_control - row.aulc_candidate)

            rows.append(
                {
                    "candidate": candidate,
                    "control": "none",
                    "cohort_seed": int(row.cohort_seed),
                    "subset_seed": int(row.subset_seed),
                    "model_seed": int(row.model_seed),
                    "candidate_aulc": float(row.aulc_candidate),
                    "control_aulc": float(row.aulc_control),
                    "effect": effect,
                    "relative_improvement": (
                        effect / float(row.aulc_control)
                    ),
                }
            )

    result = pd.DataFrame(rows)

    if len(result) != 10:
        raise RuntimeError(
            f"expected 10 flow bundle effects, found {len(result)}"
        )

    return result.sort_values(
        ["candidate", *BUNDLE_COLUMNS]
    ).reset_index(drop=True)


def _flow_n_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    control_variant = "none__none__deterministic"
    rows: list[dict[str, object]] = []

    supported_metrics = LOWER_IS_BETTER | HIGHER_IS_BETTER

    base_columns = [
        *BUNDLE_COLUMNS,
        "n_train",
        "metric",
        "value",
    ]

    control = metrics.loc[
        metrics["variant"].eq(control_variant),
        base_columns,
    ].copy()

    for candidate in ("gated", "time_scaled"):
        candidate_variant = f"{candidate}__none__deterministic"

        candidate_frame = metrics.loc[
            metrics["variant"].eq(candidate_variant),
            base_columns,
        ].copy()

        paired = candidate_frame.merge(
            control,
            on=[
                *BUNDLE_COLUMNS,
                "n_train",
                "metric",
            ],
            suffixes=("_candidate", "_control"),
            validate="one_to_one",
        )

        for row in paired.itertuples(index=False):
            metric = str(row.metric)

            if metric not in supported_metrics:
                raise ValueError(
                    f"unsupported Phase-0.5 metric direction: {metric}"
                )

            candidate_value = float(row.value_candidate)
            control_value = float(row.value_control)

            if metric in LOWER_IS_BETTER:
                effect = control_value - candidate_value
            else:
                effect = candidate_value - control_value

            if not np.isfinite(
                [candidate_value, control_value, effect]
            ).all():
                raise ValueError(
                    "Phase-0.5 paired metric effects must be finite"
                )

            rows.append(
                {
                    "candidate": candidate,
                    "control": "none",
                    "cohort_seed": int(row.cohort_seed),
                    "subset_seed": int(row.subset_seed),
                    "model_seed": int(row.model_seed),
                    "n_train": int(row.n_train),
                    "metric": metric,
                    "candidate_value": candidate_value,
                    "control_value": control_value,
                    "effect": effect,
                    "candidate_better": bool(effect > 0.0),
                }
            )

    result = pd.DataFrame(rows)

    if len(result) != 240:
        raise RuntimeError(
            f"expected 240 paired flow metric effects, found {len(result)}"
        )

    return result.sort_values(
        [
            "candidate",
            "cohort_seed",
            "subset_seed",
            "model_seed",
            "n_train",
            "metric",
        ]
    ).reset_index(drop=True)


def _flow_metric_summary(
    effects: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (candidate, n_train, metric), group in effects.groupby(
        ["candidate", "n_train", "metric"],
        sort=True,
    ):
        if len(group) != 5:
            raise RuntimeError(
                "each Phase-0.5 metric summary requires "
                "five matched development bundles"
            )

        values = group["effect"].to_numpy(dtype=float)
        wins = int(np.sum(values > 0.0))

        rows.append(
            {
                "candidate": str(candidate),
                "control": "none",
                "n_train": int(n_train),
                "metric": str(metric),
                "candidate_mean": float(
                    group["candidate_value"].mean()
                ),
                "control_mean": float(
                    group["control_value"].mean()
                ),
                "mean_effect": float(np.mean(values)),
                "median_effect": float(np.median(values)),
                "sd_effect": float(np.std(values, ddof=1)),
                "wins": wins,
                "win_fraction": wins / 5.0,
            }
        )

    result = pd.DataFrame(rows)

    if len(result) != 48:
        raise RuntimeError(
            f"expected 48 flow metric summaries, found {len(result)}"
        )

    return result.sort_values(
        ["candidate", "n_train", "metric"]
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Phase-0.5 archival result tables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Archived official Phase-0.5 output root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination directory for derived tables.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    metrics = _load_metric_rows(args.input)
    metrics.to_csv(
        args.output / "flow_cell_metrics.csv",
        index=False,
    )

    verified = _verify_flow_gate(args.input)
    verified.to_csv(
        args.output / "flow_gate_verified.csv",
        index=False,
    )

    bundle_effects = _flow_bundle_effects(metrics)
    bundle_effects.to_csv(
        args.output / "flow_bundle_effects.csv",
        index=False,
    )

    n_effects = _flow_n_effects(metrics)
    n_effects.to_csv(
        args.output / "flow_n_effects.csv",
        index=False,
    )

    metric_summary = _flow_metric_summary(n_effects)
    metric_summary.to_csv(
        args.output / "flow_metric_summary.csv",
        index=False,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
