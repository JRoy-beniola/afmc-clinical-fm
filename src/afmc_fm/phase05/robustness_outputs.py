from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from afmc_fm.phase05.execution import Phase05Job
from afmc_fm.phase05.robustness import (
    ROBUSTNESS_MODELS,
    evaluate_misspecification,
    evaluate_site_shift,
)

_ROBUSTNESS_WORLDS = ("site_shift", "misspecified")
_WORLD_RANK = {world: index for index, world in enumerate(_ROBUSTNESS_WORLDS)}
_MODEL_RANK = {model: index for index, model in enumerate(ROBUSTNESS_MODELS)}
_BUNDLE_COLUMNS = ("cohort_seed", "subset_seed", "model_seed")
_JOB_KEY_COLUMNS = (
    "world",
    *_BUNDLE_COLUMNS,
    "n_train",
    "model",
    "variant",
)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def _job_key(job: Phase05Job) -> tuple[object, ...]:
    bundle = job.shard.seed_bundle
    return (
        job.shard.world,
        bundle.cohort_seed,
        bundle.subset_seed,
        bundle.model_seed,
        job.n_train,
        job.model,
        job.variant,
    )


def _canonical_metrics(
    metrics: pd.DataFrame,
    *,
    jobs: Sequence[Phase05Job],
) -> pd.DataFrame:
    required = {
        *_JOB_KEY_COLUMNS,
        "metric",
        "value",
        "site_or_shift",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"missing robustness output columns: {sorted(missing)}")
    if metrics.empty:
        raise ValueError("robustness metrics must not be empty")

    values = metrics["value"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("robustness metric values must be finite")

    planned = tuple(jobs)
    expected_job_keys = {_job_key(job) for job in planned}
    observed_job_keys = set(
        metrics.loc[:, list(_JOB_KEY_COLUMNS)].itertuples(index=False, name=None)
    )
    unexpected = observed_job_keys - expected_job_keys
    if unexpected:
        raise ValueError("robustness metrics contain rows outside the planned cell set")

    mae = metrics.loc[metrics["metric"] == "mae"].copy()
    if "split" in mae.columns:
        mae = mae.loc[mae["split"] == "test"]
    observed_mae = set(
        mae.loc[:, [*_JOB_KEY_COLUMNS, "site_or_shift"]].itertuples(
            index=False, name=None
        )
    )
    expected_mae = {
        (*_job_key(job), site)
        for job in planned
        for site in (
            ("site_0", "site_1")
            if job.shard.world == "site_shift"
            else ("all",)
        )
    }
    if len(mae) != len(expected_mae) or observed_mae != expected_mae:
        raise ValueError("robustness MAE rows do not match the exact planned cell set")

    result = metrics.copy()
    result["_world_rank"] = result["world"].map(_WORLD_RANK)
    result["_model_rank"] = result["model"].map(_MODEL_RANK)
    if result[["_world_rank", "_model_rank"]].isna().any().any():
        raise ValueError("robustness metrics contain values outside the locked axes")
    extra_sort = [
        column
        for column in ("split", "site_or_shift", "metric")
        if column in result.columns
    ]
    return (
        result.sort_values(
            [
                "_world_rank",
                *_BUNDLE_COLUMNS,
                "n_train",
                "_model_rank",
                "variant",
                *extra_sort,
            ],
            kind="mergesort",
        )
        .drop(columns=["_world_rank", "_model_rank"])
        .reset_index(drop=True)
    )


def _sort_site_shift(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_model_rank"] = result["model"].map(_MODEL_RANK)
    return (
        result.sort_values(
            ["n_train", "_model_rank", *_BUNDLE_COLUMNS],
            kind="mergesort",
        )
        .drop(columns="_model_rank")
        .reset_index(drop=True)
    )


def _sort_site_gate(frame: pd.DataFrame) -> pd.DataFrame:
    comparator_rank = {
        "matched_gru": 0,
        "matched_representation_mlp": 1,
    }
    result = frame.copy()
    result["_comparator_rank"] = result["comparator"].map(comparator_rank)
    return (
        result.sort_values(["n_train", "_comparator_rank"], kind="mergesort")
        .drop(columns="_comparator_rank")
        .reset_index(drop=True)
    )


def persist_robustness_outputs(
    output: str | Path,
    metrics: pd.DataFrame,
    *,
    jobs: Sequence[Phase05Job],
) -> dict[str, Path]:
    planned = tuple(jobs)
    if len(planned) != 360:
        raise ValueError("robustness finalization requires exactly 360 planned cells")
    if {job.shard.world for job in planned} != set(_ROBUSTNESS_WORLDS):
        raise ValueError("robustness finalization requires the locked robustness worlds")
    if {job.model for job in planned} != set(ROBUSTNESS_MODELS):
        raise ValueError("robustness finalization requires the locked robustness models")

    canonical = _canonical_metrics(metrics, jobs=planned)
    site_shift = evaluate_site_shift(canonical)
    misspecified = evaluate_misspecification(canonical)

    site_metrics = site_shift["per_seed"]
    site_gate = site_shift["gate_summary"]
    misspecified_metrics = misspecified["per_seed"]
    misspecified_summary = pd.DataFrame([misspecified["summary"]])
    if not isinstance(site_metrics, pd.DataFrame) or not isinstance(site_gate, pd.DataFrame):
        raise TypeError("site-shift evaluation returned invalid tables")
    if not isinstance(misspecified_metrics, pd.DataFrame):
        raise TypeError("misspecification evaluation returned an invalid table")

    robustness = Path(output) / "robustness"
    return {
        "site_shift_metrics": _atomic_csv(
            robustness / "site_shift_metrics.csv",
            _sort_site_shift(site_metrics),
        ),
        "site_shift_gate_summary": _atomic_csv(
            robustness / "site_shift_gate_summary.csv",
            _sort_site_gate(site_gate),
        ),
        "misspecification_metrics": _atomic_csv(
            robustness / "misspecification_metrics.csv",
            misspecified_metrics.reset_index(drop=True),
        ),
        "misspecification_summary": _atomic_csv(
            robustness / "misspecification_summary.csv",
            misspecified_summary,
        ),
    }


__all__ = ["persist_robustness_outputs"]
