from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.confirmation import (
    CONFIRMATORY_MODELS,
    persist_confirmation_analysis,
)
from afmc_fm.phase05.execution import Phase05Job
from afmc_fm.phase05.protocol import FrozenCandidate

_TARGET_WORLD_ORDER = ("smooth", "jumps", "informative_observation")
_WORLD_RANK = {world: index for index, world in enumerate(_TARGET_WORLD_ORDER)}
_MODEL_RANK = {model: index for index, model in enumerate(CONFIRMATORY_MODELS)}
_BUNDLE_COLUMNS = ("cohort_seed", "subset_seed", "model_seed")
_MAE_KEY_COLUMNS = (
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


def _canonical_metrics(metrics: pd.DataFrame, config: Phase05Config) -> pd.DataFrame:
    required = {
        "world",
        *_BUNDLE_COLUMNS,
        "n_train",
        "model",
        "variant",
        "metric",
        "value",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"missing confirmation output columns: {sorted(missing)}")
    if metrics.empty:
        raise ValueError("confirmation metrics must not be empty")
    values = metrics["value"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("confirmation metric values must be finite")

    result = metrics.copy()
    result["_world_rank"] = result["world"].map(_WORLD_RANK)
    result["_model_rank"] = result["model"].map(_MODEL_RANK)
    train_rank = {size: index for index, size in enumerate(config.train_sizes)}
    result["_train_rank"] = result["n_train"].map(train_rank)
    if result[["_world_rank", "_model_rank", "_train_rank"]].isna().any().any():
        raise ValueError("confirmation metrics contain values outside the locked axes")

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
                "_train_rank",
                "_model_rank",
                "variant",
                *extra_sort,
            ],
            kind="mergesort",
        )
        .drop(columns=["_world_rank", "_model_rank", "_train_rank"])
        .reset_index(drop=True)
    )


def _locked_mae_rows(
    metrics: pd.DataFrame,
    *,
    jobs: Sequence[Phase05Job],
) -> pd.DataFrame:
    frame = metrics.loc[metrics["metric"] == "mae"].copy()
    if "split" in frame.columns:
        frame = frame.loc[frame["split"] == "test"]
    if "site_or_shift" in frame.columns:
        frame = frame.loc[frame["site_or_shift"] == "all"]

    if frame.duplicated(list(_MAE_KEY_COLUMNS)).any():
        raise ValueError("confirmation MAE rows contain duplicate planned cell keys")
    expected = {
        (
            job.shard.world,
            job.shard.seed_bundle.cohort_seed,
            job.shard.seed_bundle.subset_seed,
            job.shard.seed_bundle.model_seed,
            job.n_train,
            job.model,
            job.variant,
        )
        for job in jobs
    }
    observed = set(frame.loc[:, list(_MAE_KEY_COLUMNS)].itertuples(index=False, name=None))
    if observed != expected or len(frame) != len(expected):
        missing = expected - observed
        unexpected = observed - expected
        details = []
        if missing:
            details.append(f"missing={len(missing)}")
        if unexpected:
            details.append(f"unexpected={len(unexpected)}")
        raise ValueError(
            "confirmation MAE rows do not match the exact planned cell set"
            + (f" ({', '.join(details)})" if details else "")
        )
    return frame.reset_index(drop=True)


def _learning_curves(mae: pd.DataFrame) -> pd.DataFrame:
    curves = (
        mae.groupby(["world", "n_train", "model", "variant"], as_index=False)["value"]
        .agg(mean_mae="mean", sd_mae="std", n_bundles="count")
    )
    if not curves["n_bundles"].eq(10).all():
        raise ValueError("each confirmation learning-curve point requires ten bundles")
    curves["_world_rank"] = curves["world"].map(_WORLD_RANK)
    curves["_model_rank"] = curves["model"].map(_MODEL_RANK)
    return (
        curves.sort_values(
            ["_world_rank", "n_train", "_model_rank", "variant"],
            kind="mergesort",
        )
        .drop(columns=["_world_rank", "_model_rank"])
        .reset_index(drop=True)
    )


def _capacity_audit(frozen: FrozenCandidate) -> pd.DataFrame:
    rows = []
    for control, hidden_size, actual in (
        (
            "matched_gru",
            frozen.matched_gru_hidden_size,
            frozen.matched_gru_parameters,
        ),
        (
            "matched_representation_mlp",
            frozen.matched_mlp_hidden_size,
            frozen.matched_mlp_parameters,
        ),
    ):
        mismatch = abs(actual - frozen.trainable_parameters)
        rows.append(
            {
                "control": control,
                "target_parameters": frozen.trainable_parameters,
                "hidden_size": hidden_size,
                "actual_parameters": actual,
                "absolute_mismatch": mismatch,
                "relative_mismatch": mismatch / frozen.trainable_parameters,
            }
        )
    return pd.DataFrame(rows)


def persist_confirmation_outputs(
    output: str | Path,
    metrics: pd.DataFrame,
    *,
    config: Phase05Config,
    frozen: FrozenCandidate,
    jobs: Sequence[Phase05Job],
) -> dict[str, Path]:
    planned = tuple(jobs)
    if len(planned) != 1260:
        raise ValueError("confirmation finalization requires exactly 1260 planned cells")
    if tuple(config.target_worlds) != _TARGET_WORLD_ORDER:
        raise ValueError("confirmation finalization requires the locked target worlds")

    canonical = _canonical_metrics(metrics, config)
    mae = _locked_mae_rows(canonical, jobs=planned)
    confirmation = Path(output) / "confirmation"
    paths = {
        "metrics": _atomic_csv(confirmation / "metrics.csv", canonical),
        "learning_curves": _atomic_csv(
            confirmation / "learning_curves.csv",
            _learning_curves(mae),
        ),
        "capacity_audit": _atomic_csv(
            confirmation / "capacity_audit.csv",
            _capacity_audit(frozen),
        ),
    }
    paths.update(
        persist_confirmation_analysis(
            output,
            canonical,
            config=config,
        )
    )
    return paths


__all__ = ["persist_confirmation_outputs"]
