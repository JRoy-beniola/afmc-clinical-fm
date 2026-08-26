from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from afmc_fm.phase05.execution import Phase05Job, aggregate_phase05_frames
from afmc_fm.phase05.store import Phase05Store

_MECHANISM_STAGES = ("flow", "jump", "uncertainty")


def _finalized_stage_metrics(
    store: Phase05Store,
    jobs: Sequence[Phase05Job],
) -> pd.DataFrame:
    planned = tuple(jobs)
    if not planned:
        raise ValueError("development output stage plan must not be empty")
    stages = {job.shard.stage for job in planned}
    if len(stages) != 1:
        raise ValueError("development output stage plan must contain one stage")
    stage = next(iter(stages))

    expected_cell_ids = frozenset(job.cell_id for job in planned)
    expected_seed_bundles = frozenset(
        job.shard.seed_bundle.as_tuple() for job in planned
    )
    observed = store.validate_resume(
        stage,
        expected_cell_ids=expected_cell_ids,
        expected_seed_bundles=expected_seed_bundles,
    )
    if observed != expected_cell_ids:
        raise RuntimeError(
            f"{stage} mechanism output requires the exact finalized cell set"
        )

    frames: list[pd.DataFrame] = []
    for job in planned:
        path = (
            store.output
            / "stages"
            / stage
            / "cells"
            / f"{job.cell_id}.json"
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"persisted development cell is unreadable: {job.cell_id}"
            ) from error
        rows = payload.get("metric_rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows or not all(
            isinstance(row, dict) for row in rows
        ):
            raise RuntimeError(
                f"persisted development cell has invalid metric rows: {job.cell_id}"
            )
        frames.append(pd.DataFrame(rows))
    return aggregate_phase05_frames(frames)


def persist_development_mechanism_metrics(
    store: Phase05Store,
    stage_jobs: Sequence[Sequence[Phase05Job]],
) -> Path:
    plans = tuple(tuple(jobs) for jobs in stage_jobs)
    if len(plans) != len(_MECHANISM_STAGES):
        raise ValueError("mechanism metrics require flow, jump, and uncertainty plans")
    stages = tuple(plan[0].shard.stage if plan else "" for plan in plans)
    if stages != _MECHANISM_STAGES:
        raise ValueError(
            "mechanism metrics require stages in flow, jump, uncertainty order"
        )

    metrics = aggregate_phase05_frames(
        [_finalized_stage_metrics(store, plan) for plan in plans]
    )
    if metrics.empty or set(metrics["stage"]) != set(_MECHANISM_STAGES):
        raise RuntimeError("mechanism metrics do not cover all required stages")

    store.replace_development_artifact(
        "mechanism_metrics.csv",
        metrics.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    return store.output / "development" / "mechanism_metrics.csv"


__all__ = ["persist_development_mechanism_metrics"]
