from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from afmc_fm.phase07.config import Phase07Config
from afmc_fm.phase07.protocol import validate_phase07_seed_triplet

_CONTEXTS = (
    (406, 506),
    (407, 507),
    (408, 508),
    (409, 509),
    (410, 510),
)
_MODEL_SEEDS = tuple(range(1101, 1111))
_FLOW_MODES = ("none", "time_scaled")
_OPTIMIZATION_POLICIES = ("standard_early_stop", "forced_horizon")
_EXPECTED_CELL_COUNT = 200


@dataclass(frozen=True, slots=True)
class Phase07CellSpec:
    world: Literal["smooth"]
    cohort_seed: int
    subset_seed: int
    model_seed: int
    n_train: Literal[40]
    flow_mode: Literal["none", "time_scaled"]
    optimization_policy: Literal["standard_early_stop", "forced_horizon"]

    def __post_init__(self) -> None:
        if self.world != "smooth":
            raise ValueError("world must be smooth")
        validate_phase07_seed_triplet(
            self.cohort_seed,
            self.subset_seed,
            self.model_seed,
        )
        if (self.cohort_seed, self.subset_seed) not in _CONTEXTS:
            raise ValueError("context is outside the frozen Phase 0.7 scope")
        if self.model_seed not in _MODEL_SEEDS:
            raise ValueError("model_seed is outside the frozen Phase 0.7 scope")
        if self.n_train != 40:
            raise ValueError("n_train must remain 40")
        if self.flow_mode not in _FLOW_MODES:
            raise ValueError("flow_mode must be none or time_scaled")
        if self.optimization_policy not in _OPTIMIZATION_POLICIES:
            raise ValueError(
                "optimization_policy must be standard_early_stop or forced_horizon"
            )

    @property
    def cell_id(self) -> str:
        return (
            f"p07__smooth__cohort{self.cohort_seed}__subset{self.subset_seed}__"
            f"model{self.model_seed}__n40__{self.flow_mode}__{self.optimization_policy}"
        )


def _require_exact_plan(
    cells: Sequence[Phase07CellSpec],
) -> tuple[Phase07CellSpec, ...]:
    planned = tuple(cells)
    if len(planned) != _EXPECTED_CELL_COUNT:
        raise ValueError("Phase 0.7 plan must contain exactly 200 cells")
    if not all(isinstance(cell, Phase07CellSpec) for cell in planned):
        raise TypeError("Phase 0.7 plan must contain only Phase07CellSpec instances")
    if len({cell.cell_id for cell in planned}) != _EXPECTED_CELL_COUNT:
        raise ValueError("Phase 0.7 plan contains duplicate cell IDs")

    expected = {
        (cohort_seed, subset_seed, model_seed, flow_mode, optimization_policy)
        for cohort_seed, subset_seed in _CONTEXTS
        for model_seed in _MODEL_SEEDS
        for flow_mode in _FLOW_MODES
        for optimization_policy in _OPTIMIZATION_POLICIES
    }
    observed = {
        (
            cell.cohort_seed,
            cell.subset_seed,
            cell.model_seed,
            cell.flow_mode,
            cell.optimization_policy,
        )
        for cell in planned
    }
    if observed != expected:
        raise ValueError("Phase 0.7 plan does not match the frozen 200-cell matrix")
    return planned


def plan_phase07_cells(config: Phase07Config) -> tuple[Phase07CellSpec, ...]:
    if not isinstance(config, Phase07Config):
        raise TypeError("config must be a Phase07Config")

    cells = tuple(
        Phase07CellSpec(
            world=config.world,
            cohort_seed=cohort_seed,
            subset_seed=subset_seed,
            model_seed=model_seed,
            n_train=config.n_train,
            flow_mode=flow_mode,
            optimization_policy=optimization_policy,
        )
        for cohort_seed, subset_seed in config.contexts
        for model_seed in config.model_seeds
        for flow_mode in config.flow_modes
        for optimization_policy in config.optimization_policies
    )
    return _require_exact_plan(cells)


def phase07_plan_sha256(cells: Sequence[Phase07CellSpec]) -> str:
    planned = _require_exact_plan(cells)
    payload = [
        {
            "world": cell.world,
            "cohort_seed": cell.cohort_seed,
            "subset_seed": cell.subset_seed,
            "model_seed": cell.model_seed,
            "n_train": cell.n_train,
            "flow_mode": cell.flow_mode,
            "optimization_policy": cell.optimization_policy,
        }
        for cell in planned
    ]
    data = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


__all__ = ["Phase07CellSpec", "phase07_plan_sha256", "plan_phase07_cells"]
