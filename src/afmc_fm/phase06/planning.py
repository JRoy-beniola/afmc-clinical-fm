from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase06.config import Phase06Config
from afmc_fm.phase06.protocol import validate_development_seed_triplet

_D1_TRAIN_SIZES = frozenset({5, 10, 20, 40})
_D2_TRAIN_SIZES = frozenset({5, 40})
_FLOW_MODES = frozenset({"none", "time_scaled"})
_EXPECTED_DEVELOPMENT_BUNDLES = tuple(
    (400 + index, 500 + index, 600 + index) for index in range(1, 6)
)
_D2_COHORTS = (401, 402, 403, 404, 405)
_D2_SUBSETS = (501, 502, 503, 504, 505)
_D2_MODELS = (601, 602, 603, 604, 605)


@dataclass(frozen=True, slots=True)
class Phase06CellSpec:
    stage: Literal["d1", "d2a", "d2b"]
    world: Literal["smooth"]
    cohort_seed: int
    subset_seed: int
    model_seed: int
    n_train: int
    flow_mode: Literal["none", "time_scaled"]
    jump_mode: Literal["none"] = "none"
    uncertainty_mode: Literal["deterministic"] = "deterministic"

    def __post_init__(self) -> None:
        if self.stage not in {"d1", "d2a", "d2b"}:
            raise ValueError("stage must be d1, d2a, or d2b")
        if self.world != "smooth":
            raise ValueError("world must be smooth")
        validate_development_seed_triplet(
            self.cohort_seed,
            self.subset_seed,
            self.model_seed,
        )
        if type(self.n_train) is not int or self.n_train <= 0:
            raise ValueError("n_train must be a positive integer")
        allowed_train_sizes = _D1_TRAIN_SIZES if self.stage == "d1" else _D2_TRAIN_SIZES
        if self.n_train not in allowed_train_sizes:
            raise ValueError(f"n_train is outside the locked {self.stage} scope")
        if self.flow_mode not in _FLOW_MODES:
            raise ValueError("flow_mode must be none or time_scaled")
        if self.jump_mode != "none":
            raise ValueError("jump_mode must be none")
        if self.uncertainty_mode != "deterministic":
            raise ValueError("uncertainty_mode must be deterministic")

    @property
    def cell_id(self) -> str:
        return (
            f"{self.stage}__smooth__cohort{self.cohort_seed}__"
            f"subset{self.subset_seed}__model{self.model_seed}__"
            f"n{self.n_train}__{self.flow_mode}__none__deterministic"
        )


def _require_unique_cells(
    cells: tuple[Phase06CellSpec, ...],
    expected_count: int,
    stage: str,
) -> tuple[Phase06CellSpec, ...]:
    if len(cells) != expected_count:
        raise RuntimeError(f"{stage} planner produced the wrong cell count")
    if len({cell.cell_id for cell in cells}) != expected_count:
        raise RuntimeError(f"{stage} planner produced duplicate cell IDs")
    return cells


def plan_d1_cells(
    config: Phase06Config,
    phase05_config: Phase05Config,
) -> tuple[Phase06CellSpec, ...]:
    development_bundles = tuple(
        bundle.as_tuple() for bundle in phase05_config.development_bundles
    )
    if development_bundles != _EXPECTED_DEVELOPMENT_BUNDLES:
        raise ValueError("Phase 0.5 development bundles do not match the locked D1 scope")

    cells = tuple(
        Phase06CellSpec(
            stage="d1",
            world=config.world,
            cohort_seed=bundle.cohort_seed,
            subset_seed=bundle.subset_seed,
            model_seed=bundle.model_seed,
            n_train=n_train,
            flow_mode=flow_mode,
            jump_mode=config.jump_mode,
            uncertainty_mode=config.uncertainty_mode,
        )
        for bundle in phase05_config.development_bundles
        for n_train in config.d1_train_sizes
        for flow_mode in config.d1_flow_modes
    )
    return _require_unique_cells(cells, 40, "D1")


def plan_d2a_cells(config: Phase06Config) -> tuple[Phase06CellSpec, ...]:
    cells = tuple(
        Phase06CellSpec(
            stage="d2a",
            world=config.world,
            cohort_seed=cohort_seed,
            subset_seed=subset_seed,
            model_seed=_D2_MODELS[(cohort_index + subset_index) % 5],
            n_train=n_train,
            flow_mode=flow_mode,
            jump_mode=config.jump_mode,
            uncertainty_mode=config.uncertainty_mode,
        )
        for cohort_index, cohort_seed in enumerate(_D2_COHORTS)
        for subset_index, subset_seed in enumerate(_D2_SUBSETS)
        for n_train in config.d2_train_sizes
        for flow_mode in config.d2_flow_modes
    )
    return _require_unique_cells(cells, 100, "D2-A")


def plan_d2b_cells(config: Phase06Config) -> tuple[Phase06CellSpec, ...]:
    cells = tuple(
        Phase06CellSpec(
            stage="d2b",
            world=config.world,
            cohort_seed=cohort_seed,
            subset_seed=subset_seed,
            model_seed=_D2_MODELS[(cohort_index + 2 * subset_index) % 5],
            n_train=n_train,
            flow_mode=flow_mode,
            jump_mode=config.jump_mode,
            uncertainty_mode=config.uncertainty_mode,
        )
        for cohort_index, cohort_seed in enumerate(_D2_COHORTS)
        for subset_index, subset_seed in enumerate(_D2_SUBSETS)
        for n_train in config.d2_train_sizes
        for flow_mode in config.d2_flow_modes
    )
    return _require_unique_cells(cells, 100, "D2-B")


__all__ = ["Phase06CellSpec", "plan_d1_cells", "plan_d2a_cells", "plan_d2b_cells"]
