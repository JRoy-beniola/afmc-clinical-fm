from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import torch

from afmc_fm.experiments.runner import (
    ExperimentConfig,
    _run_configured_benchmarks_on_cohort,
)
from afmc_fm.simulator.cohort import simulate_cohort, simulate_world
from afmc_fm.simulator.config import SimulatorConfig


@dataclass(frozen=True, slots=True)
class ShardSpec:
    world: str
    cohort_seed: int
    subset_seed: int
    model_seed: int

    @property
    def shard_id(self) -> str:
        return (
            f"{self.world}__cohort{self.cohort_seed}"
            f"__subset{self.subset_seed}__model{self.model_seed}"
        )


@dataclass(frozen=True, slots=True)
class CellResult:
    shard: ShardSpec
    benchmark: str
    n_train: int
    model: str
    ablation: str
    metrics: pd.DataFrame

    @property
    def shard_id(self) -> str:
        return self.shard.shard_id


def plan_shards(
    experiment: ExperimentConfig,
    supplied_seed: int,
) -> tuple[ShardSpec, ...]:
    return tuple(
        ShardSpec(world, cohort_seed, subset_seed, model_seed)
        for world in experiment.worlds
        for cohort_seed, subset_seed, model_seed in experiment.seed_bundles(
            supplied_seed
        )
    )


def run_shard(
    spec: ShardSpec,
    sim_config: SimulatorConfig,
    experiment: ExperimentConfig,
    device: torch.device,
    cell_callback: Callable[[CellResult], None] | None = None,
) -> pd.DataFrame:
    cohort = (
        simulate_cohort(sim_config, spec.cohort_seed)
        if spec.world == "custom"
        else simulate_world(spec.world, sim_config, spec.cohort_seed)
    )

    def emit_cell(benchmark: str, metrics: pd.DataFrame) -> None:
        if cell_callback is None:
            return
        first = metrics.iloc[0]
        cell_callback(
            CellResult(
                shard=spec,
                benchmark=benchmark,
                n_train=int(first["n_train"]),
                model=str(first["model"]),
                ablation=str(first["ablation"]),
                metrics=metrics,
            )
        )

    return _run_configured_benchmarks_on_cohort(
        cohort,
        experiment,
        spec.subset_seed,
        spec.model_seed,
        device,
        cell_callback=emit_cell if cell_callback is not None else None,
    )
