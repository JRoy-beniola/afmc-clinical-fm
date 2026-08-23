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


def benchmark_cell_id(
    benchmark: str,
    shard: ShardSpec,
    n_train: int,
    model: str,
    ablation: str,
) -> str:
    return f"{benchmark}__{shard.shard_id}__n{n_train}__{model}__{ablation}"


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

    @property
    def cell_id(self) -> str:
        return benchmark_cell_id(
            self.benchmark,
            self.shard,
            self.n_train,
            self.model,
            self.ablation,
        )


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
    *,
    completed_cell_ids: frozenset[str] = frozenset(),
    on_cell_complete: Callable[[CellResult], None] | None = None,
) -> pd.DataFrame:
    if cell_callback is not None and on_cell_complete is not None:
        raise ValueError("provide only one cell completion callback")
    completion_callback = (
        on_cell_complete if on_cell_complete is not None else cell_callback
    )
    cohort = (
        simulate_cohort(sim_config, spec.cohort_seed)
        if spec.world == "custom"
        else simulate_world(spec.world, sim_config, spec.cohort_seed)
    )

    def cell_is_complete(
        benchmark: str, n_train: int, model: str, ablation: str
    ) -> bool:
        return (
            benchmark_cell_id(benchmark, spec, n_train, model, ablation)
            in completed_cell_ids
        )

    def emit_cell(benchmark: str, metrics: pd.DataFrame) -> None:
        if completion_callback is None:
            return
        first = metrics.iloc[0]
        completion_callback(
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
        cell_callback=emit_cell if completion_callback is not None else None,
        cell_is_complete=cell_is_complete,
    )
