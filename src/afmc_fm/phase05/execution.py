import multiprocessing
import os
from dataclasses import dataclass

import torch

from afmc_fm.execution.device import resolve_device
from afmc_fm.phase05.config import SeedBundle

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class Phase05ShardSpec:
    stage: str
    world: str
    seed_bundle: SeedBundle

    @property
    def shard_id(self) -> str:
        return (
            f"{self.stage}__{self.world}__cohort{self.seed_bundle.cohort_seed}__"
            f"subset{self.seed_bundle.subset_seed}__model{self.seed_bundle.model_seed}"
        )


@dataclass(frozen=True, slots=True)
class Phase05ExecutionOptions:
    device: str = "auto"
    workers: int = 1
    resume: bool = False
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


def phase05_cell_id(
    shard: Phase05ShardSpec,
    *,
    n_train: int,
    model: str,
    variant: str,
) -> str:
    return f"{shard.shard_id}__n{n_train}__{model}__{variant}"


def spawn_context():
    return multiprocessing.get_context("spawn")


def limit_worker_threads() -> dict[str, str]:
    for variable in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    return {variable: os.environ[variable] for variable in THREAD_ENVIRONMENT_VARIABLES}


def resolve_phase05_device(requested: str) -> torch.device:
    return resolve_device(requested)


__all__ = [
    "THREAD_ENVIRONMENT_VARIABLES",
    "Phase05ExecutionOptions",
    "Phase05ShardSpec",
    "limit_worker_threads",
    "phase05_cell_id",
    "resolve_phase05_device",
    "spawn_context",
]
