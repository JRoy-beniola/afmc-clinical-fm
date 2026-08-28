from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from afmc_fm.config import load_yaml

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
_FORBIDDEN_COHORT_SEEDS = tuple(range(701, 711))
_FORBIDDEN_SUBSET_SEEDS = tuple(range(801, 811))
_FORBIDDEN_MODEL_SEEDS = tuple(range(901, 911))


@dataclass(frozen=True, slots=True)
class Phase07Config:
    phase05_config: str = "configs/experiments/phase05.yaml"
    phase07_spec: str = (
        "docs/superpowers/specs/2026-08-28-phase0-7-optimization-horizon-intervention-design.md"
    )
    simulator_config: str = "configs/simulator/full.yaml"
    world: str = "smooth"
    n_train: int = 40
    contexts: tuple[tuple[int, int], ...] = _CONTEXTS
    model_seeds: tuple[int, ...] = _MODEL_SEEDS
    flow_modes: tuple[str, ...] = _FLOW_MODES
    optimization_policies: tuple[str, ...] = _OPTIMIZATION_POLICIES
    max_epochs: int = 100
    patience: int = 12
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20260827
    forbidden_cohort_seeds: tuple[int, ...] = _FORBIDDEN_COHORT_SEEDS
    forbidden_subset_seeds: tuple[int, ...] = _FORBIDDEN_SUBSET_SEEDS
    forbidden_model_seeds: tuple[int, ...] = _FORBIDDEN_MODEL_SEEDS

    def __post_init__(self) -> None:
        for name in ("phase05_config", "phase07_spec", "simulator_config"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be a non-empty string")

        expected = {
            "world": "smooth",
            "n_train": 40,
            "contexts": _CONTEXTS,
            "model_seeds": _MODEL_SEEDS,
            "flow_modes": _FLOW_MODES,
            "optimization_policies": _OPTIMIZATION_POLICIES,
            "max_epochs": 100,
            "patience": 12,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 20260827,
            "forbidden_cohort_seeds": _FORBIDDEN_COHORT_SEEDS,
            "forbidden_subset_seeds": _FORBIDDEN_SUBSET_SEEDS,
            "forbidden_model_seeds": _FORBIDDEN_MODEL_SEEDS,
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise ValueError(f"{name} must remain locked to the Phase 0.7 protocol")


def _tuple_field(raw: Any, field_name: str) -> tuple[Any, ...]:
    if not isinstance(raw, list):
        raise TypeError(f"{field_name} must be a list")
    return tuple(raw)


def _contexts_field(raw: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw, list):
        raise TypeError("contexts must be a list")
    contexts: list[tuple[int, int]] = []
    for value in raw:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("contexts entries must contain exactly two seeds")
        cohort_seed, subset_seed = value
        if type(cohort_seed) is not int or type(subset_seed) is not int:
            raise TypeError("contexts seeds must be integers")
        contexts.append((cohort_seed, subset_seed))
    return tuple(contexts)


def load_phase07_config(path: str | Path) -> Phase07Config:
    raw = load_yaml(path)
    normalized = dict(raw)
    if "contexts" in normalized:
        normalized["contexts"] = _contexts_field(normalized["contexts"])
    for name in (
        "model_seeds",
        "flow_modes",
        "optimization_policies",
        "forbidden_cohort_seeds",
        "forbidden_subset_seeds",
        "forbidden_model_seeds",
    ):
        if name in normalized:
            normalized[name] = _tuple_field(normalized[name], name)
    return Phase07Config(**normalized)


__all__ = ["Phase07Config", "load_phase07_config"]
