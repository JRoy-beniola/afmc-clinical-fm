from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from afmc_fm.config import load_yaml

_D1_TRAIN_SIZES = (5, 10, 20, 40)
_D2_TRAIN_SIZES = (5, 40)
_FLOW_MODES = ("none", "time_scaled")
_FORBIDDEN_COHORT_SEEDS = tuple(range(701, 711))
_FORBIDDEN_SUBSET_SEEDS = tuple(range(801, 811))
_FORBIDDEN_MODEL_SEEDS = tuple(range(901, 911))


@dataclass(frozen=True, slots=True)
class Phase06Config:
    phase05_config: str = "configs/experiments/phase05.yaml"
    phase06_spec: str = (
        "docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md"
    )
    simulator_config: str = "configs/simulator/full.yaml"
    world: str = "smooth"
    d1_train_sizes: tuple[int, ...] = _D1_TRAIN_SIZES
    d2_train_sizes: tuple[int, ...] = _D2_TRAIN_SIZES
    d1_flow_modes: tuple[str, ...] = _FLOW_MODES
    d2_flow_modes: tuple[str, ...] = _FLOW_MODES
    jump_mode: str = "none"
    uncertainty_mode: str = "deterministic"
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20260826
    forbidden_cohort_seeds: tuple[int, ...] = _FORBIDDEN_COHORT_SEEDS
    forbidden_subset_seeds: tuple[int, ...] = _FORBIDDEN_SUBSET_SEEDS
    forbidden_model_seeds: tuple[int, ...] = _FORBIDDEN_MODEL_SEEDS

    def __post_init__(self) -> None:
        for name in ("phase05_config", "phase06_spec", "simulator_config"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be a non-empty string")

        expected = {
            "world": "smooth",
            "d1_train_sizes": _D1_TRAIN_SIZES,
            "d2_train_sizes": _D2_TRAIN_SIZES,
            "d1_flow_modes": _FLOW_MODES,
            "d2_flow_modes": _FLOW_MODES,
            "jump_mode": "none",
            "uncertainty_mode": "deterministic",
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 20260826,
            "forbidden_cohort_seeds": _FORBIDDEN_COHORT_SEEDS,
            "forbidden_subset_seeds": _FORBIDDEN_SUBSET_SEEDS,
            "forbidden_model_seeds": _FORBIDDEN_MODEL_SEEDS,
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise ValueError(f"{name} must remain locked to the Phase 0.6 protocol")


def _tuple_field(raw: Any, field_name: str) -> tuple[Any, ...]:
    if not isinstance(raw, list):
        raise TypeError(f"{field_name} must be a list")
    return tuple(raw)


def load_phase06_config(path: str | Path) -> Phase06Config:
    raw = load_yaml(path)
    normalized = dict(raw)
    for name in (
        "d1_train_sizes",
        "d2_train_sizes",
        "d1_flow_modes",
        "d2_flow_modes",
        "forbidden_cohort_seeds",
        "forbidden_subset_seeds",
        "forbidden_model_seeds",
    ):
        if name in normalized:
            normalized[name] = _tuple_field(normalized[name], name)
    return Phase06Config(**normalized)


__all__ = ["Phase06Config", "load_phase06_config"]
