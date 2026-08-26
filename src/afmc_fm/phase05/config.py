from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from afmc_fm.config import load_yaml

_PHASE0_SEED_BUNDLES = frozenset(
    (100 + index, 200 + index, 300 + index) for index in range(1, 6)
)
_PRIMARY_TRAIN_SIZES = (5, 10, 20, 40)
_DEFAULT_DEVELOPMENT_BUNDLES = tuple(
    (400 + index, 500 + index, 600 + index) for index in range(1, 6)
)
_DEFAULT_CONFIRMATORY_BUNDLES = tuple(
    (700 + index, 800 + index, 900 + index) for index in range(1, 11)
)


@dataclass(frozen=True, slots=True)
class SeedBundle:
    cohort_seed: int
    subset_seed: int
    model_seed: int

    def __post_init__(self) -> None:
        values = (self.cohort_seed, self.subset_seed, self.model_seed)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("seed bundle values must be non-negative integers")

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.cohort_seed, self.subset_seed, self.model_seed)


_DEFAULT_DEVELOPMENT = tuple(SeedBundle(*values) for values in _DEFAULT_DEVELOPMENT_BUNDLES)
_DEFAULT_CONFIRMATORY = tuple(SeedBundle(*values) for values in _DEFAULT_CONFIRMATORY_BUNDLES)


@dataclass(frozen=True, slots=True)
class Phase05Config:
    train_sizes: tuple[int, ...] = (5, 10, 20, 40, 80, 100)
    primary_train_sizes: tuple[int, ...] = _PRIMARY_TRAIN_SIZES
    development_bundles: tuple[SeedBundle, ...] = _DEFAULT_DEVELOPMENT
    confirmatory_bundles: tuple[SeedBundle, ...] = _DEFAULT_CONFIRMATORY
    target_worlds: tuple[str, ...] = ("smooth", "jumps", "informative_observation")
    robustness_worlds: tuple[str, ...] = ("site_shift", "misspecified")
    max_epochs: int = 100
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_event: float = 1.0
    time_scale_days: float = 30.0
    provisional_relative_effect: float = 0.02
    development_win_requirement: int = 4
    confirmatory_win_requirement: int = 8
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20260824
    misspecification_tolerance: float = 0.05
    train_site: int = 0
    test_sites: tuple[int, ...] = (0, 1)

    def __post_init__(self) -> None:
        if self.primary_train_sizes != _PRIMARY_TRAIN_SIZES:
            raise ValueError(
                "primary_train_sizes must be exactly (5, 10, 20, 40)"
            )
        if not self.train_sizes or any(
            type(size) is not int or size <= 0 for size in self.train_sizes
        ):
            raise ValueError("train_sizes must contain positive integers")
        if len(self.train_sizes) != len(set(self.train_sizes)):
            raise ValueError("train_sizes must not contain duplicates")
        if not set(self.primary_train_sizes).issubset(self.train_sizes):
            raise ValueError("primary_train_sizes must be contained in train_sizes")

        development = set(self.development_bundles)
        confirmatory = set(self.confirmatory_bundles)
        phase0_overlap = {
            bundle.as_tuple()
            for bundle in (*self.development_bundles, *self.confirmatory_bundles)
            if bundle.as_tuple() in _PHASE0_SEED_BUNDLES
        }
        if phase0_overlap:
            raise ValueError("Phase-0.5 seed bundles must not overlap Phase-0 seed bundles")
        if development & confirmatory:
            raise ValueError(
                "development and confirmatory seed bundles must be disjoint"
            )
        if len(self.development_bundles) != 5:
            raise ValueError("development_bundles must contain exactly 5 bundles")
        if len(self.confirmatory_bundles) != 10:
            raise ValueError("confirmatory_bundles must contain exactly 10 bundles")
        if len(development) != len(self.development_bundles):
            raise ValueError("development seed bundles must be unique")
        if len(confirmatory) != len(self.confirmatory_bundles):
            raise ValueError("confirmatory seed bundles must be unique")

        if self.time_scale_days <= 0:
            raise ValueError("time_scale_days must be positive")
        for name, value in (
            ("provisional_relative_effect", self.provisional_relative_effect),
            ("misspecification_tolerance", self.misspecification_tolerance),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("max_epochs and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.lambda_event < 0:
            raise ValueError("lambda_event must be non-negative")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")
        if not 1 <= self.development_win_requirement <= len(self.development_bundles):
            raise ValueError("development_win_requirement is outside the bundle count")
        if not 1 <= self.confirmatory_win_requirement <= len(self.confirmatory_bundles):
            raise ValueError("confirmatory_win_requirement is outside the bundle count")
        if len(self.target_worlds) != len(set(self.target_worlds)) or not self.target_worlds:
            raise ValueError("target_worlds must be non-empty and unique")
        if len(self.robustness_worlds) != len(set(self.robustness_worlds)) or not self.robustness_worlds:
            raise ValueError("robustness_worlds must be non-empty and unique")
        if not self.test_sites:
            raise ValueError("test_sites must not be empty")


def _seed_bundles(raw: Any, field_name: str) -> tuple[SeedBundle, ...]:
    if not isinstance(raw, list):
        raise TypeError(f"{field_name} must be a list")
    bundles: list[SeedBundle] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError(f"{field_name} entries must contain exactly three seeds")
        bundles.append(SeedBundle(*(int(value) for value in item)))
    return tuple(bundles)


def load_phase05_config(path: str | Path) -> Phase05Config:
    raw = load_yaml(path)
    normalized = dict(raw)
    for name in (
        "train_sizes",
        "primary_train_sizes",
        "target_worlds",
        "robustness_worlds",
        "test_sites",
    ):
        if name in normalized:
            value = normalized[name]
            if not isinstance(value, list):
                raise TypeError(f"{name} must be a list")
            normalized[name] = tuple(value)
    for name in ("development_bundles", "confirmatory_bundles"):
        if name in normalized:
            normalized[name] = _seed_bundles(normalized[name], name)
    return Phase05Config(**normalized)


__all__ = ["Phase05Config", "SeedBundle", "load_phase05_config"]
