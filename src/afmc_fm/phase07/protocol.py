from __future__ import annotations

import hashlib
import re
from pathlib import Path

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase07.config import Phase07Config

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_EXPECTED_CELL_COUNT = 200


def _sha256_bytes(path: Path, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_phase07_seed_triplet(
    cohort_seed: int,
    subset_seed: int,
    model_seed: int,
) -> None:
    for name, value in (
        ("cohort", cohort_seed),
        ("subset", subset_seed),
        ("model", model_seed),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} seed must be a non-negative integer")

    if cohort_seed in range(701, 711):
        raise ValueError("confirmatory cohort seed is forbidden in Phase 0.7")
    if subset_seed in range(801, 811):
        raise ValueError("confirmatory subset seed is forbidden in Phase 0.7")
    if model_seed in range(901, 911):
        raise ValueError("confirmatory model seed is forbidden in Phase 0.7")


def build_phase07_protocol_lock(
    config: Phase07Config,
    *,
    execution_commit: str,
    phase07_spec_path: str | Path,
) -> dict[str, object]:
    if not isinstance(config, Phase07Config):
        raise TypeError("config must be a Phase07Config")
    if not isinstance(execution_commit, str) or _COMMIT_RE.fullmatch(execution_commit) is None:
        raise ValueError("execution_commit must be a 40-character hexadecimal SHA")

    for cohort_seed, subset_seed in config.contexts:
        for model_seed in config.model_seeds:
            validate_phase07_seed_triplet(cohort_seed, subset_seed, model_seed)

    expected_cell_count = (
        len(config.contexts)
        * len(config.model_seeds)
        * len(config.flow_modes)
        * len(config.optimization_policies)
    )
    if expected_cell_count != _EXPECTED_CELL_COUNT:
        raise ValueError("Phase 0.7 matrix must contain exactly 200 cells")

    spec_path = Path(phase07_spec_path)
    return {
        "schema_version": 1,
        "phase07_spec_sha256": _sha256_bytes(spec_path, "Phase 0.7 design spec"),
        "phase07_config_sha256": canonical_config_hash(config),
        "execution_commit": execution_commit,
        "world": config.world,
        "n_train": config.n_train,
        "contexts": [list(context) for context in config.contexts],
        "model_seeds": list(config.model_seeds),
        "flow_modes": list(config.flow_modes),
        "optimization_policies": list(config.optimization_policies),
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "bootstrap_resamples": config.bootstrap_resamples,
        "bootstrap_seed": config.bootstrap_seed,
        "expected_cell_count": expected_cell_count,
        "forbidden_seed_sets": {
            "cohort": list(config.forbidden_cohort_seeds),
            "subset": list(config.forbidden_subset_seeds),
            "model": list(config.forbidden_model_seeds),
        },
    }


__all__ = ["build_phase07_protocol_lock", "validate_phase07_seed_triplet"]
