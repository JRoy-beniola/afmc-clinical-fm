from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase06.config import Phase06Config

_PHASE05_EXECUTION_SHA = "50a94c06bc1c419ca55738f15f074cc06ccc3f36"
_PHASE05_STATUS = (
    "PHASE-0.5 TERMINATED AT STAGE I-A — FLOW MECHANISM GATE NOT ESTABLISHED"
)
_EXPECTED_PHASE05_PROTOCOL_SHA256 = (
    "33b25ebe46620626ae3d1bbbd6152fc3aacdb0e6200dd87e03c4fdec99e70db9"
)
_EXPECTED_PHASE05_CONFIG_SHA256 = (
    "befd7140cbf68cb981418cdc0c8880bb0652c2a6db614d05f61ad267297d519b"
)
_DEVELOPMENT_BUNDLES = tuple(
    (400 + index, 500 + index, 600 + index) for index in range(1, 6)
)
_FORBIDDEN_COHORT_SEEDS = tuple(range(701, 711))
_FORBIDDEN_SUBSET_SEEDS = tuple(range(801, 811))
_FORBIDDEN_MODEL_SEEDS = tuple(range(901, 911))
_COMMIT_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")


def _sha256_bytes(path: Path, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_development_seed_triplet(
    cohort_seed: int,
    subset_seed: int,
    model_seed: int,
) -> None:
    for value in (cohort_seed, subset_seed, model_seed):
        if type(value) is not int or value < 0:
            raise ValueError("seed values must be non-negative integers")
    if cohort_seed in _FORBIDDEN_COHORT_SEEDS:
        raise ValueError("confirmatory cohort seed is forbidden in Phase 0.6")
    if subset_seed in _FORBIDDEN_SUBSET_SEEDS:
        raise ValueError("confirmatory subset seed is forbidden in Phase 0.6")
    if model_seed in _FORBIDDEN_MODEL_SEEDS:
        raise ValueError("confirmatory model seed is forbidden in Phase 0.6")


def _validate_phase05_protocol(
    path: Path,
    phase05_config: Phase05Config,
) -> str:
    protocol_sha256 = _sha256_bytes(path, "Phase 0.5 protocol lock")
    if protocol_sha256 != _EXPECTED_PHASE05_PROTOCOL_SHA256:
        raise ValueError("Phase 0.5 protocol bytes do not match the official archive")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Phase 0.5 protocol lock is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("Phase 0.5 protocol lock must be a JSON object")

    phase05_config_sha256 = canonical_config_hash(phase05_config)
    if phase05_config_sha256 != _EXPECTED_PHASE05_CONFIG_SHA256:
        raise ValueError("Phase 0.5 config does not match the official protocol identity")
    if payload.get("phase05_config_sha256") != phase05_config_sha256:
        raise ValueError("Phase 0.5 protocol config hash does not match loaded config")
    if payload.get("development_bundles") != [list(bundle) for bundle in _DEVELOPMENT_BUNDLES]:
        raise ValueError("Phase 0.5 protocol development bundles do not match archive")
    expected_confirmatory = [
        [700 + index, 800 + index, 900 + index] for index in range(1, 11)
    ]
    if payload.get("confirmatory_bundles") != expected_confirmatory:
        raise ValueError("Phase 0.5 protocol confirmatory bundles do not match archive")
    return protocol_sha256


def build_phase06_protocol_lock(
    config: Phase06Config,
    phase05_config: Phase05Config,
    *,
    execution_commit: str,
    phase06_spec_path: str | Path,
    phase05_protocol_path: str | Path,
) -> dict[str, object]:
    if not isinstance(execution_commit, str) or _COMMIT_RE.fullmatch(execution_commit) is None:
        raise ValueError("execution_commit must be a 40- or 64-character hexadecimal SHA")

    for bundle in phase05_config.development_bundles:
        validate_development_seed_triplet(*bundle.as_tuple())
    development_bundles = tuple(
        bundle.as_tuple() for bundle in phase05_config.development_bundles
    )
    if development_bundles != _DEVELOPMENT_BUNDLES:
        raise ValueError("Phase 0.5 development bundles do not match Phase 0.6 lock")

    phase06_spec_path = Path(phase06_spec_path)
    phase05_protocol_path = Path(phase05_protocol_path)
    phase06_spec_sha256 = _sha256_bytes(phase06_spec_path, "Phase 0.6 design spec")
    phase05_protocol_sha256 = _validate_phase05_protocol(
        phase05_protocol_path, phase05_config
    )

    return {
        "schema_version": 1,
        "phase06_spec_sha256": phase06_spec_sha256,
        "phase06_config_sha256": canonical_config_hash(config),
        "phase05_config_sha256": canonical_config_hash(phase05_config),
        "phase05_protocol_sha256": phase05_protocol_sha256,
        "phase05_execution_sha": _PHASE05_EXECUTION_SHA,
        "phase05_official_status": _PHASE05_STATUS,
        "execution_commit": execution_commit,
        "forbidden_seed_sets": {
            "cohort": list(config.forbidden_cohort_seeds),
            "subset": list(config.forbidden_subset_seeds),
            "model": list(config.forbidden_model_seeds),
        },
        "d1_development_bundles": [list(bundle) for bundle in development_bundles],
        "d2_mapping": "model_index=(cohort_index+subset_index)%5",
        "bootstrap_resamples": config.bootstrap_resamples,
        "bootstrap_seed": config.bootstrap_seed,
    }


__all__ = [
    "build_phase06_protocol_lock",
    "validate_development_seed_triplet",
]
