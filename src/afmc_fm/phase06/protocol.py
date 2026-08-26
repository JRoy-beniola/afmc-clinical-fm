from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
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
_PARENT_PHASE06_EXECUTION_SHA = "1718402df1d6ef344168677e6d26ea664708e1bc"
_PARENT_PHASE06_PROTOCOL_SHA256 = (
    "c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51"
)
_PARENT_PHASE06_D3_SHA256 = (
    "6b9fffed7503fae6beeac2314238ae3d10ffdebfd27ea71ad952ab1f87916460"
)
_D2B_PHASE06_EXECUTION_SHA = "516c9e3c0e965582fa5cce976e9d8ebf32ea8404"
_D2B_PROTOCOL_CANONICAL_SHA256 = (
    "33f7cb1f6e71560f547a746cb7f5eb41130f794ab1e3a6fb1928c40e05b29f12"
)
_D2B_ADJUDICATION_CANONICAL_SHA256 = (
    "ea28fd4d5f6490a10fad20d5d3f3e76a1de08bf6b1be3b9805c9cf6c519e845f"
)
_D4B_CONTEXTS = (
    (401, 501),
    (402, 502),
    (403, 503),
    (404, 504),
    (405, 505),
)
_D4B_MODEL_SEEDS = tuple(range(1001, 1011))
_D4B_BOOTSTRAP_RESAMPLES = 10_000
_D4B_BOOTSTRAP_SEED = 20260827
_DEVELOPMENT_BUNDLES = tuple(
    (400 + index, 500 + index, 600 + index) for index in range(1, 6)
)
_FORBIDDEN_COHORT_SEEDS = tuple(range(701, 711))
_FORBIDDEN_SUBSET_SEEDS = tuple(range(801, 811))
_FORBIDDEN_MODEL_SEEDS = tuple(range(901, 911))
_COMMIT_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


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
        raise TypeError("Phase 0.5 protocol lock must be a JSON object")

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


def build_phase06_d2b_protocol_lock(
    config: Phase06Config,
    phase05_config: Phase05Config,
    *,
    execution_commit: str,
    phase06_spec_path: str | Path,
    d2b_addendum_path: str | Path,
    phase05_protocol_path: str | Path,
    parent_evidence: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(parent_evidence, Mapping):
        raise TypeError("parent_evidence must be a mapping")
    if execution_commit == _PARENT_PHASE06_EXECUTION_SHA:
        raise ValueError("D2-B child execution commit must differ from parent execution")

    phase06_spec_path = Path(phase06_spec_path)
    expected_config_hash = canonical_config_hash(config)
    expected_spec_hash = _sha256_bytes(phase06_spec_path, "Phase 0.6 design spec")
    expected_forbidden = {
        "cohort": list(config.forbidden_cohort_seeds),
        "subset": list(config.forbidden_subset_seeds),
        "model": list(config.forbidden_model_seeds),
    }

    if parent_evidence.get("parent_execution_sha") != _PARENT_PHASE06_EXECUTION_SHA:
        raise ValueError("parent Phase 0.6 execution identity drift")
    if (
        parent_evidence.get("parent_protocol_lock_sha256")
        != _PARENT_PHASE06_PROTOCOL_SHA256
    ):
        raise ValueError("parent Phase 0.6 protocol identity drift")
    parent_d3_hash = parent_evidence.get("parent_d3_sha256")
    if not isinstance(parent_d3_hash, str) or _SHA256_RE.fullmatch(parent_d3_hash) is None:
        raise ValueError("parent D3 SHA-256 is invalid")
    if parent_d3_hash != _PARENT_PHASE06_D3_SHA256:
        raise ValueError("parent D3 SHA-256 does not match frozen parent")
    if parent_evidence.get("parent_d3_next_required_stage") != "D2B":
        raise ValueError("parent D3 next_required_stage must be D2B")
    if parent_evidence.get("parent_phase06_config_sha256") != expected_config_hash:
        raise ValueError("parent Phase 0.6 config identity drift")
    if parent_evidence.get("parent_phase06_spec_sha256") != expected_spec_hash:
        raise ValueError("parent Phase 0.6 spec identity drift")
    if parent_evidence.get("parent_forbidden_seed_sets") != expected_forbidden:
        raise ValueError("parent forbidden seed sets do not match D2-B child protocol")

    lock = build_phase06_protocol_lock(
        config,
        phase05_config,
        execution_commit=execution_commit,
        phase06_spec_path=phase06_spec_path,
        phase05_protocol_path=phase05_protocol_path,
    )
    lock.update(
        {
            "schema_version": 2,
            "d2b_addendum_sha256": _sha256_bytes(
                Path(d2b_addendum_path), "Phase 0.6 D2-B execution addendum"
            ),
            "parent_execution_sha": _PARENT_PHASE06_EXECUTION_SHA,
            "parent_protocol_lock_sha256": _PARENT_PHASE06_PROTOCOL_SHA256,
            "parent_d3_sha256": parent_d3_hash,
            "parent_d3_next_required_stage": "D2B",
            "d2b_mapping": "model_index=(cohort_index+2*subset_index)%5",
        }
    )
    return lock


def build_phase06_d4b_protocol_lock(
    config: Phase06Config,
    phase05_config: Phase05Config,
    *,
    execution_commit: str,
    phase06_spec_path: str | Path,
    d4b_addendum_path: str | Path,
    phase05_protocol_path: str | Path,
    parent_evidence: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(parent_evidence, Mapping):
        raise TypeError("parent_evidence must be a mapping")
    if execution_commit == _D2B_PHASE06_EXECUTION_SHA:
        raise ValueError("D4-B child execution commit must differ from D2-B parent")

    phase06_spec_path = Path(phase06_spec_path)
    expected_config_hash = canonical_config_hash(config)
    expected_spec_hash = _sha256_bytes(phase06_spec_path, "Phase 0.6 design spec")
    expected_forbidden = {
        "cohort": list(config.forbidden_cohort_seeds),
        "subset": list(config.forbidden_subset_seeds),
        "model": list(config.forbidden_model_seeds),
    }

    frozen = {
        "core_parent_execution_sha": _PARENT_PHASE06_EXECUTION_SHA,
        "core_parent_protocol_lock_sha256": _PARENT_PHASE06_PROTOCOL_SHA256,
        "core_parent_d3_sha256": _PARENT_PHASE06_D3_SHA256,
        "d2b_parent_execution_sha": _D2B_PHASE06_EXECUTION_SHA,
        "d2b_parent_protocol_canonical_sha256": _D2B_PROTOCOL_CANONICAL_SHA256,
        "d2b_parent_adjudication_canonical_sha256": _D2B_ADJUDICATION_CANONICAL_SHA256,
        "d2b_next_required_stage": "D4_OPTIMIZATION",
        "phase06_config_sha256": expected_config_hash,
        "phase06_spec_sha256": expected_spec_hash,
        "forbidden_seed_sets": expected_forbidden,
    }
    for key, expected in frozen.items():
        observed = parent_evidence.get(key)
        if observed != expected:
            if key == "d2b_parent_adjudication_canonical_sha256":
                raise ValueError("D2-B adjudication identity drift")
            if key == "d2b_next_required_stage":
                raise ValueError("D2-B next_required_stage must be D4_OPTIMIZATION")
            raise ValueError(f"D4-B parent evidence drift: {key}")

    lock = build_phase06_protocol_lock(
        config,
        phase05_config,
        execution_commit=execution_commit,
        phase06_spec_path=phase06_spec_path,
        phase05_protocol_path=phase05_protocol_path,
    )
    lock.update(
        {
            "schema_version": 3,
            "d4b_addendum_sha256": _sha256_bytes(
                Path(d4b_addendum_path), "Phase 0.6 D4-B execution addendum"
            ),
            "core_parent_execution_sha": _PARENT_PHASE06_EXECUTION_SHA,
            "core_parent_protocol_lock_sha256": _PARENT_PHASE06_PROTOCOL_SHA256,
            "core_parent_d3_sha256": _PARENT_PHASE06_D3_SHA256,
            "d2b_parent_execution_sha": _D2B_PHASE06_EXECUTION_SHA,
            "d2b_parent_protocol_canonical_sha256": _D2B_PROTOCOL_CANONICAL_SHA256,
            "d2b_parent_adjudication_canonical_sha256": _D2B_ADJUDICATION_CANONICAL_SHA256,
            "d2b_next_required_stage": "D4_OPTIMIZATION",
            "d4b_contexts": [list(context) for context in _D4B_CONTEXTS],
            "d4b_model_seeds": list(_D4B_MODEL_SEEDS),
            "d4b_n_train": 40,
            "d4b_flow_modes": ["none", "time_scaled"],
            "d4b_bootstrap_resamples": _D4B_BOOTSTRAP_RESAMPLES,
            "d4b_bootstrap_seed": _D4B_BOOTSTRAP_SEED,
        }
    )
    return lock


__all__ = [
    "build_phase06_d2b_protocol_lock",
    "build_phase06_d4b_protocol_lock",
    "build_phase06_protocol_lock",
    "validate_development_seed_triplet",
]
