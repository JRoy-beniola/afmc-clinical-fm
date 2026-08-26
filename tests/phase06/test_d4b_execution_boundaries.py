from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase06.config import load_phase06_config
from afmc_fm.phase06.protocol import build_phase06_d4b_protocol_lock

_PHASE06_CONFIG = Path("configs/experiments/phase06.yaml")
_PHASE06_SPEC = Path("docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md")
_D4B_ADDENDUM = Path(
    "docs/superpowers/specs/2026-08-27-phase0-6-d4b-execution-addendum.md"
)
_PHASE05_PROTOCOL = Path("docs/results/phase05/raw/official_output/protocol_lock.json")
_CORE_EXECUTION_SHA = "1718402df1d6ef344168677e6d26ea664708e1bc"
_CORE_PROTOCOL_SHA256 = (
    "c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51"
)
_CORE_D3_SHA256 = "6b9fffed7503fae6beeac2314238ae3d10ffdebfd27ea71ad952ab1f87916460"
_D2B_EXECUTION_SHA = "516c9e3c0e965582fa5cce976e9d8ebf32ea8404"
_D2B_PROTOCOL_CANONICAL_SHA256 = (
    "33f7cb1f6e71560f547a746cb7f5eb41130f794ab1e3a6fb1928c40e05b29f12"
)
_D2B_ADJUDICATION_CANONICAL_SHA256 = (
    "ea28fd4d5f6490a10fad20d5d3f3e76a1de08bf6b1be3b9805c9cf6c519e845f"
)


def _parent_evidence() -> dict[str, object]:
    config = load_phase06_config(_PHASE06_CONFIG)
    return {
        "core_parent_execution_sha": _CORE_EXECUTION_SHA,
        "core_parent_protocol_lock_sha256": _CORE_PROTOCOL_SHA256,
        "core_parent_d3_sha256": _CORE_D3_SHA256,
        "d2b_parent_execution_sha": _D2B_EXECUTION_SHA,
        "d2b_parent_protocol_canonical_sha256": _D2B_PROTOCOL_CANONICAL_SHA256,
        "d2b_parent_adjudication_canonical_sha256": _D2B_ADJUDICATION_CANONICAL_SHA256,
        "d2b_next_required_stage": "D4_OPTIMIZATION",
        "phase06_config_sha256": canonical_config_hash(config),
        "phase06_spec_sha256": hashlib.sha256(_PHASE06_SPEC.read_bytes()).hexdigest(),
        "forbidden_seed_sets": {
            "cohort": list(range(701, 711)),
            "subset": list(range(801, 811)),
            "model": list(range(901, 911)),
        },
    }


def test_d4b_protocol_binds_frozen_parent_chain_and_matrix() -> None:
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)

    lock = build_phase06_d4b_protocol_lock(
        config,
        phase05_config,
        execution_commit="d" * 40,
        phase06_spec_path=_PHASE06_SPEC,
        d4b_addendum_path=_D4B_ADDENDUM,
        phase05_protocol_path=_PHASE05_PROTOCOL,
        parent_evidence=_parent_evidence(),
    )

    assert lock["schema_version"] == 3
    assert lock["core_parent_execution_sha"] == _CORE_EXECUTION_SHA
    assert lock["core_parent_protocol_lock_sha256"] == _CORE_PROTOCOL_SHA256
    assert lock["core_parent_d3_sha256"] == _CORE_D3_SHA256
    assert lock["d2b_parent_execution_sha"] == _D2B_EXECUTION_SHA
    assert (
        lock["d2b_parent_protocol_canonical_sha256"]
        == _D2B_PROTOCOL_CANONICAL_SHA256
    )
    assert (
        lock["d2b_parent_adjudication_canonical_sha256"]
        == _D2B_ADJUDICATION_CANONICAL_SHA256
    )
    assert lock["d2b_next_required_stage"] == "D4_OPTIMIZATION"
    assert lock["d4b_contexts"] == [[401, 501], [402, 502], [403, 503], [404, 504], [405, 505]]
    assert lock["d4b_model_seeds"] == list(range(1001, 1011))
    assert lock["d4b_n_train"] == 40
    assert lock["d4b_flow_modes"] == ["none", "time_scaled"]
    assert lock["d4b_bootstrap_resamples"] == 10_000
    assert lock["d4b_bootstrap_seed"] == 20260827


def test_d4b_protocol_rejects_adjudication_identity_drift() -> None:
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)
    evidence = _parent_evidence()
    evidence["d2b_parent_adjudication_canonical_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="D2-B adjudication"):
        build_phase06_d4b_protocol_lock(
            config,
            phase05_config,
            execution_commit="d" * 40,
            phase06_spec_path=_PHASE06_SPEC,
            d4b_addendum_path=_D4B_ADDENDUM,
            phase05_protocol_path=_PHASE05_PROTOCOL,
            parent_evidence=evidence,
        )


def test_d4b_protocol_rejects_non_optimization_route() -> None:
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)
    evidence = _parent_evidence()
    evidence["d2b_next_required_stage"] = "D4_CAPACITY_TIME"

    with pytest.raises(ValueError, match="D4_OPTIMIZATION"):
        build_phase06_d4b_protocol_lock(
            config,
            phase05_config,
            execution_commit="d" * 40,
            phase06_spec_path=_PHASE06_SPEC,
            d4b_addendum_path=_D4B_ADDENDUM,
            phase05_protocol_path=_PHASE05_PROTOCOL,
            parent_evidence=evidence,
        )
