import hashlib
from pathlib import Path

import pytest

import afmc_fm.phase06.protocol as protocol_module
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase06.config import load_phase06_config
from afmc_fm.phase06.protocol import (
    build_phase06_protocol_lock,
    validate_development_seed_triplet,
)

_PHASE06_CONFIG = Path("configs/experiments/phase06.yaml")
_PHASE06_SPEC = Path(
    "docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md"
)
_D2B_ADDENDUM = Path(
    "docs/superpowers/specs/2026-08-26-phase0-6-d2b-execution-addendum.md"
)
_PHASE05_PROTOCOL = Path(
    "docs/results/phase05/raw/official_output/protocol_lock.json"
)
_PHASE05_EXECUTION_SHA = "50a94c06bc1c419ca55738f15f074cc06ccc3f36"
_PHASE05_STATUS = (
    "PHASE-0.5 TERMINATED AT STAGE I-A — FLOW MECHANISM GATE NOT ESTABLISHED"
)
_PHASE05_PROTOCOL_SHA256 = (
    "33b25ebe46620626ae3d1bbbd6152fc3aacdb0e6200dd87e03c4fdec99e70db9"
)
_PARENT_EXECUTION_SHA = "1718402df1d6ef344168677e6d26ea664708e1bc"
_PARENT_PROTOCOL_SHA256 = (
    "c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51"
)


@pytest.mark.parametrize("seed", range(701, 711))
def test_reserved_confirmatory_cohort_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="confirmatory cohort seed"):
        validate_development_seed_triplet(seed, 501, 601)


@pytest.mark.parametrize("seed", range(801, 811))
def test_reserved_confirmatory_subset_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="confirmatory subset seed"):
        validate_development_seed_triplet(401, seed, 601)


@pytest.mark.parametrize("seed", range(901, 911))
def test_reserved_confirmatory_model_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="confirmatory model seed"):
        validate_development_seed_triplet(401, 501, seed)


@pytest.mark.parametrize(
    "triplet",
    [
        (401, 501, 601),
        (402, 502, 602),
        (403, 503, 603),
        (404, 504, 604),
        (405, 505, 605),
    ],
)
def test_phase05_development_triplets_pass_firewall(triplet):
    validate_development_seed_triplet(*triplet)


@pytest.mark.parametrize("triplet", [(-1, 501, 601), (401, -1, 601), (401, 501, -1)])
def test_seed_firewall_rejects_negative_values(triplet):
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_development_seed_triplet(*triplet)


def test_protocol_lock_binds_phase05_and_phase06_identity():
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)
    execution_commit = "a" * 40

    lock = build_phase06_protocol_lock(
        config,
        phase05_config,
        execution_commit=execution_commit,
        phase06_spec_path=_PHASE06_SPEC,
        phase05_protocol_path=_PHASE05_PROTOCOL,
    )

    assert lock["schema_version"] == 1
    assert lock["phase06_spec_sha256"] == hashlib.sha256(
        _PHASE06_SPEC.read_bytes()
    ).hexdigest()
    assert lock["phase06_config_sha256"] == canonical_config_hash(config)
    assert lock["phase05_config_sha256"] == canonical_config_hash(phase05_config)
    assert lock["phase05_config_sha256"] == (
        "befd7140cbf68cb981418cdc0c8880bb0652c2a6db614d05f61ad267297d519b"
    )
    assert lock["phase05_protocol_sha256"] == _PHASE05_PROTOCOL_SHA256
    assert hashlib.sha256(_PHASE05_PROTOCOL.read_bytes()).hexdigest() == (
        _PHASE05_PROTOCOL_SHA256
    )
    assert lock["phase05_execution_sha"] == _PHASE05_EXECUTION_SHA
    assert lock["phase05_official_status"] == _PHASE05_STATUS
    assert lock["execution_commit"] == execution_commit
    assert lock["d1_development_bundles"] == [
        [401, 501, 601],
        [402, 502, 602],
        [403, 503, 603],
        [404, 504, 604],
        [405, 505, 605],
    ]
    assert lock["d2_mapping"] == "model_index=(cohort_index+subset_index)%5"
    assert lock["bootstrap_resamples"] == 10_000
    assert lock["bootstrap_seed"] == 20260826
    assert lock["forbidden_seed_sets"] == {
        "cohort": list(range(701, 711)),
        "subset": list(range(801, 811)),
        "model": list(range(901, 911)),
    }


def test_protocol_lock_rejects_non_commit_execution_identity():
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)

    with pytest.raises(ValueError, match="execution_commit"):
        build_phase06_protocol_lock(
            config,
            phase05_config,
            execution_commit="not-a-commit",
            phase06_spec_path=_PHASE06_SPEC,
            phase05_protocol_path=_PHASE05_PROTOCOL,
        )


def test_protocol_lock_rejects_wrong_phase05_protocol_bytes(tmp_path):
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)
    protocol = tmp_path / "protocol_lock.json"
    protocol.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Phase 0.5 protocol"):
        build_phase06_protocol_lock(
            config,
            phase05_config,
            execution_commit="a" * 40,
            phase06_spec_path=_PHASE06_SPEC,
            phase05_protocol_path=protocol,
        )


def test_d2b_child_protocol_lock_binds_exact_parent_evidence():
    builder = getattr(protocol_module, "build_phase06_d2b_protocol_lock", None)
    assert callable(builder), "build_phase06_d2b_protocol_lock must exist"

    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)
    child_execution = "b" * 40
    parent_evidence = {
        "parent_execution_sha": _PARENT_EXECUTION_SHA,
        "parent_protocol_lock_sha256": _PARENT_PROTOCOL_SHA256,
        "parent_d3_sha256": "d" * 64,
        "parent_d3_next_required_stage": "D2B",
        "parent_phase06_config_sha256": canonical_config_hash(config),
        "parent_phase06_spec_sha256": hashlib.sha256(
            _PHASE06_SPEC.read_bytes()
        ).hexdigest(),
        "parent_forbidden_seed_sets": {
            "cohort": list(range(701, 711)),
            "subset": list(range(801, 811)),
            "model": list(range(901, 911)),
        },
    }

    lock = builder(
        config,
        phase05_config,
        execution_commit=child_execution,
        phase06_spec_path=_PHASE06_SPEC,
        d2b_addendum_path=_D2B_ADDENDUM,
        phase05_protocol_path=_PHASE05_PROTOCOL,
        parent_evidence=parent_evidence,
    )

    assert lock["schema_version"] == 2
    assert lock["execution_commit"] == child_execution
    assert lock["execution_commit"] != _PARENT_EXECUTION_SHA
    assert lock["d2b_addendum_sha256"] == hashlib.sha256(
        _D2B_ADDENDUM.read_bytes()
    ).hexdigest()
    assert lock["parent_execution_sha"] == _PARENT_EXECUTION_SHA
    assert lock["parent_protocol_lock_sha256"] == _PARENT_PROTOCOL_SHA256
    assert lock["parent_d3_sha256"] == "d" * 64
    assert lock["parent_d3_next_required_stage"] == "D2B"
    assert lock["d2b_mapping"] == "model_index=(cohort_index+2*subset_index)%5"
    assert lock["bootstrap_resamples"] == 10_000
    assert lock["bootstrap_seed"] == 20260826
    assert lock["forbidden_seed_sets"] == parent_evidence["parent_forbidden_seed_sets"]


def test_d2b_child_protocol_lock_rejects_parent_identity_drift():
    builder = getattr(protocol_module, "build_phase06_d2b_protocol_lock", None)
    assert callable(builder), "build_phase06_d2b_protocol_lock must exist"

    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)
    parent_evidence = {
        "parent_execution_sha": _PARENT_EXECUTION_SHA,
        "parent_protocol_lock_sha256": _PARENT_PROTOCOL_SHA256,
        "parent_d3_sha256": "d" * 64,
        "parent_d3_next_required_stage": "D2B",
        "parent_phase06_config_sha256": "e" * 64,
        "parent_phase06_spec_sha256": hashlib.sha256(
            _PHASE06_SPEC.read_bytes()
        ).hexdigest(),
        "parent_forbidden_seed_sets": {
            "cohort": list(range(701, 711)),
            "subset": list(range(801, 811)),
            "model": list(range(901, 911)),
        },
    }

    with pytest.raises(ValueError, match="parent Phase 0.6 config"):
        builder(
            config,
            phase05_config,
            execution_commit="b" * 40,
            phase06_spec_path=_PHASE06_SPEC,
            d2b_addendum_path=_D2B_ADDENDUM,
            phase05_protocol_path=_PHASE05_PROTOCOL,
            parent_evidence=parent_evidence,
        )
