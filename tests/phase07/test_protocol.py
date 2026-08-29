import hashlib
from pathlib import Path

import pytest

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase07.config import load_phase07_config

_CONFIG_PATH = Path("configs/experiments/phase07.yaml")
_SPEC_PATH = Path(
    "docs/superpowers/specs/2026-08-28-phase0-7-optimization-horizon-intervention-design.md"
)
_EXECUTION_SHA = "1" * 40


def _protocol_api():
    try:
        from afmc_fm.phase07.protocol import (
            build_phase07_protocol_lock,
            validate_phase07_seed_triplet,
        )
    except ModuleNotFoundError:
        pytest.fail("Phase 0.7 protocol module is not implemented")
    return build_phase07_protocol_lock, validate_phase07_seed_triplet


def test_phase07_protocol_binds_exact_frozen_spec_and_matrix():
    build_phase07_protocol_lock, _ = _protocol_api()
    config = load_phase07_config(_CONFIG_PATH)

    lock = build_phase07_protocol_lock(
        config,
        execution_commit=_EXECUTION_SHA,
        phase07_spec_path=_SPEC_PATH,
    )

    assert lock["schema_version"] == 1
    assert lock["phase07_spec_sha256"] == hashlib.sha256(_SPEC_PATH.read_bytes()).hexdigest()
    assert lock["phase07_config_sha256"] == canonical_config_hash(config)
    assert lock["execution_commit"] == _EXECUTION_SHA
    assert lock["world"] == "smooth"
    assert lock["n_train"] == 40
    assert lock["contexts"] == [
        [406, 506],
        [407, 507],
        [408, 508],
        [409, 509],
        [410, 510],
    ]
    assert lock["model_seeds"] == list(range(1101, 1111))
    assert lock["flow_modes"] == ["none", "time_scaled"]
    assert lock["optimization_policies"] == ["standard_early_stop", "forced_horizon"]
    assert lock["max_epochs"] == 100
    assert lock["patience"] == 12
    assert lock["bootstrap_resamples"] == 10_000
    assert lock["bootstrap_seed"] == 20260827
    assert lock["expected_cell_count"] == 200
    assert lock["forbidden_seed_sets"] == {
        "cohort": list(range(701, 711)),
        "subset": list(range(801, 811)),
        "model": list(range(901, 911)),
    }


def test_phase07_protocol_hash_changes_if_frozen_spec_bytes_change(tmp_path):
    build_phase07_protocol_lock, _ = _protocol_api()
    config = load_phase07_config(_CONFIG_PATH)
    changed = tmp_path / "phase07-spec.md"
    changed.write_bytes(_SPEC_PATH.read_bytes() + b"\nbyte drift\n")

    official = build_phase07_protocol_lock(
        config,
        execution_commit=_EXECUTION_SHA,
        phase07_spec_path=_SPEC_PATH,
    )
    drifted = build_phase07_protocol_lock(
        config,
        execution_commit=_EXECUTION_SHA,
        phase07_spec_path=changed,
    )

    assert drifted["phase07_spec_sha256"] != official["phase07_spec_sha256"]


@pytest.mark.parametrize("execution_commit", ["", "abc", "g" * 40, "1" * 39, "1" * 41])
def test_phase07_protocol_rejects_invalid_execution_commit(execution_commit):
    build_phase07_protocol_lock, _ = _protocol_api()
    config = load_phase07_config(_CONFIG_PATH)

    with pytest.raises(ValueError, match="execution_commit"):
        build_phase07_protocol_lock(
            config,
            execution_commit=execution_commit,
            phase07_spec_path=_SPEC_PATH,
        )


@pytest.mark.parametrize(
    ("cohort_seed", "subset_seed", "model_seed", "message"),
    [
        (701, 506, 1101, "cohort"),
        (406, 801, 1101, "subset"),
        (406, 506, 901, "model"),
    ],
)
def test_phase07_protocol_rejects_protected_confirmatory_seeds(
    cohort_seed,
    subset_seed,
    model_seed,
    message,
):
    _, validate_phase07_seed_triplet = _protocol_api()

    with pytest.raises(ValueError, match=message):
        validate_phase07_seed_triplet(cohort_seed, subset_seed, model_seed)


def test_phase07_protocol_accepts_frozen_development_seeds():
    _, validate_phase07_seed_triplet = _protocol_api()

    validate_phase07_seed_triplet(406, 506, 1101)
