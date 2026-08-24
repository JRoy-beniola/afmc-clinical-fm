import hashlib
import json
from dataclasses import replace

import pandas as pd
import pytest
from afmc_fm.phase05.store import Phase05CellResult, Phase05Store


SPEC_HASH = "1" * 64
CONFIG_HASH = "2" * 64
OTHER_PROTOCOL_HASH = "f" * 64


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _protocol_lock() -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase05_config_sha256": CONFIG_HASH,
        "locked_min_relative_effect": 0.02,
        "locked_uncertainty_mae_tolerance": 0.02,
    }


def _protocol_hash(lock: dict[str, object] | None = None) -> str:
    payload = _protocol_lock() if lock is None else lock
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _store(tmp_path) -> Phase05Store:
    store = Phase05Store(
        tmp_path / "phase05",
        _protocol_hash(),
        spec_hash=SPEC_HASH,
        config_hash=CONFIG_HASH,
    )
    store.write_protocol_lock(_protocol_lock())
    return store


def _cell(
    *,
    stage: str = "flow",
    world: str = "smooth",
    cohort_seed: int = 401,
    subset_seed: int = 501,
    model_seed: int = 601,
    frozen_candidate_hash: str | None = None,
) -> Phase05CellResult:
    return Phase05CellResult(
        stage=stage,
        world=world,
        cohort_seed=cohort_seed,
        subset_seed=subset_seed,
        model_seed=model_seed,
        n_train=5,
        model="phase05_candidate",
        variant="time_scaled__none__deterministic",
        metrics=pd.DataFrame(
            [
                {
                    "metric": "mae",
                    "value": 0.75,
                    "split": "test",
                },
                {
                    "metric": "rmse",
                    "value": 0.9,
                    "split": "test",
                },
            ]
        ),
        frozen_candidate_hash=frozen_candidate_hash,
    )


def test_write_cell_is_atomic_idempotent_and_rejects_conflicting_content(tmp_path):
    store = _store(tmp_path)
    cell = _cell()

    cell_id = store.write_cell(cell)

    assert cell_id == (
        "flow__smooth__cohort401__subset501__model601__n5__"
        "phase05_candidate__time_scaled__none__deterministic"
    )
    path = tmp_path / "phase05" / "stages" / "flow" / "cells" / f"{cell_id}.json"
    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns
    assert not list(path.parent.glob("*.tmp"))

    assert store.write_cell(cell) == cell_id
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime

    changed_metrics = cell.metrics.copy()
    changed_metrics.loc[0, "value"] = 0.5
    with pytest.raises(ValueError, match="conflicting persisted cell"):
        store.write_cell(replace(cell, metrics=changed_metrics))

    assert path.read_bytes() == original_bytes
    assert not list(path.parent.glob("*.tmp"))


def test_cell_payload_contains_stage_and_phase05_protocol_identity(tmp_path):
    store = _store(tmp_path)
    cell = _cell()
    cell_id = store.write_cell(cell)
    path = tmp_path / "phase05" / "stages" / "flow" / "cells" / f"{cell_id}.json"

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["stage"] == "flow"
    assert payload["identity"] == {
        "schema_version": 1,
        "spec_sha256": SPEC_HASH,
        "config_sha256": CONFIG_HASH,
        "protocol_lock_sha256": _protocol_hash(),
    }
    assert payload["seed_bundle"] == {
        "cohort_seed": 401,
        "subset_seed": 501,
        "model_seed": 601,
    }


def test_protocol_lock_hash_must_match_store_identity(tmp_path):
    store = Phase05Store(
        tmp_path / "phase05",
        OTHER_PROTOCOL_HASH,
        spec_hash=SPEC_HASH,
        config_hash=CONFIG_HASH,
    )

    with pytest.raises(ValueError, match="protocol lock hash"):
        store.write_protocol_lock(_protocol_lock())

    assert not (tmp_path / "phase05" / "protocol_lock.json").exists()


def test_confirmation_start_creates_hard_mutation_boundary(tmp_path):
    store = _store(tmp_path)
    candidate = {
        "flow_mode": "time_scaled",
        "jump_mode": "residual",
        "uncertainty_mode": "deterministic",
    }
    store.replace_development_artifact("flow_gate.csv", b"candidate,passed\ntime_scaled,true\n")
    store.write_frozen_candidate(candidate)

    store.mark_confirmation_started()

    marker = tmp_path / "phase05" / "confirmation" / "STARTED"
    assert marker.is_file()
    with pytest.raises(RuntimeError, match="confirmation has started"):
        store.write_protocol_lock(_protocol_lock())
    with pytest.raises(RuntimeError, match="confirmation has started"):
        store.write_frozen_candidate({**candidate, "flow_mode": "gated"})
    with pytest.raises(RuntimeError, match="confirmation has started"):
        store.replace_development_artifact("flow_gate.csv", b"different")


def test_confirmation_start_requires_protocol_lock_and_frozen_candidate(tmp_path):
    empty_store = Phase05Store(
        tmp_path / "empty",
        _protocol_hash(),
        spec_hash=SPEC_HASH,
        config_hash=CONFIG_HASH,
    )
    with pytest.raises(RuntimeError, match="protocol_lock.json"):
        empty_store.mark_confirmation_started()

    no_candidate_store = _store(tmp_path / "no_candidate")
    with pytest.raises(RuntimeError, match="frozen_candidate.json"):
        no_candidate_store.mark_confirmation_started()


def test_frozen_candidate_write_is_idempotent_but_conflicts_before_confirmation(tmp_path):
    store = _store(tmp_path)
    candidate = {"flow_mode": "time_scaled", "jump_mode": "residual"}

    candidate_hash = store.write_frozen_candidate(candidate)
    path = tmp_path / "phase05" / "frozen_candidate.json"
    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns

    assert candidate_hash == hashlib.sha256(original_bytes).hexdigest()
    assert store.write_frozen_candidate(candidate) == candidate_hash
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime

    with pytest.raises(ValueError, match="conflicting frozen candidate"):
        store.write_frozen_candidate({"flow_mode": "gated", "jump_mode": "residual"})


def test_validate_resume_rejects_unexpected_cell_ids_and_seed_bundles(tmp_path):
    store = _store(tmp_path)
    candidate_hash = store.write_frozen_candidate({"flow_mode": "time_scaled"})
    store.mark_confirmation_started()
    cell = _cell(
        stage="confirmation",
        world="smooth",
        cohort_seed=701,
        subset_seed=801,
        model_seed=901,
        frozen_candidate_hash=candidate_hash,
    )
    cell_id = store.write_cell(cell)

    assert store.validate_resume(
        "confirmation",
        expected_cell_ids={cell_id},
        expected_seed_bundles={(701, 801, 901)},
        frozen_candidate_hash=candidate_hash,
    ) == frozenset({cell_id})

    with pytest.raises(ValueError, match="unexpected persisted cell IDs"):
        store.validate_resume(
            "confirmation",
            expected_cell_ids={"different-cell"},
            expected_seed_bundles={(701, 801, 901)},
            frozen_candidate_hash=candidate_hash,
        )
    with pytest.raises(ValueError, match="unexpected seed bundle"):
        store.validate_resume(
            "confirmation",
            expected_cell_ids={cell_id},
            expected_seed_bundles={(702, 802, 902)},
            frozen_candidate_hash=candidate_hash,
        )


def test_validate_resume_rejects_candidate_or_protocol_drift(tmp_path):
    store = _store(tmp_path)
    candidate_hash = store.write_frozen_candidate({"flow_mode": "time_scaled"})
    store.mark_confirmation_started()
    cell = _cell(
        stage="confirmation",
        world="jumps",
        cohort_seed=701,
        subset_seed=801,
        model_seed=901,
        frozen_candidate_hash=candidate_hash,
    )
    cell_id = store.write_cell(cell)

    with pytest.raises(ValueError, match="frozen candidate hash"):
        store.validate_resume(
            "confirmation",
            expected_cell_ids={cell_id},
            expected_seed_bundles={(701, 801, 901)},
            frozen_candidate_hash="a" * 64,
        )

    incompatible = Phase05Store(
        tmp_path / "phase05",
        OTHER_PROTOCOL_HASH,
        spec_hash=SPEC_HASH,
        config_hash=CONFIG_HASH,
    )
    with pytest.raises(ValueError, match="protocol identity"):
        incompatible.validate_resume(
            "confirmation",
            expected_cell_ids={cell_id},
            expected_seed_bundles={(701, 801, 901)},
            frozen_candidate_hash=candidate_hash,
        )


def test_complete_marker_is_valid_only_for_exact_expected_cell_set(tmp_path):
    store = _store(tmp_path)
    first = _cell()
    first_id = store.write_cell(first)

    store.mark_stage_complete("flow", {first_id})
    assert store.validate_resume(
        "flow",
        expected_cell_ids={first_id},
        expected_seed_bundles={(401, 501, 601)},
    ) == frozenset({first_id})

    second = replace(first, n_train=10)
    second_id = store.write_cell(second)
    marker = tmp_path / "phase05" / "stages" / "flow" / "COMPLETE"
    assert not marker.exists()

    with pytest.raises(ValueError, match="unexpected persisted cell IDs"):
        store.validate_resume(
            "flow",
            expected_cell_ids={first_id},
            expected_seed_bundles={(401, 501, 601)},
        )

    store.mark_stage_complete("flow", {first_id, second_id})
    assert marker.is_file()
