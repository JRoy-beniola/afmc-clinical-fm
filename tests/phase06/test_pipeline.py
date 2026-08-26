from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

import afmc_fm.phase06.pipeline as pipeline_module
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase06.config import load_phase06_config
from afmc_fm.phase06.planning import plan_d1_cells, plan_d2a_cells
from afmc_fm.phase06.protocol import build_phase06_protocol_lock

_PHASE06_CONFIG = Path("configs/experiments/phase06.yaml")
_PHASE06_SPEC = Path(
    "docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md"
)
_PHASE05_PROTOCOL = Path(
    "docs/results/phase05/raw/official_output/protocol_lock.json"
)
_PARENT_EXECUTION_SHA = "1718402df1d6ef344168677e6d26ea664708e1bc"
_PARENT_PROTOCOL_SHA256 = (
    "c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51"
)


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _write_parent_root(tmp_path: Path, *, next_stage: str = "D2B") -> Path:
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)
    lock = build_phase06_protocol_lock(
        config,
        phase05_config,
        execution_commit=_PARENT_EXECUTION_SHA,
        phase06_spec_path=_PHASE06_SPEC,
        phase05_protocol_path=_PHASE05_PROTOCOL,
    )
    lock_bytes = _canonical_json_bytes(lock)
    assert hashlib.sha256(lock_bytes).hexdigest() == _PARENT_PROTOCOL_SHA256

    root = tmp_path / "parent"
    root.mkdir(parents=True)
    (root / "protocol_lock.json").write_bytes(lock_bytes)
    identity = {
        "protocol_lock_sha256": _PARENT_PROTOCOL_SHA256,
        "phase06_config_sha256": canonical_config_hash(config),
        "execution_commit": _PARENT_EXECUTION_SHA,
    }

    stage_plans = {
        "d1": plan_d1_cells(config, phase05_config),
        "d2a": plan_d2a_cells(config),
    }
    for stage, cells in stage_plans.items():
        marker = root / "stages" / stage / "COMPLETE"
        marker.parent.mkdir(parents=True)
        marker.write_bytes(
            _canonical_json_bytes(
                {
                    "schema_version": 1,
                    "identity": identity,
                    "stage": stage,
                    "cell_ids": sorted(cell.cell_id for cell in cells),
                }
            )
        )

    analysis = root / "analysis"
    analysis.mkdir()
    d3_payload = {
        "phenomenon_reproduction": "reproduced",
        "interaction_ambiguity": True,
        "next_required_stage": next_stage,
        "triggered_escalations": ["D2B", "D4_OPTIMIZATION", "D4_CAPACITY_TIME"],
        "input_artifact_hashes": {},
    }
    (analysis / "phase06_d3_adjudication.json").write_bytes(
        _canonical_json_bytes(d3_payload)
    )
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _load_parent(root: Path):
    loader = getattr(pipeline_module, "load_phase06_d2b_parent_evidence", None)
    assert callable(loader), "load_phase06_d2b_parent_evidence must exist"
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)
    d3_path = root / "analysis" / "phase06_d3_adjudication.json"
    expected_input_hashes: dict[str, object] = {}
    if d3_path.is_file():
        try:
            d3_payload = json.loads(d3_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            d3_payload = None
        if isinstance(d3_payload, dict) and isinstance(
            d3_payload.get("input_artifact_hashes"), dict
        ):
            expected_input_hashes = d3_payload["input_artifact_hashes"]

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(pipeline_module, "open_bound_store", return_value=object())
        )
        stack.enter_context(
            patch.object(
                pipeline_module,
                "require_completed_stage",
                side_effect=lambda _store, cells: frozenset(
                    cell.cell_id for cell in cells
                ),
            )
        )
        stack.enter_context(
            patch.object(
                pipeline_module,
                "_expected_parent_d3_input_hashes",
                return_value=expected_input_hashes,
            )
        )
        if d3_path.is_file():
            stack.enter_context(
                patch.object(
                    pipeline_module,
                    "_PARENT_PHASE06_D3_SHA256",
                    hashlib.sha256(d3_path.read_bytes()).hexdigest(),
                )
            )
        return loader(root, config=config, phase05_config=phase05_config)


def test_d2b_parent_loader_validates_and_hashes_read_only_evidence(tmp_path):
    root = _write_parent_root(tmp_path)
    before = _snapshot(root)

    evidence = _load_parent(root)

    assert evidence["parent_execution_sha"] == _PARENT_EXECUTION_SHA
    assert evidence["parent_protocol_lock_sha256"] == _PARENT_PROTOCOL_SHA256
    assert evidence["parent_d3_next_required_stage"] == "D2B"
    assert evidence["parent_d3_sha256"] == hashlib.sha256(
        (root / "analysis" / "phase06_d3_adjudication.json").read_bytes()
    ).hexdigest()
    assert evidence["parent_phase06_config_sha256"] == canonical_config_hash(
        load_phase06_config(_PHASE06_CONFIG)
    )
    assert evidence["parent_phase06_spec_sha256"] == hashlib.sha256(
        _PHASE06_SPEC.read_bytes()
    ).hexdigest()
    assert evidence["parent_forbidden_seed_sets"] == {
        "cohort": list(range(701, 711)),
        "subset": list(range(801, 811)),
        "model": list(range(901, 911)),
    }
    assert _snapshot(root) == before


def test_d2b_parent_loader_rejects_missing_or_invalid_d3(tmp_path):
    root = _write_parent_root(tmp_path)
    d3 = root / "analysis" / "phase06_d3_adjudication.json"
    d3.unlink()
    with pytest.raises(ValueError, match="parent D3"):
        _load_parent(root)

    root = _write_parent_root(tmp_path / "second")
    d3 = root / "analysis" / "phase06_d3_adjudication.json"
    d3.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parent D3"):
        _load_parent(root)


def test_d2b_parent_loader_rejects_wrong_d3_route(tmp_path):
    root = _write_parent_root(tmp_path, next_stage="D4_OPTIMIZATION")
    with pytest.raises(ValueError, match="next_required_stage.*D2B"):
        _load_parent(root)


def test_d2b_parent_loader_rejects_mutated_parent_protocol_bytes(tmp_path):
    root = _write_parent_root(tmp_path)
    protocol = root / "protocol_lock.json"
    protocol.write_bytes(protocol.read_bytes() + b" ")
    with pytest.raises(ValueError, match="parent protocol"):
        _load_parent(root)


def test_d2b_parent_loader_requires_both_parent_completion_markers(tmp_path):
    root = _write_parent_root(tmp_path)
    (root / "stages" / "d2a" / "COMPLETE").unlink()
    with pytest.raises(ValueError, match="d2a.*complete"):
        _load_parent(root)
