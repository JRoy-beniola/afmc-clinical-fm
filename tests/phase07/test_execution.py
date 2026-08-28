from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase07.config import load_phase07_config
from afmc_fm.phase07.planning import phase07_plan_sha256, plan_phase07_cells

_CONFIG_PATH = Path("configs/experiments/phase07.yaml")
_SPEC_PATH = Path(
    "docs/superpowers/specs/2026-08-28-phase0-7-optimization-horizon-intervention-design.md"
)
_EXECUTION_SHA = "2" * 40


def _execution_api():
    try:
        return importlib.import_module("afmc_fm.phase07.execution")
    except ModuleNotFoundError as error:
        raise AssertionError("Phase 0.7 execution module is not implemented") from error


def test_execution_manifest_binds_protocol_config_plan_and_exact_cell_count():
    module = _execution_api()
    config = load_phase07_config(_CONFIG_PATH)
    cells = plan_phase07_cells(config)

    manifest = module.build_phase07_execution_manifest(
        config,
        execution_commit=_EXECUTION_SHA,
        phase07_spec_path=_SPEC_PATH,
    )

    assert manifest["schema_version"] == 1
    assert manifest["phase"] == "phase07"
    assert manifest["execution_commit"] == _EXECUTION_SHA
    assert manifest["phase07_spec_sha256"] == hashlib.sha256(_SPEC_PATH.read_bytes()).hexdigest()
    assert manifest["phase07_config_sha256"] == canonical_config_hash(config)
    assert manifest["phase07_plan_sha256"] == phase07_plan_sha256(cells)
    assert manifest["expected_cell_count"] == 200
    assert len(manifest["protocol_lock_sha256"]) == 64
    assert manifest["forbidden_seed_sets"] == {
        "cohort": list(range(701, 711)),
        "subset": list(range(801, 811)),
        "model": list(range(901, 911)),
    }


def test_official_execution_authorization_fails_closed_when_artifact_is_absent(tmp_path):
    module = _execution_api()
    config = load_phase07_config(_CONFIG_PATH)
    manifest = module.build_phase07_execution_manifest(
        config,
        execution_commit=_EXECUTION_SHA,
        phase07_spec_path=_SPEC_PATH,
    )

    with pytest.raises(ValueError, match="authorization"):
        module.require_phase07_official_authorization(
            tmp_path / "missing-authorization.json",
            manifest,
        )


def test_official_execution_authorization_is_exactly_hash_bound(tmp_path):
    module = _execution_api()
    config = load_phase07_config(_CONFIG_PATH)
    manifest = module.build_phase07_execution_manifest(
        config,
        execution_commit=_EXECUTION_SHA,
        phase07_spec_path=_SPEC_PATH,
    )
    path = tmp_path / "authorization.json"
    payload = {
        "schema_version": 1,
        "phase": "phase07",
        "authorization": "OFFICIAL_EXECUTION_AUTHORIZED",
        "execution_commit": manifest["execution_commit"],
        "protocol_lock_sha256": manifest["protocol_lock_sha256"],
        "phase07_plan_sha256": manifest["phase07_plan_sha256"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert module.require_phase07_official_authorization(path, manifest) == payload

    payload["phase07_plan_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization"):
        module.require_phase07_official_authorization(path, manifest)


def test_official_execution_checks_authorization_before_any_cell_callback(tmp_path):
    module = _execution_api()
    config = load_phase07_config(_CONFIG_PATH)
    cells = plan_phase07_cells(config)
    called = False

    def forbidden_callback(_cell):
        nonlocal called
        called = True
        raise AssertionError("official cells must not run without authorization")

    with pytest.raises(ValueError, match="authorization"):
        module.run_phase07_official_cells(
            cells,
            config=config,
            execution_commit=_EXECUTION_SHA,
            phase07_spec_path=_SPEC_PATH,
            authorization_path=tmp_path / "missing.json",
            execute_cell=forbidden_callback,
        )

    assert called is False


def test_cpu_device_smoke_is_non_official_and_never_materializes_phase07_plan(monkeypatch):
    module = _execution_api()

    def forbidden_planner(*_args, **_kwargs):
        raise AssertionError("validation smoke must not materialize the official plan")

    monkeypatch.setattr(module, "plan_phase07_cells", forbidden_planner)
    result = module.run_phase07_device_smoke("cpu")

    assert result["mode"] == "non_official_validation_smoke"
    assert result["device"] == "cpu"
    assert result["official_cells_executed"] == 0
    assert result["standard_epochs_run"] >= 1
    assert result["forced_epochs_run"] >= result["standard_epochs_run"]
    assert result["prefix_identical"] is True
