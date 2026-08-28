from __future__ import annotations

import importlib
import json
from pathlib import Path


def _provenance_api():
    try:
        return importlib.import_module("afmc_fm.phase07.provenance")
    except ModuleNotFoundError as error:
        raise AssertionError("Phase 0.7 execution provenance is not implemented") from error


def _identity() -> dict[str, object]:
    return {
        "execution_commit": "a" * 40,
        "phase07_spec_sha256": "1" * 64,
        "phase07_config_sha256": "2" * 64,
        "phase05_config_sha256": "3" * 64,
        "simulator_config_sha256": "4" * 64,
        "protocol_lock_sha256": "5" * 64,
        "phase07_plan_sha256": "6" * 64,
        "expected_cell_count": 200,
    }


def _payload(root: Path) -> dict[str, object]:
    return json.loads((root / "execution_provenance.json").read_text(encoding="utf-8"))


def test_failure_provenance_is_atomic_and_preserves_resume_history(tmp_path):
    module = _provenance_api()
    recorder = module.Phase07ExecutionProvenance(
        tmp_path,
        identity=_identity(),
        device="cuda",
        planned_cell_count=200,
    )

    recorder.start(completed_before=17)
    recorder.attempt("p07__cell018")
    error = RuntimeError("simulated CUDA failure")
    recorder.fail(completed_after=17, error=error)

    first = _payload(tmp_path)
    assert first["schema_version"] == 1
    assert first["phase"] == "phase07"
    assert first["identity"] == _identity()
    assert first["planned_cell_count"] == 200
    assert len(first["invocations"]) == 1
    failed = first["invocations"][0]
    assert failed["status"] == "failed"
    assert failed["completed_before_count"] == 17
    assert failed["completed_after_count"] == 17
    assert failed["last_attempted_cell_id"] == "p07__cell018"
    assert failed["failure"] == {
        "type": "RuntimeError",
        "message": "simulated CUDA failure",
    }
    assert isinstance(failed["started_at_utc"], str)
    assert isinstance(failed["ended_at_utc"], str)
    assert float(failed["wall_time_seconds"]) >= 0
    assert isinstance(failed["runtime"], dict)
    assert failed["device"] == "cuda"

    resumed = module.Phase07ExecutionProvenance(
        tmp_path,
        identity=_identity(),
        device="cuda",
        planned_cell_count=200,
    )
    resumed.start(completed_before=17)
    resumed.attempt("p07__cell018")
    resumed.finish(completed_after=200)

    final = _payload(tmp_path)
    assert len(final["invocations"]) == 2
    assert final["invocations"][0]["status"] == "failed"
    assert final["invocations"][1]["status"] == "complete"
    assert final["invocations"][1]["completed_before_count"] == 17
    assert final["invocations"][1]["completed_after_count"] == 200
    assert not list(tmp_path.glob("execution_provenance.json.*.tmp"))


def test_provenance_identity_drift_fails_closed(tmp_path):
    module = _provenance_api()
    recorder = module.Phase07ExecutionProvenance(
        tmp_path,
        identity=_identity(),
        device="cpu",
        planned_cell_count=200,
    )
    recorder.start(completed_before=0)
    recorder.finish(completed_after=200)

    changed = dict(_identity())
    changed["phase07_plan_sha256"] = "f" * 64
    drifted = module.Phase07ExecutionProvenance(
        tmp_path,
        identity=changed,
        device="cpu",
        planned_cell_count=200,
    )

    try:
        drifted.start(completed_before=0)
    except ValueError as error:
        assert "identity" in str(error)
    else:
        raise AssertionError("provenance identity drift must fail closed")
