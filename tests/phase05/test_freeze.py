import json
from pathlib import Path

import pytest

from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.protocol import freeze_candidate


def _write_gate(output: Path, name: str, rows: str) -> None:
    development = output / "development"
    development.mkdir(parents=True, exist_ok=True)
    (development / name).write_text(rows, encoding="utf-8")


def _failure_payload(output: Path) -> dict[str, object]:
    return json.loads((output / "development_failure.json").read_text(encoding="utf-8"))


def test_freeze_refuses_failed_flow_gate_and_records_machine_readable_failure(tmp_path):
    output = tmp_path / "phase05"
    _write_gate(
        output,
        "flow_gate.csv",
        "candidate,passed,trainable_parameters\n"
        "gated,false,6000\n"
        "time_scaled,false,5900\n",
    )

    with pytest.raises(RuntimeError, match="flow gate failed"):
        freeze_candidate(output, Phase05Config())

    payload = _failure_payload(output)
    assert payload["status"] == "development_failed"
    assert payload["failed_gates"] == ["flow"]
    assert not (output / "frozen_candidate.json").exists()


def test_freeze_refuses_failed_jump_gate_after_passing_flow(tmp_path):
    output = tmp_path / "phase05"
    _write_gate(
        output,
        "flow_gate.csv",
        "candidate,passed,trainable_parameters\n"
        "gated,false,6000\n"
        "time_scaled,true,5900\n",
    )
    _write_gate(
        output,
        "jump_gate.csv",
        "candidate,passed,trainable_parameters\n"
        "gru,false,6500\n"
        "residual,false,6400\n",
    )

    with pytest.raises(RuntimeError, match="jump gate failed"):
        freeze_candidate(output, Phase05Config())

    payload = _failure_payload(output)
    assert payload["status"] == "development_failed"
    assert payload["failed_gates"] == ["jump"]
    assert not (output / "frozen_candidate.json").exists()
