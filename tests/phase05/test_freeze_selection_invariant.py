from pathlib import Path

import pytest

from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.protocol import freeze_candidate


def _write_gate(output: Path, name: str, rows: str) -> None:
    development = output / "development"
    development.mkdir(parents=True, exist_ok=True)
    (development / name).write_text(rows, encoding="utf-8")


def _write_success_inputs(output: Path) -> None:
    _write_gate(
        output,
        "flow_gate.csv",
        "candidate,passed,selected,trainable_parameters\n"
        "gated,true,false,6000\n"
        "time_scaled,true,true,5900\n",
    )
    _write_gate(
        output,
        "jump_gate.csv",
        "candidate,passed,selected,trainable_parameters\n"
        "gru,true,false,6500\n"
        "residual,true,true,6400\n",
    )
    _write_gate(
        output,
        "uncertainty_gate.csv",
        "candidate,selected,trainable_parameters\n"
        "joint,false,7000\n"
        "decoupled,true,6900\n"
        "deterministic,false,6400\n",
    )
    _write_gate(
        output,
        "representation_timing_audit.csv",
        "strict_history,mean_delta_inclusive_minus_strict\ntrue,-0.03\n",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol_lock.json").write_text(
        '{"schema_version":1,"locked_min_relative_effect":0.02}\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("gate_name", "rows"),
    [
        (
            "flow",
            (
                "candidate,passed,selected,trainable_parameters\n"
                "gated,true,false,6000\n"
                "time_scaled,false,true,5900\n"
            ),
        ),
        (
            "jump",
            (
                "candidate,passed,selected,trainable_parameters\n"
                "gru,true,false,6500\n"
                "residual,false,true,6400\n"
            ),
        ),
    ],
)
def test_freeze_rejects_selected_mechanism_that_did_not_pass(tmp_path, gate_name, rows):
    output = tmp_path / "phase05"
    _write_success_inputs(output)
    _write_gate(output, f"{gate_name}_gate.csv", rows)

    with pytest.raises(RuntimeError, match=rf"selected {gate_name} candidate must have passed"):
        freeze_candidate(output, Phase05Config())

    assert not (output / "frozen_candidate.json").exists()
