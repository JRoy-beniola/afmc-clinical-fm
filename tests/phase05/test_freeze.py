import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import pytest

from afmc_fm.phase05.baselines import build_capacity_audit
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.development import select_flow
from afmc_fm.phase05.protocol import freeze_candidate


def _write_gate(output: Path, name: str, rows: str) -> None:
    development = output / "development"
    development.mkdir(parents=True, exist_ok=True)
    (development / name).write_text(rows, encoding="utf-8")


def _failure_payload(output: Path) -> dict[str, object]:
    return json.loads((output / "development_failure.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _flow_selection_frame() -> pd.DataFrame:
    rows = []
    for variant, value, parameters in (
        ("none__none__deterministic", 1.0, 5000),
        ("gated__none__deterministic", 0.960, 6000),
        ("time_scaled__none__deterministic", 0.956, 5900),
    ):
        for seed_index in range(1, 6):
            for n_train in (5, 10, 20, 40):
                rows.append(
                    {
                        "world": "smooth",
                        "metric": "mae",
                        "variant": variant,
                        "cohort_seed": 400 + seed_index,
                        "subset_seed": 500 + seed_index,
                        "model_seed": 600 + seed_index,
                        "n_train": n_train,
                        "value": value,
                        "trainable_parameters": parameters,
                    }
                )
    return pd.DataFrame(rows)


def test_development_gate_marks_the_selected_candidate_when_multiple_candidates_pass():
    result = select_flow(
        _flow_selection_frame(),
        locked_min_relative_effect=0.02,
    )

    gate = result["gate_table"].set_index("candidate")
    assert bool(gate.loc["gated", "passed"]) is True
    assert bool(gate.loc["time_scaled", "passed"]) is True
    assert bool(gate.loc["gated", "selected"]) is False
    assert bool(gate.loc["time_scaled", "selected"]) is True
    assert result["selected"] == "time_scaled"


@pytest.mark.parametrize(
    ("gate_name", "rows"),
    [
        (
            "flow",
            "candidate,passed,selected,trainable_parameters\n"
            "gated,true,false,6000\n"
            "time_scaled,false,true,5900\n",
        ),
        (
            "jump",
            "candidate,passed,selected,trainable_parameters\n"
            "gru,true,false,6500\n"
            "residual,false,true,6400\n",
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


def test_successful_freeze_records_architecture_capacity_controls_and_provenance(tmp_path):
    output = tmp_path / "phase05"
    _write_success_inputs(output)
    config = Phase05Config()

    frozen = freeze_candidate(output, config)

    assert frozen.flow_mode == "time_scaled"
    assert frozen.jump_mode == "residual"
    assert frozen.uncertainty_mode == "decoupled"
    assert frozen.strict_history is True
    assert frozen.state_dim == 24
    assert frozen.time_scale_days == 30.0
    assert frozen.jump_eligible_event_codes == ("SYNTHETIC_INTERVENTION",)
    assert frozen.assimilation_semantics == "phase0_grucell_unchanged"
    assert frozen.trainable_parameters == 6900

    expected_audit = build_capacity_audit(
        target_parameters=6900,
        value_dim=3,
        event_dim=3,
        representation_input_dim=19,
    ).set_index("control")
    assert frozen.matched_gru_hidden_size == int(
        expected_audit.loc["matched_gru", "hidden_size"]
    )
    assert frozen.matched_gru_parameters == int(
        expected_audit.loc["matched_gru", "actual_parameters"]
    )
    assert frozen.matched_mlp_hidden_size == int(
        expected_audit.loc["matched_representation_mlp", "hidden_size"]
    )
    assert frozen.matched_mlp_parameters == int(
        expected_audit.loc["matched_representation_mlp", "actual_parameters"]
    )
    assert frozen.protocol_lock_sha256 == _sha256(output / "protocol_lock.json")
    assert frozen.development_artifact_hashes == {
        name: _sha256(output / "development" / name)
        for name in (
            "flow_gate.csv",
            "jump_gate.csv",
            "uncertainty_gate.csv",
            "representation_timing_audit.csv",
        )
    }

    persisted = json.loads((output / "frozen_candidate.json").read_text(encoding="utf-8"))
    assert persisted == frozen.to_dict()
    assert not (output / "development_failure.json").exists()


def test_identical_refreeze_is_idempotent_and_does_not_rewrite_artifact(tmp_path):
    output = tmp_path / "phase05"
    _write_success_inputs(output)
    config = Phase05Config()
    first = freeze_candidate(output, config)
    frozen_path = output / "frozen_candidate.json"
    before_bytes = frozen_path.read_bytes()
    before_mtime = frozen_path.stat().st_mtime_ns
    time.sleep(0.01)

    second = freeze_candidate(output, config)

    assert second == first
    assert frozen_path.read_bytes() == before_bytes
    assert frozen_path.stat().st_mtime_ns == before_mtime


def test_refreeze_rejects_changed_decision_input_and_preserves_original_candidate(tmp_path):
    output = tmp_path / "phase05"
    _write_success_inputs(output)
    config = Phase05Config()
    freeze_candidate(output, config)
    frozen_path = output / "frozen_candidate.json"
    before_bytes = frozen_path.read_bytes()

    _write_gate(
        output,
        "representation_timing_audit.csv",
        "strict_history,mean_delta_inclusive_minus_strict\ntrue,-0.04\n",
    )

    with pytest.raises(RuntimeError, match="frozen candidate conflicts"):
        freeze_candidate(output, config)

    assert frozen_path.read_bytes() == before_bytes
