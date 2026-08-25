import json
from pathlib import Path

import pytest

from afmc_fm import cli
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase05.protocol import freeze_candidate


def _locked_frozen_output(tmp_path: Path) -> Path:
    config = load_phase05_config("configs/experiments/phase05.yaml")
    output = tmp_path / "phase05"
    output.mkdir(parents=True)
    lock = {"phase05_config_sha256": canonical_config_hash(config)}
    (output / "protocol_lock.json").write_text(
        json.dumps(lock, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    development = output / "development"
    development.mkdir(parents=True)
    (development / "flow_gate.csv").write_text(
        "candidate,passed,selected,trainable_parameters\n"
        "gated,true,false,6000\n"
        "time_scaled,true,true,5900\n",
        encoding="utf-8",
    )
    (development / "jump_gate.csv").write_text(
        "candidate,passed,selected,trainable_parameters\n"
        "gru,true,false,6500\n"
        "residual,true,true,6400\n",
        encoding="utf-8",
    )
    (development / "uncertainty_gate.csv").write_text(
        "candidate,selected,trainable_parameters\n"
        "joint,false,7000\n"
        "decoupled,true,6900\n"
        "deterministic,false,6400\n",
        encoding="utf-8",
    )
    (development / "representation_timing_audit.csv").write_text(
        "strict_history,mean_delta_inclusive_minus_strict\ntrue,-0.03\n",
        encoding="utf-8",
    )
    freeze_candidate(output, config)
    return output


def _robustness_argv(output: Path) -> list[str]:
    return [
        "phase05",
        "robustness",
        "--sim-config",
        "configs/simulator/smoke.yaml",
        "--exp-config",
        "configs/experiments/phase05.yaml",
        "--output",
        str(output),
        "--device",
        "cpu",
        "--workers",
        "1",
    ]


def test_phase05_robustness_requires_finalized_confirmation_before_execution(
    tmp_path: Path,
):
    output = _locked_frozen_output(tmp_path)

    with pytest.raises(RuntimeError, match="finalized confirmation"):
        cli.main(_robustness_argv(output))

    assert not (output / "stages" / "robustness" / "COMPLETE").exists()
