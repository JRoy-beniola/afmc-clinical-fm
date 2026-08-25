import json
from pathlib import Path

import pytest

from afmc_fm import cli
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import load_phase05_config


def _locked_output(tmp_path: Path) -> Path:
    config = load_phase05_config("configs/experiments/phase05.yaml")
    lock = {"phase05_config_sha256": canonical_config_hash(config)}
    output = tmp_path / "phase05"
    output.mkdir(parents=True)
    data = (
        json.dumps(lock, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    (output / "protocol_lock.json").write_bytes(data)
    return output


def _confirm_argv(output: Path) -> list[str]:
    return [
        "phase05",
        "confirm",
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


def test_phase05_confirm_requires_frozen_candidate_before_start(tmp_path: Path):
    output = _locked_output(tmp_path)

    with pytest.raises(RuntimeError, match="frozen_candidate"):
        cli.main(_confirm_argv(output))

    assert not (output / "confirmation" / "STARTED").exists()
