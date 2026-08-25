import hashlib
import json
from pathlib import Path

import pytest

from afmc_fm import cli
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase05.protocol import freeze_candidate


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


def _write_successful_development_artifacts(output: Path) -> None:
    development = output / "development"
    development.mkdir(parents=True, exist_ok=True)
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


def _freeze(output: Path):
    config = load_phase05_config("configs/experiments/phase05.yaml")
    _write_successful_development_artifacts(output)
    return freeze_candidate(output, config)


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


def test_phase05_confirm_binds_frozen_candidate_before_any_fit(
    tmp_path: Path,
    monkeypatch,
):
    output = _locked_output(tmp_path)
    _freeze(output)
    config = load_phase05_config("configs/experiments/phase05.yaml")
    frozen_hash = hashlib.sha256(
        (output / "frozen_candidate.json").read_bytes()
    ).hexdigest()
    expected_bundles = {bundle.as_tuple() for bundle in config.confirmatory_bundles}
    development_bundles = {bundle.as_tuple() for bundle in config.development_bundles}

    def intercept_execution(jobs, *, store, **_kwargs):
        planned = tuple(jobs)
        assert (output / "confirmation" / "STARTED").is_file()
        assert len(planned) == 1260
        assert {job.shard.stage for job in planned} == {"confirmation"}
        assert {job.shard.world for job in planned} == set(config.target_worlds)
        observed_bundles = {
            job.shard.seed_bundle.as_tuple() for job in planned
        }
        assert observed_bundles == expected_bundles
        assert observed_bundles.isdisjoint(development_bundles)
        assert {job.frozen_candidate_hash for job in planned} == {frozen_hash}
        assert store.output == output
        raise RuntimeError("injected stop before confirmation fit")

    monkeypatch.setattr(cli, "run_phase05_jobs", intercept_execution)

    with pytest.raises(RuntimeError, match="injected stop before confirmation fit"):
        cli.main(_confirm_argv(output))
