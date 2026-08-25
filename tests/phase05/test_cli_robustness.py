import hashlib
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


def _finalize_confirmation_fixture(output: Path) -> Path:
    confirmation = output / "confirmation"
    confirmation.mkdir(parents=True, exist_ok=True)
    (confirmation / "STARTED").write_text("fixture\n", encoding="utf-8")
    gate = confirmation / "primary_gate_summary.csv"
    gate.write_text(
        "world,headline_passed\n"
        "smooth,true\n"
        "jumps,true\n"
        "informative_observation,true\n",
        encoding="utf-8",
    )
    complete = output / "stages" / "confirmation" / "COMPLETE"
    complete.parent.mkdir(parents=True, exist_ok=True)
    complete.write_text("fixture\n", encoding="utf-8")
    return gate


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


def test_phase05_robustness_binds_exact_stage_iv_plan_before_any_fit(
    tmp_path: Path,
    monkeypatch,
):
    output = _locked_frozen_output(tmp_path)
    gate = _finalize_confirmation_fixture(output)
    gate_before = gate.read_bytes()
    config = load_phase05_config("configs/experiments/phase05.yaml")
    frozen_hash = hashlib.sha256(
        (output / "frozen_candidate.json").read_bytes()
    ).hexdigest()
    expected_bundles = {bundle.as_tuple() for bundle in config.confirmatory_bundles}

    def intercept_execution(jobs, *, store, **_kwargs):
        planned = tuple(jobs)
        assert len(planned) == 360
        assert {job.shard.stage for job in planned} == {"robustness"}
        assert {job.shard.world for job in planned} == set(config.robustness_worlds)
        assert {
            job.shard.seed_bundle.as_tuple() for job in planned
        } == expected_bundles
        assert {job.n_train for job in planned} == set(config.train_sizes)
        assert {job.model for job in planned} == {
            "phase05_candidate",
            "matched_gru",
            "matched_representation_mlp",
        }
        assert {job.frozen_candidate_hash for job in planned} == {frozen_hash}
        assert store.output == output
        assert gate.read_bytes() == gate_before
        raise RuntimeError("injected stop before robustness fit")

    monkeypatch.setattr(cli, "run_phase05_jobs", intercept_execution)

    with pytest.raises(RuntimeError, match="injected stop before robustness fit"):
        cli.main(_robustness_argv(output))

    assert gate.read_bytes() == gate_before
