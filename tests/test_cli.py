import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from afmc_fm.cli import _experiment_from_yaml, _parser, main
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import load_phase05_config

_PHASE05_SPEC_COMMIT = "67c6662c64e69606bcbd3eca2bc8139548013051"
_PHASE0_EXECUTION_SHA = "d6f105eee73fcb8e9cc5987d292b1bb98a687382"


def test_simulate_cli_writes_synthetic_outputs(tmp_path: Path):
    rc = main([
        "simulate",
        "--config", "configs/simulator/smoke.yaml",
        "--output", str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / "events.csv").exists()
    assert (tmp_path / "latent_truth.npz").exists()
    assert (tmp_path / "simulation_manifest.json").exists()


def test_experiment_config_honors_world_model_ablation_and_site_fields(tmp_path: Path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
train_sizes: [5]
cohort_seeds: [11]
subset_seeds: [12]
model_seeds: [13]
worlds: [smooth, site_shift]
models: [representation_linear, flow_jump]
ablations: [none, no_flow]
train_site: 1
test_sites: [1, 0]
max_epochs: 1
patience: 1
""".strip()
    )
    config = _experiment_from_yaml(config_path)
    assert config.worlds == ("smooth", "site_shift")
    assert config.models == ("representation_linear", "flow_jump")
    assert config.ablations == ("none", "no_flow")
    assert config.train_site == 1
    assert config.test_sites == (1, 0)


def test_benchmark_cli_runs_configured_worlds_and_writes_gate_outputs(tmp_path: Path):
    simulator_path = tmp_path / "simulator.yaml"
    simulator_path.write_text(
        """
seed: 7
cohort_size: 36
n_sites: 2
latent_dim: 4
followup_days: 30.0
mean_event_interval_days: 7.0
observation_regime: mnar
""".strip()
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """
train_sizes: [5]
cohort_seeds: [11]
subset_seeds: [12]
model_seeds: [13]
worlds: [smooth, site_shift]
models: [engineered_linear]
ablations: [none]
max_epochs: 1
patience: 1
""".strip()
    )
    output = tmp_path / "benchmark"
    rc = main(
        [
            "benchmark",
            "--sim-config",
            str(simulator_path),
            "--exp-config",
            str(experiment_path),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--workers",
            "1",
            "--fail-fast",
        ]
    )
    assert rc == 0
    metrics = pd.read_csv(output / "metrics.csv")
    assert set(metrics["world"]) == {"smooth", "site_shift"}
    assert (output / "ablation_metrics.csv").exists()
    assert (output / "gate_summary.csv").exists()
    assert (output / "learning_curves.png").exists()
    assert len(list(output.glob("shards/*/cells/*.json"))) == 2
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["resolved_device"] == "cpu"
    assert manifest["worker_count"] == 1
    assert manifest["expected_shard_count"] == 2
    assert manifest["completed_shard_count"] == 2
    assert manifest["expected_cell_count"] == 2
    assert manifest["completed_cell_count"] == 2
    assert manifest["failed_shard_count"] == manifest["failed_cell_count"] == 0


def test_benchmark_parser_exposes_execution_options_with_compatible_defaults():
    defaults = _parser().parse_args(
        [
            "benchmark",
            "--sim-config",
            "sim.yaml",
            "--exp-config",
            "exp.yaml",
            "--output",
            "output",
        ]
    )
    selected = _parser().parse_args(
        [
            "benchmark",
            "--sim-config",
            "sim.yaml",
            "--exp-config",
            "exp.yaml",
            "--output",
            "output",
            "--device",
            "cuda",
            "--workers",
            "3",
            "--resume",
            "--fail-fast",
        ]
    )

    assert (defaults.device, defaults.workers, defaults.resume, defaults.fail_fast) == (
        "auto",
        1,
        False,
        False,
    )
    assert (selected.device, selected.workers, selected.resume, selected.fail_fast) == (
        "cuda",
        3,
        True,
        True,
    )


def test_benchmark_cli_explicit_cuda_unavailable_hard_fails_without_output(
    tmp_path,
    monkeypatch,
):
    simulator_path = tmp_path / "simulator.yaml"
    simulator_path.write_text("cohort_size: 30\nfollowup_days: 45.0\n")
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """train_sizes: [5]
cohort_seeds: [17]
subset_seeds: [23]
model_seeds: [31]
worlds: [smooth]
models: [engineered_linear]
ablations: [none]
max_epochs: 1
patience: 1
"""
    )
    output = tmp_path / "run"
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested but is not available"):
        main(
            [
                "benchmark",
                "--sim-config",
                str(simulator_path),
                "--exp-config",
                str(experiment_path),
                "--output",
                str(output),
                "--device",
                "cuda",
            ]
        )

    assert not output.exists()


def test_phase05_parser_exposes_locked_stage_specific_arguments():
    parser = _parser()

    calibrate = parser.parse_args(
        [
            "phase05",
            "calibrate",
            "--phase0-metrics",
            "phase0.csv",
            "--exp-config",
            "phase05.yaml",
            "--output",
            "run",
        ]
    )
    freeze = parser.parse_args(
        [
            "phase05",
            "freeze",
            "--exp-config",
            "phase05.yaml",
            "--output",
            "run",
        ]
    )
    report = parser.parse_args(["phase05", "report", "--output", "run"])

    for args, stage in (
        (calibrate, "calibrate"),
        (freeze, "freeze"),
        (report, "report"),
    ):
        assert args.phase05_command == stage
        assert not hasattr(args, "device")
        assert not hasattr(args, "workers")
        assert not hasattr(args, "resume")
        assert not hasattr(args, "fail_fast")

    for stage in ("develop", "confirm", "robustness"):
        defaults = parser.parse_args(
            [
                "phase05",
                stage,
                "--sim-config",
                "sim.yaml",
                "--exp-config",
                "phase05.yaml",
                "--output",
                "run",
            ]
        )
        selected = parser.parse_args(
            [
                "phase05",
                stage,
                "--sim-config",
                "sim.yaml",
                "--exp-config",
                "phase05.yaml",
                "--output",
                "run",
                "--device",
                "cuda",
                "--workers",
                "3",
                "--resume",
                "--fail-fast",
            ]
        )
        assert defaults.phase05_command == stage
        assert (defaults.device, defaults.workers, defaults.resume, defaults.fail_fast) == (
            "auto",
            1,
            False,
            False,
        )
        assert (selected.device, selected.workers, selected.resume, selected.fail_fast) == (
            "cuda",
            3,
            True,
            True,
        )


def _phase05_calibration_metrics() -> pd.DataFrame:
    rows = []
    worlds = ("smooth", "jumps", "informative_observation")
    train_sizes = (5, 10, 20, 40)
    models = ("flow_jump", "representation_linear", "gru_from_scratch")
    bundles = tuple((100 + index, 200 + index, 300 + index) for index in range(1, 6))
    for world_index, world in enumerate(worlds):
        for n_index, n_train in enumerate(train_sizes):
            for model_index, model in enumerate(models):
                base = 1.0 + 0.1 * world_index + 0.01 * n_index + 0.001 * model_index
                for seed_index, bundle in enumerate(bundles):
                    rows.append(
                        {
                            "benchmark": "low_n",
                            "site_or_shift": "all",
                            "ablation": "none",
                            "metric": "mae",
                            "world": world,
                            "n_train": n_train,
                            "model": model,
                            "cohort_seed": bundle[0],
                            "subset_seed": bundle[1],
                            "model_seed": bundle[2],
                            "value": base * (1.0 + 0.01 * (seed_index - 2)),
                        }
                    )
    return pd.DataFrame(rows)


def test_phase05_calibrate_persists_locked_protocol_and_is_idempotent(tmp_path: Path):
    metrics_path = tmp_path / "phase0_metrics.csv"
    _phase05_calibration_metrics().to_csv(metrics_path, index=False)
    output = tmp_path / "phase05"
    argv = [
        "phase05",
        "calibrate",
        "--phase0-metrics",
        str(metrics_path),
        "--exp-config",
        "configs/experiments/phase05.yaml",
        "--output",
        str(output),
    ]

    assert main(argv) == 0
    lock_path = output / "protocol_lock.json"
    assert lock_path.is_file()
    before = lock_path.read_bytes()
    lock = json.loads(before)
    assert lock["phase0_metrics_sha256"] == hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    assert lock["spec_commit"] == _PHASE05_SPEC_COMMIT
    assert lock["phase0_execution_sha"] == _PHASE0_EXECUTION_SHA
    assert lock["phase05_config_sha256"] == canonical_config_hash(
        load_phase05_config("configs/experiments/phase05.yaml")
    )

    assert main(argv) == 0
    assert lock_path.read_bytes() == before


def _write_successful_phase05_development_artifacts(output: Path) -> None:
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


def test_phase05_freeze_consumes_completed_development_and_is_idempotent(tmp_path: Path):
    metrics_path = tmp_path / "phase0_metrics.csv"
    _phase05_calibration_metrics().to_csv(metrics_path, index=False)
    output = tmp_path / "phase05"
    assert main(
        [
            "phase05",
            "calibrate",
            "--phase0-metrics",
            str(metrics_path),
            "--exp-config",
            "configs/experiments/phase05.yaml",
            "--output",
            str(output),
        ]
    ) == 0
    _write_successful_phase05_development_artifacts(output)
    argv = [
        "phase05",
        "freeze",
        "--exp-config",
        "configs/experiments/phase05.yaml",
        "--output",
        str(output),
    ]

    assert main(argv) == 0
    frozen_path = output / "frozen_candidate.json"
    assert frozen_path.is_file()
    before = frozen_path.read_bytes()
    frozen = json.loads(before)
    assert frozen["flow_mode"] == "time_scaled"
    assert frozen["jump_mode"] == "residual"
    assert frozen["uncertainty_mode"] == "decoupled"
    assert frozen["strict_history"] is True
    assert frozen["trainable_parameters"] == 6900

    assert main(argv) == 0
    assert frozen_path.read_bytes() == before
