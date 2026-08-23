import json
from pathlib import Path

import pandas as pd
import pytest

from afmc_fm.cli import _experiment_from_yaml, _parser, main


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
