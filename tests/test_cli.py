from pathlib import Path

import pandas as pd

from afmc_fm.cli import _experiment_from_yaml, main


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
        ]
    )
    assert rc == 0
    metrics = pd.read_csv(output / "metrics.csv")
    assert set(metrics["world"]) == {"smooth", "site_shift"}
    assert (output / "ablation_metrics.csv").exists()
    assert (output / "gate_summary.csv").exists()
