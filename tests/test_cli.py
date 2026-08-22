from pathlib import Path

from afmc_fm.cli import main


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
