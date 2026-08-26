from pathlib import Path

import numpy as np
import torch

from afmc_fm.config import load_yaml
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.runner import prepare_phase05_cohort, run_phase05_variant
from afmc_fm.simulator.cohort import simulate_world
from afmc_fm.simulator.config import SimulatorConfig


def _smoke_simulator_config() -> tuple[SimulatorConfig, int]:
    raw = dict(load_yaml(Path("configs/simulator/smoke.yaml")))
    seed = int(raw.pop("seed"))
    return SimulatorConfig(**raw), seed


def test_phase05_cpu_validation_smoke_one_bundle_n5_n10_locked_mechanisms():
    simulator_config, _ = _smoke_simulator_config()
    config = Phase05Config(max_epochs=1, patience=1)
    bundle = config.development_bundles[0]
    device = torch.device("cpu")

    cases = (
        ("smooth", "gated", "none", "deterministic"),
        ("smooth", "time_scaled", "none", "deterministic"),
        ("jumps", "time_scaled", "residual", "deterministic"),
        ("informative_observation", "time_scaled", "residual", "decoupled"),
    )
    prepared_by_world = {
        world: prepare_phase05_cohort(
            simulate_world(world, simulator_config, seed=bundle.cohort_seed)
        )
        for world in {case[0] for case in cases}
    }

    observed = []
    for world, flow_mode, jump_mode, uncertainty_mode in cases:
        for n_train in (5, 10):
            result = run_phase05_variant(
                prepared_by_world[world],
                config,
                world=world,
                seed_bundle=bundle,
                n_train=n_train,
                flow_mode=flow_mode,
                jump_mode=jump_mode,
                uncertainty_mode=uncertainty_mode,
                device=device,
                stage="validation_smoke",
            )
            assert not result.empty
            assert set(result["backend"]) == {"torch"}
            assert set(result["n_train"]) == {n_train}
            assert set(result["variant"]) == {
                f"{flow_mode}__{jump_mode}__{uncertainty_mode}"
            }
            defined = result.loc[result["metric"] != "event_roc_auc", "value"]
            assert np.isfinite(defined.to_numpy(dtype=float)).all()
            if uncertainty_mode == "decoupled":
                probabilistic = result.loc[
                    result["metric"].isin(("nll", "coverage_90")), "value"
                ]
                assert len(probabilistic) == 2
                assert np.isfinite(probabilistic.to_numpy(dtype=float)).all()
            observed.append((world, flow_mode, jump_mode, uncertainty_mode, n_train))

    assert len(observed) == 8
