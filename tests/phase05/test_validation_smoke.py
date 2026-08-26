from pathlib import Path

import numpy as np
import pytest
import torch

import afmc_fm.phase05.runner as phase05_runner
from afmc_fm.config import load_yaml
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.runner import prepare_phase05_cohort, run_phase05_variant
from afmc_fm.simulator.cohort import simulate_world
from afmc_fm.simulator.config import SimulatorConfig

_VALIDATION_CASES = (
    ("smooth", "gated", "none", "deterministic"),
    ("smooth", "time_scaled", "none", "deterministic"),
    ("jumps", "time_scaled", "residual", "deterministic"),
    ("informative_observation", "time_scaled", "residual", "decoupled"),
)


def _smoke_simulator_config() -> tuple[SimulatorConfig, int]:
    raw = dict(load_yaml(Path("configs/simulator/smoke.yaml")))
    seed = int(raw.pop("seed"))
    return SimulatorConfig(**raw), seed


def test_phase05_cpu_validation_smoke_one_bundle_n5_n10_locked_mechanisms():
    simulator_config, _ = _smoke_simulator_config()
    config = Phase05Config(max_epochs=1, patience=1)
    bundle = config.development_bundles[0]
    device = torch.device("cpu")

    prepared_by_world = {
        world: prepare_phase05_cohort(
            simulate_world(world, simulator_config, seed=bundle.cohort_seed)
        )
        for world in {case[0] for case in _VALIDATION_CASES}
    }

    observed = []
    for world, flow_mode, jump_mode, uncertainty_mode in _VALIDATION_CASES:
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_phase05_cuda_validation_smoke_four_real_fits_use_cuda(monkeypatch):
    simulator_config, _ = _smoke_simulator_config()
    config = Phase05Config(max_epochs=1, patience=1)
    bundle = config.development_bundles[0]
    device = torch.device("cuda")
    prepared_by_world = {
        world: prepare_phase05_cohort(
            simulate_world(world, simulator_config, seed=bundle.cohort_seed)
        )
        for world in {case[0] for case in _VALIDATION_CASES}
    }
    original_fit = phase05_runner.fit_phase05_model
    fitted_parameter_devices = []

    def verifying_fit(model, train, validation, fit_config, fit_device):
        assert fit_device.type == "cuda"
        fitted = original_fit(model, train, validation, fit_config, fit_device)
        parameter_devices = {parameter.device.type for parameter in fitted.parameters()}
        assert parameter_devices == {"cuda"}
        fitted_parameter_devices.append(parameter_devices)
        return fitted

    monkeypatch.setattr(phase05_runner, "fit_phase05_model", verifying_fit)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    observed = []
    for world, flow_mode, jump_mode, uncertainty_mode in _VALIDATION_CASES:
        result = run_phase05_variant(
            prepared_by_world[world],
            config,
            world=world,
            seed_bundle=bundle,
            n_train=5,
            flow_mode=flow_mode,
            jump_mode=jump_mode,
            uncertainty_mode=uncertainty_mode,
            device=device,
            stage="cuda_validation_smoke",
        )
        assert set(result["backend"]) == {"torch"}
        defined = result.loc[result["metric"] != "event_roc_auc", "value"]
        assert np.isfinite(defined.to_numpy(dtype=float)).all()
        if uncertainty_mode == "decoupled":
            probabilistic = result.loc[
                result["metric"].isin(("nll", "coverage_90")), "value"
            ]
            assert len(probabilistic) == 2
            assert np.isfinite(probabilistic.to_numpy(dtype=float)).all()
        observed.append((world, flow_mode, jump_mode, uncertainty_mode))

    torch.cuda.synchronize(device)
    assert len(observed) == len(_VALIDATION_CASES) == 4
    assert fitted_parameter_devices == [{"cuda"}] * 4
    assert torch.cuda.max_memory_allocated(device) > 0
