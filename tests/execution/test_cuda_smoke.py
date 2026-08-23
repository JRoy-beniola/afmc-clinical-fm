import numpy as np
import pytest
import torch

from afmc_fm.experiments.runner import ExperimentConfig, run_low_n_benchmark
from afmc_fm.simulator.cohort import simulate_cohort
from afmc_fm.simulator.config import SimulatorConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_neural_models_complete_one_epoch_on_cuda():
    cohort = simulate_cohort(
        SimulatorConfig(
            cohort_size=36,
            followup_days=90.0,
            intervention_rate=0.15,
        ),
        seed=31,
    )
    config = ExperimentConfig(
        train_sizes=(5,),
        subset_seeds=(2,),
        model_seeds=(3,),
        max_epochs=1,
        patience=1,
    )

    results = run_low_n_benchmark(
        cohort,
        config,
        model_names=("gru_from_scratch", "flow_jump"),
        device=torch.device("cuda"),
    )

    assert set(results["model"]) == {"gru_from_scratch", "flow_jump"}
    assert np.isfinite(results["value"]).all()
