import numpy as np
import pytest

from afmc_fm.simulator.config import SimulatorConfig
from afmc_fm.simulator.dynamics import (
    apply_intervention_jump,
    sample_patient_parameters,
    simulate_latent_trajectory,
)


def test_same_seed_produces_same_latent_trajectory():
    config = SimulatorConfig(cohort_size=4, latent_dim=4, followup_days=90.0)
    rng_a = np.random.default_rng(11)
    params_a = sample_patient_parameters(config, rng_a)
    traj_a = simulate_latent_trajectory(config, params_a, rng_a)

    rng_b = np.random.default_rng(11)
    params_b = sample_patient_parameters(config, rng_b)
    traj_b = simulate_latent_trajectory(config, params_b, rng_b)

    np.testing.assert_allclose(traj_a.times, traj_b.times)
    np.testing.assert_allclose(traj_a.states, traj_b.states)


def test_patient_parameters_are_heterogeneous_across_draws():
    config = SimulatorConfig(cohort_size=4, latent_dim=4)
    rng = np.random.default_rng(3)
    a = sample_patient_parameters(config, rng)
    b = sample_patient_parameters(config, rng)
    assert not np.allclose(a.baseline_state, b.baseline_state)


def test_latent_trajectory_is_finite_and_monotonic_in_time():
    config = SimulatorConfig(cohort_size=4, latent_dim=4, followup_days=120.0)
    rng = np.random.default_rng(5)
    params = sample_patient_parameters(config, rng)
    trajectory = simulate_latent_trajectory(config, params, rng)
    assert np.all(np.diff(trajectory.times) > 0)
    assert np.isfinite(trajectory.states).all()
    assert trajectory.states.shape[1] == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latent_dim": 0},
        {"followup_days": 0.0},
        {"mean_event_interval_days": -1.0},
        {"observation_regime": "unknown"},
    ],
)
def test_simulator_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SimulatorConfig(**kwargs)


def test_intervention_jump_changes_state_without_mutating_input():
    config = SimulatorConfig(latent_dim=4)
    rng = np.random.default_rng(7)
    params = sample_patient_parameters(config, rng)
    state = np.zeros(4)
    jumped = apply_intervention_jump(state, strength=1.0, params=params, rng=rng)
    assert not np.allclose(jumped, state)
    np.testing.assert_array_equal(state, np.zeros(4))
