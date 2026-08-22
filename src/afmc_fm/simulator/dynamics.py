from dataclasses import dataclass

import numpy as np

from afmc_fm.simulator.config import SimulatorConfig


@dataclass(frozen=True)
class PatientParameters:
    baseline_state: np.ndarray
    progression_scale: np.ndarray
    response_scale: np.ndarray
    noise_scale: float
    site_id: int


@dataclass(frozen=True)
class LatentTrajectory:
    times: np.ndarray
    states: np.ndarray


def sample_patient_parameters(
    config: SimulatorConfig,
    rng: np.random.Generator,
) -> PatientParameters:
    dimension = config.latent_dim
    return PatientParameters(
        baseline_state=rng.normal(0.0, 0.7 * config.heterogeneity_scale, size=dimension),
        progression_scale=rng.lognormal(0.0, 0.2 * config.heterogeneity_scale, size=dimension),
        response_scale=rng.lognormal(0.0, 0.25 * config.heterogeneity_scale, size=dimension),
        noise_scale=float(config.process_noise * rng.lognormal(0.0, 0.15)),
        site_id=int(rng.integers(config.n_sites)),
    )


def _drift(state: np.ndarray, progression: np.ndarray, t_days: float) -> np.ndarray:
    coupling = 0.08 * np.tanh(np.roll(state, 1) - state)
    seasonal = 0.01 * np.sin(t_days / 30.0)
    return -0.03 * progression * state + coupling + seasonal


def simulate_latent_trajectory(
    config: SimulatorConfig,
    params: PatientParameters,
    rng: np.random.Generator,
) -> LatentTrajectory:
    times = [0.0]
    states = [np.array(params.baseline_state, dtype=float, copy=True)]
    current_time = 0.0

    while current_time < config.followup_days:
        interval = float(rng.exponential(config.mean_event_interval_days))
        next_time = min(current_time + max(interval, 1e-6), config.followup_days)
        dt = next_time - current_time
        if dt <= 0:
            break

        state = states[-1]
        next_state = state + dt * _drift(state, params.progression_scale, current_time) / 30.0
        next_state += (
            np.sqrt(max(dt, 1e-6) / 30.0)
            * params.noise_scale
            * rng.normal(size=config.latent_dim)
        )
        times.append(next_time)
        states.append(next_state)
        current_time = next_time

    return LatentTrajectory(times=np.asarray(times), states=np.asarray(states))


def apply_intervention_jump(
    state: np.ndarray,
    strength: float,
    params: PatientParameters,
    rng: np.random.Generator,
) -> np.ndarray:
    direction = np.zeros_like(state)
    direction[0] = 1.0
    if state.size > 1:
        direction[1] = -0.35
    noise = 0.02 * rng.normal(size=state.shape)
    return state + strength * params.response_scale * direction + noise
