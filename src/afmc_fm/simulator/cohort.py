from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import numpy as np

from afmc_fm.schema.events import ClinicalEvent, EventType, PatientTimeline
from afmc_fm.simulator.config import SimulatorConfig
from afmc_fm.simulator.dynamics import (
    LatentTrajectory,
    PatientParameters,
    apply_intervention_jump,
    sample_patient_parameters,
    simulate_latent_trajectory,
)
from afmc_fm.simulator.observation import emit_labs, observation_probability

_ORIGIN = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class SimulatedPatient:
    patient_id: str
    site_id: int
    parameters: PatientParameters
    latent: LatentTrajectory
    timeline: PatientTimeline


@dataclass(frozen=True)
class SimulatedCohort:
    patients: list[SimulatedPatient]
    config: SimulatorConfig
    seed: int


def _intervention_probability(state: np.ndarray, base_rate: float) -> float:
    state_factor = 1.0 + 0.25 * np.tanh(float(state[0]))
    return float(np.clip(base_rate * state_factor, 0.0, 0.95))


def simulate_patient(
    patient_id: str,
    config: SimulatorConfig,
    rng: np.random.Generator,
) -> SimulatedPatient:
    parameters = sample_patient_parameters(config, rng)
    initial_latent = simulate_latent_trajectory(config, parameters, rng)
    states = np.array(initial_latent.states, copy=True)
    if config.hidden_regime_switch and len(states) > 2:
        midpoint = len(states) // 2
        regime_offset = np.linspace(0.0, 0.6, len(states) - midpoint)[:, None]
        states[midpoint:] += regime_offset
    events: list[ClinicalEvent] = []
    previous_observed_value: float | None = None

    for index, elapsed_days in enumerate(initial_latent.times):
        state = states[index]
        timestamp = _ORIGIN + timedelta(days=float(elapsed_days))

        if rng.random() < _intervention_probability(state, config.intervention_rate):
            strength = float(rng.uniform(0.5, 1.5))
            jumped_state = apply_intervention_jump(state, strength, parameters, rng)
            shift = jumped_state - state
            effect_index = index + 1 if config.delayed_intervention_effect else index
            states[effect_index:] += shift
            state = states[index]
            events.append(
                ClinicalEvent(
                    patient_id=patient_id,
                    start_time=timestamp,
                    code="SYNTHETIC_INTERVENTION",
                    value=strength,
                    unit="arb",
                    event_type=EventType.INTERVENTION,
                    source="synthetic",
                    metadata={"site_id": parameters.site_id},
                )
            )

        candidate_labs = emit_labs(state, rng, config.measurement_noise)
        probability = observation_probability(
            config.observation_regime,
            state,
            previous_observed_value,
            parameters.site_id,
        )
        for code, value in candidate_labs.items():
            if rng.random() >= probability:
                continue
            events.append(
                ClinicalEvent(
                    patient_id=patient_id,
                    start_time=timestamp,
                    code=code,
                    value=value,
                    unit="arb",
                    event_type=EventType.OBSERVATION,
                    source="synthetic",
                    metadata={"site_id": parameters.site_id},
                )
            )
            previous_observed_value = value

    latent = LatentTrajectory(times=np.array(initial_latent.times, copy=True), states=states)
    return SimulatedPatient(
        patient_id=patient_id,
        site_id=parameters.site_id,
        parameters=parameters,
        latent=latent,
        timeline=PatientTimeline(patient_id=patient_id, events=events),
    )



def simulate_cohort(config: SimulatorConfig, seed: int) -> SimulatedCohort:
    rng = np.random.default_rng(seed)
    patients = [
        simulate_patient(f"synthetic-{index:06d}", config, rng)
        for index in range(config.cohort_size)
    ]
    return SimulatedCohort(patients=patients, config=config, seed=seed)


def simulate_world(
    world_name: str,
    config: SimulatorConfig,
    seed: int,
) -> SimulatedCohort:
    overrides: dict[str, object]
    if world_name == "smooth":
        overrides = {"intervention_rate": 0.01, "observation_regime": "mcar"}
    elif world_name == "jumps":
        overrides = {"intervention_rate": 0.15, "observation_regime": "mar"}
    elif world_name == "informative_observation":
        overrides = {"observation_regime": "mnar"}
    elif world_name == "site_shift":
        overrides = {"observation_regime": "site_shift"}
    elif world_name == "misspecified":
        overrides = {
            "intervention_rate": 0.12,
            "observation_regime": "mnar",
            "heterogeneity_scale": 2.0,
            "delayed_intervention_effect": True,
            "hidden_regime_switch": True,
        }
    else:
        raise ValueError(f"unknown simulation world: {world_name}")
    world_config = replace(config, world_name=world_name, **overrides)
    return simulate_cohort(world_config, seed)
