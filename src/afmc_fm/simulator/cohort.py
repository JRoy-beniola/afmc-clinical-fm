from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import numpy as np

from afmc_fm.schema.events import ClinicalEvent, EventType, PatientTimeline
from afmc_fm.simulator.config import SimulatorConfig
from afmc_fm.simulator.dynamics import (
    LatentTrajectory,
    PatientParameters,
    advance_latent_state,
    apply_intervention_jump,
    sample_event_times,
    sample_patient_parameters,
)
from afmc_fm.simulator.observation import LAB_CODES, emit_labs, observation_probability

_ORIGIN = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class CompleteOutcomeTruth:
    origin_time: datetime
    times: np.ndarray
    value_codes: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class SimulatedPatient:
    patient_id: str
    site_id: int
    parameters: PatientParameters
    latent: LatentTrajectory
    complete_outcomes: CompleteOutcomeTruth
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
    site_id: int | None = None,
) -> SimulatedPatient:
    parameters = sample_patient_parameters(config, rng)
    if site_id is not None:
        parameters = replace(parameters, site_id=site_id)
    times = sample_event_times(config, rng)
    states: list[np.ndarray] = []
    current_state = np.array(parameters.baseline_state, dtype=float, copy=True)
    events: list[ClinicalEvent] = []
    complete_values: list[list[float]] = []
    previous_observed_value: float | None = None
    midpoint = len(times) // 2
    previous_regime_offset = 0.0

    for index, elapsed_days in enumerate(times):
        if config.hidden_regime_switch and index >= midpoint:
            denominator = max(len(times) - midpoint - 1, 1)
            regime_offset = 0.6 * (index - midpoint) / denominator
            current_state = current_state + (regime_offset - previous_regime_offset)
            previous_regime_offset = regime_offset

        timestamp = _ORIGIN + timedelta(days=float(elapsed_days))
        events.append(
            ClinicalEvent(
                patient_id=patient_id,
                start_time=timestamp,
                code="SYNTHETIC_MEASUREMENT_OPPORTUNITY",
                value=None,
                unit=None,
                event_type=EventType.ENCOUNTER,
                source="synthetic",
                metadata={
                    "site_id": parameters.site_id,
                    "measurement_opportunity": True,
                },
            )
        )
        delayed_shift: np.ndarray | None = None
        if rng.random() < _intervention_probability(
            current_state, config.intervention_rate
        ):
            strength = float(rng.uniform(0.5, 1.5))
            jumped_state = apply_intervention_jump(
                current_state, strength, parameters, rng
            )
            if config.delayed_intervention_effect:
                delayed_shift = jumped_state - current_state
            else:
                current_state = jumped_state
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

        states.append(np.array(current_state, copy=True))
        candidate_labs = emit_labs(
            current_state,
            rng,
            config.measurement_noise,
            regime=config.observation_regime,
            site_id=parameters.site_id,
        )
        complete_values.append([candidate_labs[code] for code in LAB_CODES])
        for code, value in candidate_labs.items():
            probability = observation_probability(
                config.observation_regime,
                current_state,
                previous_observed_value,
                parameters.site_id,
                code,
            )
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

        if index + 1 < len(times):
            next_state = advance_latent_state(
                current_state,
                float(elapsed_days),
                float(times[index + 1]),
                parameters,
                rng,
            )
            if delayed_shift is not None:
                next_state = next_state + delayed_shift
            current_state = next_state

    latent = LatentTrajectory(times=np.array(times, copy=True), states=np.asarray(states))
    return SimulatedPatient(
        patient_id=patient_id,
        site_id=parameters.site_id,
        parameters=parameters,
        latent=latent,
        complete_outcomes=CompleteOutcomeTruth(
            origin_time=_ORIGIN,
            times=np.array(times, copy=True),
            value_codes=LAB_CODES,
            values=np.asarray(complete_values, dtype=float),
        ),
        timeline=PatientTimeline(patient_id=patient_id, events=events),
    )



def simulate_cohort(config: SimulatorConfig, seed: int) -> SimulatedCohort:
    rng = np.random.default_rng(seed)
    patients = [
        simulate_patient(
            f"synthetic-{index:06d}",
            config,
            rng,
            site_id=index % config.n_sites,
        )
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
