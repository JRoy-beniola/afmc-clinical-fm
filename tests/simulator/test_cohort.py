import numpy as np

import afmc_fm.simulator.cohort as cohort_module
from afmc_fm.schema.events import EventType
from afmc_fm.simulator.cohort import simulate_cohort, simulate_patient, simulate_world
from afmc_fm.simulator.config import SimulatorConfig


def test_cohort_contains_unique_patients_and_sorted_events():
    config = SimulatorConfig(cohort_size=12, followup_days=120.0, latent_dim=4)
    cohort = simulate_cohort(config, seed=17)
    assert len(cohort.patients) == 12
    assert len({p.patient_id for p in cohort.patients}) == 12
    for patient in cohort.patients:
        times = [e.start_time for e in patient.timeline.events]
        assert times == sorted(times)


def test_cohort_contains_observations_and_interventions():
    config = SimulatorConfig(cohort_size=32, followup_days=180.0, intervention_rate=0.15)
    cohort = simulate_cohort(config, seed=19)
    types = [e.event_type for p in cohort.patients for e in p.timeline.events]
    assert EventType.OBSERVATION in types
    assert EventType.INTERVENTION in types


def test_same_seed_reproduces_events_and_latent_truth():
    config = SimulatorConfig(cohort_size=2, followup_days=30.0)
    a = simulate_cohort(config, seed=23)
    b = simulate_cohort(config, seed=23)
    for patient_a, patient_b in zip(a.patients, b.patients, strict=True):
        assert patient_a.timeline.events == patient_b.timeline.events
        np.testing.assert_allclose(patient_a.latent.times, patient_b.latent.times)
        np.testing.assert_allclose(patient_a.latent.states, patient_b.latent.states)


def test_complete_outcome_truth_is_retained_before_observation_masking():
    config = SimulatorConfig(
        cohort_size=1,
        followup_days=120.0,
        observation_regime="mcar",
    )
    patient = simulate_cohort(config, seed=31).patients[0]
    truth = patient.complete_outcomes
    assert truth.values.shape == (len(patient.latent.times), 3)
    assert np.isfinite(truth.values).all()
    observed = [
        event for event in patient.timeline.events if event.event_type == EventType.OBSERVATION
    ]
    assert len(observed) < truth.values.size
    for event in observed:
        elapsed = (event.start_time - truth.origin_time).total_seconds() / 86400.0
        time_index = int(np.argmin(np.abs(truth.times - elapsed)))
        value_index = truth.value_codes.index(event.code)
        assert event.value == truth.values[time_index, value_index]


def test_immediate_intervention_state_drives_next_dynamics_step(monkeypatch):
    advance_inputs: list[np.ndarray] = []
    intervention_calls = 0

    def intervention_probability(*_args):
        nonlocal intervention_calls
        intervention_calls += 1
        return 1.0 if intervention_calls == 1 else 0.0

    def advance_state(state, *_args):
        advance_inputs.append(state.copy())
        return state + 0.5

    monkeypatch.setattr(cohort_module, "_intervention_probability", intervention_probability)
    monkeypatch.setattr(cohort_module, "advance_latent_state", advance_state)
    monkeypatch.setattr(
        cohort_module,
        "apply_intervention_jump",
        lambda state, *_args: state + 2.0,
    )
    patient = simulate_patient(
        "patient",
        SimulatorConfig(
            followup_days=30.0,
            mean_event_interval_days=5.0,
            process_noise=0.0,
            intervention_rate=1.0,
        ),
        np.random.default_rng(7),
    )
    np.testing.assert_allclose(advance_inputs[0], patient.latent.states[0])
    np.testing.assert_allclose(patient.latent.states[1], patient.latent.states[0] + 0.5)


def test_delayed_intervention_enters_at_next_state(monkeypatch):
    advance_inputs: list[np.ndarray] = []
    intervention_calls = 0

    def intervention_probability(*_args):
        nonlocal intervention_calls
        intervention_calls += 1
        return 1.0 if intervention_calls == 1 else 0.0

    def advance_state(state, *_args):
        advance_inputs.append(state.copy())
        return state + 0.5

    monkeypatch.setattr(cohort_module, "_intervention_probability", intervention_probability)
    monkeypatch.setattr(cohort_module, "advance_latent_state", advance_state)
    monkeypatch.setattr(
        cohort_module,
        "apply_intervention_jump",
        lambda state, *_args: state + 2.0,
    )
    patient = simulate_patient(
        "patient",
        SimulatorConfig(
            followup_days=30.0,
            mean_event_interval_days=5.0,
            process_noise=0.0,
            intervention_rate=1.0,
            delayed_intervention_effect=True,
        ),
        np.random.default_rng(7),
    )
    np.testing.assert_allclose(advance_inputs[0], patient.latent.states[0])
    np.testing.assert_allclose(patient.latent.states[1], patient.latent.states[0] + 2.5)


def test_site_shift_balances_sites_without_shifting_latent_physiology():
    cohort = simulate_world(
        "site_shift",
        SimulatorConfig(
            cohort_size=400,
            n_sites=2,
            followup_days=30.0,
        ),
        seed=41,
    )
    site_zero = [patient for patient in cohort.patients if patient.site_id == 0]
    site_one = [patient for patient in cohort.patients if patient.site_id == 1]
    assert len(site_zero) == len(site_one) == 200
    mean_zero = np.mean([patient.parameters.baseline_state for patient in site_zero], axis=0)
    mean_one = np.mean([patient.parameters.baseline_state for patient in site_one], axis=0)
    np.testing.assert_allclose(mean_zero, mean_one, atol=0.2)


def test_every_candidate_time_has_a_measurement_opportunity_anchor():
    patient = simulate_cohort(
        SimulatorConfig(
            cohort_size=1,
            followup_days=60.0,
            intervention_rate=0.0,
        ),
        seed=43,
    ).patients[0]
    anchors = [
        event
        for event in patient.timeline.events
        if event.event_type == EventType.ENCOUNTER
        and event.metadata.get("measurement_opportunity") is True
    ]
    anchor_days = np.asarray(
        [
            (event.start_time - patient.complete_outcomes.origin_time).total_seconds()
            / 86400.0
            for event in anchors
        ]
    )
    np.testing.assert_allclose(anchor_days, patient.complete_outcomes.times)
