import numpy as np

from afmc_fm.schema.events import EventType
from afmc_fm.simulator.cohort import simulate_cohort
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
