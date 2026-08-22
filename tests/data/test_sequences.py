from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import afmc_fm.data.encoding as encoding_module
import afmc_fm.data.sequences as sequences_module
import afmc_fm.simulator.cohort as cohort_module
from afmc_fm.data.encoding import SummaryHistoryEncoder
from afmc_fm.data.sequences import build_patient_sequence
from afmc_fm.data.tasks import LongitudinalTask
from afmc_fm.schema.events import ClinicalEvent, EventType, PatientTimeline
from afmc_fm.simulator.cohort import simulate_cohort, simulate_patient
from afmc_fm.simulator.config import SimulatorConfig

SYNTHETIC_TASK = LongitudinalTask(("LAB_FAST", "LAB_SLOW", "LAB_BURDEN"))


def test_patient_sequence_contains_aligned_time_and_mask_arrays():
    patient = simulate_cohort(
        SimulatorConfig(cohort_size=1, followup_days=120), seed=2
    ).patients[0]
    encoder = SummaryHistoryEncoder(SYNTHETIC_TASK, representation_dim=16, seed=9)
    sequence = build_patient_sequence(patient, encoder, SYNTHETIC_TASK)
    steps = len(sequence.times)
    assert sequence.representations.shape == (steps, 16)
    assert sequence.values.shape[0] == steps
    assert sequence.masks.shape == sequence.values.shape
    assert sequence.target_event_valid.shape == (steps,)
    assert not sequence.target_event_valid[-1]
    assert sequence.target_event_valid[:-1].all()
    assert np.all(np.diff(sequence.times) >= 0)


def test_data_pipeline_accepts_generic_timeline_and_task_channels():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    timeline = PatientTimeline(
        "generic",
        [
            ClinicalEvent(
                "generic",
                origin,
                "CUSTOM_VALUE",
                1.0,
                "arb",
                EventType.OBSERVATION,
                "example",
            ),
            ClinicalEvent(
                "generic",
                origin + timedelta(days=1),
                "CUSTOM_ACTION",
                1.0,
                "arb",
                EventType.INTERVENTION,
                "example",
            ),
        ],
    )
    patient = SimpleNamespace(patient_id="generic", timeline=timeline)
    task = LongitudinalTask(
        value_codes=("CUSTOM_VALUE",),
        event_target_type=EventType.INTERVENTION,
    )
    encoder = SummaryHistoryEncoder(task, representation_dim=4, seed=1)
    sequence = build_patient_sequence(patient, encoder, task)
    assert sequence.values.shape == (2, 1)
    assert sequence.target_event_within_horizon.tolist() == [1.0, 0.0]


def test_data_modules_do_not_import_simulator_internals():
    assert "afmc_fm.simulator" not in Path(encoding_module.__file__).read_text()
    assert "afmc_fm.simulator" not in Path(sequences_module.__file__).read_text()


def test_all_zero_measurement_opportunities_survive_sequence_construction(monkeypatch):
    monkeypatch.setattr(
        cohort_module,
        "observation_probability",
        lambda *_args: 0.0,
    )
    patient = simulate_patient(
        "zero-observation-patient",
        SimulatorConfig(
            followup_days=60.0,
            intervention_rate=0.0,
        ),
        np.random.default_rng(44),
        site_id=0,
    )
    encoder = SummaryHistoryEncoder(
        SYNTHETIC_TASK,
        representation_dim=8,
        seed=2,
    )
    sequence = build_patient_sequence(patient, encoder, SYNTHETIC_TASK)
    assert len(sequence.times) == len(patient.complete_outcomes.times)
    assert np.all(sequence.masks.sum(axis=1) == 0)
    encounter_index = list(EventType).index(EventType.ENCOUNTER)
    assert np.all(sequence.event_features[:, encounter_index] == 1)
