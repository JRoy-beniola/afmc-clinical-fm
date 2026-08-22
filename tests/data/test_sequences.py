import numpy as np

from afmc_fm.data.encoding import SummaryHistoryEncoder
from afmc_fm.data.sequences import build_patient_sequence
from afmc_fm.simulator.cohort import simulate_cohort
from afmc_fm.simulator.config import SimulatorConfig


def test_patient_sequence_contains_aligned_time_and_mask_arrays():
    patient = simulate_cohort(
        SimulatorConfig(cohort_size=1, followup_days=120), seed=2
    ).patients[0]
    encoder = SummaryHistoryEncoder(representation_dim=16, seed=9)
    sequence = build_patient_sequence(patient, encoder)
    steps = len(sequence.times)
    assert sequence.representations.shape == (steps, 16)
    assert sequence.values.shape[0] == steps
    assert sequence.masks.shape == sequence.values.shape
    assert sequence.target_event_valid.shape == (steps,)
    assert not sequence.target_event_valid[-1]
    assert sequence.target_event_valid[:-1].all()
    assert np.all(np.diff(sequence.times) >= 0)
