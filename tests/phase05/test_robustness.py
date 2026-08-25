import numpy as np
import torch

from afmc_fm.experiments.runner import build_complete_truth_targets
from afmc_fm.phase05.robustness import complete_truth_phase05_batch
from afmc_fm.phase05.runner import prepare_phase05_cohort
from afmc_fm.simulator.cohort import simulate_world
from afmc_fm.simulator.config import SimulatorConfig


def test_complete_truth_batch_uses_latent_emissions_not_observed_next_lab_masks():
    cohort = simulate_world(
        "site_shift",
        SimulatorConfig(cohort_size=20, followup_days=60.0),
        seed=701,
    )
    prepared = prepare_phase05_cohort(cohort)
    patient = cohort.patients[0]
    sequence = prepared.sequences[patient.patient_id]

    batch = complete_truth_phase05_batch([sequence], [patient])
    expected_values, expected_masks = build_complete_truth_targets(patient)
    length = len(expected_masks)

    torch.testing.assert_close(
        batch["target_values"][0, :length],
        torch.from_numpy(expected_values),
    )
    torch.testing.assert_close(
        batch["target_masks"][0, :length],
        torch.from_numpy(expected_masks),
    )
    assert np.any(expected_masks != sequence.target_next_masks)
    assert expected_masks.sum() > sequence.target_next_masks.sum()
