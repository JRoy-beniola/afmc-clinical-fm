from __future__ import annotations

import torch

from afmc_fm.experiments.runner import build_complete_truth_targets
from afmc_fm.phase05.runner import padded_phase05_batch
from afmc_fm.phase05.sequences import Phase05Sequence
from afmc_fm.simulator.cohort import SimulatedPatient


def complete_truth_phase05_batch(
    sequences: list[Phase05Sequence],
    patients: list[SimulatedPatient],
) -> dict[str, torch.Tensor]:
    if len(sequences) != len(patients):
        raise ValueError("patients and sequences must have equal lengths")
    batch = padded_phase05_batch(sequences, patients=patients)
    max_steps = batch["target_values"].shape[1]
    for row, patient in enumerate(patients):
        targets, masks = build_complete_truth_targets(patient)
        length = len(targets)
        if length > max_steps:
            raise ValueError("complete-truth targets exceed Phase-0.5 sequence length")
        batch["target_values"][row, :length] = torch.from_numpy(targets)
        batch["target_masks"][row, :length] = torch.from_numpy(masks)
    return batch


__all__ = ["complete_truth_phase05_batch"]
