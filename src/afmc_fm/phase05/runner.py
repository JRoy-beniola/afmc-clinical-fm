from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from afmc_fm.data.encoding import SummaryHistoryEncoder
from afmc_fm.data.sequences import PatientSequence, build_patient_sequence
from afmc_fm.data.splits import split_patient_ids
from afmc_fm.data.tasks import LongitudinalTask
from afmc_fm.experiments.runner import (
    _attach_latent_truth,
    _evaluate_neural,
    _fit_neural,
    _padded_batch,
    select_low_n_budget,
)
from afmc_fm.models.flow_jump import FlowJumpAdapter
from afmc_fm.phase05.config import Phase05Config, SeedBundle
from afmc_fm.phase05.model import (
    FlowMode,
    JumpMode,
    Phase05FlowJumpAdapter,
    UncertaintyMode,
)
from afmc_fm.phase05.sequences import Phase05Sequence, build_phase05_sequence
from afmc_fm.phase05.training import evaluate_phase05_model, fit_phase05_model
from afmc_fm.schema.events import EventType
from afmc_fm.simulator.cohort import SimulatedCohort, SimulatedPatient


@dataclass(frozen=True)
class PreparedPhase05Cohort:
    patient_by_id: dict[str, SimulatedPatient]
    task: LongitudinalTask
    sequences: dict[str, Phase05Sequence]
    historical_sequences: dict[str, PatientSequence] | None = None


def prepare_phase05_cohort(
    cohort: SimulatedCohort,
    *,
    include_historical: bool = False,
) -> PreparedPhase05Cohort:
    patient_by_id = {patient.patient_id: patient for patient in cohort.patients}
    task = LongitudinalTask(
        value_codes=cohort.patients[0].complete_outcomes.value_codes
    )
    encoder = SummaryHistoryEncoder(task, representation_dim=16, seed=0)
    sequences = {
        patient_id: build_phase05_sequence(patient, encoder, task)
        for patient_id, patient in patient_by_id.items()
    }
    historical_sequences = (
        {
            patient_id: build_patient_sequence(patient, encoder, task)
            for patient_id, patient in patient_by_id.items()
        }
        if include_historical
        else None
    )
    return PreparedPhase05Cohort(
        patient_by_id=patient_by_id,
        task=task,
        sequences=sequences,
        historical_sequences=historical_sequences,
    )


def padded_phase05_batch(
    sequences: list[Phase05Sequence],
    *,
    patients: list[SimulatedPatient] | None = None,
) -> dict[str, torch.Tensor]:
    if not sequences:
        raise ValueError("sequences must not be empty")
    batch = len(sequences)
    steps = max(len(sequence.times) for sequence in sequences)
    rep_dim = sequences[0].representations.shape[1]
    value_dim = sequences[0].values.shape[1]
    event_dim = sequences[0].event_features.shape[1]
    arrays = {
        "representations": np.zeros((batch, steps, rep_dim), dtype=np.float32),
        "values": np.zeros((batch, steps, value_dim), dtype=np.float32),
        "masks": np.zeros((batch, steps, value_dim), dtype=np.float32),
        "event_features": np.zeros((batch, steps, event_dim), dtype=np.float32),
        "update_mask": np.zeros((batch, steps), dtype=np.float32),
        "jump_eligible_mask": np.zeros((batch, steps), dtype=np.float32),
        "times": np.zeros((batch, steps), dtype=np.float32),
        "target_values": np.zeros((batch, steps, value_dim), dtype=np.float32),
        "target_masks": np.zeros((batch, steps, value_dim), dtype=np.float32),
        "target_events": np.zeros((batch, steps), dtype=np.float32),
        "event_valid": np.zeros((batch, steps), dtype=np.float32),
        "valid": np.zeros((batch, steps), dtype=np.float32),
    }
    for row, sequence in enumerate(sequences):
        length = len(sequence.times)
        if length == 0:
            continue
        for name, source in (
            ("representations", sequence.representations),
            ("values", sequence.values),
            ("masks", sequence.masks),
            ("event_features", sequence.event_features),
            ("update_mask", sequence.update_mask),
            ("jump_eligible_mask", sequence.jump_eligible_mask),
            ("times", sequence.times),
            ("target_values", sequence.target_next_values),
            ("target_masks", sequence.target_next_masks),
            ("target_events", sequence.target_event_within_horizon),
            ("event_valid", sequence.target_event_valid),
        ):
            arrays[name][row, :length] = source
        arrays["valid"][row, :length] = 1.0
    result = {name: torch.from_numpy(value) for name, value in arrays.items()}
    if patients is not None:
        if len(patients) != len(sequences):
            raise ValueError("patients and sequences must have equal lengths")
        latent_dim = patients[0].latent.states.shape[1]
        latent_targets = torch.zeros((batch, steps, latent_dim), dtype=torch.float32)
        latent_valid = torch.zeros((batch, steps), dtype=torch.float32)
        for row, patient in enumerate(patients):
            event_times = sorted({event.start_time for event in patient.timeline.events})
            for step, timestamp in enumerate(event_times[:steps]):
                elapsed_days = (
                    timestamp - patient.complete_outcomes.origin_time
                ).total_seconds() / 86400
                latent_index = int(
                    np.argmin(np.abs(patient.latent.times - elapsed_days))
                )
                latent_targets[row, step] = torch.from_numpy(
                    patient.latent.states[latent_index].astype(np.float32)
                )
                latent_valid[row, step] = 1.0
        result["latent_targets"] = latent_targets
        result["latent_valid"] = latent_valid
    return result


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _historical_reference_metrics(
    prepared: PreparedPhase05Cohort,
    config: Phase05Config,
    budget,
    test_ids: list[str],
    device: torch.device,
) -> tuple[dict[str, float], int]:
    if prepared.historical_sequences is None:
        raise ValueError("historical reference requested without historical sequences")
    sequences = prepared.historical_sequences
    train_batch = _padded_batch([sequences[patient_id] for patient_id in budget.fit_ids])
    validation_batch = _padded_batch(
        [sequences[patient_id] for patient_id in budget.validation_ids]
    )
    test_sequences = [sequences[patient_id] for patient_id in test_ids]
    test_patients = [prepared.patient_by_id[patient_id] for patient_id in test_ids]
    test_batch = _attach_latent_truth(_padded_batch(test_sequences), test_patients)
    model = FlowJumpAdapter(
        16,
        len(prepared.task.value_codes),
        len(EventType),
    )
    model = _fit_neural(
        model,
        train_batch,
        validation_batch,
        config,
        False,
        device,
    )
    return _evaluate_neural(model, test_batch, device), _parameter_count(model)


def run_phase05_variant(
    prepared: PreparedPhase05Cohort,
    config: Phase05Config,
    *,
    world: str,
    seed_bundle: SeedBundle,
    n_train: int,
    flow_mode: FlowMode,
    jump_mode: JumpMode,
    uncertainty_mode: UncertaintyMode,
    device: torch.device,
    stage: str = "development",
    historical_reference: bool = False,
) -> pd.DataFrame:
    all_ids = list(prepared.patient_by_id)
    development_pool, _, test_ids = split_patient_ids(
        all_ids,
        seed=0,
        train_fraction=0.8,
        val_fraction=0.0,
    )
    budget = select_low_n_budget(
        development_pool,
        n_train,
        seed_bundle.subset_seed,
    )
    np.random.seed(seed_bundle.model_seed)
    torch.manual_seed(seed_bundle.model_seed)

    if historical_reference:
        metrics, parameters = _historical_reference_metrics(
            prepared,
            config,
            budget,
            test_ids,
            device,
        )
        model_name = "phase0_flow_jump_reference"
        variant = "historical_inclusive_history"
    else:
        train_sequences = [prepared.sequences[patient_id] for patient_id in budget.fit_ids]
        validation_sequences = [
            prepared.sequences[patient_id] for patient_id in budget.validation_ids
        ]
        test_sequences = [prepared.sequences[patient_id] for patient_id in test_ids]
        test_patients = [prepared.patient_by_id[patient_id] for patient_id in test_ids]
        model = Phase05FlowJumpAdapter(
            representation_dim=16,
            value_dim=len(prepared.task.value_codes),
            event_dim=len(EventType),
            state_dim=24,
            flow_mode=flow_mode,
            jump_mode=jump_mode,
            uncertainty_mode=uncertainty_mode,
            time_scale_days=config.time_scale_days,
        )
        fit_phase05_model(
            model,
            padded_phase05_batch(train_sequences),
            padded_phase05_batch(validation_sequences),
            config,
            device,
        )
        metrics = evaluate_phase05_model(
            model,
            padded_phase05_batch(test_sequences, patients=test_patients),
            device,
        )
        parameters = _parameter_count(model)
        model_name = "phase05_flow_jump"
        variant = f"{flow_mode}__{jump_mode}__{uncertainty_mode}"

    rows = [
        {
            "stage": stage,
            "world": world,
            "cohort_seed": seed_bundle.cohort_seed,
            "subset_seed": seed_bundle.subset_seed,
            "model_seed": seed_bundle.model_seed,
            "n_train": n_train,
            "n_fit": len(budget.fit_ids),
            "n_validation": len(budget.validation_ids),
            "model": model_name,
            "variant": variant,
            "split": "test",
            "site_or_shift": "all",
            "metric": metric,
            "value": value,
            "trainable_parameters": parameters,
            "backend": "torch",
        }
        for metric, value in metrics.items()
    ]
    return pd.DataFrame(rows)


__all__ = [
    "PreparedPhase05Cohort",
    "padded_phase05_batch",
    "prepare_phase05_cohort",
    "run_phase05_variant",
]
