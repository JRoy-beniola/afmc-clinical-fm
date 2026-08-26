from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from afmc_fm.data.splits import split_patient_ids
from afmc_fm.experiments.runner import select_low_n_budget
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.model import Phase05FlowJumpAdapter
from afmc_fm.phase05.runner import (
    PreparedPhase05Cohort,
    padded_phase05_batch,
    prepare_phase05_cohort,
)
from afmc_fm.phase05.training import evaluate_phase05_model, fit_phase05_model
from afmc_fm.phase06.diagnostics import Phase06DiagnosticRecorder
from afmc_fm.phase06.planning import Phase06CellSpec
from afmc_fm.schema.events import EventType
from afmc_fm.simulator.cohort import simulate_world
from afmc_fm.simulator.config import SimulatorConfig


@dataclass(frozen=True, slots=True)
class Phase06CellRun:
    metrics: pd.DataFrame
    trace: pd.DataFrame
    summary: dict[str, int | float | str]
    production_state_dict: dict[str, torch.Tensor]
    shadow_state_dict: dict[str, torch.Tensor]


def prepare_phase06_cohort(
    simulator_config: SimulatorConfig,
    cell: Phase06CellSpec,
) -> PreparedPhase05Cohort:
    if not isinstance(simulator_config, SimulatorConfig):
        raise TypeError("simulator_config must be a SimulatorConfig")
    if not isinstance(cell, Phase06CellSpec):
        raise TypeError("cell must be a Phase06CellSpec")
    cohort = simulate_world(
        "smooth",
        simulator_config,
        seed=cell.cohort_seed,
    )
    return prepare_phase05_cohort(cohort, include_historical=False)


def run_phase06_cell(
    prepared: PreparedPhase05Cohort,
    phase05_config: Phase05Config,
    cell: Phase06CellSpec,
    device: torch.device,
) -> Phase06CellRun:
    if not isinstance(prepared, PreparedPhase05Cohort):
        raise TypeError("prepared must be a PreparedPhase05Cohort")
    if not isinstance(phase05_config, Phase05Config):
        raise TypeError("phase05_config must be a Phase05Config")
    if not isinstance(cell, Phase06CellSpec):
        raise TypeError("cell must be a Phase06CellSpec")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if cell.n_train not in phase05_config.train_sizes:
        raise ValueError("Phase 0.6 cell n_train is absent from the Phase 0.5 config")

    all_ids = list(prepared.patient_by_id)
    development_pool, _, test_ids = split_patient_ids(
        all_ids,
        seed=0,
        train_fraction=0.8,
        val_fraction=0.0,
    )
    budget = select_low_n_budget(
        development_pool,
        cell.n_train,
        cell.subset_seed,
    )

    np.random.seed(cell.model_seed)
    torch.manual_seed(cell.model_seed)

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
        flow_mode=cell.flow_mode,
        jump_mode="none",
        uncertainty_mode="deterministic",
        time_scale_days=phase05_config.time_scale_days,
    )
    recorder = Phase06DiagnosticRecorder()
    fit_phase05_model(
        model,
        padded_phase05_batch(train_sequences),
        padded_phase05_batch(validation_sequences),
        phase05_config,
        device,
        diagnostics=recorder,
    )

    metrics = evaluate_phase05_model(
        model,
        padded_phase05_batch(test_sequences, patients=test_patients),
        device,
    )
    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    variant = f"{cell.flow_mode}__none__deterministic"
    rows = [
        {
            "stage": cell.stage,
            "world": cell.world,
            "cohort_seed": cell.cohort_seed,
            "subset_seed": cell.subset_seed,
            "model_seed": cell.model_seed,
            "n_train": cell.n_train,
            "n_fit": len(budget.fit_ids),
            "n_validation": len(budget.validation_ids),
            "model": "phase05_flow_jump",
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

    return Phase06CellRun(
        metrics=pd.DataFrame(rows),
        trace=recorder.trace_frame(),
        summary=recorder.summary_payload(),
        production_state_dict=recorder.production_state_dict,
        shadow_state_dict=recorder.shadow_state_dict,
    )


__all__ = [
    "Phase06CellRun",
    "prepare_phase06_cohort",
    "run_phase06_cell",
]
