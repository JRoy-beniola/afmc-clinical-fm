from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from afmc_fm.data.encoding import SummaryHistoryEncoder
from afmc_fm.data.sequences import PatientSequence, build_patient_sequence
from afmc_fm.data.splits import split_patient_ids
from afmc_fm.data.tasks import LongitudinalTask
from afmc_fm.models.baselines import (
    GradientBoostingRegressorBaseline,
    GRUBaseline,
    ProbeRegressor,
)
from afmc_fm.models.flow_jump import FlowJumpAdapter
from afmc_fm.models.losses import masked_gaussian_nll, observation_bce
from afmc_fm.schema.events import EventType
from afmc_fm.simulator.cohort import SimulatedCohort, SimulatedPatient

MODEL_NAMES = (
    "probe_linear",
    "gradient_boosting",
    "gru_from_scratch",
    "flow_jump",
    "flow_jump_observation",
)


@dataclass(frozen=True)
class ExperimentConfig:
    train_sizes: tuple[int, ...] = (5, 10, 20, 40, 80, 100)
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    max_epochs: int = 100
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    lambda_event: float = 1.0
    lambda_obs: float = 0.2


@dataclass(frozen=True)
class LowNBudget:
    fit_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]


def sample_low_n_train_ids(train_pool: Sequence[str], n: int, seed: int) -> list[str]:
    if n <= 0 or n > len(train_pool):
        raise ValueError("n must be positive and no larger than the training pool")
    indices = np.random.default_rng(seed).choice(len(train_pool), size=n, replace=False)
    return [train_pool[index] for index in indices]


def select_low_n_budget(
    development_pool: Sequence[str],
    n: int,
    subset_seed: int,
    validation_fraction: float = 0.2,
) -> LowNBudget:
    if n < 2:
        raise ValueError("low-N budget must contain at least two patients")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    budget_ids = sample_low_n_train_ids(development_pool, n, subset_seed)
    shuffled = np.asarray(budget_ids, dtype=object)
    np.random.default_rng(subset_seed + 1).shuffle(shuffled)
    validation_count = max(1, min(n - 1, round(n * validation_fraction)))
    return LowNBudget(
        fit_ids=tuple(shuffled[validation_count:].tolist()),
        validation_ids=tuple(shuffled[:validation_count].tolist()),
    )


def _static_examples(sequences: Sequence[PatientSequence]) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    for sequence in sequences:
        for step, lab in np.argwhere(sequence.target_next_masks > 0):
            lab_indicator = np.zeros(sequence.target_next_masks.shape[1])
            lab_indicator[lab] = 1.0
            features.append(np.concatenate([sequence.representations[step], lab_indicator]))
            targets.append(float(sequence.target_next_values[step, lab]))
    if not features:
        raise ValueError("sequences contain no observed forecasting targets")
    return np.asarray(features), np.asarray(targets)


def _padded_batch(sequences: Sequence[PatientSequence]) -> dict[str, torch.Tensor]:
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
            ("times", sequence.times),
            ("target_values", sequence.target_next_values),
            ("target_masks", sequence.target_next_masks),
            ("target_events", sequence.target_event_within_horizon),
            ("event_valid", sequence.target_event_valid),
        ):
            arrays[name][row, :length] = source
        arrays["valid"][row, :length] = 1.0
    return {name: torch.from_numpy(value) for name, value in arrays.items()}


def build_complete_truth_targets(
    patient: SimulatedPatient,
) -> tuple[np.ndarray, np.ndarray]:
    event_times = sorted({event.start_time for event in patient.timeline.events})
    value_dim = len(patient.complete_outcomes.value_codes)
    targets = np.zeros((len(event_times), value_dim), dtype=np.float32)
    masks = np.zeros_like(targets)
    for step, timestamp in enumerate(event_times):
        elapsed_days = (
            timestamp - patient.complete_outcomes.origin_time
        ).total_seconds() / 86400
        target_index = int(
            np.searchsorted(patient.complete_outcomes.times, elapsed_days, side="right")
        )
        if target_index < len(patient.complete_outcomes.times):
            targets[step] = patient.complete_outcomes.values[target_index]
            masks[step] = 1.0
    return targets, masks


def _complete_truth_batch(
    sequences: Sequence[PatientSequence],
    patients: Sequence[SimulatedPatient],
) -> dict[str, torch.Tensor]:
    batch = _padded_batch(sequences)
    for row, patient in enumerate(patients):
        targets, masks = build_complete_truth_targets(patient)
        length = len(targets)
        batch["target_values"][row, :length] = torch.from_numpy(targets)
        batch["target_masks"][row, :length] = torch.from_numpy(masks)
    return batch


def _neural_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    config: ExperimentConfig,
    observation_aware: bool,
) -> torch.Tensor:
    output = model(
        representations=batch["representations"],
        values=batch["values"],
        masks=batch["masks"],
        event_features=batch["event_features"],
        times=batch["times"],
    )
    loss = masked_gaussian_nll(
        output.value_mean,
        output.value_log_scale,
        batch["target_values"],
        batch["target_masks"],
    )
    event_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.event_logits,
        batch["target_events"],
        weight=batch["event_valid"],
        reduction="sum",
    ) / batch["event_valid"].sum().clamp_min(1.0)
    loss = loss + config.lambda_event * event_loss
    if observation_aware:
        assert output.observation_logits is not None
        loss = loss + config.lambda_obs * observation_bce(
            output.observation_logits, batch["masks"], batch["valid"]
        )
    return loss


def _fit_neural(
    model: nn.Module,
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    config: ExperimentConfig,
    observation_aware: bool,
) -> nn.Module:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0
    for _ in range(config.max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = _neural_loss(model, train, config, observation_aware)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                _neural_loss(model, validation, config, observation_aware).item()
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    model.load_state_dict(best_state)
    return model


def _evaluate_neural(model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        output = model(
            representations=batch["representations"],
            values=batch["values"],
            masks=batch["masks"],
            event_features=batch["event_features"],
            times=batch["times"],
        )
    selected = batch["target_masks"].bool()
    error = output.value_mean[selected] - batch["target_values"][selected]
    return {"mae": float(error.abs().mean()), "rmse": float(error.square().mean().sqrt())}


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_low_n_benchmark(
    cohort: SimulatedCohort,
    config: ExperimentConfig,
    model_names: Sequence[str] = MODEL_NAMES,
) -> pd.DataFrame:
    unknown = set(model_names).difference(MODEL_NAMES)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    torch.set_num_threads(1)
    patient_by_id = {patient.patient_id: patient for patient in cohort.patients}
    all_ids = list(patient_by_id)
    development_pool, _, test_ids = split_patient_ids(
        all_ids, seed=0, train_fraction=0.8, val_fraction=0.0
    )
    task = LongitudinalTask(
        value_codes=cohort.patients[0].complete_outcomes.value_codes
    )
    encoder = SummaryHistoryEncoder(task, representation_dim=16, seed=0)
    sequences = {
        patient_id: build_patient_sequence(patient, encoder, task)
        for patient_id, patient in patient_by_id.items()
    }
    rows: list[dict[str, object]] = []
    for seed in config.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        for n_train in config.train_sizes:
            budget = select_low_n_budget(development_pool, n_train, seed)
            train_ids = budget.fit_ids
            train_sequences = [sequences[patient_id] for patient_id in train_ids]
            validation_sequences = [
                sequences[patient_id] for patient_id in budget.validation_ids
            ]
            test_sequences = [sequences[patient_id] for patient_id in test_ids]
            for model_name in model_names:
                parameters = 0
                if model_name in {"probe_linear", "gradient_boosting"}:
                    train_x, train_y = _static_examples(train_sequences)
                    test_x, test_y = _static_examples(test_sequences)
                    estimator = (
                        ProbeRegressor()
                        if model_name == "probe_linear"
                        else GradientBoostingRegressorBaseline()
                    )
                    prediction = estimator.fit(train_x, train_y).predict(test_x)
                    error = prediction - test_y
                    metrics = {
                        "mae": float(np.mean(np.abs(error))),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                    }
                else:
                    train_batch = _padded_batch(train_sequences)
                    validation_batch = _padded_batch(validation_sequences)
                    test_batch = _padded_batch(test_sequences)
                    if model_name == "gru_from_scratch":
                        model: nn.Module = GRUBaseline(
                            16, len(task.value_codes), len(EventType)
                        )
                        observation_aware = False
                    else:
                        observation_aware = model_name == "flow_jump_observation"
                        model = FlowJumpAdapter(
                            16,
                            len(task.value_codes),
                            len(EventType),
                            model_observation_process=observation_aware,
                        )
                    model = _fit_neural(
                        model, train_batch, validation_batch, config, observation_aware
                    )
                    metrics = _evaluate_neural(model, test_batch)
                    parameters = _parameter_count(model)
                for metric, value in metrics.items():
                    rows.append(
                        {
                            "model": model_name,
                            "n_train": n_train,
                            "n_fit": len(budget.fit_ids),
                            "n_validation": len(budget.validation_ids),
                            "seed": seed,
                            "split": "test",
                            "site_or_shift": "all",
                            "metric": metric,
                            "value": value,
                            "trainable_parameters": parameters,
                        }
                    )
    return pd.DataFrame(rows)


def run_observation_shift_benchmark(
    cohort: SimulatedCohort,
    config: ExperimentConfig,
) -> pd.DataFrame:
    site_zero_ids = [
        patient.patient_id for patient in cohort.patients if patient.site_id == 0
    ]
    site_one_ids = [
        patient.patient_id for patient in cohort.patients if patient.site_id == 1
    ]
    if not site_zero_ids or not site_one_ids:
        raise ValueError("observation-shift evaluation requires patients from sites 0 and 1")
    development_pool, _, site_zero_test_ids = split_patient_ids(
        site_zero_ids, seed=0, train_fraction=0.8, val_fraction=0.0
    )
    patient_by_id = {patient.patient_id: patient for patient in cohort.patients}
    task = LongitudinalTask(
        value_codes=cohort.patients[0].complete_outcomes.value_codes
    )
    encoder = SummaryHistoryEncoder(task, representation_dim=16, seed=0)
    sequences = {
        patient_id: build_patient_sequence(patient, encoder, task)
        for patient_id, patient in patient_by_id.items()
    }
    rows: list[dict[str, object]] = []
    for seed in config.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        for n_train in config.train_sizes:
            budget = select_low_n_budget(development_pool, n_train, seed)
            train_ids = budget.fit_ids
            train_batch = _padded_batch([sequences[patient_id] for patient_id in train_ids])
            validation_batch = _padded_batch(
                [sequences[patient_id] for patient_id in budget.validation_ids]
            )
            site_zero_patients = [
                patient_by_id[patient_id] for patient_id in site_zero_test_ids
            ]
            site_one_patients = [
                patient_by_id[patient_id] for patient_id in site_one_ids
            ]
            evaluation_batches = {
                "site_0": _complete_truth_batch(
                    [sequences[patient.patient_id] for patient in site_zero_patients],
                    site_zero_patients,
                ),
                "site_1": _complete_truth_batch(
                    [sequences[patient.patient_id] for patient in site_one_patients],
                    site_one_patients,
                ),
            }
            for model_name in ("flow_jump", "flow_jump_observation"):
                observation_aware = model_name == "flow_jump_observation"
                model = FlowJumpAdapter(
                    16,
                    len(task.value_codes),
                    len(EventType),
                    model_observation_process=observation_aware,
                )
                model = _fit_neural(
                    model, train_batch, validation_batch, config, observation_aware
                )
                by_site = {
                    site: _evaluate_neural(model, batch)
                    for site, batch in evaluation_batches.items()
                }
                parameters = _parameter_count(model)
                for site, metrics in by_site.items():
                    for metric, value in metrics.items():
                        rows.append(
                            {
                                "model": model_name,
                                "n_train": n_train,
                            "n_fit": len(budget.fit_ids),
                            "n_validation": len(budget.validation_ids),
                                "seed": seed,
                                "split": "test",
                                "site_or_shift": site,
                                "metric": metric,
                                "value": value,
                                "trainable_parameters": parameters,
                            }
                        )
                for metric in by_site["site_0"]:
                    rows.append(
                        {
                            "model": model_name,
                            "n_train": n_train,
                            "n_fit": len(budget.fit_ids),
                            "n_validation": len(budget.validation_ids),
                            "seed": seed,
                            "split": "test",
                            "site_or_shift": "site_1_minus_site_0",
                            "metric": metric,
                            "value": by_site["site_1"][metric] - by_site["site_0"][metric],
                            "trainable_parameters": parameters,
                        }
                    )
    return pd.DataFrame(rows)
