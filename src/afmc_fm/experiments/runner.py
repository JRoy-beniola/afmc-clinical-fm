from collections.abc import Callable, Sequence
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
from afmc_fm.execution.device import move_batch
from afmc_fm.metrics.forecasting import (
    binary_metrics,
    gaussian_forecasting_metrics,
    regression_metrics,
)
from afmc_fm.metrics.latent import aligned_latent_r2
from afmc_fm.models.baselines import (
    GradientBoostingRegressorBaseline,
    GRUBaseline,
    MLPRegressorBaseline,
    TorchMLPRegressorBaseline,
    TorchRidgeRegressor,
)
from afmc_fm.models.flow_jump import FlowJumpAdapter
from afmc_fm.models.losses import masked_gaussian_nll, observation_bce
from afmc_fm.schema.events import EventType
from afmc_fm.simulator.cohort import (
    SimulatedCohort,
    SimulatedPatient,
    simulate_cohort,
    simulate_world,
)

MODEL_NAMES = (
    "engineered_linear",
    "gradient_boosting",
    "gru_from_scratch",
    "representation_linear",
    "representation_mlp",
    "flow_jump",
    "flow_jump_observation",
)

ABLATION_IDS = (
    "none",
    "no_representation",
    "no_flow",
    "no_jump",
    "no_observation_head",
    "no_prob_scale",
)

_CPU_DEVICE = torch.device("cpu")


@dataclass(frozen=True)
class ExperimentConfig:
    train_sizes: tuple[int, ...] = (5, 10, 20, 40, 80, 100)
    cohort_seeds: tuple[int, ...] = ()
    subset_seeds: tuple[int, ...] = (1,)
    model_seeds: tuple[int, ...] = (1,)
    max_epochs: int = 100
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_event: float = 1.0
    lambda_obs: float = 0.2
    worlds: tuple[str, ...] = ("custom",)
    models: tuple[str, ...] = MODEL_NAMES
    ablations: tuple[str, ...] = ("none",)
    train_site: int = 0
    test_sites: tuple[int, ...] = (0, 1)

    def seed_bundles(self, supplied_cohort_seed: int) -> tuple[tuple[int, int, int], ...]:
        cohort_seeds = self.cohort_seeds or (supplied_cohort_seed,)
        if not (
            len(cohort_seeds) == len(self.subset_seeds) == len(self.model_seeds)
        ):
            raise ValueError(
                "cohort_seeds, subset_seeds, and model_seeds must have equal lengths"
            )
        return tuple(
            zip(cohort_seeds, self.subset_seeds, self.model_seeds, strict=True)
        )


@dataclass(frozen=True)
class LowNBudget:
    fit_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedCohort:
    patient_by_id: dict[str, SimulatedPatient]
    task: LongitudinalTask
    sequences: dict[str, PatientSequence]


def _prepare_cohort(cohort: SimulatedCohort) -> _PreparedCohort:
    patient_by_id = {patient.patient_id: patient for patient in cohort.patients}
    task = LongitudinalTask(
        value_codes=cohort.patients[0].complete_outcomes.value_codes
    )
    encoder = SummaryHistoryEncoder(task, representation_dim=16, seed=0)
    sequences = {
        patient_id: build_patient_sequence(patient, encoder, task)
        for patient_id, patient in patient_by_id.items()
    }
    return _PreparedCohort(patient_by_id, task, sequences)


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


def _representation_examples(sequences: Sequence[PatientSequence]) -> tuple[np.ndarray, np.ndarray]:
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


def _engineered_examples(
    sequences: Sequence[PatientSequence],
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    for sequence in sequences:
        value_dim = sequence.values.shape[1]
        last_values = np.zeros(value_dim)
        observation_counts = np.zeros(value_dim)
        for step in range(len(sequence.times)):
            observed = sequence.masks[step] > 0
            last_values[observed] = sequence.values[step, observed]
            observation_counts += observed
            history = np.concatenate(
                [
                    last_values,
                    observation_counts,
                    sequence.masks[step],
                    sequence.event_features[step],
                    np.asarray([np.log1p(sequence.times[step])]),
                ]
            )
            for lab in np.flatnonzero(sequence.target_next_masks[step] > 0):
                lab_indicator = np.zeros(value_dim)
                lab_indicator[lab] = 1.0
                features.append(np.concatenate([history, lab_indicator]))
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
        "update_mask": np.zeros((batch, steps), dtype=np.float32),
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


def _attach_latent_truth(
    batch: dict[str, torch.Tensor],
    patients: Sequence[SimulatedPatient],
) -> dict[str, torch.Tensor]:
    latent_dim = patients[0].latent.states.shape[1]
    targets = torch.zeros(
        (len(patients), batch["valid"].shape[1], latent_dim),
        dtype=torch.float32,
    )
    valid = torch.zeros_like(batch["valid"])
    for row, patient in enumerate(patients):
        event_times = sorted({event.start_time for event in patient.timeline.events})
        for step, timestamp in enumerate(event_times):
            elapsed_days = (
                timestamp - patient.complete_outcomes.origin_time
            ).total_seconds() / 86400
            latent_index = int(np.argmin(np.abs(patient.latent.times - elapsed_days)))
            targets[row, step] = torch.from_numpy(
                patient.latent.states[latent_index].astype(np.float32)
            )
            valid[row, step] = 1.0
    batch["latent_targets"] = targets
    batch["latent_valid"] = valid
    return batch


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
    return _attach_latent_truth(batch, patients)


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
        **(
            {"update_mask": batch["update_mask"]}
            if isinstance(model, FlowJumpAdapter)
            else {}
        ),
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
    device: torch.device,
) -> nn.Module:
    model = model.to(device)
    train = move_batch(train, device)
    validation = move_batch(validation, device)
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


def _evaluate_neural(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    batch = move_batch(batch, device)
    model.eval()
    with torch.no_grad():
        output = model(
            representations=batch["representations"],
            values=batch["values"],
            masks=batch["masks"],
            event_features=batch["event_features"],
            times=batch["times"],
            **(
                {"update_mask": batch["update_mask"]}
                if isinstance(model, FlowJumpAdapter)
                else {}
            ),
        )
    selected = batch["target_masks"].bool()
    truth = batch["target_values"][selected].detach().cpu().numpy()
    mean = output.value_mean[selected].detach().cpu().numpy()
    log_scale = output.value_log_scale[selected].detach().cpu().numpy()
    metrics = regression_metrics(truth, mean)
    metrics.update(gaussian_forecasting_metrics(truth, mean, log_scale))

    event_selected = batch["event_valid"].bool()
    event_metrics = binary_metrics(
        batch["target_events"][event_selected].detach().cpu().numpy(),
        torch.sigmoid(output.event_logits[event_selected]).detach().cpu().numpy(),
    )
    metrics.update({f"event_{name}": value for name, value in event_metrics.items()})

    if "latent_targets" in batch:
        latent_selected = batch["latent_valid"].bool()
        learned_states = (
            output.post_event_states
            if hasattr(output, "post_event_states")
            else output.states
        )
        metrics["latent_aligned_r2"] = aligned_latent_r2(
            batch["latent_targets"][latent_selected].detach().cpu().numpy(),
            learned_states[latent_selected].detach().cpu().numpy(),
        )
    return metrics


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)



def _flow_jump_variant(
    model_name: str,
    ablation: str,
    representation_dim: int,
    value_dim: int,
    event_dim: int,
) -> tuple[FlowJumpAdapter, bool]:
    observation_aware = model_name == "flow_jump_observation"
    model = FlowJumpAdapter(
        representation_dim,
        value_dim,
        event_dim,
        model_observation_process=observation_aware,
        no_representation=ablation == "no_representation",
        no_flow=ablation == "no_flow",
        no_jump=ablation == "no_jump",
        no_observation_head=ablation == "no_observation_head",
        no_probabilistic_scale=ablation == "no_prob_scale",
    )
    return model, model.model_observation_process


def _run_low_n_on_cohort(
    cohort: SimulatedCohort,
    config: ExperimentConfig,
    subset_seed: int,
    model_seed: int,
    model_names: Sequence[str] = MODEL_NAMES,
    ablations: Sequence[str] = ("none",),
    device: torch.device = _CPU_DEVICE,
    mlp_backend: str = "torch",
    prepared: _PreparedCohort | None = None,
    cell_callback: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    unknown = set(model_names).difference(MODEL_NAMES)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    unknown_ablations = set(ablations).difference(ABLATION_IDS)
    if unknown_ablations:
        raise ValueError(f"unknown ablations: {sorted(unknown_ablations)}")
    if mlp_backend not in {"torch", "sklearn"}:
        raise ValueError("mlp_backend must be 'torch' or 'sklearn'")
    variants = [
        (model_name, ablation)
        for model_name in model_names
        for ablation in (
            ablations if model_name.startswith("flow_jump") else ("none",)
        )
        if not (
            ablation == "no_observation_head"
            and model_name != "flow_jump_observation"
        )
    ]
    torch.set_num_threads(1)
    prepared = _prepare_cohort(cohort) if prepared is None else prepared
    patient_by_id = prepared.patient_by_id
    all_ids = list(patient_by_id)
    development_pool, _, test_ids = split_patient_ids(
        all_ids, seed=0, train_fraction=0.8, val_fraction=0.0
    )
    task = prepared.task
    sequences = prepared.sequences
    rows: list[dict[str, object]] = []
    for n_train in config.train_sizes:
        budget = select_low_n_budget(development_pool, n_train, subset_seed)
        train_ids = budget.fit_ids
        train_sequences = [sequences[patient_id] for patient_id in train_ids]
        validation_sequences = [
            sequences[patient_id] for patient_id in budget.validation_ids
        ]
        test_sequences = [sequences[patient_id] for patient_id in test_ids]
        test_patients = [patient_by_id[patient_id] for patient_id in test_ids]
        for model_name, ablation in variants:
            np.random.seed(model_seed)
            torch.manual_seed(model_seed)
            parameters = 0
            backend: str | None = None
            if model_name in {
                "engineered_linear",
                "gradient_boosting",
                "representation_linear",
                "representation_mlp",
            }:
                example_builder = (
                    _representation_examples
                    if model_name.startswith("representation_")
                    else _engineered_examples
                )
                train_x, train_y = example_builder(train_sequences)
                test_x, test_y = example_builder(test_sequences)
                if model_name in {"engineered_linear", "representation_linear"}:
                    estimator = TorchRidgeRegressor(device=device)
                elif model_name == "representation_mlp":
                    if mlp_backend == "torch":
                        estimator = TorchMLPRegressorBaseline(
                            input_dim=train_x.shape[1],
                            seed=model_seed,
                            device=device,
                        )
                        backend = "torch_lbfgs"
                    else:
                        estimator = MLPRegressorBaseline(seed=model_seed)
                        backend = "sklearn_lbfgs"
                else:
                    estimator = GradientBoostingRegressorBaseline()
                prediction = estimator.fit(train_x, train_y).predict(test_x)
                metrics = regression_metrics(test_y, prediction)
                if isinstance(
                    estimator,
                    (
                        MLPRegressorBaseline,
                        TorchMLPRegressorBaseline,
                        TorchRidgeRegressor,
                    ),
                ):
                    parameters = estimator.trainable_parameter_count()
            else:
                train_batch = _padded_batch(train_sequences)
                validation_batch = _padded_batch(validation_sequences)
                test_batch = _attach_latent_truth(
                    _padded_batch(test_sequences), test_patients
                )
                if model_name == "gru_from_scratch":
                    model: nn.Module = GRUBaseline(
                        len(task.value_codes), len(EventType)
                    )
                    observation_aware = False
                else:
                    model, observation_aware = _flow_jump_variant(
                        model_name,
                        ablation,
                        16,
                        len(task.value_codes),
                        len(EventType),
                    )
                model = _fit_neural(
                    model,
                    train_batch,
                    validation_batch,
                    config,
                    observation_aware,
                    device,
                )
                metrics = _evaluate_neural(model, test_batch, device)
                parameters = _parameter_count(model)
            cell_rows = [
                {
                    "model": model_name,
                    "ablation": ablation,
                    "n_train": n_train,
                    "n_fit": len(budget.fit_ids),
                    "n_validation": len(budget.validation_ids),
                    "seed": subset_seed,
                    "cohort_seed": cohort.seed,
                    "subset_seed": subset_seed,
                    "model_seed": model_seed,
                    "split": "test",
                    "site_or_shift": "all",
                    "metric": metric,
                    "value": value,
                    "trainable_parameters": parameters,
                    "backend": backend,
                }
                for metric, value in metrics.items()
            ]
            rows.extend(cell_rows)
            if cell_callback is not None:
                cell_callback(pd.DataFrame(cell_rows))
    return pd.DataFrame(rows)


def _cohort_for_seed(template: SimulatedCohort, cohort_seed: int) -> SimulatedCohort:
    if cohort_seed == template.seed:
        return template
    if template.config.world_name == "custom":
        return simulate_cohort(template.config, cohort_seed)
    return simulate_world(template.config.world_name, template.config, cohort_seed)


def run_low_n_benchmark(
    cohort: SimulatedCohort,
    config: ExperimentConfig,
    model_names: Sequence[str] | None = None,
    ablations: Sequence[str] | None = None,
    device: torch.device = _CPU_DEVICE,
    mlp_backend: str = "torch",
) -> pd.DataFrame:
    selected_models = tuple(config.models if model_names is None else model_names)
    selected_ablations = tuple(
        config.ablations if ablations is None else ablations
    )
    frames = [
        _run_low_n_on_cohort(
            _cohort_for_seed(cohort, cohort_seed),
            config,
            subset_seed,
            model_seed,
            selected_models,
            selected_ablations,
            device,
            mlp_backend,
        )
        for cohort_seed, subset_seed, model_seed in config.seed_bundles(cohort.seed)
    ]
    return pd.concat(frames, ignore_index=True)


def _run_observation_shift_on_cohort(
    cohort: SimulatedCohort,
    config: ExperimentConfig,
    subset_seed: int,
    model_seed: int,
    model_names: Sequence[str],
    ablations: Sequence[str],
    device: torch.device,
    prepared: _PreparedCohort | None = None,
    cell_callback: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    ids_by_site = {
        site_id: [
            patient.patient_id
            for patient in cohort.patients
            if patient.site_id == site_id
        ]
        for site_id in set(config.test_sites) | {config.train_site}
    }
    missing_sites = [site for site, ids in ids_by_site.items() if not ids]
    if missing_sites:
        raise ValueError(f"observation-shift sites have no patients: {missing_sites}")
    development_pool, _, train_site_test_ids = split_patient_ids(
        ids_by_site[config.train_site],
        seed=0,
        train_fraction=0.8,
        val_fraction=0.0,
    )
    prepared = _prepare_cohort(cohort) if prepared is None else prepared
    patient_by_id = prepared.patient_by_id
    task = prepared.task
    sequences = prepared.sequences
    variants = [
        (model_name, ablation)
        for model_name in model_names
        for ablation in ablations
        if not (
            ablation == "no_observation_head"
            and model_name != "flow_jump_observation"
        )
    ]
    rows: list[dict[str, object]] = []
    for n_train in config.train_sizes:
        budget = select_low_n_budget(development_pool, n_train, subset_seed)
        train_batch = _padded_batch(
            [sequences[patient_id] for patient_id in budget.fit_ids]
        )
        validation_batch = _padded_batch(
            [sequences[patient_id] for patient_id in budget.validation_ids]
        )
        evaluation_patients = {
            site: [
                patient_by_id[patient_id]
                for patient_id in (
                    train_site_test_ids
                    if site == config.train_site
                    else ids_by_site[site]
                )
            ]
            for site in config.test_sites
        }
        evaluation_batches = {
            site: _complete_truth_batch(
                [sequences[patient.patient_id] for patient in patients],
                patients,
            )
            for site, patients in evaluation_patients.items()
        }
        for model_name, ablation in variants:
            np.random.seed(model_seed)
            torch.manual_seed(model_seed)
            model, observation_aware = _flow_jump_variant(
                model_name,
                ablation,
                16,
                len(task.value_codes),
                len(EventType),
            )
            model = _fit_neural(
                model,
                train_batch,
                validation_batch,
                config,
                observation_aware,
                device,
            )
            by_site = {
                site: _evaluate_neural(model, batch, device)
                for site, batch in evaluation_batches.items()
            }
            parameters = _parameter_count(model)
            cell_rows: list[dict[str, object]] = []
            for site, metrics in by_site.items():
                for metric, value in metrics.items():
                    cell_rows.append(
                        {
                            "model": model_name,
                            "ablation": ablation,
                            "n_train": n_train,
                            "n_fit": len(budget.fit_ids),
                            "n_validation": len(budget.validation_ids),
                            "seed": subset_seed,
                            "cohort_seed": cohort.seed,
                            "subset_seed": subset_seed,
                            "model_seed": model_seed,
                            "split": "test",
                            "site_or_shift": f"site_{site}",
                            "metric": metric,
                            "value": value,
                            "trainable_parameters": parameters,
                        }
                    )
            reference = by_site[config.train_site]
            for site, metrics in by_site.items():
                if site == config.train_site:
                    continue
                for metric, value in metrics.items():
                    cell_rows.append(
                        {
                            "model": model_name,
                            "ablation": ablation,
                            "n_train": n_train,
                            "n_fit": len(budget.fit_ids),
                            "n_validation": len(budget.validation_ids),
                            "seed": subset_seed,
                            "cohort_seed": cohort.seed,
                            "subset_seed": subset_seed,
                            "model_seed": model_seed,
                            "split": "test",
                            "site_or_shift": (
                                f"site_{site}_minus_site_{config.train_site}"
                            ),
                            "metric": metric,
                            "value": value - reference[metric],
                            "trainable_parameters": parameters,
                        }
                    )
            rows.extend(cell_rows)
            if cell_callback is not None:
                cell_callback(pd.DataFrame(cell_rows))
    return pd.DataFrame(rows)


def _run_configured_benchmarks_on_cohort(
    cohort: SimulatedCohort,
    config: ExperimentConfig,
    subset_seed: int,
    model_seed: int,
    device: torch.device,
    cell_callback: Callable[[str, pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    prepared = _prepare_cohort(cohort)

    def emit_cell(benchmark: str, metrics: pd.DataFrame) -> None:
        metrics["world"] = cohort.config.world_name
        metrics["benchmark"] = benchmark
        if cell_callback is not None:
            cell_callback(benchmark, metrics)

    def emit_low_n_cell(metrics: pd.DataFrame) -> None:
        emit_cell("low_n", metrics)

    def emit_observation_shift_cell(metrics: pd.DataFrame) -> None:
        emit_cell("observation_shift", metrics)

    low_n = _run_low_n_on_cohort(
        cohort,
        config,
        subset_seed,
        model_seed,
        config.models,
        config.ablations,
        device,
        prepared=prepared,
        cell_callback=(emit_low_n_cell if cell_callback is not None else None),
    )
    low_n["world"] = cohort.config.world_name
    low_n["benchmark"] = "low_n"
    frames = [low_n]
    observation_models = tuple(
        model for model in config.models if model.startswith("flow_jump")
    )
    if cohort.config.world_name == "site_shift" and observation_models:
        observation_shift = _run_observation_shift_on_cohort(
            cohort,
            config,
            subset_seed,
            model_seed,
            observation_models,
            config.ablations,
            device,
            prepared=prepared,
            cell_callback=(
                emit_observation_shift_cell
                if cell_callback is not None
                else None
            ),
        )
        observation_shift["world"] = cohort.config.world_name
        observation_shift["benchmark"] = "observation_shift"
        frames.append(observation_shift)
    return pd.concat(frames, ignore_index=True)


def run_observation_shift_benchmark(
    cohort: SimulatedCohort,
    config: ExperimentConfig,
    model_names: Sequence[str] | None = None,
    ablations: Sequence[str] | None = None,
    device: torch.device = _CPU_DEVICE,
) -> pd.DataFrame:
    configured_models = config.models if model_names is None else model_names
    selected_models = tuple(
        model for model in configured_models if model.startswith("flow_jump")
    )
    if not selected_models:
        raise ValueError("observation-shift benchmark requires a flow-jump model")
    selected_ablations = tuple(
        config.ablations if ablations is None else ablations
    )
    frames = [
        _run_observation_shift_on_cohort(
            _cohort_for_seed(cohort, cohort_seed),
            config,
            subset_seed,
            model_seed,
            selected_models,
            selected_ablations,
            device,
        )
        for cohort_seed, subset_seed, model_seed in config.seed_bundles(cohort.seed)
    ]
    return pd.concat(frames, ignore_index=True)
