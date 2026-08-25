from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from afmc_fm.data.splits import split_patient_ids
from afmc_fm.experiments.runner import (
    _evaluate_neural,
    _fit_neural,
    _representation_examples,
    build_complete_truth_targets,
    select_low_n_budget,
)
from afmc_fm.metrics.forecasting import regression_metrics
from afmc_fm.models.baselines import GRUBaseline, TorchMLPRegressorBaseline
from afmc_fm.phase05.analysis import normalized_log_n_aulc
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.confirmation import run_confirmation_job
from afmc_fm.phase05.execution import Phase05Job, Phase05ShardSpec, plan_phase05_shards
from afmc_fm.phase05.model import Phase05FlowJumpAdapter
from afmc_fm.phase05.protocol import FrozenCandidate
from afmc_fm.phase05.runner import PreparedPhase05Cohort, padded_phase05_batch
from afmc_fm.phase05.sequences import Phase05Sequence
from afmc_fm.phase05.training import evaluate_phase05_model, fit_phase05_model
from afmc_fm.schema.events import EventType
from afmc_fm.simulator.cohort import SimulatedPatient

_BUNDLE_COLUMNS = ["cohort_seed", "subset_seed", "model_seed"]
_DEFAULT_CANDIDATE = "phase05_candidate"
_DEFAULT_COMPARATORS = ("matched_gru", "matched_representation_mlp")
ROBUSTNESS_MODELS = (
    "phase05_candidate",
    "matched_gru",
    "matched_representation_mlp",
)


def _robustness_variants(frozen: FrozenCandidate) -> dict[str, str]:
    return {
        "phase05_candidate": (
            f"{frozen.flow_mode}__{frozen.jump_mode}__{frozen.uncertainty_mode}"
        ),
        "matched_gru": f"hidden{frozen.matched_gru_hidden_size}",
        "matched_representation_mlp": f"hidden{frozen.matched_mlp_hidden_size}",
    }


def build_robustness_jobs(
    config: Phase05Config,
    frozen: FrozenCandidate,
    *,
    frozen_candidate_hash: str,
) -> tuple[Phase05Job, ...]:
    variants = _robustness_variants(frozen)
    return tuple(
        Phase05Job(
            shard=shard,
            n_train=n_train,
            model=model,
            variant=variants[model],
            frozen_candidate_hash=frozen_candidate_hash,
        )
        for shard in plan_phase05_shards(config, "robustness")
        for n_train in config.train_sizes
        for model in ROBUSTNESS_MODELS
    )


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


def _complete_truth_representation_examples(
    sequences: list[Phase05Sequence],
    patients: list[SimulatedPatient],
) -> tuple[np.ndarray, np.ndarray]:
    if len(sequences) != len(patients):
        raise ValueError("patients and sequences must have equal lengths")
    features: list[np.ndarray] = []
    targets: list[float] = []
    for sequence, patient in zip(sequences, patients, strict=True):
        complete_values, complete_masks = build_complete_truth_targets(patient)
        if len(complete_values) != len(sequence.representations):
            raise ValueError(
                "complete-truth targets must align with Phase-0.5 representation steps"
            )
        value_dim = complete_masks.shape[1]
        for step, lab in np.argwhere(complete_masks > 0):
            lab_indicator = np.zeros(value_dim)
            lab_indicator[lab] = 1.0
            features.append(
                np.concatenate([sequence.representations[step], lab_indicator])
            )
            targets.append(float(complete_values[step, lab]))
    if not features:
        raise ValueError("complete-truth evaluation contains no forecasting targets")
    return np.asarray(features), np.asarray(targets)


def _site_shift_budget(
    prepared: PreparedPhase05Cohort,
    job: Phase05Job,
    config: Phase05Config,
):
    sites = set(config.test_sites) | {config.train_site}
    ids_by_site = {
        site: [
            patient.patient_id
            for patient in prepared.patient_by_id.values()
            if patient.site_id == site
        ]
        for site in sites
    }
    missing_sites = [site for site, patient_ids in ids_by_site.items() if not patient_ids]
    if missing_sites:
        raise ValueError(f"site-shift sites have no patients: {missing_sites}")
    if config.train_site not in config.test_sites:
        raise ValueError("site-shift test sites must include the training site")
    development_pool, _, train_site_test_ids = split_patient_ids(
        ids_by_site[config.train_site],
        seed=0,
        train_fraction=0.8,
        val_fraction=0.0,
    )
    budget = select_low_n_budget(
        development_pool,
        job.n_train,
        job.shard.seed_bundle.subset_seed,
    )
    evaluation_ids = {
        site: (
            train_site_test_ids
            if site == config.train_site
            else ids_by_site[site]
        )
        for site in config.test_sites
    }
    return budget, evaluation_ids


def _robustness_rows(
    job: Phase05Job,
    *,
    metrics_by_site: dict[int, dict[str, float]],
    n_fit: int,
    n_validation: int,
    trainable_parameters: int,
    backend: str,
) -> pd.DataFrame:
    bundle = job.shard.seed_bundle
    return pd.DataFrame(
        [
            {
                "stage": "robustness",
                "world": job.shard.world,
                "cohort_seed": bundle.cohort_seed,
                "subset_seed": bundle.subset_seed,
                "model_seed": bundle.model_seed,
                "n_train": job.n_train,
                "n_fit": n_fit,
                "n_validation": n_validation,
                "model": job.model,
                "variant": job.variant,
                "split": "test",
                "site_or_shift": f"site_{site}",
                "metric": metric,
                "value": value,
                "trainable_parameters": trainable_parameters,
                "backend": backend,
            }
            for site, metrics in metrics_by_site.items()
            for metric, value in metrics.items()
        ]
    )


def _expected_parameter_count(job: Phase05Job, frozen: FrozenCandidate) -> int:
    if job.model == "phase05_candidate":
        return frozen.trainable_parameters
    if job.model == "matched_gru":
        return frozen.matched_gru_parameters
    if job.model == "matched_representation_mlp":
        return frozen.matched_mlp_parameters
    raise ValueError(f"unknown robustness model: {job.model}")


def _validate_parameter_count(
    job: Phase05Job,
    frozen: FrozenCandidate,
    actual: int,
) -> None:
    expected = _expected_parameter_count(job, frozen)
    if actual != expected:
        raise RuntimeError(
            f"{job.model} parameter count differs from frozen capacity definition"
        )


def _run_site_shift_job(
    job: Phase05Job,
    prepared: PreparedPhase05Cohort,
    *,
    config: Phase05Config,
    frozen: FrozenCandidate,
    device: torch.device,
) -> pd.DataFrame:
    budget, evaluation_ids = _site_shift_budget(prepared, job, config)
    train_sequences = [prepared.sequences[patient_id] for patient_id in budget.fit_ids]
    validation_sequences = [
        prepared.sequences[patient_id] for patient_id in budget.validation_ids
    ]
    evaluation_sequences = {
        site: [prepared.sequences[patient_id] for patient_id in patient_ids]
        for site, patient_ids in evaluation_ids.items()
    }
    evaluation_patients = {
        site: [prepared.patient_by_id[patient_id] for patient_id in patient_ids]
        for site, patient_ids in evaluation_ids.items()
    }
    bundle = job.shard.seed_bundle
    np.random.seed(bundle.model_seed)
    torch.manual_seed(bundle.model_seed)

    if job.model == "phase05_candidate":
        model = Phase05FlowJumpAdapter(
            representation_dim=16,
            value_dim=len(prepared.task.value_codes),
            event_dim=len(EventType),
            state_dim=frozen.state_dim,
            flow_mode=frozen.flow_mode,
            jump_mode=frozen.jump_mode,
            uncertainty_mode=frozen.uncertainty_mode,
            time_scale_days=frozen.time_scale_days,
        )
        fit_phase05_model(
            model,
            padded_phase05_batch(train_sequences),
            padded_phase05_batch(validation_sequences),
            config,
            device,
        )
        metrics_by_site = {
            site: evaluate_phase05_model(
                model,
                complete_truth_phase05_batch(
                    evaluation_sequences[site], evaluation_patients[site]
                ),
                device,
            )
            for site in config.test_sites
        }
        parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        backend = "torch"
    elif job.model == "matched_gru":
        model = GRUBaseline(
            value_dim=len(prepared.task.value_codes),
            event_dim=len(EventType),
            hidden_size=frozen.matched_gru_hidden_size,
        )
        model = _fit_neural(
            model,
            padded_phase05_batch(train_sequences),
            padded_phase05_batch(validation_sequences),
            config,
            False,
            device,
        )
        metrics_by_site = {
            site: _evaluate_neural(
                model,
                complete_truth_phase05_batch(
                    evaluation_sequences[site], evaluation_patients[site]
                ),
                device,
            )
            for site in config.test_sites
        }
        parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        backend = "torch"
    else:
        train_x, train_y = _representation_examples(train_sequences)
        estimator = TorchMLPRegressorBaseline(
            input_dim=train_x.shape[1],
            seed=bundle.model_seed,
            device=device,
            hidden_size=frozen.matched_mlp_hidden_size,
        )
        estimator.fit(train_x, train_y)
        metrics_by_site = {}
        for site in config.test_sites:
            test_x, test_y = _complete_truth_representation_examples(
                evaluation_sequences[site], evaluation_patients[site]
            )
            metrics_by_site[site] = regression_metrics(
                test_y, estimator.predict(test_x)
            )
        parameters = estimator.trainable_parameter_count()
        backend = "torch_lbfgs"

    _validate_parameter_count(job, frozen, parameters)
    return _robustness_rows(
        job,
        metrics_by_site=metrics_by_site,
        n_fit=len(budget.fit_ids),
        n_validation=len(budget.validation_ids),
        trainable_parameters=parameters,
        backend=backend,
    )


def _run_misspecified_job(
    job: Phase05Job,
    prepared: PreparedPhase05Cohort,
    *,
    config: Phase05Config,
    frozen: FrozenCandidate,
    device: torch.device,
) -> pd.DataFrame:
    surrogate = Phase05Job(
        shard=Phase05ShardSpec(
            "confirmation",
            job.shard.world,
            job.shard.seed_bundle,
        ),
        n_train=job.n_train,
        model=job.model,
        variant=job.variant,
        frozen_candidate_hash=job.frozen_candidate_hash,
    )
    frame = run_confirmation_job(
        surrogate,
        prepared,
        config=config,
        frozen=frozen,
        device=device,
    ).copy()
    frame["stage"] = "robustness"
    frame["world"] = job.shard.world
    frame["model"] = job.model
    frame["variant"] = job.variant
    parameters = {int(value) for value in frame["trainable_parameters"]}
    if len(parameters) != 1:
        raise RuntimeError("robustness job emitted inconsistent parameter counts")
    _validate_parameter_count(job, frozen, next(iter(parameters)))
    return frame


def run_robustness_job(
    job: Phase05Job,
    prepared: PreparedPhase05Cohort,
    *,
    config: Phase05Config,
    frozen: FrozenCandidate,
    device: torch.device,
) -> pd.DataFrame:
    if job.shard.stage != "robustness":
        raise ValueError("run_robustness_job requires a robustness-stage job")
    if job.shard.world not in config.robustness_worlds:
        raise ValueError("robustness job world is outside the locked robustness worlds")
    expected_variants = _robustness_variants(frozen)
    if job.model not in expected_variants:
        raise ValueError(f"unknown robustness model: {job.model}")
    if job.variant != expected_variants[job.model]:
        raise ValueError("robustness job variant does not match the frozen model definition")
    if job.shard.world == "site_shift":
        return _run_site_shift_job(
            job,
            prepared,
            config=config,
            frozen=frozen,
            device=device,
        )
    if job.shard.world == "misspecified":
        return _run_misspecified_job(
            job,
            prepared,
            config=config,
            frozen=frozen,
            device=device,
        )
    raise ValueError(f"unsupported robustness world: {job.shard.world}")


def _site_metric_value(
    group: pd.DataFrame,
    metric: str,
    site: int,
    *,
    required: bool,
) -> float | None:
    rows = group.loc[
        (group["metric"] == metric) & (group["site_or_shift"] == f"site_{site}"),
        "value",
    ]
    if rows.empty and not required:
        return None
    if len(rows) != 1:
        expectation = "exactly one" if required else "at most one"
        raise ValueError(
            f"site-shift evaluation requires {expectation} {metric} row for site {site}"
        )
    value = float(rows.iloc[0])
    if not np.isfinite(value):
        raise ValueError("site-shift metric values must be finite")
    return value


def evaluate_site_shift(
    metrics: pd.DataFrame,
    *,
    candidate: str = _DEFAULT_CANDIDATE,
    comparators: tuple[str, str] = _DEFAULT_COMPARATORS,
    win_requirement: int = 8,
) -> dict[str, pd.DataFrame]:
    if len(comparators) != 2 or len(set(comparators)) != 2:
        raise ValueError("site-shift gate requires exactly two distinct comparators")
    if candidate in comparators:
        raise ValueError("candidate must be distinct from site-shift comparators")
    if not 1 <= win_requirement <= 10:
        raise ValueError("win_requirement must be between one and ten")

    required_columns = {
        "world",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "n_train",
        "model",
        "site_or_shift",
        "metric",
        "value",
    }
    missing = required_columns - set(metrics.columns)
    if missing:
        raise ValueError(f"missing site-shift metric columns: {sorted(missing)}")

    selected_models = (candidate, *comparators)
    frame = metrics.loc[
        (metrics["world"] == "site_shift")
        & metrics["model"].isin(selected_models)
        & metrics["site_or_shift"].isin(("site_0", "site_1"))
    ].copy()
    if "split" in frame.columns:
        frame = frame.loc[frame["split"] == "test"]
    if frame.empty:
        raise ValueError("site-shift metrics contain no supported rows")

    rows: list[dict[str, object]] = []
    group_columns = ["n_train", "model", *_BUNDLE_COLUMNS]
    for key, group in frame.groupby(group_columns, sort=True):
        n_train, model, cohort_seed, subset_seed, model_seed = key
        site0_mae = _site_metric_value(group, "mae", 0, required=True)
        site1_mae = _site_metric_value(group, "mae", 1, required=True)
        assert site0_mae is not None and site1_mae is not None
        if site0_mae <= 0:
            raise ValueError("site-0 MAE must be positive for relative degradation")
        row: dict[str, object] = {
            "world": "site_shift",
            "cohort_seed": int(cohort_seed),
            "subset_seed": int(subset_seed),
            "model_seed": int(model_seed),
            "n_train": int(n_train),
            "model": str(model),
            "mae_site_0": site0_mae,
            "mae_site_1": site1_mae,
            "mae_absolute_degradation": site1_mae - site0_mae,
            "mae_relative_degradation": (site1_mae - site0_mae) / site0_mae,
        }

        site0_nll = _site_metric_value(group, "nll", 0, required=False)
        site1_nll = _site_metric_value(group, "nll", 1, required=False)
        if (site0_nll is None) != (site1_nll is None):
            raise ValueError("site-shift NLL must be present for both sites or neither")
        if site0_nll is not None and site1_nll is not None:
            row["nll_site_0"] = site0_nll
            row["nll_site_1"] = site1_nll
            row["nll_absolute_degradation"] = site1_nll - site0_nll

        site0_coverage = _site_metric_value(group, "coverage_90", 0, required=False)
        site1_coverage = _site_metric_value(group, "coverage_90", 1, required=False)
        if (site0_coverage is None) != (site1_coverage is None):
            raise ValueError(
                "site-shift 90% coverage must be present for both sites or neither"
            )
        if site0_coverage is not None and site1_coverage is not None:
            ce90_site0 = abs(site0_coverage - 0.90)
            ce90_site1 = abs(site1_coverage - 0.90)
            row["coverage_90_site_0"] = site0_coverage
            row["coverage_90_site_1"] = site1_coverage
            row["ce90_site_0"] = ce90_site0
            row["ce90_site_1"] = ce90_site1
            row["ce90_absolute_degradation"] = ce90_site1 - ce90_site0
        rows.append(row)

    per_seed = pd.DataFrame(rows).sort_values(
        ["n_train", "model", *_BUNDLE_COLUMNS], kind="mergesort"
    ).reset_index(drop=True)

    gate_rows: list[dict[str, object]] = []
    for n_train in sorted(per_seed["n_train"].unique()):
        candidate_rows = per_seed.loc[
            (per_seed["n_train"] == n_train) & (per_seed["model"] == candidate),
            [*_BUNDLE_COLUMNS, "mae_absolute_degradation"],
        ]
        candidate_bundles = set(
            candidate_rows[_BUNDLE_COLUMNS].itertuples(index=False, name=None)
        )
        if len(candidate_bundles) != 10:
            raise ValueError(
                "site-shift gate requires exactly ten candidate confirmatory bundles per N"
            )
        for comparator in comparators:
            comparator_rows = per_seed.loc[
                (per_seed["n_train"] == n_train)
                & (per_seed["model"] == comparator),
                [*_BUNDLE_COLUMNS, "mae_absolute_degradation"],
            ]
            comparator_bundles = set(
                comparator_rows[_BUNDLE_COLUMNS].itertuples(index=False, name=None)
            )
            if comparator_bundles != candidate_bundles:
                raise ValueError(
                    "site-shift gate requires ten exactly matched confirmatory bundles"
                )
            paired = candidate_rows.merge(
                comparator_rows,
                on=_BUNDLE_COLUMNS,
                suffixes=("_candidate", "_comparator"),
                validate="one_to_one",
            )
            advantages = (
                paired["mae_absolute_degradation_comparator"].to_numpy(dtype=float)
                - paired["mae_absolute_degradation_candidate"].to_numpy(dtype=float)
            )
            wins = int(np.sum(advantages > 0))
            candidate_mean = float(
                paired["mae_absolute_degradation_candidate"].mean()
            )
            comparator_mean = float(
                paired["mae_absolute_degradation_comparator"].mean()
            )
            gate_rows.append(
                {
                    "n_train": int(n_train),
                    "comparator": comparator,
                    "candidate_mean_mae_degradation": candidate_mean,
                    "comparator_mean_mae_degradation": comparator_mean,
                    "mean_degradation_advantage": float(np.mean(advantages)),
                    "wins": wins,
                    "win_fraction": wins / 10.0,
                    "passed": bool(
                        candidate_mean < comparator_mean
                        and wins >= win_requirement
                    ),
                }
            )

    gate_summary = pd.DataFrame(gate_rows).reset_index(drop=True)
    return {"per_seed": per_seed, "gate_summary": gate_summary}


def _misspecified_model_aulcs(metrics: pd.DataFrame, model: str) -> pd.DataFrame:
    frame = metrics.loc[
        (metrics["world"] == "misspecified")
        & (metrics["model"] == model)
        & (metrics["metric"] == "mae")
    ].copy()
    if "split" in frame.columns:
        frame = frame.loc[frame["split"] == "test"]
    if "site_or_shift" in frame.columns:
        frame = frame.loc[frame["site_or_shift"] == "all"]
    rows: list[dict[str, object]] = []
    for bundle, group in frame.groupby(_BUNDLE_COLUMNS, sort=True):
        rows.append(
            {
                "cohort_seed": int(bundle[0]),
                "subset_seed": int(bundle[1]),
                "model_seed": int(bundle[2]),
                "naulc": normalized_log_n_aulc(group),
            }
        )
    return pd.DataFrame(rows)


def evaluate_misspecification(
    metrics: pd.DataFrame,
    *,
    candidate: str = _DEFAULT_CANDIDATE,
    comparators: tuple[str, str] = _DEFAULT_COMPARATORS,
    tolerance: float = 0.05,
) -> dict[str, object]:
    if len(comparators) != 2 or len(set(comparators)) != 2:
        raise ValueError("misspecification evaluation requires exactly two comparators")
    if candidate in comparators:
        raise ValueError("candidate must be distinct from misspecification comparators")
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("misspecification tolerance must be finite and between zero and one")
    required_columns = {
        "world",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "n_train",
        "model",
        "metric",
        "value",
    }
    missing = required_columns - set(metrics.columns)
    if missing:
        raise ValueError(f"missing misspecification metric columns: {sorted(missing)}")

    candidate_aulc = _misspecified_model_aulcs(metrics, candidate)
    comparator_aulcs = [
        _misspecified_model_aulcs(metrics, comparator) for comparator in comparators
    ]
    candidate_bundles = set(
        candidate_aulc[_BUNDLE_COLUMNS].itertuples(index=False, name=None)
    )
    if len(candidate_bundles) != 10:
        raise ValueError(
            "misspecification evaluation requires exactly ten candidate confirmatory bundles"
        )
    for comparator_aulc in comparator_aulcs:
        comparator_bundles = set(
            comparator_aulc[_BUNDLE_COLUMNS].itertuples(index=False, name=None)
        )
        if comparator_bundles != candidate_bundles:
            raise ValueError(
                "misspecification evaluation requires ten exactly matched confirmatory bundles"
            )

    per_seed = candidate_aulc.rename(columns={"naulc": "candidate_naulc"})
    for comparator, comparator_aulc in zip(
        comparators, comparator_aulcs, strict=True
    ):
        per_seed = per_seed.merge(
            comparator_aulc.rename(columns={"naulc": f"{comparator}_naulc"}),
            on=_BUNDLE_COLUMNS,
            validate="one_to_one",
        )
    control_columns = [f"{comparator}_naulc" for comparator in comparators]
    control_values = per_seed[control_columns].to_numpy(dtype=float)
    if not np.isfinite(control_values).all() or np.any(control_values <= 0):
        raise ValueError("misspecification control nAULC values must be finite and positive")
    best_indices = np.argmin(control_values, axis=1)
    per_seed["best_control_model"] = [comparators[index] for index in best_indices]
    per_seed["best_control_naulc"] = control_values[
        np.arange(len(per_seed)), best_indices
    ]
    per_seed["relative_excess"] = (
        per_seed["candidate_naulc"] - per_seed["best_control_naulc"]
    ) / per_seed["best_control_naulc"]
    per_seed = per_seed.sort_values(_BUNDLE_COLUMNS, kind="mergesort").reset_index(drop=True)
    relative_excess = per_seed["relative_excess"].to_numpy(dtype=float)
    if not np.isfinite(relative_excess).all():
        raise ValueError("misspecification relative excess must be finite")
    mean_relative_excess = float(np.mean(relative_excess))
    summary = {
        "mean_relative_excess": mean_relative_excess,
        "median_relative_excess": float(np.median(relative_excess)),
        "tolerance": float(tolerance),
        "passed": bool(mean_relative_excess <= tolerance),
    }
    return {"per_seed": per_seed, "summary": summary}


__all__ = [
    "ROBUSTNESS_MODELS",
    "build_robustness_jobs",
    "complete_truth_phase05_batch",
    "evaluate_misspecification",
    "evaluate_site_shift",
    "run_robustness_job",
]
