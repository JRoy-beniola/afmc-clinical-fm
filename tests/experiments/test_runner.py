import numpy as np

from afmc_fm.data.encoding import SummaryHistoryEncoder
from afmc_fm.data.sequences import build_patient_sequence
from afmc_fm.data.splits import split_patient_ids
from afmc_fm.data.tasks import LongitudinalTask
from afmc_fm.experiments.runner import (
    ExperimentConfig,
    build_complete_truth_targets,
    run_low_n_benchmark,
    run_observation_shift_benchmark,
    sample_low_n_train_ids,
    select_low_n_budget,
)
from afmc_fm.simulator.cohort import simulate_cohort, simulate_world
from afmc_fm.simulator.config import SimulatorConfig


def test_low_n_sampling_is_exact_and_reproducible():
    pool = [f"p{i}" for i in range(200)]
    a = sample_low_n_train_ids(pool, n=40, seed=3)
    b = sample_low_n_train_ids(pool, n=40, seed=3)
    assert a == b
    assert len(a) == 40
    assert len(set(a)) == 40


def test_low_n_ids_are_sampled_only_from_training_pool():
    ids = [f"p{i}" for i in range(100)]
    train, validation, test = split_patient_ids(ids, seed=4)
    sampled = set(sample_low_n_train_ids(train, n=20, seed=2))
    assert sampled.isdisjoint(validation)
    assert sampled.isdisjoint(test)


def test_tiny_benchmark_returns_unique_tidy_metric_rows():
    cohort = simulate_cohort(
        SimulatorConfig(cohort_size=36, followup_days=45.0), seed=8
    )
    config = ExperimentConfig(train_sizes=(5,), seeds=(1,), max_epochs=2, patience=1)
    results = run_low_n_benchmark(
        cohort,
        config,
        model_names=("probe_linear", "gradient_boosting"),
    )
    assert not results.empty
    assert results["model"].nunique() == 2
    assert (results["n_fit"] + results["n_validation"] == results["n_train"]).all()
    assert (results["n_validation"] >= 1).all()
    key = ["model", "n_train", "seed", "site_or_shift", "metric"]
    assert not results.duplicated(key).any()


def test_named_worlds_preserve_interface_but_change_generating_assumptions():
    config = SimulatorConfig(cohort_size=4, followup_days=45.0)
    smooth = simulate_world("smooth", config, seed=5)
    misspecified = simulate_world("misspecified", config, seed=5)
    assert type(smooth) is type(misspecified)
    assert len(smooth.patients) == len(misspecified.patients) == 4
    assert smooth.config.world_name == "smooth"
    assert misspecified.config.world_name == "misspecified"
    assert not np.allclose(
        smooth.patients[0].parameters.progression_scale,
        misspecified.patients[0].parameters.progression_scale,
    )


def test_observation_shift_reports_both_sites_for_observation_models():
    cohort = simulate_world(
        "site_shift",
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        seed=9,
    )
    config = ExperimentConfig(train_sizes=(5,), seeds=(1,), max_epochs=1, patience=1)
    results = run_observation_shift_benchmark(cohort, config)
    assert set(results["model"]) == {"flow_jump", "flow_jump_observation"}
    raw = results[results["site_or_shift"].isin(["site_0", "site_1"])]
    for model_name in ("flow_jump", "flow_jump_observation"):
        assert set(raw.loc[raw["model"] == model_name, "site_or_shift"]) == {
            "site_0",
            "site_1",
        }


def test_fit_and_validation_patients_stay_within_total_low_n_budget():
    development = [f"dev-{index}" for index in range(80)]
    final_test = {f"test-{index}" for index in range(20)}
    budget = select_low_n_budget(
        development,
        n=5,
        subset_seed=11,
        validation_fraction=0.2,
    )
    labelled = set(budget.fit_ids) | set(budget.validation_ids)
    assert len(labelled) == 5
    assert set(budget.fit_ids).isdisjoint(budget.validation_ids)
    assert labelled <= set(development)
    assert labelled.isdisjoint(final_test)


def test_complete_truth_targets_do_not_condition_on_observation_masks():
    patient = simulate_world(
        "site_shift",
        SimulatorConfig(cohort_size=1, followup_days=90.0),
        seed=14,
    ).patients[0]
    task = LongitudinalTask(patient.complete_outcomes.value_codes)
    sequence = build_patient_sequence(
        patient,
        SummaryHistoryEncoder(task, representation_dim=16, seed=0),
        task,
    )
    targets, masks = build_complete_truth_targets(patient)
    assert targets.shape == sequence.target_next_values.shape
    assert masks.shape == sequence.target_next_masks.shape
    assert masks.sum() >= sequence.target_next_masks.sum()
    assert np.all(masks.sum(axis=1) % 3 == 0)
