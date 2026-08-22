from afmc_fm.data.splits import split_patient_ids
from afmc_fm.experiments.runner import (
    ExperimentConfig,
    run_low_n_benchmark,
    sample_low_n_train_ids,
)
from afmc_fm.simulator.cohort import simulate_cohort
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
    key = ["model", "n_train", "seed", "site_or_shift", "metric"]
    assert not results.duplicated(key).any()
