from dataclasses import FrozenInstanceError

import pytest
import torch

from afmc_fm.cli import _experiment_from_yaml
from afmc_fm.execution import jobs
from afmc_fm.execution.jobs import ShardSpec, plan_shards, run_shard
from afmc_fm.experiments import runner
from afmc_fm.experiments.runner import ExperimentConfig
from afmc_fm.simulator.config import SimulatorConfig


def test_full_experiment_plans_25_unique_shards_in_config_order():
    experiment = _experiment_from_yaml("configs/experiments/low_n.yaml")

    shards = plan_shards(experiment, supplied_seed=42)

    assert len(shards) == 25
    assert len({shard.shard_id for shard in shards}) == 25
    assert tuple(shard.shard_id for shard in shards) == (
        "smooth__cohort101__subset201__model301",
        "smooth__cohort102__subset202__model302",
        "smooth__cohort103__subset203__model303",
        "smooth__cohort104__subset204__model304",
        "smooth__cohort105__subset205__model305",
        "jumps__cohort101__subset201__model301",
        "jumps__cohort102__subset202__model302",
        "jumps__cohort103__subset203__model303",
        "jumps__cohort104__subset204__model304",
        "jumps__cohort105__subset205__model305",
        "informative_observation__cohort101__subset201__model301",
        "informative_observation__cohort102__subset202__model302",
        "informative_observation__cohort103__subset203__model303",
        "informative_observation__cohort104__subset204__model304",
        "informative_observation__cohort105__subset205__model305",
        "site_shift__cohort101__subset201__model301",
        "site_shift__cohort102__subset202__model302",
        "site_shift__cohort103__subset203__model303",
        "site_shift__cohort104__subset204__model304",
        "site_shift__cohort105__subset205__model305",
        "misspecified__cohort101__subset201__model301",
        "misspecified__cohort102__subset202__model302",
        "misspecified__cohort103__subset203__model303",
        "misspecified__cohort104__subset204__model304",
        "misspecified__cohort105__subset205__model305",
    )


def test_plan_shards_preserves_matched_seed_tuple_order_without_cross_products():
    experiment = ExperimentConfig(
        worlds=("smooth", "jumps"),
        cohort_seeds=(41, 7),
        subset_seeds=(900, 4),
        model_seeds=(12, 808),
    )

    shards = plan_shards(experiment, supplied_seed=999)

    assert tuple(
        (shard.world, shard.cohort_seed, shard.subset_seed, shard.model_seed)
        for shard in shards
    ) == (
        ("smooth", 41, 900, 12),
        ("smooth", 7, 4, 808),
        ("jumps", 41, 900, 12),
        ("jumps", 7, 4, 808),
    )
    assert {
        (shard.cohort_seed, shard.subset_seed, shard.model_seed) for shard in shards
    } == {(41, 900, 12), (7, 4, 808)}


def test_plan_shards_uses_supplied_seed_when_cohort_seeds_are_not_configured():
    experiment = ExperimentConfig(
        worlds=("smooth",),
        cohort_seeds=(),
        subset_seeds=(23,),
        model_seeds=(29,),
    )

    assert plan_shards(experiment, supplied_seed=17) == (
        ShardSpec("smooth", 17, 23, 29),
    )


def test_shard_spec_is_immutable():
    shard = ShardSpec("smooth", 101, 201, 301)

    with pytest.raises(FrozenInstanceError):
        shard.model_seed = 999  # type: ignore[misc]


def test_run_shard_runs_every_configured_low_n_cell():
    experiment = ExperimentConfig(
        train_sizes=(5, 10),
        worlds=("smooth",),
        models=("engineered_linear", "flow_jump_observation"),
        ablations=("none", "no_flow"),
        max_epochs=1,
        patience=1,
    )
    spec = ShardSpec("smooth", 17, 23, 29)

    results = run_shard(
        spec,
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        experiment,
        torch.device("cpu"),
    )

    assert set(
        results[["n_train", "model", "ablation"]].itertuples(
            index=False, name=None
        )
    ) == {
        (5, "engineered_linear", "none"),
        (10, "engineered_linear", "none"),
        (5, "flow_jump_observation", "none"),
        (5, "flow_jump_observation", "no_flow"),
        (10, "flow_jump_observation", "none"),
        (10, "flow_jump_observation", "no_flow"),
    }
    assert set(results["world"]) == {"smooth"}
    assert set(results["benchmark"]) == {"low_n"}
    assert set(results["cohort_seed"]) == {17}
    assert set(results["subset_seed"]) == {23}
    assert set(results["model_seed"]) == {29}


def test_run_shard_simulates_once_and_encodes_each_patient_once(monkeypatch):
    simulation_calls: list[tuple[str, int]] = []
    encoded_patient_ids: list[str] = []
    real_simulate_world = jobs.simulate_world
    real_build_patient_sequence = runner.build_patient_sequence

    def counted_simulate_world(world, config, seed):
        simulation_calls.append((world, seed))
        return real_simulate_world(world, config, seed)

    def counted_build_patient_sequence(patient, encoder, task):
        encoded_patient_ids.append(patient.patient_id)
        return real_build_patient_sequence(patient, encoder, task)

    monkeypatch.setattr(jobs, "simulate_world", counted_simulate_world)
    monkeypatch.setattr(
        runner, "build_patient_sequence", counted_build_patient_sequence
    )
    experiment = ExperimentConfig(
        train_sizes=(5,),
        worlds=("site_shift",),
        models=("flow_jump",),
        ablations=("none",),
        max_epochs=1,
        patience=1,
    )

    results = run_shard(
        ShardSpec("site_shift", 31, 37, 41),
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        experiment,
        torch.device("cpu"),
    )

    assert simulation_calls == [("site_shift", 31)]
    assert len(encoded_patient_ids) == 30
    assert len(set(encoded_patient_ids)) == 30
    assert set(
        results[["benchmark", "n_train", "model", "ablation"]].itertuples(
            index=False, name=None
        )
    ) == {
        ("low_n", 5, "flow_jump", "none"),
        ("observation_shift", 5, "flow_jump", "none"),
    }
