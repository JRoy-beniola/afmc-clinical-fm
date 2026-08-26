from dataclasses import FrozenInstanceError

import pandas as pd
import pytest
import torch

from afmc_fm.cli import _experiment_from_yaml
from afmc_fm.execution import jobs
from afmc_fm.execution.jobs import (
    CellResult,
    ShardSpec,
    plan_shards,
    run_shard,
)
from afmc_fm.execution.persistence import RunIdentity, RunStore
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


def test_benchmark_cell_id_preserves_the_task_6_persistence_schema():
    shard = ShardSpec("jumps", 101, 201, 301)

    assert jobs.benchmark_cell_id(
        "low_n", shard, 5, "engineered_linear", "none"
    ) == "low_n__jumps__cohort101__subset201__model301__n5__engineered_linear__none"


def test_run_shard_persists_before_interrupt_and_resumes_without_refitting_cell(
    tmp_path, monkeypatch
):
    experiment = ExperimentConfig(
        train_sizes=(5, 10),
        worlds=("smooth",),
        models=("engineered_linear",),
        ablations=("none",),
        max_epochs=1,
        patience=1,
    )
    spec = ShardSpec("smooth", 17, 23, 29)
    store = RunStore(
        tmp_path / "run",
        RunIdentity(
            protocol_anchor="be5a66b2e45362f60c90844e4e25673fb7bb3e21",
            simulator_config_hash="simulator-sha256",
            experiment_config_hash="experiment-sha256",
        ),
    )
    fit_calls: list[int] = []
    active_n_train: int | None = None
    real_fit = runner.TorchRidgeRegressor.fit
    real_select_budget = runner.select_low_n_budget

    def tracked_select_budget(development_pool, n, subset_seed, **kwargs):
        nonlocal active_n_train
        active_n_train = n
        return real_select_budget(development_pool, n, subset_seed, **kwargs)

    def counted_fit(estimator, features, targets):
        assert active_n_train is not None
        fit_calls.append(active_n_train)
        return real_fit(estimator, features, targets)

    monkeypatch.setattr(runner, "select_low_n_budget", tracked_select_budget)
    monkeypatch.setattr(runner.TorchRidgeRegressor, "fit", counted_fit)
    persisted_before_interrupt: list[str] = []

    def persist_then_interrupt(cell: CellResult) -> None:
        persisted_before_interrupt.append(store.write_cell(cell))
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_shard(
            spec,
            SimulatorConfig(cohort_size=30, followup_days=45.0),
            experiment,
            torch.device("cpu"),
            completed_cell_ids=frozenset(),
            on_cell_complete=persist_then_interrupt,
        )

    assert fit_calls == [5]
    completed = store.load_completed_cell_ids(spec.shard_id)
    assert completed == frozenset(persisted_before_interrupt)
    assert completed == frozenset(
        {"low_n__smooth__cohort17__subset23__model29__n5__engineered_linear__none"}
    )

    resumed_callbacks: list[str] = []
    fit_calls.clear()

    def persist_resumed(cell: CellResult) -> None:
        resumed_callbacks.append(store.write_cell(cell))

    resumed = run_shard(
        spec,
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        experiment,
        torch.device("cpu"),
        completed_cell_ids=completed,
        on_cell_complete=persist_resumed,
    )

    assert fit_calls == [10]
    assert resumed_callbacks == [
        "low_n__smooth__cohort17__subset23__model29__n10__engineered_linear__none"
    ]
    assert set(resumed["n_train"]) == {10}
    assert completed.isdisjoint(resumed_callbacks)


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


def test_run_shard_emits_one_typed_callback_per_completed_cell():
    experiment = ExperimentConfig(
        train_sizes=(5, 10),
        worlds=("smooth",),
        models=("engineered_linear", "flow_jump_observation"),
        ablations=("none", "no_flow"),
        max_epochs=1,
        patience=1,
    )
    spec = ShardSpec("smooth", 17, 23, 29)
    emitted: list[CellResult] = []

    aggregate = run_shard(
        spec,
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        experiment,
        torch.device("cpu"),
        cell_callback=emitted.append,
    )

    assert len(emitted) == 6
    assert {
        (cell.benchmark, cell.n_train, cell.model, cell.ablation)
        for cell in emitted
    } == {
        ("low_n", 5, "engineered_linear", "none"),
        ("low_n", 10, "engineered_linear", "none"),
        ("low_n", 5, "flow_jump_observation", "none"),
        ("low_n", 5, "flow_jump_observation", "no_flow"),
        ("low_n", 10, "flow_jump_observation", "none"),
        ("low_n", 10, "flow_jump_observation", "no_flow"),
    }
    for cell in emitted:
        assert cell.shard == spec
        assert cell.shard_id == spec.shard_id
        assert not cell.metrics.empty
        assert cell.metrics["benchmark"].eq(cell.benchmark).all()
        assert cell.metrics["n_train"].eq(cell.n_train).all()
        assert cell.metrics["model"].eq(cell.model).all()
        assert cell.metrics["ablation"].eq(cell.ablation).all()
    pd.testing.assert_frame_equal(
        pd.concat([cell.metrics for cell in emitted], ignore_index=True),
        aggregate,
    )


def test_run_shard_preserves_legacy_fifth_positional_callback():
    experiment = ExperimentConfig(
        train_sizes=(5,),
        worlds=("smooth",),
        models=("engineered_linear",),
        ablations=("none",),
        max_epochs=1,
        patience=1,
    )
    emitted: list[CellResult] = []

    aggregate = run_shard(
        ShardSpec("smooth", 17, 23, 29),
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        experiment,
        torch.device("cpu"),
        emitted.append,
    )

    assert len(emitted) == 1
    pd.testing.assert_frame_equal(emitted[0].metrics, aggregate)


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
    emitted: list[CellResult] = []

    results = run_shard(
        ShardSpec("site_shift", 31, 37, 41),
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        experiment,
        torch.device("cpu"),
        cell_callback=emitted.append,
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
    assert [(cell.benchmark, cell.model, cell.ablation) for cell in emitted] == [
        ("low_n", "flow_jump", "none"),
        ("observation_shift", "flow_jump", "none"),
    ]
    shift = emitted[1].metrics
    assert {"site_0", "site_1", "site_1_minus_site_0"} <= set(
        shift["site_or_shift"]
    )


def test_run_shard_skips_completed_observation_shift_before_neural_fit(
    monkeypatch,
):
    spec = ShardSpec("site_shift", 31, 37, 41)
    experiment = ExperimentConfig(
        train_sizes=(5,),
        worlds=("site_shift",),
        models=("flow_jump",),
        ablations=("none",),
        max_epochs=1,
        patience=1,
    )
    completed_observation_shift = jobs.benchmark_cell_id(
        "observation_shift", spec, 5, "flow_jump", "none"
    )
    fit_calls = 0
    real_fit = runner._fit_neural

    def counted_fit(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(runner, "_fit_neural", counted_fit)
    emitted: list[CellResult] = []

    results = run_shard(
        spec,
        SimulatorConfig(cohort_size=30, followup_days=45.0),
        experiment,
        torch.device("cpu"),
        completed_cell_ids=frozenset({completed_observation_shift}),
        on_cell_complete=emitted.append,
    )

    assert fit_calls == 1
    assert set(results["benchmark"]) == {"low_n"}
    assert [(cell.benchmark, cell.cell_id) for cell in emitted] == [
        (
            "low_n",
            "low_n__site_shift__cohort31__subset37__model41__n5__flow_jump__none",
        )
    ]
