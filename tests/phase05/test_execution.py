from importlib import import_module

import pandas as pd
import pytest
import torch
from threadpoolctl import threadpool_info

from afmc_fm.phase05.config import Phase05Config, SeedBundle

_execution = import_module("afmc_fm.phase05.execution")
Phase05ExecutionOptions = _execution.Phase05ExecutionOptions
Phase05ShardSpec = _execution.Phase05ShardSpec
phase05_cell_id = _execution.phase05_cell_id
resolve_phase05_device = _execution.resolve_phase05_device
spawn_context = _execution.spawn_context
limit_worker_threads = _execution.limit_worker_threads


def test_shard_and_cell_ids_bind_stage_world_seed_bundle_and_variant():
    shard = Phase05ShardSpec(
        stage="flow",
        world="smooth",
        seed_bundle=SeedBundle(401, 501, 601),
    )

    assert shard.shard_id == "flow__smooth__cohort401__subset501__model601"
    assert phase05_cell_id(
        shard,
        n_train=20,
        model="phase05_flow_jump",
        variant="time_scaled__none__deterministic",
    ) == (
        "flow__smooth__cohort401__subset501__model601__n20__"
        "phase05_flow_jump__time_scaled__none__deterministic"
    )


def test_execution_options_lock_supported_device_and_worker_domain():
    assert Phase05ExecutionOptions() == Phase05ExecutionOptions(
        device="auto",
        workers=1,
        resume=False,
        fail_fast=False,
    )

    with pytest.raises(ValueError, match="workers must be at least 1"):
        Phase05ExecutionOptions(workers=0)
    with pytest.raises(ValueError, match="device must be auto, cpu, or cuda"):
        Phase05ExecutionOptions(device="mps")


def test_phase05_execution_uses_spawn_context():
    assert spawn_context().get_start_method() == "spawn"


def test_worker_thread_limits_set_torch_and_cpu_environment(monkeypatch):
    for variable in _execution.THREAD_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    previous = torch.get_num_threads()
    try:
        environment = limit_worker_threads()

        assert torch.get_num_threads() == 1
        assert environment == {
            variable: "1" for variable in _execution.THREAD_ENVIRONMENT_VARIABLES
        }
    finally:
        torch.set_num_threads(previous)


def test_worker_threadpool_context_caps_loaded_numeric_libraries():
    with _execution.worker_thread_limits():
        pools = threadpool_info()
        assert all(int(pool["num_threads"]) <= 1 for pool in pools)


def test_explicit_unavailable_cuda_hard_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested but is not available"):
        resolve_phase05_device("cuda")


def test_stage_planning_uses_locked_seed_roles_and_world_roles():
    config = Phase05Config()

    flow = _execution.plan_phase05_shards(config, "flow")
    jump = _execution.plan_phase05_shards(config, "jump")
    uncertainty = _execution.plan_phase05_shards(config, "uncertainty")
    confirmation = _execution.plan_phase05_shards(config, "confirmation")
    robustness = _execution.plan_phase05_shards(config, "robustness")

    assert {shard.world for shard in flow} == {"smooth"}
    assert {shard.seed_bundle for shard in flow} == set(config.development_bundles)
    assert len(flow) == len(config.development_bundles)

    assert {shard.world for shard in jump} == {"jumps"}
    assert {shard.seed_bundle for shard in jump} == set(config.development_bundles)
    assert len(jump) == len(config.development_bundles)

    assert {shard.world for shard in uncertainty} == set(config.target_worlds)
    assert {shard.seed_bundle for shard in uncertainty} == set(
        config.development_bundles
    )
    assert len(uncertainty) == len(config.target_worlds) * len(
        config.development_bundles
    )

    assert {shard.world for shard in confirmation} == set(config.target_worlds)
    assert {shard.seed_bundle for shard in confirmation} == set(
        config.confirmatory_bundles
    )
    assert len(confirmation) == len(config.target_worlds) * len(
        config.confirmatory_bundles
    )

    assert {shard.world for shard in robustness} == set(config.robustness_worlds)
    assert {shard.seed_bundle for shard in robustness} == set(
        config.confirmatory_bundles
    )
    assert len(robustness) == len(config.robustness_worlds) * len(
        config.confirmatory_bundles
    )


def test_final_aggregation_order_is_completion_order_independent():
    first = pd.DataFrame(
        [
            {
                "stage": "flow",
                "world": "smooth",
                "cohort_seed": 401,
                "subset_seed": 501,
                "model_seed": 601,
                "n_train": 5,
                "model": "phase05_flow_jump",
                "variant": "time_scaled__none__deterministic",
                "metric": "mae",
                "value": 2.0,
            }
        ]
    )
    second = first.assign(n_train=10, value=1.5)

    forward = _execution.aggregate_phase05_frames([first, second])
    reverse = _execution.aggregate_phase05_frames([second, first])

    pd.testing.assert_frame_equal(forward, reverse)
    assert forward["n_train"].tolist() == [5, 10]
