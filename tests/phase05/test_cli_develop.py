import pytest

import afmc_fm.cli as cli
from afmc_fm.phase05.config import load_phase05_config


def _job_variants(jobs):
    return {job.variant for job in jobs}


def _job_worlds(jobs):
    return {job.shard.world for job in jobs}


def _job_bundles(jobs):
    return {job.shard.seed_bundle for job in jobs}


def test_development_job_plans_are_sequential_and_use_only_primary_n():
    config = load_phase05_config("configs/experiments/phase05.yaml")

    flow = cli._phase05_development_jobs(config, "flow")
    assert len(flow) == 60
    assert {job.shard.stage for job in flow} == {"flow"}
    assert _job_worlds(flow) == {"smooth"}
    assert _job_bundles(flow) == set(config.development_bundles)
    assert {job.n_train for job in flow} == set(config.primary_train_sizes)
    assert _job_variants(flow) == {
        "none__none__deterministic",
        "gated__none__deterministic",
        "time_scaled__none__deterministic",
    }

    jump = cli._phase05_development_jobs(
        config,
        "jump",
        selected_flow="time_scaled",
    )
    assert len(jump) == 60
    assert {job.shard.stage for job in jump} == {"jump"}
    assert _job_worlds(jump) == {"jumps"}
    assert _job_bundles(jump) == set(config.development_bundles)
    assert {job.n_train for job in jump} == set(config.primary_train_sizes)
    assert _job_variants(jump) == {
        "time_scaled__none__deterministic",
        "time_scaled__gru__deterministic",
        "time_scaled__residual__deterministic",
    }

    uncertainty = cli._phase05_development_jobs(
        config,
        "uncertainty",
        selected_flow="time_scaled",
        selected_jump="residual",
    )
    assert len(uncertainty) == 180
    assert {job.shard.stage for job in uncertainty} == {"uncertainty"}
    assert _job_worlds(uncertainty) == set(config.target_worlds)
    assert _job_bundles(uncertainty) == set(config.development_bundles)
    assert {job.n_train for job in uncertainty} == set(config.primary_train_sizes)
    assert _job_variants(uncertainty) == {
        "time_scaled__residual__joint",
        "time_scaled__residual__decoupled",
        "time_scaled__residual__deterministic",
    }

    timing = cli._phase05_development_jobs(
        config,
        "timing_audit",
        selected_flow="time_scaled",
        selected_jump="residual",
    )
    assert len(timing) == 120
    assert {job.shard.stage for job in timing} == {"timing_audit"}
    assert _job_worlds(timing) == set(config.target_worlds)
    assert _job_bundles(timing) == set(config.development_bundles)
    assert {job.n_train for job in timing} == set(config.primary_train_sizes)
    assert _job_variants(timing) == {
        "time_scaled__residual__deterministic__strict_history",
        "time_scaled__residual__deterministic__inclusive_history",
    }


def test_development_job_plans_refuse_missing_prior_selection():
    config = load_phase05_config("configs/experiments/phase05.yaml")

    with pytest.raises(RuntimeError, match="selected flow"):
        cli._phase05_development_jobs(config, "jump")
    with pytest.raises(RuntimeError, match="selected flow and jump"):
        cli._phase05_development_jobs(
            config,
            "uncertainty",
            selected_flow="time_scaled",
        )
    with pytest.raises(RuntimeError, match="selected flow and jump"):
        cli._phase05_development_jobs(
            config,
            "timing_audit",
            selected_flow="time_scaled",
        )
