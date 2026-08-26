import json
from pathlib import Path

import pandas as pd
import pytest

from afmc_fm import cli
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase05.store import Phase05CellResult


def _locked_output(tmp_path: Path) -> Path:
    config = load_phase05_config("configs/experiments/phase05.yaml")
    lock = {
        "phase05_config_sha256": canonical_config_hash(config),
        "locked_min_relative_effect": 0.01,
        "locked_uncertainty_mae_tolerance": 0.01,
    }
    output = tmp_path / "phase05"
    output.mkdir(parents=True)
    data = (
        json.dumps(lock, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    (output / "protocol_lock.json").write_bytes(data)
    return output


def _develop_argv(output: Path, *, resume: bool = False) -> list[str]:
    argv = [
        "phase05",
        "develop",
        "--sim-config",
        "configs/simulator/smoke.yaml",
        "--exp-config",
        "configs/experiments/phase05.yaml",
        "--output",
        str(output),
        "--device",
        "cpu",
        "--workers",
        "1",
    ]
    if resume:
        argv.append("--resume")
    return argv


def _parameter_count(variant: str) -> int:
    if variant.endswith("__joint"):
        return 7000
    if variant.endswith("__decoupled"):
        return 6900
    if "__residual__" in variant:
        return 6400
    if "__gru__" in variant:
        return 6500
    if variant.startswith("time_scaled__"):
        return 5900
    if variant.startswith("gated__"):
        return 6000
    return 5000


def _metric_rows(job) -> list[dict[str, object]]:
    stage = job.shard.stage
    variant = job.variant
    if stage == "flow":
        metrics = [
            (
                "mae",
                {
                    "none__none__deterministic": 1.00,
                    "gated__none__deterministic": 0.95,
                    "time_scaled__none__deterministic": 0.90,
                }[variant],
            )
        ]
    elif stage == "jump":
        metrics = [
            (
                "mae",
                {
                    "time_scaled__none__deterministic": 1.00,
                    "time_scaled__gru__deterministic": 0.95,
                    "time_scaled__residual__deterministic": 0.90,
                }[variant],
            )
        ]
    elif stage == "uncertainty":
        if variant.endswith("__joint"):
            metrics = [("mae", 1.01), ("nll", 0.60), ("coverage_90", 0.82)]
        elif variant.endswith("__decoupled"):
            metrics = [("mae", 1.00), ("nll", 0.50), ("coverage_90", 0.88)]
        else:
            metrics = [("mae", 1.00)]
    elif stage == "timing_audit":
        value = 1.00 if variant.endswith("__strict_history") else 0.90
        metrics = [("mae", value)]
    else:
        raise AssertionError(f"unexpected synthetic stage: {stage}")

    bundle = job.shard.seed_bundle
    return [
        {
            "stage": stage,
            "world": job.shard.world,
            "cohort_seed": bundle.cohort_seed,
            "subset_seed": bundle.subset_seed,
            "model_seed": bundle.model_seed,
            "n_train": job.n_train,
            "model": job.model,
            "variant": variant,
            "split": "test",
            "site_or_shift": "all",
            "metric": metric,
            "value": value,
            "trainable_parameters": _parameter_count(variant),
            "backend": "synthetic-test",
        }
        for metric, value in metrics
    ]


def _persist_synthetic_jobs(jobs, store) -> pd.DataFrame:
    frames = []
    for job in jobs:
        frame = pd.DataFrame(_metric_rows(job))
        bundle = job.shard.seed_bundle
        store.write_cell(
            Phase05CellResult(
                stage=job.shard.stage,
                world=job.shard.world,
                cohort_seed=bundle.cohort_seed,
                subset_seed=bundle.subset_seed,
                model_seed=bundle.model_seed,
                n_train=job.n_train,
                model=job.model,
                variant=job.variant,
                metrics=frame,
            )
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_develop_resume_skips_finalized_flow_and_jump(tmp_path: Path, monkeypatch):
    output = _locked_output(tmp_path)
    first_calls: list[str] = []

    def interrupted_run(jobs, *, store, **_kwargs):
        planned = tuple(jobs)
        stage = planned[0].shard.stage
        first_calls.append(stage)
        if stage == "uncertainty":
            raise RuntimeError("injected interruption after jump")
        return _persist_synthetic_jobs(planned, store)

    monkeypatch.setattr(cli, "run_phase05_jobs", interrupted_run)
    with pytest.raises(RuntimeError, match="injected interruption"):
        cli.main(_develop_argv(output))
    assert first_calls == ["flow", "jump", "uncertainty"]

    development = output / "development"
    flow_gate = development / "flow_gate.csv"
    jump_gate = development / "jump_gate.csv"
    flow_cell = next((output / "stages" / "flow" / "cells").glob("*.json"))
    jump_cell = next((output / "stages" / "jump" / "cells").glob("*.json"))
    snapshots = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (flow_gate, jump_gate, flow_cell, jump_cell)
    }

    resume_calls: list[str] = []

    def resumed_run(jobs, *, store, **_kwargs):
        planned = tuple(jobs)
        stage = planned[0].shard.stage
        resume_calls.append(stage)
        if stage in {"flow", "jump"}:
            raise AssertionError(f"finalized {stage} stage was rerun")
        return _persist_synthetic_jobs(planned, store)

    monkeypatch.setattr(cli, "run_phase05_jobs", resumed_run)
    assert cli.main(_develop_argv(output, resume=True)) == 0
    assert resume_calls == ["uncertainty", "timing_audit"]
    for path, (payload, mtime) in snapshots.items():
        assert path.read_bytes() == payload
        assert path.stat().st_mtime_ns == mtime


def test_develop_resume_skips_finalized_uncertainty(tmp_path: Path, monkeypatch):
    output = _locked_output(tmp_path)
    first_calls: list[str] = []

    def interrupted_run(jobs, *, store, **_kwargs):
        planned = tuple(jobs)
        stage = planned[0].shard.stage
        first_calls.append(stage)
        if stage == "timing_audit":
            raise RuntimeError("injected interruption after uncertainty")
        return _persist_synthetic_jobs(planned, store)

    monkeypatch.setattr(cli, "run_phase05_jobs", interrupted_run)
    with pytest.raises(RuntimeError, match="injected interruption"):
        cli.main(_develop_argv(output))
    assert first_calls == ["flow", "jump", "uncertainty", "timing_audit"]

    uncertainty_gate = output / "development" / "uncertainty_gate.csv"
    uncertainty_cell = next(
        (output / "stages" / "uncertainty" / "cells").glob("*.json")
    )
    snapshots = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (uncertainty_gate, uncertainty_cell)
    }

    resume_calls: list[str] = []

    def resumed_run(jobs, *, store, **_kwargs):
        planned = tuple(jobs)
        stage = planned[0].shard.stage
        resume_calls.append(stage)
        if stage in {"flow", "jump", "uncertainty"}:
            raise AssertionError(f"finalized {stage} stage was rerun")
        return _persist_synthetic_jobs(planned, store)

    monkeypatch.setattr(cli, "run_phase05_jobs", resumed_run)
    assert cli.main(_develop_argv(output, resume=True)) == 0
    assert resume_calls == ["timing_audit"]
    for path, (payload, mtime) in snapshots.items():
        assert path.read_bytes() == payload
        assert path.stat().st_mtime_ns == mtime


def test_develop_resume_is_noop_after_timing_audit_finalization(
    tmp_path: Path,
    monkeypatch,
):
    output = _locked_output(tmp_path)

    def first_run(jobs, *, store, **_kwargs):
        return _persist_synthetic_jobs(tuple(jobs), store)

    monkeypatch.setattr(cli, "run_phase05_jobs", first_run)
    assert cli.main(_develop_argv(output)) == 0

    timing_artifact = output / "development" / "representation_timing_audit.csv"
    timing_cell = next(
        (output / "stages" / "timing_audit" / "cells").glob("*.json")
    )
    snapshots = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (timing_artifact, timing_cell)
    }
    resume_calls: list[str] = []

    def resumed_run(jobs, **_kwargs):
        stage = next(iter(jobs)).shard.stage
        resume_calls.append(stage)
        raise AssertionError(f"finalized {stage} stage was rerun")

    monkeypatch.setattr(cli, "run_phase05_jobs", resumed_run)
    assert cli.main(_develop_argv(output, resume=True)) == 0
    assert resume_calls == []
    for path, (payload, mtime) in snapshots.items():
        assert path.read_bytes() == payload
        assert path.stat().st_mtime_ns == mtime
