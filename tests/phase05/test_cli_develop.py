from pathlib import Path

import pandas as pd
import pytest

from afmc_fm import cli
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase05.store import Phase05CellResult


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


def _calibration_metrics() -> pd.DataFrame:
    rows = []
    worlds = ("smooth", "jumps", "informative_observation")
    models = ("flow_jump", "representation_linear", "gru_from_scratch")
    for world_index, world in enumerate(worlds):
        for n_index, n_train in enumerate((5, 10, 20, 40)):
            for model_index, model in enumerate(models):
                base = 1.0 + 0.1 * world_index + 0.01 * n_index + 0.001 * model_index
                for seed_index in range(1, 6):
                    rows.append(
                        {
                            "benchmark": "low_n",
                            "site_or_shift": "all",
                            "ablation": "none",
                            "metric": "mae",
                            "world": world,
                            "n_train": n_train,
                            "model": model,
                            "cohort_seed": 100 + seed_index,
                            "subset_seed": 200 + seed_index,
                            "model_seed": 300 + seed_index,
                            "value": base * (1.0 + 0.01 * (seed_index - 3)),
                        }
                    )
    return pd.DataFrame(rows)


def _calibrated_output(tmp_path: Path) -> Path:
    metrics = tmp_path / "phase0_metrics.csv"
    _calibration_metrics().to_csv(metrics, index=False)
    output = tmp_path / "phase05"
    assert cli.main(
        [
            "phase05",
            "calibrate",
            "--phase0-metrics",
            str(metrics),
            "--exp-config",
            "configs/experiments/phase05.yaml",
            "--output",
            str(output),
        ]
    ) == 0
    return output


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
        mae = {
            "none__none__deterministic": 1.00,
            "gated__none__deterministic": 0.95,
            "time_scaled__none__deterministic": 0.90,
        }[variant]
        metrics = [("mae", mae)]
    elif stage == "jump":
        mae = {
            "time_scaled__none__deterministic": 1.00,
            "time_scaled__gru__deterministic": 0.95,
            "time_scaled__residual__deterministic": 0.90,
        }[variant]
        metrics = [("mae", mae)]
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


def test_phase05_develop_requires_existing_protocol_lock(tmp_path: Path):
    output = tmp_path / "phase05"

    with pytest.raises(RuntimeError, match="protocol_lock.json"):
        cli.main(
            [
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
            ]
        )

    assert not (output / "protocol_lock.json").exists()


def test_phase05_develop_runs_sequential_stages_and_finalizes_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    output = _calibrated_output(tmp_path)
    lock_path = output / "protocol_lock.json"
    lock_before = lock_path.read_bytes()
    calls: list[tuple[str, int]] = []

    def fake_run_phase05_jobs(jobs, *, store, **_kwargs):
        planned = tuple(jobs)
        stage = planned[0].shard.stage
        calls.append((stage, len(planned)))
        frames = []
        for job in planned:
            frame = pd.DataFrame(_metric_rows(job))
            bundle = job.shard.seed_bundle
            store.write_cell(
                Phase05CellResult(
                    stage=stage,
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

    monkeypatch.setattr(cli, "run_phase05_jobs", fake_run_phase05_jobs, raising=False)

    assert cli.main(
        [
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
    ) == 0

    assert calls == [
        ("flow", 60),
        ("jump", 60),
        ("uncertainty", 180),
        ("timing_audit", 120),
    ]
    assert lock_path.read_bytes() == lock_before

    development = output / "development"
    flow = pd.read_csv(development / "flow_gate.csv")
    jump = pd.read_csv(development / "jump_gate.csv")
    uncertainty = pd.read_csv(development / "uncertainty_gate.csv")
    timing = pd.read_csv(development / "representation_timing_audit.csv")

    assert flow.loc[flow["selected"], "candidate"].tolist() == ["time_scaled"]
    assert jump.loc[jump["selected"], "candidate"].tolist() == ["residual"]
    assert uncertainty.loc[uncertainty["selected"], "candidate"].tolist() == [
        "decoupled"
    ]
    assert len(timing) == 15
    assert set(timing["world"]) == {"smooth", "jumps", "informative_observation"}
    assert set(timing["strict_timing_required"]) == {True}
    assert (timing["delta_inclusive_minus_strict"] < 0).all()

    for stage in ("flow", "jump", "uncertainty", "timing_audit"):
        assert (output / "stages" / stage / "COMPLETE").is_file()
