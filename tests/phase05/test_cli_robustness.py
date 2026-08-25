import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from afmc_fm import cli
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase05.protocol import freeze_candidate
from afmc_fm.phase05.store import Phase05CellResult, Phase05Store


def _locked_frozen_output(tmp_path: Path) -> Path:
    config = load_phase05_config("configs/experiments/phase05.yaml")
    output = tmp_path / "phase05"
    output.mkdir(parents=True)
    lock = {"phase05_config_sha256": canonical_config_hash(config)}
    (output / "protocol_lock.json").write_text(
        json.dumps(lock, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    development = output / "development"
    development.mkdir(parents=True)
    (development / "flow_gate.csv").write_text(
        "candidate,passed,selected,trainable_parameters\n"
        "gated,true,false,6000\n"
        "time_scaled,true,true,5900\n",
        encoding="utf-8",
    )
    (development / "jump_gate.csv").write_text(
        "candidate,passed,selected,trainable_parameters\n"
        "gru,true,false,6500\n"
        "residual,true,true,6400\n",
        encoding="utf-8",
    )
    (development / "uncertainty_gate.csv").write_text(
        "candidate,selected,trainable_parameters\n"
        "joint,false,7000\n"
        "decoupled,true,6900\n"
        "deterministic,false,6400\n",
        encoding="utf-8",
    )
    (development / "representation_timing_audit.csv").write_text(
        "strict_history,mean_delta_inclusive_minus_strict\ntrue,-0.03\n",
        encoding="utf-8",
    )
    freeze_candidate(output, config)
    return output


def _write_confirmation_gate(output: Path) -> Path:
    confirmation = output / "confirmation"
    confirmation.mkdir(parents=True, exist_ok=True)
    gate = confirmation / "primary_gate_summary.csv"
    gate.write_text(
        "world,headline_passed\n"
        "smooth,true\n"
        "jumps,true\n"
        "informative_observation,true\n",
        encoding="utf-8",
    )
    complete = output / "stages" / "confirmation" / "COMPLETE"
    complete.parent.mkdir(parents=True, exist_ok=True)
    complete.write_text("fixture\n", encoding="utf-8")
    return gate


def _finalize_confirmation_fixture(output: Path) -> Path:
    confirmation = output / "confirmation"
    confirmation.mkdir(parents=True, exist_ok=True)
    (confirmation / "STARTED").write_text("fixture\n", encoding="utf-8")
    return _write_confirmation_gate(output)


def _finalize_bound_confirmation_fixture(output: Path) -> Path:
    config = load_phase05_config("configs/experiments/phase05.yaml")
    lock_bytes = (output / "protocol_lock.json").read_bytes()
    store = Phase05Store(
        output,
        hashlib.sha256(lock_bytes).hexdigest(),
        config_hash=canonical_config_hash(config),
    )
    store.mark_confirmation_started()
    return _write_confirmation_gate(output)


def _robustness_argv(output: Path) -> list[str]:
    return [
        "phase05",
        "robustness",
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


def _synthetic_robustness_metric(job) -> pd.DataFrame:
    bundle = job.shard.seed_bundle
    train_rank = {5: 0, 10: 1, 20: 2, 40: 3, 80: 4, 100: 5}[job.n_train]
    bundle_offset = 0.001 * (bundle.cohort_seed - 700)
    common = {
        "stage": "robustness",
        "world": job.shard.world,
        "cohort_seed": bundle.cohort_seed,
        "subset_seed": bundle.subset_seed,
        "model_seed": bundle.model_seed,
        "n_train": job.n_train,
        "model": job.model,
        "variant": job.variant,
        "split": "test",
    }
    if job.shard.world == "site_shift":
        degradation = {
            "phase05_candidate": 0.05,
            "matched_gru": 0.10,
            "matched_representation_mlp": 0.12,
        }[job.model]
        site0 = 1.2 - 0.04 * train_rank + bundle_offset
        rows = [
            {**common, "site_or_shift": "site_0", "metric": "mae", "value": site0},
            {
                **common,
                "site_or_shift": "site_1",
                "metric": "mae",
                "value": site0 + degradation,
            },
        ]
        if job.model == "phase05_candidate":
            rows.extend(
                [
                    {
                        **common,
                        "site_or_shift": "site_0",
                        "metric": "nll",
                        "value": 0.60,
                    },
                    {
                        **common,
                        "site_or_shift": "site_1",
                        "metric": "nll",
                        "value": 0.65,
                    },
                    {
                        **common,
                        "site_or_shift": "site_0",
                        "metric": "coverage_90",
                        "value": 0.86,
                    },
                    {
                        **common,
                        "site_or_shift": "site_1",
                        "metric": "coverage_90",
                        "value": 0.82,
                    },
                ]
            )
        return pd.DataFrame(rows)

    multiplier = {
        "phase05_candidate": 1.03,
        "matched_gru": 1.00,
        "matched_representation_mlp": 1.08,
    }[job.model]
    base = 1.2 - 0.04 * train_rank + bundle_offset
    return pd.DataFrame(
        [
            {
                **common,
                "site_or_shift": "all",
                "metric": "mae",
                "value": base * multiplier,
            }
        ]
    )


def test_phase05_robustness_requires_finalized_confirmation_before_execution(
    tmp_path: Path,
):
    output = _locked_frozen_output(tmp_path)

    with pytest.raises(RuntimeError, match="finalized confirmation"):
        cli.main(_robustness_argv(output))

    assert not (output / "stages" / "robustness" / "COMPLETE").exists()


def test_phase05_robustness_binds_exact_stage_iv_plan_before_any_fit(
    tmp_path: Path,
    monkeypatch,
):
    output = _locked_frozen_output(tmp_path)
    gate = _finalize_confirmation_fixture(output)
    gate_before = gate.read_bytes()
    config = load_phase05_config("configs/experiments/phase05.yaml")
    frozen_hash = hashlib.sha256(
        (output / "frozen_candidate.json").read_bytes()
    ).hexdigest()
    expected_bundles = {bundle.as_tuple() for bundle in config.confirmatory_bundles}

    def intercept_execution(jobs, *, store, **_kwargs):
        planned = tuple(jobs)
        assert len(planned) == 360
        assert {job.shard.stage for job in planned} == {"robustness"}
        assert {job.shard.world for job in planned} == set(config.robustness_worlds)
        assert {
            job.shard.seed_bundle.as_tuple() for job in planned
        } == expected_bundles
        assert {job.n_train for job in planned} == set(config.train_sizes)
        assert {job.model for job in planned} == {
            "phase05_candidate",
            "matched_gru",
            "matched_representation_mlp",
        }
        assert {job.frozen_candidate_hash for job in planned} == {frozen_hash}
        assert store.output == output
        assert gate.read_bytes() == gate_before
        raise RuntimeError("injected stop before robustness fit")

    monkeypatch.setattr(cli, "run_phase05_jobs", intercept_execution)

    with pytest.raises(RuntimeError, match="injected stop before robustness fit"):
        cli.main(_robustness_argv(output))

    assert gate.read_bytes() == gate_before


def test_phase05_robustness_finalizes_exact_cells_and_persists_outputs(
    tmp_path: Path,
    monkeypatch,
):
    output = _locked_frozen_output(tmp_path)
    gate = _finalize_bound_confirmation_fixture(output)
    gate_before = gate.read_bytes()

    def persist_synthetic_results(jobs, *, store, **_kwargs):
        frames = []
        for job in tuple(jobs):
            frame = _synthetic_robustness_metric(job)
            bundle = job.shard.seed_bundle
            store.write_cell(
                Phase05CellResult(
                    stage="robustness",
                    world=job.shard.world,
                    cohort_seed=bundle.cohort_seed,
                    subset_seed=bundle.subset_seed,
                    model_seed=bundle.model_seed,
                    n_train=job.n_train,
                    model=job.model,
                    variant=job.variant,
                    metrics=frame,
                    frozen_candidate_hash=job.frozen_candidate_hash,
                )
            )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    monkeypatch.setattr(cli, "run_phase05_jobs", persist_synthetic_results)

    assert cli.main(_robustness_argv(output)) == 0

    assert gate.read_bytes() == gate_before
    assert (output / "stages" / "robustness" / "COMPLETE").is_file()
    robustness = output / "robustness"
    expected_files = {
        "site_shift_metrics.csv",
        "site_shift_gate_summary.csv",
        "misspecification_metrics.csv",
        "misspecification_summary.csv",
    }
    assert expected_files.issubset({path.name for path in robustness.iterdir()})

    site_shift = pd.read_csv(robustness / "site_shift_metrics.csv")
    assert len(site_shift) == 6 * 3 * 10
    assert {
        "mae_site_0",
        "mae_site_1",
        "mae_absolute_degradation",
        "mae_relative_degradation",
    }.issubset(site_shift.columns)

    site_gate = pd.read_csv(robustness / "site_shift_gate_summary.csv")
    assert len(site_gate) == 6 * 2
    assert site_gate["passed"].all()

    misspecified = pd.read_csv(robustness / "misspecification_metrics.csv")
    assert len(misspecified) == 10
    assert misspecified["relative_excess"].between(0.0, 0.05).all()

    misspecified_summary = pd.read_csv(robustness / "misspecification_summary.csv")
    assert len(misspecified_summary) == 1
    assert bool(misspecified_summary.loc[0, "passed"])
