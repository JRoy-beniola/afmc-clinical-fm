import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from afmc_fm import cli
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase05.protocol import freeze_candidate
from afmc_fm.phase05.store import Phase05CellResult


def _locked_output(tmp_path: Path) -> Path:
    config = load_phase05_config("configs/experiments/phase05.yaml")
    lock = {"phase05_config_sha256": canonical_config_hash(config)}
    output = tmp_path / "phase05"
    output.mkdir(parents=True)
    data = (
        json.dumps(lock, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    (output / "protocol_lock.json").write_bytes(data)
    return output


def _write_successful_development_artifacts(output: Path) -> None:
    development = output / "development"
    development.mkdir(parents=True, exist_ok=True)
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


def _freeze(output: Path):
    config = load_phase05_config("configs/experiments/phase05.yaml")
    _write_successful_development_artifacts(output)
    return freeze_candidate(output, config)


def _confirm_argv(output: Path) -> list[str]:
    return [
        "phase05",
        "confirm",
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


def _synthetic_confirmation_metric(job) -> pd.DataFrame:
    bundle = job.shard.seed_bundle
    train_rank = {5: 0, 10: 1, 20: 2, 40: 3, 80: 4, 100: 5}[job.n_train]
    model_offset = {
        "phase05_candidate": -0.12,
        "matched_gru": 0.02,
        "matched_representation_mlp": 0.04,
        "representation_linear": 0.08,
        "representation_mlp_original": 0.07,
        "gru_original": 0.06,
        "phase0_flow_jump_reference": 0.05,
    }[job.model]
    bundle_index = bundle.cohort_seed - 700
    value = 1.2 - 0.05 * train_rank + model_offset + 0.001 * bundle_index
    return pd.DataFrame(
        [
            {
                "stage": "confirmation",
                "world": job.shard.world,
                "cohort_seed": bundle.cohort_seed,
                "subset_seed": bundle.subset_seed,
                "model_seed": bundle.model_seed,
                "n_train": job.n_train,
                "model": job.model,
                "variant": job.variant,
                "split": "test",
                "site_or_shift": "all",
                "metric": "mae",
                "value": value,
            }
        ]
    )


def test_phase05_confirm_requires_frozen_candidate_before_start(tmp_path: Path):
    output = _locked_output(tmp_path)

    with pytest.raises(RuntimeError, match="frozen_candidate"):
        cli.main(_confirm_argv(output))

    assert not (output / "confirmation" / "STARTED").exists()


def test_phase05_confirm_binds_frozen_candidate_before_any_fit(
    tmp_path: Path,
    monkeypatch,
):
    output = _locked_output(tmp_path)
    _freeze(output)
    config = load_phase05_config("configs/experiments/phase05.yaml")
    frozen_hash = hashlib.sha256(
        (output / "frozen_candidate.json").read_bytes()
    ).hexdigest()
    expected_bundles = {bundle.as_tuple() for bundle in config.confirmatory_bundles}
    development_bundles = {bundle.as_tuple() for bundle in config.development_bundles}

    def intercept_execution(jobs, *, store, **_kwargs):
        planned = tuple(jobs)
        assert (output / "confirmation" / "STARTED").is_file()
        assert len(planned) == 1260
        assert {job.shard.stage for job in planned} == {"confirmation"}
        assert {job.shard.world for job in planned} == set(config.target_worlds)
        observed_bundles = {
            job.shard.seed_bundle.as_tuple() for job in planned
        }
        assert observed_bundles == expected_bundles
        assert observed_bundles.isdisjoint(development_bundles)
        assert {job.frozen_candidate_hash for job in planned} == {frozen_hash}
        assert store.output == output
        raise RuntimeError("injected stop before confirmation fit")

    monkeypatch.setattr(cli, "run_phase05_jobs", intercept_execution)

    with pytest.raises(RuntimeError, match="injected stop before confirmation fit"):
        cli.main(_confirm_argv(output))


def test_phase05_confirm_finalizes_exact_cells_and_persists_outputs(
    tmp_path: Path,
    monkeypatch,
):
    output = _locked_output(tmp_path)
    frozen = _freeze(output)
    config = load_phase05_config("configs/experiments/phase05.yaml")

    def persist_synthetic_results(jobs, *, store, **_kwargs):
        frames = []
        for job in tuple(jobs):
            frame = _synthetic_confirmation_metric(job)
            bundle = job.shard.seed_bundle
            store.write_cell(
                Phase05CellResult(
                    stage="confirmation",
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

    assert cli.main(_confirm_argv(output)) == 0

    assert (output / "stages" / "confirmation" / "COMPLETE").is_file()
    confirmation = output / "confirmation"
    expected_files = {
        "metrics.csv",
        "learning_curves.csv",
        "paired_naulc_effects.csv",
        "primary_gate_summary.csv",
        "bootstrap_intervals.csv",
        "sign_tests.csv",
        "capacity_audit.csv",
    }
    assert expected_files.issubset({path.name for path in confirmation.iterdir()})

    metrics = pd.read_csv(confirmation / "metrics.csv")
    assert len(metrics) == 1260
    assert set(metrics["model"]) == {
        "phase05_candidate",
        "matched_gru",
        "matched_representation_mlp",
        "representation_linear",
        "representation_mlp_original",
        "gru_original",
        "phase0_flow_jump_reference",
    }

    curves = pd.read_csv(confirmation / "learning_curves.csv")
    assert len(curves) == 3 * 6 * 7
    assert {
        "world",
        "n_train",
        "model",
        "variant",
        "mean_mae",
        "sd_mae",
        "n_bundles",
    }.issubset(curves.columns)
    assert set(curves["n_bundles"]) == {10}

    capacity = pd.read_csv(confirmation / "capacity_audit.csv")
    assert set(capacity["control"]) == {
        "matched_gru",
        "matched_representation_mlp",
    }
    assert set(capacity["target_parameters"]) == {frozen.trainable_parameters}

    primary = pd.read_csv(confirmation / "primary_gate_summary.csv")
    assert len(primary) == 6
    assert set(primary["headline_passed"].astype(str).str.lower()) == {"true"}
