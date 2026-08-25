import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.experiments.runner import build_complete_truth_targets
from afmc_fm.phase05.config import Phase05Config, SeedBundle
from afmc_fm.phase05.execution import (
    Phase05ExecutionOptions,
    Phase05Job,
    Phase05ShardSpec,
    run_phase05_jobs,
)
from afmc_fm.phase05.robustness import (
    complete_truth_phase05_batch,
    evaluate_misspecification,
    evaluate_site_shift,
)
from afmc_fm.phase05.runner import prepare_phase05_cohort
from afmc_fm.phase05.store import Phase05CellResult, Phase05Store
from afmc_fm.simulator.cohort import simulate_world
from afmc_fm.simulator.config import SimulatorConfig

MODELS = (
    "phase05_candidate",
    "matched_gru",
    "matched_representation_mlp",
)
PRIMARY_SIZES = (5, 10, 20, 40)


def test_complete_truth_batch_uses_latent_emissions_not_observed_next_lab_masks():
    cohort = simulate_world(
        "site_shift",
        SimulatorConfig(cohort_size=20, followup_days=60.0),
        seed=701,
    )
    prepared = prepare_phase05_cohort(cohort)
    patient = cohort.patients[0]
    sequence = prepared.sequences[patient.patient_id]

    batch = complete_truth_phase05_batch([sequence], [patient])
    expected_values, expected_masks = build_complete_truth_targets(patient)
    length = len(expected_masks)

    torch.testing.assert_close(
        batch["target_values"][0, :length],
        torch.from_numpy(expected_values),
    )
    torch.testing.assert_close(
        batch["target_masks"][0, :length],
        torch.from_numpy(expected_masks),
    )
    assert np.any(expected_masks != sequence.target_next_masks)
    assert expected_masks.sum() > sequence.target_next_masks.sum()


def _site_shift_metrics() -> pd.DataFrame:
    rows = []
    degradation = {
        "phase05_candidate": 0.05,
        "matched_gru": 0.10,
        "matched_representation_mlp": 0.12,
    }
    for index in range(10):
        bundle = (701 + index, 801 + index, 901 + index)
        for model in MODELS:
            site0_mae = 1.0 + 0.01 * index
            for site, value in (
                (0, site0_mae),
                (1, site0_mae + degradation[model]),
            ):
                rows.append(
                    {
                        "world": "site_shift",
                        "cohort_seed": bundle[0],
                        "subset_seed": bundle[1],
                        "model_seed": bundle[2],
                        "n_train": 20,
                        "model": model,
                        "site_or_shift": f"site_{site}",
                        "metric": "mae",
                        "value": value,
                    }
                )
            if model == "phase05_candidate":
                for site, nll, coverage in ((0, 0.60, 0.86), (1, 0.65, 0.82)):
                    for metric, value in (("nll", nll), ("coverage_90", coverage)):
                        rows.append(
                            {
                                "world": "site_shift",
                                "cohort_seed": bundle[0],
                                "subset_seed": bundle[1],
                                "model_seed": bundle[2],
                                "n_train": 20,
                                "model": model,
                                "site_or_shift": f"site_{site}",
                                "metric": metric,
                                "value": value,
                            }
                        )
    return pd.DataFrame(rows)


def test_site_shift_reports_raw_absolute_relative_and_calibration_degradation():
    result = evaluate_site_shift(_site_shift_metrics(), win_requirement=8)
    per_seed = result["per_seed"]

    assert len(per_seed) == 30
    candidate = per_seed.loc[per_seed["model"] == "phase05_candidate"]
    assert candidate["mae_absolute_degradation"].tolist() == pytest.approx([0.05] * 10)
    assert candidate["mae_relative_degradation"].gt(0).all()
    assert candidate["nll_absolute_degradation"].tolist() == pytest.approx([0.05] * 10)
    assert candidate["ce90_site_0"].tolist() == pytest.approx([0.04] * 10)
    assert candidate["ce90_site_1"].tolist() == pytest.approx([0.08] * 10)
    assert candidate["ce90_absolute_degradation"].tolist() == pytest.approx([0.04] * 10)

    gate = result["gate_summary"]
    assert set(gate["comparator"]) == {
        "matched_gru",
        "matched_representation_mlp",
    }
    assert gate["wins"].tolist() == [10, 10]
    assert gate["passed"].all()


def _misspecified_metrics(candidate_multiplier: float) -> pd.DataFrame:
    rows = []
    for index in range(10):
        bundle = (701 + index, 801 + index, 901 + index)
        seed_scale = 1.0 + 0.01 * index
        multipliers = {
            "phase05_candidate": candidate_multiplier,
            "matched_gru": 1.0,
            "matched_representation_mlp": 1.08,
        }
        for n_train in PRIMARY_SIZES:
            n_scale = 1.0 - 0.04 * np.log2(n_train / 5)
            for model in MODELS:
                rows.append(
                    {
                        "world": "misspecified",
                        "cohort_seed": bundle[0],
                        "subset_seed": bundle[1],
                        "model_seed": bundle[2],
                        "n_train": n_train,
                        "model": model,
                        "split": "test",
                        "site_or_shift": "all",
                        "metric": "mae",
                        "value": seed_scale * n_scale * multipliers[model],
                    }
                )
    return pd.DataFrame(rows)


def test_misspecification_uses_per_seed_best_control_and_five_percent_tolerance():
    passing = evaluate_misspecification(_misspecified_metrics(1.03), tolerance=0.05)
    per_seed = passing["per_seed"]
    summary = passing["summary"]

    assert len(per_seed) == 10
    assert set(per_seed["best_control_model"]) == {"matched_gru"}
    assert per_seed["relative_excess"].tolist() == pytest.approx([0.03] * 10)
    assert summary["mean_relative_excess"] == pytest.approx(0.03)
    assert summary["tolerance"] == pytest.approx(0.05)
    assert summary["passed"] is True

    failing = evaluate_misspecification(_misspecified_metrics(1.06), tolerance=0.05)
    assert failing["summary"]["mean_relative_excess"] == pytest.approx(0.06)
    assert failing["summary"]["passed"] is False


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _robustness_store(tmp_path):
    config = Phase05Config(max_epochs=1, patience=1)
    config_hash = canonical_config_hash(config)
    lock = {
        "schema_version": 1,
        "phase05_config_sha256": config_hash,
    }
    protocol_hash = hashlib.sha256(_canonical_json_bytes(lock)).hexdigest()
    store = Phase05Store(
        tmp_path / "phase05",
        protocol_hash,
        spec_hash="1" * 64,
        config_hash=config_hash,
    )
    store.write_protocol_lock(lock)
    candidate_hash = store.write_frozen_candidate(
        {
            "flow_mode": "time_scaled",
            "jump_mode": "residual",
            "uncertainty_mode": "deterministic",
        }
    )
    store.mark_confirmation_started()
    confirmation_result = Phase05CellResult(
        stage="confirmation",
        world="smooth",
        cohort_seed=701,
        subset_seed=801,
        model_seed=901,
        n_train=5,
        model="phase05_candidate",
        variant="time_scaled__residual__deterministic",
        metrics=pd.DataFrame([{"metric": "mae", "value": 1.0}]),
        frozen_candidate_hash=candidate_hash,
    )
    store.write_cell(confirmation_result)
    store.mark_stage_complete("confirmation", {confirmation_result.cell_id})
    return store, config, candidate_hash


def test_robustness_requires_finalized_primary_gate_and_does_not_mutate_it(tmp_path):
    store, config, candidate_hash = _robustness_store(tmp_path)
    job = Phase05Job(
        shard=Phase05ShardSpec(
            "robustness", "site_shift", SeedBundle(701, 801, 901)
        ),
        n_train=5,
        model="phase05_candidate",
        variant="time_scaled__residual__deterministic",
        frozen_candidate_hash=candidate_hash,
    )
    options = Phase05ExecutionOptions(device="cpu", workers=1)

    with pytest.raises(
        RuntimeError,
        match="primary confirmatory gate must be finalized before robustness",
    ):
        run_phase05_jobs(
            (job,),
            store=store,
            config=config,
            options=options,
            prepare_shard=lambda spec: object(),
            run_job=lambda job, prepared, device: pd.DataFrame(
                [{"metric": "mae", "value": 1.0}]
            ),
        )

    gate_path = store.output / "confirmation" / "primary_gate_summary.csv"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_bytes(b"world,headline_passed\nsmooth,false\n")
    before = gate_path.read_bytes()

    result = run_phase05_jobs(
        (job,),
        store=store,
        config=config,
        options=options,
        prepare_shard=lambda spec: object(),
        run_job=lambda job, prepared, device: pd.DataFrame(
            [{"metric": "mae", "value": 1.0}]
        ),
    )

    assert not result.empty
    assert gate_path.read_bytes() == before
