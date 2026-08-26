import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.experiments.runner import build_complete_truth_targets
from afmc_fm.phase05.baselines import build_capacity_audit
from afmc_fm.phase05.config import Phase05Config, SeedBundle
from afmc_fm.phase05.execution import (
    Phase05ExecutionOptions,
    Phase05Job,
    Phase05ShardSpec,
    run_phase05_jobs,
)
from afmc_fm.phase05.model import Phase05FlowJumpAdapter
from afmc_fm.phase05.protocol import FrozenCandidate
from afmc_fm.phase05.robustness import (
    ROBUSTNESS_MODELS,
    build_robustness_jobs,
    complete_truth_phase05_batch,
    evaluate_misspecification,
    evaluate_site_shift,
    run_robustness_job,
)
from afmc_fm.phase05.runner import prepare_phase05_cohort
from afmc_fm.phase05.store import Phase05CellResult, Phase05Store
from afmc_fm.schema.events import EventType
from afmc_fm.simulator.cohort import simulate_world
from afmc_fm.simulator.config import SimulatorConfig

MODELS = (
    "phase05_candidate",
    "matched_gru",
    "matched_representation_mlp",
)
PRIMARY_SIZES = (5, 10, 20, 40)


def _audited_frozen_candidate() -> FrozenCandidate:
    candidate = Phase05FlowJumpAdapter(
        representation_dim=16,
        value_dim=3,
        event_dim=len(EventType),
        state_dim=24,
        flow_mode="time_scaled",
        jump_mode="residual",
        uncertainty_mode="decoupled",
        time_scale_days=30.0,
    )
    target_parameters = sum(
        parameter.numel()
        for parameter in candidate.parameters()
        if parameter.requires_grad
    )
    audit = build_capacity_audit(
        target_parameters=target_parameters,
        value_dim=3,
        event_dim=len(EventType),
        representation_input_dim=19,
    ).set_index("control")
    return FrozenCandidate(
        flow_mode="time_scaled",
        jump_mode="residual",
        uncertainty_mode="decoupled",
        strict_history=True,
        state_dim=24,
        time_scale_days=30.0,
        jump_eligible_event_codes=("SYNTHETIC_INTERVENTION",),
        assimilation_semantics="phase0_grucell_unchanged",
        trainable_parameters=target_parameters,
        matched_gru_hidden_size=int(audit.loc["matched_gru", "hidden_size"]),
        matched_gru_parameters=int(audit.loc["matched_gru", "actual_parameters"]),
        matched_mlp_hidden_size=int(
            audit.loc["matched_representation_mlp", "hidden_size"]
        ),
        matched_mlp_parameters=int(
            audit.loc["matched_representation_mlp", "actual_parameters"]
        ),
        protocol_lock_sha256="a" * 64,
        development_artifact_hashes={"flow_gate.csv": "b" * 64},
    )


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


def test_robustness_job_plan_is_locked_to_three_models_two_worlds_and_ten_bundles():
    config = Phase05Config()
    frozen = _audited_frozen_candidate()

    jobs = build_robustness_jobs(
        config,
        frozen,
        frozen_candidate_hash="c" * 64,
    )

    assert ROBUSTNESS_MODELS == MODELS
    assert len(jobs) == 2 * 10 * 6 * 3
    assert {job.shard.world for job in jobs} == set(config.robustness_worlds)
    assert {job.shard.seed_bundle for job in jobs} == set(config.confirmatory_bundles)
    assert {job.n_train for job in jobs} == set(config.train_sizes)
    assert {job.model for job in jobs} == set(MODELS)
    assert all(job.frozen_candidate_hash == "c" * 64 for job in jobs)
    candidate_variants = {job.variant for job in jobs if job.model == "phase05_candidate"}
    assert candidate_variants == {"time_scaled__residual__decoupled"}


def test_site_shift_dispatcher_trains_locked_models_and_emits_both_complete_truth_sites():
    config = Phase05Config(max_epochs=1, patience=1)
    frozen = _audited_frozen_candidate()
    cohort = simulate_world(
        "site_shift",
        SimulatorConfig(cohort_size=80, followup_days=45.0),
        seed=701,
    )
    prepared = prepare_phase05_cohort(cohort)
    jobs = [
        job
        for job in build_robustness_jobs(
            config,
            frozen,
            frozen_candidate_hash="c" * 64,
        )
        if job.shard.world == "site_shift"
        and job.shard.seed_bundle == config.confirmatory_bundles[0]
        and job.n_train == 5
    ]

    assert len(jobs) == 3
    results = [
        run_robustness_job(
            job,
            prepared,
            config=config,
            frozen=frozen,
            device=torch.device("cpu"),
        )
        for job in jobs
    ]

    for job, result in zip(jobs, results, strict=True):
        assert set(result["stage"]) == {"robustness"}
        assert set(result["world"]) == {"site_shift"}
        assert set(result["cohort_seed"]) == {701}
        assert set(result["subset_seed"]) == {801}
        assert set(result["model_seed"]) == {901}
        assert set(result["n_train"]) == {5}
        assert set(result["model"]) == {job.model}
        assert set(result["variant"]) == {job.variant}
        assert set(result["site_or_shift"]) == {"site_0", "site_1"}
        assert set(result.loc[result["metric"] == "mae", "site_or_shift"]) == {
            "site_0",
            "site_1",
        }
        assert not result.duplicated(
            [
                "stage",
                "world",
                "cohort_seed",
                "subset_seed",
                "model_seed",
                "n_train",
                "model",
                "variant",
                "split",
                "site_or_shift",
                "metric",
            ]
        ).any()

    by_model = {job.model: result for job, result in zip(jobs, results, strict=True)}
    assert set(by_model["matched_gru"]["trainable_parameters"]) == {
        frozen.matched_gru_parameters
    }
    assert set(by_model["matched_representation_mlp"]["trainable_parameters"]) == {
        frozen.matched_mlp_parameters
    }


def test_misspecified_dispatcher_uses_standard_test_split_for_all_three_models():
    config = Phase05Config(max_epochs=1, patience=1)
    frozen = _audited_frozen_candidate()
    cohort = simulate_world(
        "misspecified",
        SimulatorConfig(cohort_size=40, followup_days=45.0),
        seed=701,
    )
    prepared = prepare_phase05_cohort(cohort)
    jobs = [
        job
        for job in build_robustness_jobs(
            config,
            frozen,
            frozen_candidate_hash="c" * 64,
        )
        if job.shard.world == "misspecified"
        and job.shard.seed_bundle == config.confirmatory_bundles[0]
        and job.n_train == 5
    ]

    assert len(jobs) == 3
    results = [
        run_robustness_job(
            job,
            prepared,
            config=config,
            frozen=frozen,
            device=torch.device("cpu"),
        )
        for job in jobs
    ]
    for job, result in zip(jobs, results, strict=True):
        assert set(result["stage"]) == {"robustness"}
        assert set(result["world"]) == {"misspecified"}
        assert set(result["model"]) == {job.model}
        assert set(result["site_or_shift"]) == {"all"}
        assert "mae" in set(result["metric"])


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
