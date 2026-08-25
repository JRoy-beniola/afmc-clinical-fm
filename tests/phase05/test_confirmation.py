import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import binomtest

from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.confirmation import (
    CONFIRMATORY_MODELS,
    PRIMARY_COMPARATORS,
    build_confirmation_jobs,
    evaluate_primary_gate,
    exact_sign_test,
    holm_adjust,
    paired_bootstrap_mean_ci,
    paired_naulc_effects,
    persist_confirmation_analysis,
    run_confirmation_job,
)
from afmc_fm.phase05.protocol import FrozenCandidate
from afmc_fm.phase05.runner import prepare_phase05_cohort
from afmc_fm.simulator.cohort import simulate_world
from afmc_fm.simulator.config import SimulatorConfig

WORLDS = ("smooth", "jumps", "informative_observation")
PRIMARY_SIZES = (5, 10, 20, 40)
ALL_SIZES = (5, 10, 20, 40, 80, 100)


def _frozen_candidate() -> FrozenCandidate:
    return FrozenCandidate(
        flow_mode="time_scaled",
        jump_mode="residual",
        uncertainty_mode="decoupled",
        strict_history=True,
        state_dim=24,
        time_scale_days=30.0,
        jump_eligible_event_codes=("SYNTHETIC_INTERVENTION",),
        assimilation_semantics="phase0_grucell_unchanged",
        trainable_parameters=6900,
        matched_gru_hidden_size=38,
        matched_gru_parameters=6888,
        matched_mlp_hidden_size=216,
        matched_mlp_parameters=6915,
        protocol_lock_sha256="a" * 64,
        development_artifact_hashes={"flow_gate.csv": "b" * 64},
    )


def _metric_rows(
    *,
    candidate_offsets: dict[str, float] | None = None,
    early_n_loss: bool = False,
) -> pd.DataFrame:
    candidate_offsets = candidate_offsets or {world: -0.10 for world in WORLDS}
    rows = []
    for world_index, world in enumerate(WORLDS):
        for bundle_index in range(10):
            seeds = (701 + bundle_index, 801 + bundle_index, 901 + bundle_index)
            for n_train in ALL_SIZES:
                base = 1.2 - 0.08 * np.log2(n_train / 5) + 0.01 * world_index
                for model in (
                    "phase05_candidate",
                    "matched_gru",
                    "matched_representation_mlp",
                ):
                    value = base
                    if model == "phase05_candidate":
                        value += candidate_offsets[world]
                        if early_n_loss and n_train in (5, 10, 20):
                            value += 0.20
                        elif early_n_loss and n_train == 40:
                            value -= 0.80
                    elif model == "matched_representation_mlp":
                        value += 0.01
                    rows.append(
                        {
                            "world": world,
                            "cohort_seed": seeds[0],
                            "subset_seed": seeds[1],
                            "model_seed": seeds[2],
                            "n_train": n_train,
                            "model": model,
                            "metric": "mae",
                            "value": value,
                        }
                    )
    return pd.DataFrame(rows)


def test_paired_naulc_effects_returns_ten_seed_level_effects_per_world():
    metrics = _metric_rows()

    effects = paired_naulc_effects(metrics, "phase05_candidate", "matched_gru")

    assert len(effects) == 30
    assert set(effects["world"]) == set(WORLDS)
    assert set(effects.groupby("world").size()) == {10}
    assert np.all(effects["effect"].to_numpy() > 0)
    assert effects[["cohort_seed", "subset_seed", "model_seed"]].drop_duplicates().shape[0] == 10


def test_paired_naulc_effects_requires_exactly_ten_matched_bundles():
    metrics = _metric_rows()
    metrics = metrics.loc[metrics["cohort_seed"] != 710]

    with pytest.raises(ValueError, match="exactly ten matched confirmatory bundles"):
        paired_naulc_effects(metrics, "phase05_candidate", "matched_gru")


def test_paired_bootstrap_is_deterministic_and_resamples_seed_effects():
    effects = np.linspace(-0.02, 0.07, 10)

    first = paired_bootstrap_mean_ci(effects, resamples=10_000, seed=20260824)
    second = paired_bootstrap_mean_ci(effects, resamples=10_000, seed=20260824)

    assert first == second
    assert len(first) == 2
    assert first[0] < np.mean(effects) < first[1]


def test_exact_sign_test_matches_scipy_for_eight_of_ten_wins():
    effects = np.array([1.0] * 8 + [-1.0] * 2)

    result = exact_sign_test(effects)
    expected = binomtest(8, 10, 0.5, alternative="greater")

    assert result["wins"] == 8
    assert result["n"] == 10
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["p_value"] >= 0.05


def test_holm_adjust_is_monotone_in_sorted_p_value_order_and_capped():
    p_values = np.array([0.001, 0.04, 0.02, 0.5, 0.005, 0.9])

    adjusted = holm_adjust(p_values)

    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all((0 <= adjusted) & (adjusted <= 1))
    np.testing.assert_allclose(adjusted, [0.006, 0.12, 0.08, 1.0, 0.025, 1.0])


def test_confirmation_model_plan_contains_exact_locked_model_set():
    config = Phase05Config()
    frozen = _frozen_candidate()
    jobs = build_confirmation_jobs(config, frozen, frozen_candidate_hash="c" * 64)

    assert CONFIRMATORY_MODELS == (
        "phase05_candidate",
        "matched_gru",
        "matched_representation_mlp",
        "representation_linear",
        "representation_mlp_original",
        "gru_original",
        "phase0_flow_jump_reference",
    )
    assert PRIMARY_COMPARATORS == ("matched_gru", "matched_representation_mlp")
    assert len(jobs) == 3 * 10 * 6 * 7
    assert {job.shard.world for job in jobs} == set(WORLDS)
    assert {job.n_train for job in jobs} == set(ALL_SIZES)
    assert {job.model for job in jobs} == set(CONFIRMATORY_MODELS)
    assert all(job.frozen_candidate_hash == "c" * 64 for job in jobs)
    candidate_variants = {job.variant for job in jobs if job.model == "phase05_candidate"}
    assert candidate_variants == {"time_scaled__residual__decoupled"}


def test_confirmation_dispatcher_executes_every_locked_model_with_exact_provenance():
    config = Phase05Config(max_epochs=1, patience=1)
    frozen = _frozen_candidate()
    cohort = simulate_world(
        "jumps",
        SimulatorConfig(
            cohort_size=36,
            followup_days=45.0,
            intervention_rate=0.2,
        ),
        seed=701,
    )
    prepared = prepare_phase05_cohort(cohort, include_historical=True)
    jobs = [
        job
        for job in build_confirmation_jobs(
            config,
            frozen,
            frozen_candidate_hash="c" * 64,
        )
        if job.shard.world == "jumps"
        and job.shard.seed_bundle == config.confirmatory_bundles[0]
        and job.n_train == 5
    ]

    assert len(jobs) == 7
    results = [
        run_confirmation_job(
            job,
            prepared,
            config=config,
            frozen=frozen,
            device=torch.device("cpu"),
        )
        for job in jobs
    ]

    for job, result in zip(jobs, results, strict=True):
        assert set(result["stage"]) == {"confirmation"}
        assert set(result["world"]) == {"jumps"}
        assert set(result["cohort_seed"]) == {701}
        assert set(result["subset_seed"]) == {801}
        assert set(result["model_seed"]) == {901}
        assert set(result["n_train"]) == {5}
        assert set(result["model"]) == {job.model}
        assert set(result["variant"]) == {job.variant}
        assert "mae" in set(result["metric"])
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


def test_primary_gate_requires_same_two_worlds_against_both_controls_and_early_n_win():
    metrics = _metric_rows(
        candidate_offsets={
            "smooth": -0.10,
            "jumps": -0.08,
            "informative_observation": 0.04,
        }
    )

    result = evaluate_primary_gate(metrics, win_requirement=8)

    assert result["headline_passed"] is True
    world_summary = result["world_summary"]
    assert world_summary["passed"].sum() == 4
    assert set(world_summary.loc[world_summary["passed"], "world"]) == {"smooth", "jumps"}
    assert result["shared_passing_worlds"] == ("smooth", "jumps")
    assert result["early_n_passed"] is True


def test_primary_gate_cannot_be_rescued_without_an_early_n_joint_win():
    metrics = _metric_rows(early_n_loss=True)

    result = evaluate_primary_gate(metrics, win_requirement=8)

    assert result["world_summary"]["passed"].all()
    assert result["early_n_passed"] is False
    assert result["headline_passed"] is False


def test_persist_confirmation_analysis_writes_only_six_primary_contrasts(tmp_path):
    metrics = _metric_rows()
    config = Phase05Config()

    result = persist_confirmation_analysis(tmp_path, metrics, config=config)

    expected = {
        "paired_naulc_effects.csv",
        "bootstrap_intervals.csv",
        "sign_tests.csv",
        "primary_gate_summary.csv",
    }
    assert {path.name for path in result.values()} == expected
    confirmation = tmp_path / "confirmation"
    assert expected == {path.name for path in confirmation.glob("*.csv")}

    sign_tests = pd.read_csv(confirmation / "sign_tests.csv")
    assert len(sign_tests) == 6
    assert set(sign_tests["comparator"]) == set(PRIMARY_COMPARATORS)
    assert sign_tests["holm_p_value"].between(0, 1).all()

    effects = pd.read_csv(confirmation / "paired_naulc_effects.csv")
    assert len(effects) == 60
    assert set(effects.groupby(["world", "comparator"]).size()) == {10}
