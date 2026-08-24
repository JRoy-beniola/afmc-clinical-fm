import numpy as np
import torch

from afmc_fm.data.splits import split_patient_ids
from afmc_fm.experiments.runner import select_low_n_budget
from afmc_fm.phase05.config import Phase05Config, SeedBundle
from afmc_fm.phase05.runner import (
    padded_phase05_batch,
    prepare_phase05_cohort,
    run_phase05_variant,
)
from afmc_fm.simulator.cohort import simulate_world
from afmc_fm.simulator.config import SimulatorConfig


def _tiny_cohort(seed: int = 41):
    return simulate_world(
        "jumps",
        SimulatorConfig(
            cohort_size=36,
            followup_days=45.0,
            intervention_rate=0.2,
        ),
        seed=seed,
    )


def _tiny_config() -> Phase05Config:
    return Phase05Config(
        train_sizes=(5, 10, 20, 40, 80, 100),
        development_bundles=(
            SeedBundle(401, 501, 601),
            SeedBundle(402, 502, 602),
            SeedBundle(403, 503, 603),
            SeedBundle(404, 504, 604),
            SeedBundle(405, 505, 605),
        ),
        confirmatory_bundles=tuple(
            SeedBundle(700 + index, 800 + index, 900 + index)
            for index in range(1, 11)
        ),
        max_epochs=1,
        patience=1,
    )


def test_phase05_prepare_uses_strict_sequences_and_preserves_patient_map():
    cohort = _tiny_cohort()

    prepared = prepare_phase05_cohort(cohort)

    assert set(prepared.patient_by_id) == {patient.patient_id for patient in cohort.patients}
    assert set(prepared.sequences) == set(prepared.patient_by_id)
    assert all(
        sequence.jump_eligible_mask.shape == sequence.update_mask.shape
        for sequence in prepared.sequences.values()
    )


def test_phase05_low_n_budget_matches_historical_split_rules():
    cohort = _tiny_cohort()
    prepared = prepare_phase05_cohort(cohort)
    all_ids = list(prepared.patient_by_id)
    development_pool, _, test_ids = split_patient_ids(
        all_ids,
        seed=0,
        train_fraction=0.8,
        val_fraction=0.0,
    )

    expected = select_low_n_budget(development_pool, 5, 501)
    result = run_phase05_variant(
        prepared,
        _tiny_config(),
        world="jumps",
        seed_bundle=SeedBundle(401, 501, 601),
        n_train=5,
        flow_mode="none",
        jump_mode="none",
        uncertainty_mode="deterministic",
        device=torch.device("cpu"),
    )

    assert set(result["n_fit"]) == {len(expected.fit_ids)}
    assert set(result["n_validation"]) == {len(expected.validation_ids)}
    assert set(result["n_train"]) == {5}
    assert len(test_ids) > 0


def test_padded_phase05_batch_contains_jump_eligibility_and_latent_truth():
    cohort = _tiny_cohort()
    prepared = prepare_phase05_cohort(cohort)
    patient_ids = list(prepared.patient_by_id)[:3]
    sequences = [prepared.sequences[patient_id] for patient_id in patient_ids]
    patients = [prepared.patient_by_id[patient_id] for patient_id in patient_ids]

    batch = padded_phase05_batch(sequences, patients=patients)

    assert "jump_eligible_mask" in batch
    assert batch["jump_eligible_mask"].shape == batch["update_mask"].shape
    assert "latent_targets" in batch
    assert "latent_valid" in batch
    assert batch["latent_targets"].shape[:2] == batch["update_mask"].shape
    assert torch.isfinite(batch["latent_targets"]).all()


def test_run_phase05_variant_returns_unique_tidy_metric_rows():
    cohort = _tiny_cohort(seed=42)
    prepared = prepare_phase05_cohort(cohort)

    result = run_phase05_variant(
        prepared,
        _tiny_config(),
        world="jumps",
        seed_bundle=SeedBundle(401, 501, 601),
        n_train=5,
        flow_mode="time_scaled",
        jump_mode="residual",
        uncertainty_mode="deterministic",
        device=torch.device("cpu"),
        stage="development",
    )

    required = {
        "stage",
        "world",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "n_train",
        "n_fit",
        "n_validation",
        "model",
        "variant",
        "split",
        "site_or_shift",
        "metric",
        "value",
        "trainable_parameters",
        "backend",
    }
    assert required <= set(result.columns)
    assert set(result["stage"]) == {"development"}
    assert set(result["world"]) == {"jumps"}
    assert set(result["model"]) == {"phase05_flow_jump"}
    assert set(result["backend"]) == {"torch"}
    key = [
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
    assert not result.duplicated(key).any()
    defined = result[result["metric"] != "event_roc_auc"]
    assert np.isfinite(defined["value"]).all()


def test_historical_reference_is_explicitly_labelled_and_separate():
    cohort = _tiny_cohort(seed=43)
    prepared = prepare_phase05_cohort(cohort, include_historical=True)

    result = run_phase05_variant(
        prepared,
        _tiny_config(),
        world="jumps",
        seed_bundle=SeedBundle(401, 501, 601),
        n_train=5,
        flow_mode="gated",
        jump_mode="gru",
        uncertainty_mode="joint",
        device=torch.device("cpu"),
        historical_reference=True,
    )

    assert set(result["model"]) == {"phase0_flow_jump_reference"}
    assert set(result["variant"]) == {"historical_inclusive_history"}
