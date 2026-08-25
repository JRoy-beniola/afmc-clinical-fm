import numpy as np
import pandas as pd
import pytest
import torch

from afmc_fm.experiments.runner import build_complete_truth_targets
from afmc_fm.phase05.robustness import (
    complete_truth_phase05_batch,
    evaluate_site_shift,
)
from afmc_fm.phase05.runner import prepare_phase05_cohort
from afmc_fm.simulator.cohort import simulate_world
from afmc_fm.simulator.config import SimulatorConfig

MODELS = (
    "phase05_candidate",
    "matched_gru",
    "matched_representation_mlp",
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
