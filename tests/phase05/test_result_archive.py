from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "analysis" / "phase05" / "generate_tables.py"
RAW = ROOT / "docs" / "results" / "phase05" / "raw" / "official_output"


def test_phase05_result_generator_reconstructs_official_flow_gate(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(RAW),
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr

    generated = pd.read_csv(tmp_path / "flow_gate_verified.csv")

    assert set(generated["candidate"]) == {"gated", "time_scaled"}
    assert generated["verification_passed"].all()

    gated = generated.loc[generated["candidate"] == "gated"].iloc[0]
    timed = generated.loc[generated["candidate"] == "time_scaled"].iloc[0]

    assert bool(gated["recomputed_passed"]) is False
    assert int(gated["recomputed_wins"]) == 2

    assert bool(timed["recomputed_passed"]) is False
    assert int(timed["recomputed_wins"]) == 2


def test_phase05_result_generator_preserves_complete_flow_metric_surface(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(RAW),
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr

    metrics = pd.read_csv(
        tmp_path / "flow_cell_metrics.csv",
        float_precision="round_trip",
    )

    # 60 cells × 6 persisted metrics per cell.
    assert len(metrics) == 360

    assert set(metrics["variant"]) == {
        "none__none__deterministic",
        "gated__none__deterministic",
        "time_scaled__none__deterministic",
    }
    assert set(metrics["n_train"]) == {5, 10, 20, 40}
    assert set(metrics["metric"]) == {
        "mae",
        "rmse",
        "event_roc_auc",
        "event_brier",
        "event_log_loss",
        "latent_aligned_r2",
    }

    assert set(metrics["stage"]) == {"flow"}
    assert set(metrics["world"]) == {"smooth"}
    assert set(metrics["split"]) == {"test"}
    assert set(metrics["site_or_shift"]) == {"all"}
    assert set(metrics["model"]) == {"phase05_flow_jump"}
    assert set(metrics["backend"]) == {"torch"}

    bundles = (
        metrics[
            [
                "cohort_seed",
                "subset_seed",
                "model_seed",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "cohort_seed",
                "subset_seed",
                "model_seed",
            ]
        )
        .reset_index(drop=True)
    )

    assert bundles.to_records(index=False).tolist() == [
        (401, 501, 601),
        (402, 502, 602),
        (403, 503, 603),
        (404, 504, 604),
        (405, 505, 605),
    ]

    cell_key = [
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "n_train",
        "model",
        "variant",
        "world",
        "stage",
    ]
    assert len(metrics[cell_key].drop_duplicates()) == 60

    scientific_metric_key = [
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "n_train",
        "model",
        "variant",
        "world",
        "stage",
        "split",
        "site_or_shift",
        "metric",
    ]
    assert not metrics.duplicated(scientific_metric_key).any()

    assert metrics["value"].notna().all()
    assert np.isfinite(metrics["value"].to_numpy(dtype=float)).all()

    expected_budget = {
        5: (4, 1),
        10: (8, 2),
        20: (16, 4),
        40: (32, 8),
    }

    observed_budget = (
        metrics[
            [
                "n_train",
                "n_fit",
                "n_validation",
            ]
        ]
        .drop_duplicates()
        .sort_values("n_train")
    )

    assert {
        int(row.n_train): (
            int(row.n_fit),
            int(row.n_validation),
        )
        for row in observed_budget.itertuples(index=False)
    } == expected_budget


def test_phase05_result_generator_emits_exploratory_flow_effect_tables(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(RAW),
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr

    bundle = pd.read_csv(
        tmp_path / "flow_bundle_effects.csv",
        float_precision="round_trip",
    )
    n_effects = pd.read_csv(
        tmp_path / "flow_n_effects.csv",
        float_precision="round_trip",
    )
    summary = pd.read_csv(
        tmp_path / "flow_metric_summary.csv",
        float_precision="round_trip",
    )

    # 2 candidates × 5 matched development bundles.
    assert len(bundle) == 10
    assert set(bundle["candidate"]) == {"gated", "time_scaled"}

    bundle_key = [
        "candidate",
        "cohort_seed",
        "subset_seed",
        "model_seed",
    ]
    assert not bundle.duplicated(bundle_key).any()

    assert set(bundle["control"]) == {"none"}
    assert bundle["candidate_aulc"].notna().all()
    assert bundle["control_aulc"].notna().all()
    assert bundle["effect"].notna().all()
    assert bundle["relative_improvement"].notna().all()

    np.testing.assert_allclose(
        bundle["effect"].to_numpy(dtype=float),
        (
            bundle["control_aulc"].to_numpy(dtype=float)
            - bundle["candidate_aulc"].to_numpy(dtype=float)
        ),
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        bundle["relative_improvement"].to_numpy(dtype=float),
        (
            bundle["effect"].to_numpy(dtype=float)
            / bundle["control_aulc"].to_numpy(dtype=float)
        ),
        rtol=0.0,
        atol=1e-12,
    )

    # 2 candidates × 5 bundles × 4 N × 6 metrics.
    assert len(n_effects) == 240

    assert set(n_effects["candidate"]) == {"gated", "time_scaled"}
    assert set(n_effects["n_train"]) == {5, 10, 20, 40}
    assert set(n_effects["metric"]) == {
        "mae",
        "rmse",
        "event_roc_auc",
        "event_brier",
        "event_log_loss",
        "latent_aligned_r2",
    }

    paired_key = [
        "candidate",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "n_train",
        "metric",
    ]
    assert not n_effects.duplicated(paired_key).any()

    assert np.isfinite(
        n_effects[
            [
                "candidate_value",
                "control_value",
                "effect",
            ]
        ].to_numpy(dtype=float)
    ).all()

    lower_is_better = {
        "mae",
        "rmse",
        "event_brier",
        "event_log_loss",
    }
    higher_is_better = {
        "event_roc_auc",
        "latent_aligned_r2",
    }

    lower = n_effects["metric"].isin(lower_is_better)
    higher = n_effects["metric"].isin(higher_is_better)

    np.testing.assert_allclose(
        n_effects.loc[lower, "effect"].to_numpy(dtype=float),
        (
            n_effects.loc[lower, "control_value"].to_numpy(dtype=float)
            - n_effects.loc[lower, "candidate_value"].to_numpy(dtype=float)
        ),
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        n_effects.loc[higher, "effect"].to_numpy(dtype=float),
        (
            n_effects.loc[higher, "candidate_value"].to_numpy(dtype=float)
            - n_effects.loc[higher, "control_value"].to_numpy(dtype=float)
        ),
        rtol=0.0,
        atol=1e-12,
    )

    assert (
        n_effects["candidate_better"]
        == (n_effects["effect"] > 0)
    ).all()

    # 2 candidates × 4 N × 6 metrics.
    assert len(summary) == 48

    summary_key = [
        "candidate",
        "n_train",
        "metric",
    ]
    assert not summary.duplicated(summary_key).any()

    required_summary_columns = {
        "candidate",
        "control",
        "n_train",
        "metric",
        "candidate_mean",
        "control_mean",
        "mean_effect",
        "median_effect",
        "sd_effect",
        "wins",
        "win_fraction",
    }
    assert required_summary_columns.issubset(summary.columns)

    assert set(summary["wins"]).issubset({0, 1, 2, 3, 4, 5})

    np.testing.assert_allclose(
        summary["win_fraction"].to_numpy(dtype=float),
        summary["wins"].to_numpy(dtype=float) / 5.0,
        rtol=0.0,
        atol=1e-12,
    )

    # Lock the already-observed exploratory N=40 time-scaled phenomenon.
    time_scaled_n40_mae = summary.loc[
        (summary["candidate"] == "time_scaled")
        & (summary["n_train"] == 40)
        & (summary["metric"] == "mae")
    ]

    assert len(time_scaled_n40_mae) == 1

    row = time_scaled_n40_mae.iloc[0]
    assert int(row["wins"]) == 4
    assert float(row["mean_effect"]) == pytest.approx(
        0.029968,
        abs=1e-6,
    )

    # And preserve the mechanistic mismatch at the same N.
    time_scaled_n40_latent = summary.loc[
        (summary["candidate"] == "time_scaled")
        & (summary["n_train"] == 40)
        & (summary["metric"] == "latent_aligned_r2")
    ]

    assert len(time_scaled_n40_latent) == 1

    latent = time_scaled_n40_latent.iloc[0]
    assert int(latent["wins"]) == 1
    assert float(latent["mean_effect"]) == pytest.approx(
        -0.012063,
        abs=1e-6,
    )
