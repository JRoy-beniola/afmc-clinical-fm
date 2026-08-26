import numpy as np
import pandas as pd

from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.confirmation import evaluate_primary_gate, persist_confirmation_analysis

WORLDS = ("smooth", "jumps", "informative_observation")
TRAIN_SIZES = (5, 10, 20, 40, 80, 100)
COMPARATORS = ("matched_gru", "matched_representation_mlp")


def _metrics() -> pd.DataFrame:
    rows = []
    for world_index, world in enumerate(WORLDS):
        for bundle_index in range(10):
            seeds = (701 + bundle_index, 801 + bundle_index, 901 + bundle_index)
            seed_shift = 0.002 * bundle_index
            for n_train in TRAIN_SIZES:
                base = 1.2 - 0.08 * np.log2(n_train / 5) + 0.01 * world_index
                for model in ("phase05_candidate", *COMPARATORS):
                    value = base + seed_shift
                    if model == "phase05_candidate":
                        value -= 0.08 - 0.001 * bundle_index
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


def test_primary_gate_reports_preregistered_effect_statistics_in_locked_world_order():
    result = evaluate_primary_gate(_metrics(), win_requirement=8)
    summary = result["world_summary"]

    required = {
        "mean_effect",
        "median_effect",
        "sd_effect",
        "wins",
        "win_fraction",
        "mean_relative_improvement_pct",
        "passed",
    }
    assert required.issubset(summary.columns)
    assert list(dict.fromkeys(summary["world"])) == list(WORLDS)
    assert summary["sd_effect"].gt(0).all()
    assert summary["mean_relative_improvement_pct"].gt(0).all()


def test_persisted_primary_outputs_keep_relative_effects_and_locked_world_order(tmp_path):
    persist_confirmation_analysis(tmp_path, _metrics(), config=Phase05Config())
    confirmation = tmp_path / "confirmation"

    effects = pd.read_csv(confirmation / "paired_naulc_effects.csv")
    assert "relative_improvement_pct" in effects.columns
    assert np.isfinite(effects["relative_improvement_pct"]).all()
    assert list(dict.fromkeys(effects["world"])) == list(WORLDS)

    summary = pd.read_csv(confirmation / "primary_gate_summary.csv")
    assert {
        "median_effect",
        "sd_effect",
        "mean_relative_improvement_pct",
    }.issubset(summary.columns)
    assert list(dict.fromkeys(summary["world"])) == list(WORLDS)

    for filename in ("bootstrap_intervals.csv", "sign_tests.csv"):
        frame = pd.read_csv(confirmation / filename)
        assert list(dict.fromkeys(frame["world"])) == list(WORLDS)
