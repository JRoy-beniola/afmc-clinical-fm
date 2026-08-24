import numpy as np
import pandas as pd
import pytest

from afmc_fm.phase05.analysis import normalized_log_n_aulc, paired_development_gate


def test_normalized_log_n_aulc_matches_log_space_trapezoid():
    frame = pd.DataFrame(
        {
            "n_train": [5, 10, 20, 40],
            "value": [4.0, 3.0, 2.0, 1.0],
        }
    )

    result = normalized_log_n_aulc(frame)

    x = np.log(np.array([5.0, 10.0, 20.0, 40.0]))
    y = np.array([4.0, 3.0, 2.0, 1.0])
    expected = np.trapezoid(y, x) / (x[-1] - x[0])
    assert result == pytest.approx(expected)


def test_normalized_log_n_aulc_ignores_secondary_train_sizes():
    frame = pd.DataFrame(
        {
            "n_train": [5, 10, 20, 40, 80, 100],
            "value": [4.0, 3.0, 2.0, 1.0, 1000.0, 2000.0],
        }
    )

    result = normalized_log_n_aulc(frame)

    x = np.log(np.array([5.0, 10.0, 20.0, 40.0]))
    expected = np.trapezoid(np.array([4.0, 3.0, 2.0, 1.0]), x) / (x[-1] - x[0])
    assert result == pytest.approx(expected)


def test_normalized_log_n_aulc_rejects_missing_primary_point():
    frame = pd.DataFrame(
        {
            "n_train": [5, 10, 40],
            "value": [4.0, 3.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="exactly one value at each primary train size"):
        normalized_log_n_aulc(frame)


def test_normalized_log_n_aulc_rejects_duplicate_primary_point():
    frame = pd.DataFrame(
        {
            "n_train": [5, 10, 20, 20, 40],
            "value": [4.0, 3.0, 2.0, 2.1, 1.0],
        }
    )

    with pytest.raises(ValueError, match="exactly one value at each primary train size"):
        normalized_log_n_aulc(frame)


def test_normalized_log_n_aulc_rejects_nonfinite_primary_value():
    frame = pd.DataFrame(
        {
            "n_train": [5, 10, 20, 40],
            "value": [4.0, np.nan, 2.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        normalized_log_n_aulc(frame)


def test_paired_development_gate_passes_only_with_positive_effect_four_wins_and_threshold():
    control = np.ones(5)
    candidate = np.array([0.95, 0.96, 0.97, 0.98, 1.01])

    result = paired_development_gate(
        candidate,
        control,
        min_relative_effect=0.02,
    )

    assert result["passed"] is True
    assert result["wins"] == 4
    assert result["mean_improvement"] == pytest.approx(np.mean(control - candidate))
    assert result["mean_relative_improvement"] == pytest.approx(
        np.mean((control - candidate) / control)
    )
    assert result["candidate_mean_aulc"] == pytest.approx(candidate.mean())
    assert result["control_mean_aulc"] == pytest.approx(control.mean())


def test_paired_development_gate_fails_with_only_three_seed_wins():
    control = np.ones(5)
    candidate = np.array([0.80, 0.80, 0.80, 1.01, 1.01])

    result = paired_development_gate(
        candidate,
        control,
        min_relative_effect=0.02,
    )

    assert result["mean_improvement"] > 0
    assert result["mean_relative_improvement"] > 0.02
    assert result["wins"] == 3
    assert result["passed"] is False


def test_paired_development_gate_fails_below_locked_relative_effect():
    control = np.ones(5)
    candidate = np.array([0.99, 0.99, 0.99, 0.99, 1.0])

    result = paired_development_gate(
        candidate,
        control,
        min_relative_effect=0.02,
    )

    assert result["wins"] == 4
    assert result["mean_improvement"] > 0
    assert result["mean_relative_improvement"] < 0.02
    assert result["passed"] is False


def test_paired_development_gate_requires_exactly_five_finite_paired_bundles():
    with pytest.raises(ValueError, match="exactly five paired development bundles"):
        paired_development_gate(
            np.array([0.9, 0.9, 0.9, 0.9]),
            np.ones(4),
            min_relative_effect=0.02,
        )

    with pytest.raises(ValueError, match="finite"):
        paired_development_gate(
            np.array([0.9, 0.9, np.nan, 0.9, 0.9]),
            np.ones(5),
            min_relative_effect=0.02,
        )
