import pandas as pd
import pytest
from afmc_fm.phase05.baselines import (
    build_capacity_audit,
    closest_gru_hidden_size,
    closest_mlp_hidden_size,
)

from afmc_fm.models.baselines import GRUBaseline, TorchMLPRegressorBaseline


def _count_gru(value_dim: int, event_dim: int, hidden_size: int) -> int:
    model = GRUBaseline(value_dim=value_dim, event_dim=event_dim, hidden_size=hidden_size)
    return sum(parameter.numel() for parameter in model.parameters())


def _count_mlp(input_dim: int, hidden_size: int) -> int:
    return TorchMLPRegressorBaseline(
        input_dim=input_dim,
        seed=0,
        device="cpu",
        hidden_size=hidden_size,
    ).trainable_parameter_count()


def _brute_force_best(target: int, counter) -> int:
    return min(range(1, 257), key=lambda width: (abs(counter(width) - target), width))


def test_closest_gru_hidden_size_matches_brute_force_module_counts():
    target = 6967
    expected = _brute_force_best(target, lambda width: _count_gru(3, 3, width))

    result = closest_gru_hidden_size(target, value_dim=3, event_dim=3)

    assert result == expected


def test_closest_mlp_hidden_size_matches_brute_force_module_counts():
    target = 6967
    expected = _brute_force_best(target, lambda width: _count_mlp(19, width))

    result = closest_mlp_hidden_size(target, input_dim=19)

    assert result == expected


def test_capacity_search_breaks_exact_mismatch_ties_toward_smaller_width():
    counts = [_count_mlp(19, width) for width in range(1, 257)]
    for width in range(1, 256):
        left = counts[width - 1]
        right = counts[width]
        if (left + right) % 2 == 0:
            target = (left + right) // 2
            assert abs(left - target) == abs(right - target)
            assert closest_mlp_hidden_size(target, input_dim=19) == width
            return
    pytest.fail("expected to find an exact adjacent-width midpoint tie")


def test_capacity_search_rejects_nonpositive_target_or_dimensions():
    with pytest.raises(ValueError, match="positive"):
        closest_gru_hidden_size(0, value_dim=3, event_dim=3)
    with pytest.raises(ValueError, match="positive"):
        closest_gru_hidden_size(1000, value_dim=0, event_dim=3)
    with pytest.raises(ValueError, match="positive"):
        closest_mlp_hidden_size(1000, input_dim=0)


def test_capacity_audit_reports_exact_matching_metadata():
    target = 6967

    audit = build_capacity_audit(
        target_parameters=target,
        value_dim=3,
        event_dim=3,
        representation_input_dim=19,
    )

    assert isinstance(audit, pd.DataFrame)
    assert list(audit["control"]) == ["matched_gru", "matched_representation_mlp"]
    assert set(audit.columns) == {
        "control",
        "target_parameters",
        "hidden_size",
        "actual_parameters",
        "absolute_mismatch",
        "relative_mismatch",
    }
    assert (audit["target_parameters"] == target).all()
    assert (audit["hidden_size"] >= 1).all()
    assert (audit["hidden_size"] <= 256).all()
    assert (audit["absolute_mismatch"] == (audit["actual_parameters"] - target).abs()).all()
    assert audit.loc[0, "actual_parameters"] == _count_gru(
        3, 3, int(audit.loc[0, "hidden_size"])
    )
    assert audit.loc[1, "actual_parameters"] == _count_mlp(
        19, int(audit.loc[1, "hidden_size"])
    )
    assert audit.loc[0, "relative_mismatch"] == pytest.approx(
        audit.loc[0, "absolute_mismatch"] / target
    )
    assert audit.loc[1, "relative_mismatch"] == pytest.approx(
        audit.loc[1, "absolute_mismatch"] / target
    )
