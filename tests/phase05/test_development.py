import numpy as np
import pandas as pd
import pytest

from afmc_fm.phase05.development import (
    representation_timing_audit,
    select_flow,
    select_jump,
    select_uncertainty,
)


def _candidate_frame(
    *,
    world: str,
    metric: str,
    variants: dict[str, list[float]],
    train_sizes=(5, 10, 20, 40),
) -> pd.DataFrame:
    rows = []
    for variant, seed_aulcs in variants.items():
        for seed_index, target_aulc in enumerate(seed_aulcs, start=1):
            for n_train in train_sizes:
                rows.append(
                    {
                        "world": world,
                        "metric": metric,
                        "variant": variant,
                        "cohort_seed": 400 + seed_index,
                        "subset_seed": 500 + seed_index,
                        "model_seed": 600 + seed_index,
                        "n_train": n_train,
                        "value": target_aulc,
                        "trainable_parameters": {
                            "none": 5000,
                            "gated": 6500,
                            "time_scaled": 6200,
                            "gru": 6800,
                            "residual": 6300,
                            "joint": 7000,
                            "decoupled": 6900,
                            "deterministic": 6200,
                        }.get(variant, 6000),
                    }
                )
    return pd.DataFrame(rows)


def test_select_flow_uses_no_jump_deterministic_isolation_and_gate():
    frame = _candidate_frame(
        world="smooth",
        metric="mae",
        variants={
            "none__none__deterministic": [1.0] * 5,
            "gated__none__deterministic": [0.95, 0.95, 0.95, 0.95, 1.01],
            "time_scaled__none__deterministic": [0.90, 0.91, 0.92, 0.93, 0.94],
        },
    )

    result = select_flow(frame, locked_min_relative_effect=0.02)

    assert result["passed"] is True
    assert result["selected"] == "time_scaled"
    assert set(result["gate_table"]["candidate"]) == {"gated", "time_scaled"}
    assert set(result["gate_table"]["control"]) == {"none"}


def test_select_flow_prefers_lower_parameter_candidate_when_effects_within_half_percent():
    frame = _candidate_frame(
        world="smooth",
        metric="mae",
        variants={
            "none__none__deterministic": [1.0] * 5,
            "gated__none__deterministic": [0.960] * 5,
            "time_scaled__none__deterministic": [0.956] * 5,
        },
    )

    result = select_flow(frame, locked_min_relative_effect=0.02)

    assert result["passed"] is True
    assert result["selected"] == "time_scaled"
    gated = result["gate_table"].set_index("candidate").loc["gated"]
    scaled = result["gate_table"].set_index("candidate").loc["time_scaled"]
    assert abs(gated["mean_relative_improvement"] - scaled["mean_relative_improvement"]) < 0.005
    assert scaled["trainable_parameters"] < gated["trainable_parameters"]


def test_select_flow_fails_when_no_learned_flow_passes():
    frame = _candidate_frame(
        world="smooth",
        metric="mae",
        variants={
            "none__none__deterministic": [1.0] * 5,
            "gated__none__deterministic": [0.99, 0.99, 0.99, 0.99, 1.0],
            "time_scaled__none__deterministic": [1.01, 1.0, 0.99, 1.0, 1.01],
        },
    )

    result = select_flow(frame, locked_min_relative_effect=0.02)

    assert result["passed"] is False
    assert result["selected"] is None


def test_select_jump_refuses_when_flow_stage_failed():
    frame = _candidate_frame(
        world="jumps",
        metric="mae",
        variants={
            "time_scaled__none__deterministic": [1.0] * 5,
            "time_scaled__gru__deterministic": [0.9] * 5,
            "time_scaled__residual__deterministic": [0.85] * 5,
        },
    )

    with pytest.raises(RuntimeError, match="flow selection must pass"):
        select_jump(
            frame,
            selected_flow=None,
            flow_passed=False,
            locked_min_relative_effect=0.02,
        )


def test_select_jump_uses_selected_flow_and_no_jump_control():
    frame = _candidate_frame(
        world="jumps",
        metric="mae",
        variants={
            "time_scaled__none__deterministic": [1.0] * 5,
            "time_scaled__gru__deterministic": [0.95, 0.95, 0.95, 0.95, 1.01],
            "time_scaled__residual__deterministic": [0.90, 0.91, 0.92, 0.93, 0.94],
            "gated__residual__deterministic": [0.1] * 5,
        },
    )

    result = select_jump(
        frame,
        selected_flow="time_scaled",
        flow_passed=True,
        locked_min_relative_effect=0.02,
    )

    assert result["passed"] is True
    assert result["selected"] == "residual"
    assert set(result["gate_table"]["candidate"]) == {"gru", "residual"}


def _uncertainty_frame() -> pd.DataFrame:
    rows = []
    worlds = ("smooth", "jumps", "informative_observation")
    for world in worlds:
        for seed_index in range(1, 6):
            for n_train in (5, 10, 20, 40):
                for variant, mae in (
                    ("joint", 1.01),
                    ("decoupled", 1.00),
                    ("deterministic", 1.00),
                ):
                    rows.append(
                        {
                            "world": world,
                            "metric": "mae",
                            "variant": f"time_scaled__residual__{variant}",
                            "cohort_seed": 400 + seed_index,
                            "subset_seed": 500 + seed_index,
                            "model_seed": 600 + seed_index,
                            "n_train": n_train,
                            "value": mae,
                        }
                    )
            for variant, nll, coverage in (
                ("joint", 0.60, 0.82),
                ("decoupled", 0.50, 0.88),
            ):
                rows.extend(
                    [
                        {
                            "world": world,
                            "metric": "nll",
                            "variant": f"time_scaled__residual__{variant}",
                            "cohort_seed": 400 + seed_index,
                            "subset_seed": 500 + seed_index,
                            "model_seed": 600 + seed_index,
                            "n_train": 5,
                            "value": nll,
                        },
                        {
                            "world": world,
                            "metric": "coverage_90",
                            "variant": f"time_scaled__residual__{variant}",
                            "cohort_seed": 400 + seed_index,
                            "subset_seed": 500 + seed_index,
                            "model_seed": 600 + seed_index,
                            "n_train": 5,
                            "value": coverage,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def test_select_uncertainty_keeps_point_admissible_probabilistic_winner():
    result = select_uncertainty(
        _uncertainty_frame(),
        selected_flow="time_scaled",
        selected_jump="residual",
        locked_uncertainty_mae_tolerance=0.02,
    )

    assert result["selected"] == "decoupled"
    assert result["point_admissible"]["decoupled"] is True
    assert result["point_admissible"]["joint"] is True
    assert result["probabilistic_world_wins"]["decoupled"] == 3


def test_select_uncertainty_falls_back_to_deterministic_if_point_error_exceeds_tolerance():
    frame = _uncertainty_frame()
    mask = frame["variant"].str.endswith("__decoupled") & (frame["metric"] == "mae")
    frame.loc[mask, "value"] = 1.05
    mask = frame["variant"].str.endswith("__joint") & (frame["metric"] == "mae")
    frame.loc[mask, "value"] = 1.04

    result = select_uncertainty(
        frame,
        selected_flow="time_scaled",
        selected_jump="residual",
        locked_uncertainty_mae_tolerance=0.02,
    )

    assert result["selected"] == "deterministic"
    assert result["point_admissible"]["decoupled"] is False
    assert result["point_admissible"]["joint"] is False


def test_representation_timing_audit_reports_paired_deltas_without_selection():
    strict = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
    inclusive = np.array([0.8, 0.9, 1.0, 1.1, 1.2])

    result = representation_timing_audit(strict, inclusive)

    np.testing.assert_allclose(result["paired_delta_inclusive_minus_strict"], inclusive - strict)
    assert result["strict_timing_required"] is True
    assert "selected" not in result
