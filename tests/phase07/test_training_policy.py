from copy import deepcopy

import pandas as pd
import torch

from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.model import Phase05FlowJumpAdapter
from afmc_fm.phase05.training import fit_phase05_model
from afmc_fm.phase06.diagnostics import Phase06DiagnosticRecorder


def _model(flow_mode: str = "time_scaled") -> Phase05FlowJumpAdapter:
    torch.manual_seed(41)
    return Phase05FlowJumpAdapter(
        representation_dim=16,
        value_dim=3,
        event_dim=3,
        state_dim=24,
        flow_mode=flow_mode,
        jump_mode="none",
        uncertainty_mode="deterministic",
        time_scale_days=30.0,
    )


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(73)
    batch = 3
    steps = 4
    value_dim = 3
    return {
        "representations": torch.randn(batch, steps, 16),
        "values": torch.randn(batch, steps, value_dim),
        "masks": torch.ones(batch, steps, value_dim),
        "event_features": torch.randn(batch, steps, 3),
        "times": torch.tensor(
            [
                [0.0, 2.0, 8.0, 14.0],
                [0.0, 3.0, 7.0, 20.0],
                [0.0, 1.0, 9.0, 18.0],
            ]
        ),
        "update_mask": torch.ones(batch, steps),
        "jump_eligible_mask": torch.zeros(batch, steps),
        "target_values": torch.randn(batch, steps, value_dim),
        "target_masks": torch.ones(batch, steps, value_dim),
        "target_events": torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ),
        "event_valid": torch.ones(batch, steps),
        "valid": torch.ones(batch, steps, dtype=torch.bool),
    }


def _config() -> Phase05Config:
    return Phase05Config(
        max_epochs=4,
        patience=1,
        learning_rate=1e-30,
        weight_decay=0.0,
    )


def test_default_policy_preserves_historical_patience_termination():
    recorder = Phase06DiagnosticRecorder()
    batch = _batch()

    fit_phase05_model(
        _model(),
        batch,
        deepcopy(batch),
        _config(),
        torch.device("cpu"),
        diagnostics=recorder,
    )

    summary = recorder.summary_payload()
    assert summary["epochs_run"] == 2
    assert summary["stop_epoch"] == 2
    assert summary["early_stop_reason"] == "patience_exhausted"
    assert recorder.trace_frame()["stale_epochs"].tolist() == [0, 1]


def test_forced_horizon_continues_and_is_prefix_identical_to_standard_policy():
    batch = _batch()
    initial_model = _model()
    standard_model = deepcopy(initial_model)
    forced_model = deepcopy(initial_model)
    standard = Phase06DiagnosticRecorder()
    forced = Phase06DiagnosticRecorder()

    fit_phase05_model(
        standard_model,
        batch,
        deepcopy(batch),
        _config(),
        torch.device("cpu"),
        diagnostics=standard,
    )
    fit_phase05_model(
        forced_model,
        batch,
        deepcopy(batch),
        _config(),
        torch.device("cpu"),
        diagnostics=forced,
        stop_on_patience=False,
    )

    standard_summary = standard.summary_payload()
    forced_summary = forced.summary_payload()
    assert standard_summary["epochs_run"] == 2
    assert standard_summary["early_stop_reason"] == "patience_exhausted"
    assert forced_summary["epochs_run"] == 4
    assert forced_summary["stop_epoch"] == 4
    assert forced_summary["early_stop_reason"] == "max_epochs_reached"

    standard_trace = standard.trace_frame().reset_index(drop=True)
    forced_prefix = forced.trace_frame().iloc[: len(standard_trace)].reset_index(drop=True)
    pd.testing.assert_frame_equal(standard_trace, forced_prefix, check_exact=True)
    assert forced.trace_frame()["stale_epochs"].tolist() == [0, 1, 2, 3]


def test_frozen_100_epoch_patience_12_policy_invariants_cover_both_architectures():
    config = Phase05Config(
        max_epochs=100,
        patience=12,
        learning_rate=1e-30,
        weight_decay=0.0,
    )

    for flow_mode in ("none", "time_scaled"):
        batch = _batch()
        initial_model = _model(flow_mode)
        standard_model = deepcopy(initial_model)
        forced_model = deepcopy(initial_model)
        standard = Phase06DiagnosticRecorder()
        forced = Phase06DiagnosticRecorder()

        fit_phase05_model(
            standard_model,
            deepcopy(batch),
            deepcopy(batch),
            config,
            torch.device("cpu"),
            diagnostics=standard,
            stop_on_patience=True,
        )
        fit_phase05_model(
            forced_model,
            deepcopy(batch),
            deepcopy(batch),
            config,
            torch.device("cpu"),
            diagnostics=forced,
            stop_on_patience=False,
        )

        standard_summary = standard.summary_payload()
        forced_summary = forced.summary_payload()
        assert standard_summary["early_stop_reason"] == "patience_exhausted"
        assert int(standard_summary["epochs_run"]) < 100
        assert forced_summary["epochs_run"] == 100
        assert forced_summary["stop_epoch"] == 100
        assert forced_summary["early_stop_reason"] == "max_epochs_reached"

        standard_trace = standard.trace_frame().reset_index(drop=True)
        forced_prefix = forced.trace_frame().iloc[: len(standard_trace)].reset_index(drop=True)
        pd.testing.assert_frame_equal(standard_trace, forced_prefix, check_exact=True)
