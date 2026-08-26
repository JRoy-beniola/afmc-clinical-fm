from copy import deepcopy
import math
from pathlib import Path

import pytest
import torch

from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.model import Phase05FlowJumpAdapter
from afmc_fm.phase05.training import (
    TrainingDiagnosticEpochRecord,
    TrainingDiagnosticSummary,
    evaluate_phase05_model,
    fit_phase05_model,
)


class _CollectingObserver:
    def __init__(self) -> None:
        self.epochs: list[TrainingDiagnosticEpochRecord] = []
        self.summary: TrainingDiagnosticSummary | None = None

    def on_epoch(self, record: TrainingDiagnosticEpochRecord) -> None:
        self.epochs.append(record)

    def on_training_end(self, summary: TrainingDiagnosticSummary) -> None:
        self.summary = summary


class _FailingObserver(_CollectingObserver):
    def on_epoch(self, record: TrainingDiagnosticEpochRecord) -> None:
        raise ValueError("observer exploded")


def _model(flow_mode: str = "time_scaled") -> Phase05FlowJumpAdapter:
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
    torch.manual_seed(17)
    batch = 3
    steps = 4
    value_dim = 3
    return {
        "representations": torch.randn(batch, steps, 16),
        "values": torch.randn(batch, steps, value_dim),
        "masks": torch.randint(0, 2, (batch, steps, value_dim)).float(),
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
        "valid": torch.ones(batch, steps),
    }


def _tiny_config() -> Phase05Config:
    return Phase05Config(
        max_epochs=5,
        patience=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )


def _parameter_bytes(model: torch.nn.Module) -> dict[str, bytes]:
    return {
        name: parameter.detach().cpu().numpy().tobytes()
        for name, parameter in model.named_parameters()
    }


def _state_bytes(state: dict[str, torch.Tensor]) -> dict[str, bytes]:
    return {
        name: tensor.detach().cpu().numpy().tobytes()
        for name, tensor in state.items()
    }


def _fit_with_recorder(
    model: Phase05FlowJumpAdapter | None = None,
) -> tuple[Phase05FlowJumpAdapter, _CollectingObserver, dict[str, torch.Tensor]]:
    train = _batch()
    validation = deepcopy(train)
    recorder = _CollectingObserver()
    fitted = _model() if model is None else model
    fit_phase05_model(
        fitted,
        deepcopy(train),
        deepcopy(validation),
        _tiny_config(),
        torch.device("cpu"),
        diagnostics=recorder,
    )
    return fitted, recorder, validation


def test_diagnostics_are_byte_identical_to_default_training_path():
    torch.manual_seed(29)
    baseline = _model()
    observed = deepcopy(baseline)
    train = _batch()
    validation = deepcopy(train)

    fit_phase05_model(
        baseline,
        deepcopy(train),
        deepcopy(validation),
        _tiny_config(),
        torch.device("cpu"),
    )
    recorder = _CollectingObserver()
    fit_phase05_model(
        observed,
        deepcopy(train),
        deepcopy(validation),
        _tiny_config(),
        torch.device("cpu"),
        diagnostics=recorder,
    )

    assert _parameter_bytes(observed) == _parameter_bytes(baseline)
    assert recorder.summary is not None
    assert evaluate_phase05_model(
        observed, validation, torch.device("cpu")
    ) == evaluate_phase05_model(baseline, validation, torch.device("cpu"))


def test_observer_gets_one_record_per_completed_epoch():
    _, recorder, _ = _fit_with_recorder()

    assert recorder.summary is not None
    assert len(recorder.epochs) == recorder.summary.epochs_run
    assert [record.epoch for record in recorder.epochs] == list(
        range(1, recorder.summary.epochs_run + 1)
    )
    assert 1 <= recorder.summary.selected_checkpoint_epoch <= recorder.summary.epochs_run
    assert 1 <= recorder.summary.shadow_mae_checkpoint_epoch <= recorder.summary.epochs_run
    assert recorder.summary.stop_epoch == recorder.summary.epochs_run
    assert recorder.summary.early_stop_reason in {
        "patience_exhausted",
        "max_epochs_reached",
    }


def test_none_flow_reports_exact_zero_displacement():
    _, recorder, _ = _fit_with_recorder(_model("none"))

    assert recorder.epochs
    for record in recorder.epochs:
        assert record.mean_flow_displacement == 0.0
        assert record.median_flow_displacement == 0.0
        assert record.p95_flow_displacement == 0.0


def test_diagnostic_losses_and_norms_are_finite_and_non_negative():
    _, recorder, _ = _fit_with_recorder()

    assert recorder.epochs
    for record in recorder.epochs:
        for value in (
            record.train_core_loss,
            record.validation_core_loss,
            record.validation_mae,
            record.validation_rmse,
            record.gradient_l2_norm,
            record.parameter_l2_norm,
            record.mean_flow_displacement,
            record.median_flow_displacement,
            record.p95_flow_displacement,
        ):
            assert math.isfinite(value)
        assert record.gradient_l2_norm >= 0.0
        assert record.parameter_l2_norm >= 0.0
        assert record.mean_flow_displacement >= 0.0
        assert record.median_flow_displacement >= 0.0
        assert record.p95_flow_displacement >= 0.0


def test_production_snapshot_matches_returned_model():
    model, recorder, _ = _fit_with_recorder()

    assert recorder.summary is not None
    assert _state_bytes(recorder.summary.production_state_dict) == _parameter_bytes(model)
    assert recorder.summary.shadow_state_dict


def test_diagnostic_training_requires_valid_positions_before_optimization():
    model = _model()
    train = _batch()
    validation = deepcopy(train)
    validation.pop("valid")

    with pytest.raises(ValueError, match="valid"):
        fit_phase05_model(
            model,
            train,
            validation,
            _tiny_config(),
            torch.device("cpu"),
            diagnostics=_CollectingObserver(),
        )


def test_observer_error_fails_the_diagnostic_run_clearly():
    train = _batch()

    with pytest.raises(RuntimeError, match="on_epoch") as caught:
        fit_phase05_model(
            _model(),
            train,
            deepcopy(train),
            _tiny_config(),
            torch.device("cpu"),
            diagnostics=_FailingObserver(),
        )

    assert isinstance(caught.value.__cause__, ValueError)


def test_phase05_training_does_not_import_phase06():
    root = Path(__file__).parents[2]
    source = (root / "src" / "afmc_fm" / "phase05" / "training.py").read_text(
        encoding="utf-8"
    )
    assert "afmc_fm.phase06" not in source
