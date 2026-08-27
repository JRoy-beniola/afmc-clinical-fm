import json
import math
from dataclasses import fields
from importlib import import_module

import pytest
import torch

from afmc_fm.phase05.training import (
    TrainingDiagnosticEpochRecord,
    TrainingDiagnosticSummary,
)

Phase06DiagnosticRecorder = import_module(
    "afmc_fm.phase06.diagnostics"
).Phase06DiagnosticRecorder


def _epoch_record(
    epoch: int = 1,
    **overrides: float,
) -> TrainingDiagnosticEpochRecord:
    values: dict[str, float] = {
        "epoch": epoch,
        "train_core_loss": 1.0,
        "validation_core_loss": 0.9,
        "validation_mae": 0.8,
        "validation_rmse": 0.95,
        "gradient_l2_norm": 1.2,
        "parameter_l2_norm": 3.4,
        "mean_flow_displacement": 0.1,
        "median_flow_displacement": 0.08,
        "p95_flow_displacement": 0.2,
        "best_validation_core_loss_so_far": 0.9,
        "best_core_epoch": 1,
        "shadow_best_validation_mae_so_far": 0.8,
        "shadow_best_mae_epoch": 1,
        "stale_epochs": 0,
    }
    values.update(overrides)
    return TrainingDiagnosticEpochRecord(**values)


def _summary(
    *,
    epochs_run: int = 2,
    selected_checkpoint_epoch: int = 1,
    shadow_mae_checkpoint_epoch: int = 2,
    early_stop_reason: str = "max_epochs_reached",
) -> TrainingDiagnosticSummary:
    return TrainingDiagnosticSummary(
        epochs_run=epochs_run,
        stop_epoch=epochs_run,
        selected_checkpoint_epoch=selected_checkpoint_epoch,
        selected_validation_core_loss=0.9,
        shadow_mae_checkpoint_epoch=shadow_mae_checkpoint_epoch,
        shadow_validation_mae=0.7,
        early_stop_reason=early_stop_reason,
        production_state_dict={"weight": torch.tensor([1.0, 2.0])},
        shadow_state_dict={"weight": torch.tensor([3.0, 4.0])},
    )


def test_recorder_preserves_epoch_order_and_scalar_summary():
    recorder = Phase06DiagnosticRecorder()
    recorder.on_epoch(_epoch_record(epoch=1))
    recorder.on_epoch(
        _epoch_record(
            epoch=2,
            validation_core_loss=0.85,
            validation_mae=0.7,
            best_validation_core_loss_so_far=0.85,
            best_core_epoch=2,
            shadow_best_validation_mae_so_far=0.7,
            shadow_best_mae_epoch=2,
        )
    )
    recorder.on_training_end(_summary())

    trace = recorder.trace_frame()
    assert trace["epoch"].tolist() == [1, 2]
    assert trace.columns.tolist() == [
        field.name for field in fields(TrainingDiagnosticEpochRecord)
    ]
    payload = recorder.summary_payload()
    assert payload["selected_checkpoint_epoch"] == 1
    assert payload["shadow_mae_checkpoint_epoch"] == 2
    assert "production_state_dict" not in payload
    assert "shadow_state_dict" not in payload
    json.dumps(payload)


def test_recorder_rejects_duplicate_or_out_of_order_epochs():
    recorder = Phase06DiagnosticRecorder()
    recorder.on_epoch(_epoch_record(epoch=2))

    with pytest.raises(ValueError, match="strictly increasing"):
        recorder.on_epoch(_epoch_record(epoch=2))
    with pytest.raises(ValueError, match="strictly increasing"):
        recorder.on_epoch(_epoch_record(epoch=1))


@pytest.mark.parametrize(
    "field_name",
    [
        "train_core_loss",
        "validation_core_loss",
        "validation_mae",
        "validation_rmse",
        "gradient_l2_norm",
        "parameter_l2_norm",
        "mean_flow_displacement",
        "median_flow_displacement",
        "p95_flow_displacement",
        "best_validation_core_loss_so_far",
        "shadow_best_validation_mae_so_far",
    ],
)
def test_recorder_rejects_non_finite_epoch_values(field_name):
    recorder = Phase06DiagnosticRecorder()

    with pytest.raises(ValueError, match="finite"):
        recorder.on_epoch(_epoch_record(**{field_name: math.nan}))


def test_recorder_rejects_invalid_training_summary():
    recorder = Phase06DiagnosticRecorder()
    recorder.on_epoch(_epoch_record(epoch=1))
    recorder.on_epoch(_epoch_record(epoch=2))

    with pytest.raises(ValueError, match="early_stop_reason"):
        recorder.on_training_end(_summary(early_stop_reason="other"))

    with pytest.raises(ValueError, match="selected_checkpoint_epoch"):
        recorder.on_training_end(_summary(selected_checkpoint_epoch=3))

    with pytest.raises(ValueError, match="shadow_mae_checkpoint_epoch"):
        recorder.on_training_end(_summary(shadow_mae_checkpoint_epoch=0))


def test_recorder_rejects_summary_that_disagrees_with_recorded_epochs():
    recorder = Phase06DiagnosticRecorder()
    recorder.on_epoch(_epoch_record(epoch=1))

    with pytest.raises(ValueError, match="epochs_run"):
        recorder.on_training_end(_summary(epochs_run=2))


def test_recorder_rejects_second_training_summary():
    recorder = Phase06DiagnosticRecorder()
    recorder.on_epoch(_epoch_record(epoch=1))
    recorder.on_training_end(_summary(epochs_run=1, shadow_mae_checkpoint_epoch=1))

    with pytest.raises(RuntimeError, match="already recorded"):
        recorder.on_training_end(_summary(epochs_run=1, shadow_mae_checkpoint_epoch=1))


def test_checkpoint_properties_return_defensive_cpu_clones():
    recorder = Phase06DiagnosticRecorder()
    recorder.on_epoch(_epoch_record(epoch=1))
    summary = _summary(epochs_run=1, shadow_mae_checkpoint_epoch=1)
    recorder.on_training_end(summary)

    production = recorder.production_state_dict
    shadow = recorder.shadow_state_dict
    assert production["weight"].device.type == "cpu"
    assert shadow["weight"].device.type == "cpu"

    production["weight"].zero_()
    shadow["weight"].zero_()
    summary.production_state_dict["weight"].fill_(99.0)
    summary.shadow_state_dict["weight"].fill_(99.0)

    torch.testing.assert_close(
        recorder.production_state_dict["weight"], torch.tensor([1.0, 2.0])
    )
    torch.testing.assert_close(
        recorder.shadow_state_dict["weight"], torch.tensor([3.0, 4.0])
    )


def test_summary_and_checkpoint_access_require_training_end():
    recorder = Phase06DiagnosticRecorder()

    with pytest.raises(RuntimeError, match="not recorded"):
        recorder.summary_payload()
    with pytest.raises(RuntimeError, match="not recorded"):
        _ = recorder.production_state_dict
    with pytest.raises(RuntimeError, match="not recorded"):
        _ = recorder.shadow_state_dict
