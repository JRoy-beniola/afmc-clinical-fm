from __future__ import annotations

import math
from dataclasses import asdict, fields

import pandas as pd
import torch

from afmc_fm.phase05.training import (
    TrainingDiagnosticEpochRecord,
    TrainingDiagnosticSummary,
)

_ALLOWED_STOP_REASONS = frozenset({"max_epochs_reached", "patience_exhausted"})
_STATE_FIELDS = frozenset({"production_state_dict", "shadow_state_dict"})


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state_dict.items()
    }


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


class Phase06DiagnosticRecorder:
    """In-memory recorder for Phase 0.6 observational training diagnostics."""

    def __init__(self) -> None:
        self._epochs: list[TrainingDiagnosticEpochRecord] = []
        self._summary: TrainingDiagnosticSummary | None = None
        self._production_state_dict: dict[str, torch.Tensor] | None = None
        self._shadow_state_dict: dict[str, torch.Tensor] | None = None

    def on_epoch(self, record: TrainingDiagnosticEpochRecord) -> None:
        if self._summary is not None:
            raise RuntimeError("training summary already recorded")
        if self._epochs and record.epoch <= self._epochs[-1].epoch:
            raise ValueError("diagnostic epochs must be strictly increasing")
        if record.epoch <= 0:
            raise ValueError("epoch must be positive")

        for field in fields(TrainingDiagnosticEpochRecord):
            value = getattr(record, field.name)
            if field.name in {"epoch", "best_core_epoch", "shadow_best_mae_epoch", "stale_epochs"}:
                continue
            _require_finite(field.name, value)

        self._epochs.append(record)

    def on_training_end(self, summary: TrainingDiagnosticSummary) -> None:
        if self._summary is not None:
            raise RuntimeError("training summary already recorded")
        if not self._epochs:
            raise ValueError("cannot record training summary without epoch diagnostics")
        if summary.epochs_run != len(self._epochs):
            raise ValueError("epochs_run does not match recorded epochs")
        if summary.stop_epoch != self._epochs[-1].epoch:
            raise ValueError("stop_epoch does not match the final recorded epoch")
        if summary.early_stop_reason not in _ALLOWED_STOP_REASONS:
            raise ValueError("early_stop_reason is not recognized")

        for name, epoch in (
            ("selected_checkpoint_epoch", summary.selected_checkpoint_epoch),
            ("shadow_mae_checkpoint_epoch", summary.shadow_mae_checkpoint_epoch),
        ):
            if not 1 <= epoch <= summary.epochs_run:
                raise ValueError(f"{name} must reference a recorded epoch")

        _require_finite(
            "selected_validation_core_loss", summary.selected_validation_core_loss
        )
        _require_finite("shadow_validation_mae", summary.shadow_validation_mae)

        self._production_state_dict = _clone_state_dict(summary.production_state_dict)
        self._shadow_state_dict = _clone_state_dict(summary.shadow_state_dict)
        self._summary = summary

    def trace_frame(self) -> pd.DataFrame:
        columns = [field.name for field in fields(TrainingDiagnosticEpochRecord)]
        return pd.DataFrame.from_records(
            [asdict(record) for record in self._epochs],
            columns=columns,
        )

    def summary_payload(self) -> dict[str, int | float | str]:
        summary = self._require_summary()
        payload = {
            key: value
            for key, value in asdict(summary).items()
            if key not in _STATE_FIELDS
        }
        return payload

    @property
    def production_state_dict(self) -> dict[str, torch.Tensor]:
        self._require_summary()
        assert self._production_state_dict is not None
        return _clone_state_dict(self._production_state_dict)

    @property
    def shadow_state_dict(self) -> dict[str, torch.Tensor]:
        self._require_summary()
        assert self._shadow_state_dict is not None
        return _clone_state_dict(self._shadow_state_dict)

    def _require_summary(self) -> TrainingDiagnosticSummary:
        if self._summary is None:
            raise RuntimeError("training summary not recorded")
        return self._summary


__all__ = ["Phase06DiagnosticRecorder"]
