from copy import deepcopy

import torch

from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.model import Phase05FlowJumpAdapter
from afmc_fm.phase05.training import evaluate_phase05_model, fit_phase05_model


class _CollectingObserver:
    def __init__(self) -> None:
        self.epochs = []
        self.summary = None

    def on_epoch(self, record) -> None:
        self.epochs.append(record)

    def on_training_end(self, summary) -> None:
        self.summary = summary


def _model() -> Phase05FlowJumpAdapter:
    return Phase05FlowJumpAdapter(
        representation_dim=16,
        value_dim=3,
        event_dim=3,
        state_dim=24,
        flow_mode="time_scaled",
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
        max_epochs=3,
        patience=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )


def _parameter_bytes(model: torch.nn.Module) -> dict[str, bytes]:
    return {
        name: parameter.detach().cpu().numpy().tobytes()
        for name, parameter in model.named_parameters()
    }


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
