from copy import deepcopy

import pytest
import torch

from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.losses import masked_half_mse, masked_residual_gaussian_nll
from afmc_fm.phase05.model import Phase05FlowJumpAdapter
from afmc_fm.phase05.training import (
    evaluate_phase05_model,
    fit_decoupled_scale_head,
    fit_phase05_model,
)


def _model(uncertainty_mode: str = "decoupled") -> Phase05FlowJumpAdapter:
    return Phase05FlowJumpAdapter(
        representation_dim=16,
        value_dim=3,
        event_dim=3,
        state_dim=24,
        flow_mode="time_scaled",
        jump_mode="residual",
        uncertainty_mode=uncertainty_mode,
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
        "jump_eligible_mask": torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ),
        "target_values": torch.randn(batch, steps, value_dim),
        "target_masks": torch.tensor(
            [
                [[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
                [[1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1]],
            ],
            dtype=torch.float32,
        ),
        "target_events": torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ),
        "event_valid": torch.ones(batch, steps),
    }


def _tiny_config() -> Phase05Config:
    return Phase05Config(
        max_epochs=3,
        patience=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )


def _parameter_bytes(
    model: torch.nn.Module,
    *,
    exclude_prefix: str | None = None,
) -> dict[str, bytes]:
    snapshots: dict[str, bytes] = {}
    for name, parameter in model.named_parameters():
        if exclude_prefix is not None and name.startswith(exclude_prefix):
            continue
        snapshots[name] = parameter.detach().cpu().numpy().tobytes()
    return snapshots


def test_masked_half_mse_matches_manual_observed_mean():
    mean = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[0.0, 4.0], [5.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    loss = masked_half_mse(mean, target, mask)

    expected = 0.5 * torch.tensor([1.0, 4.0, 9.0]).mean()
    torch.testing.assert_close(loss, expected)


def test_decoupled_scale_loss_has_no_gradient_path_into_core_or_mean():
    model = _model("decoupled")
    batch = _batch()
    output = model(
        representations=batch["representations"],
        values=batch["values"],
        masks=batch["masks"],
        event_features=batch["event_features"],
        times=batch["times"],
        update_mask=batch["update_mask"],
        jump_eligible_mask=batch["jump_eligible_mask"],
    )
    assert output.value_log_scale is not None
    residual = (batch["target_values"] - output.value_mean).detach()

    model.zero_grad(set_to_none=True)
    loss = masked_residual_gaussian_nll(
        output.value_log_scale,
        residual,
        batch["target_masks"],
    )
    loss.backward()

    scale_gradients = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if name.startswith("scale_head."):
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            scale_gradients.append(gradient)
        else:
            assert gradient is None or torch.count_nonzero(gradient) == 0
    assert scale_gradients
    assert any(torch.count_nonzero(gradient) > 0 for gradient in scale_gradients)


def test_scale_phase_leaves_every_non_scale_parameter_byte_identical():
    model = _model("decoupled")
    train = _batch()
    validation = deepcopy(train)
    before = _parameter_bytes(model, exclude_prefix="scale_head.")

    fit_decoupled_scale_head(
        model,
        train,
        validation,
        _tiny_config(),
        torch.device("cpu"),
    )

    after = _parameter_bytes(model, exclude_prefix="scale_head.")
    assert after == before


def test_full_decoupled_fit_updates_parameters_and_reports_probabilistic_metrics():
    model = _model("decoupled")
    train = _batch()
    validation = deepcopy(train)
    before = _parameter_bytes(model)

    fitted = fit_phase05_model(
        model,
        train,
        validation,
        _tiny_config(),
        torch.device("cpu"),
    )

    assert fitted is model
    assert _parameter_bytes(model) != before
    metrics = evaluate_phase05_model(model, validation, torch.device("cpu"))
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "nll" in metrics
    assert "coverage_90" in metrics


@pytest.mark.parametrize("mode", ["joint", "deterministic"])
def test_non_decoupled_modes_fit_and_report_expected_metric_contract(mode):
    model = _model(mode)
    train = _batch()
    validation = deepcopy(train)

    fit_phase05_model(
        model,
        train,
        validation,
        _tiny_config(),
        torch.device("cpu"),
    )
    metrics = evaluate_phase05_model(model, validation, torch.device("cpu"))

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "event_brier" in metrics
    assert "event_log_loss" in metrics
    assert "event_roc_auc" in metrics
    if mode == "joint":
        assert "nll" in metrics
        assert "coverage_90" in metrics
    else:
        assert "nll" not in metrics
        assert "coverage_90" not in metrics
