from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from afmc_fm.execution.device import move_batch
from afmc_fm.metrics.forecasting import (
    binary_metrics,
    gaussian_forecasting_metrics,
    regression_metrics,
)
from afmc_fm.models.losses import masked_gaussian_nll
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.losses import masked_half_mse, masked_residual_gaussian_nll
from afmc_fm.phase05.model import Phase05FlowJumpAdapter


def _forward(
    model: Phase05FlowJumpAdapter,
    batch: dict[str, torch.Tensor],
):
    return model(
        representations=batch["representations"],
        values=batch["values"],
        masks=batch["masks"],
        event_features=batch["event_features"],
        times=batch["times"],
        update_mask=batch["update_mask"],
        jump_eligible_mask=batch["jump_eligible_mask"],
    )


def _event_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    terms = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    weights = valid.to(dtype=terms.dtype)
    return (terms * weights).sum() / weights.sum().clamp_min(1.0)


def _core_loss(
    model: Phase05FlowJumpAdapter,
    batch: dict[str, torch.Tensor],
    config: Phase05Config,
) -> torch.Tensor:
    output = _forward(model, batch)
    if model.uncertainty_mode == "joint":
        assert output.value_log_scale is not None
        value_loss = masked_gaussian_nll(
            output.value_mean,
            output.value_log_scale,
            batch["target_values"],
            batch["target_masks"],
        )
    else:
        value_loss = masked_half_mse(
            output.value_mean,
            batch["target_values"],
            batch["target_masks"],
        )
    return value_loss + config.lambda_event * _event_loss(
        output.event_logits,
        batch["target_events"],
        batch["event_valid"],
    )


def _fit_core(
    model: Phase05FlowJumpAdapter,
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    config: Phase05Config,
    device: torch.device,
) -> Phase05FlowJumpAdapter:
    model.to(device)
    train = move_batch(train, device)
    validation = move_batch(validation, device)
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("scale_head.")
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0
    for _ in range(config.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _core_loss(model, train, config)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(_core_loss(model, validation, config).item())
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    model.load_state_dict(best_state)
    return model


def fit_decoupled_scale_head(
    model: Phase05FlowJumpAdapter,
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    config: Phase05Config,
    device: torch.device,
) -> Phase05FlowJumpAdapter:
    if model.uncertainty_mode != "decoupled" or model.scale_head is None:
        raise ValueError("scale-head fitting requires uncertainty_mode='decoupled'")

    model.to(device)
    train = move_batch(train, device)
    validation = move_batch(validation, device)
    requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("scale_head."))

    optimizer = torch.optim.AdamW(
        model.scale_head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_scale_state = deepcopy(model.scale_head.state_dict())
    best_loss = float("inf")
    stale_epochs = 0
    try:
        for _ in range(config.max_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, train)
            assert output.value_log_scale is not None
            residual = (train["target_values"] - output.value_mean).detach()
            loss = masked_residual_gaussian_nll(
                output.value_log_scale,
                residual,
                train["target_masks"],
            )
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_output = _forward(model, validation)
                assert val_output.value_log_scale is not None
                val_residual = (
                    validation["target_values"] - val_output.value_mean
                ).detach()
                validation_loss = float(
                    masked_residual_gaussian_nll(
                        val_output.value_log_scale,
                        val_residual,
                        validation["target_masks"],
                    ).item()
                )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_scale_state = deepcopy(model.scale_head.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    break
        model.scale_head.load_state_dict(best_scale_state)
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(requires_grad[name])
    return model


def fit_phase05_model(
    model: Phase05FlowJumpAdapter,
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    config: Phase05Config,
    device: torch.device,
) -> Phase05FlowJumpAdapter:
    _fit_core(model, train, validation, config, device)
    if model.uncertainty_mode == "decoupled":
        fit_decoupled_scale_head(model, train, validation, config, device)
    return model


def evaluate_phase05_model(
    model: Phase05FlowJumpAdapter,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    batch = move_batch(batch, device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = _forward(model, batch)

    selected = batch["target_masks"].bool()
    truth = batch["target_values"][selected].detach().cpu().numpy()
    mean = output.value_mean[selected].detach().cpu().numpy()
    metrics = regression_metrics(truth, mean)

    if output.value_log_scale is not None:
        log_scale = output.value_log_scale[selected].detach().cpu().numpy()
        metrics.update(gaussian_forecasting_metrics(truth, mean, log_scale))

    event_selected = batch["event_valid"].bool()
    event_truth = batch["target_events"][event_selected].detach().cpu().numpy()
    event_probability = (
        torch.sigmoid(output.event_logits[event_selected]).detach().cpu().numpy()
    )
    event_metrics = binary_metrics(event_truth, event_probability)
    metrics.update({f"event_{name}": value for name, value in event_metrics.items()})

    return {name: float(np.asarray(value)) for name, value in metrics.items()}


__all__ = [
    "evaluate_phase05_model",
    "fit_decoupled_scale_head",
    "fit_phase05_model",
]
