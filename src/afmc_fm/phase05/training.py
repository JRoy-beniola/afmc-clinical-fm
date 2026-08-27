from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch.nn import functional as F

from afmc_fm.execution.device import move_batch
from afmc_fm.metrics.forecasting import (
    binary_metrics,
    gaussian_forecasting_metrics,
    regression_metrics,
)
from afmc_fm.metrics.latent import aligned_latent_r2
from afmc_fm.models.losses import masked_gaussian_nll
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.losses import masked_half_mse, masked_residual_gaussian_nll
from afmc_fm.phase05.model import Phase05FlowJumpAdapter


@dataclass(frozen=True, slots=True)
class TrainingDiagnosticEpochRecord:
    epoch: int
    train_core_loss: float
    validation_core_loss: float
    validation_mae: float
    validation_rmse: float
    gradient_l2_norm: float
    parameter_l2_norm: float
    mean_flow_displacement: float
    median_flow_displacement: float
    p95_flow_displacement: float
    best_validation_core_loss_so_far: float
    best_core_epoch: int
    shadow_best_validation_mae_so_far: float
    shadow_best_mae_epoch: int
    stale_epochs: int


@dataclass(frozen=True, slots=True)
class TrainingDiagnosticSummary:
    epochs_run: int
    stop_epoch: int
    selected_checkpoint_epoch: int
    selected_validation_core_loss: float
    shadow_mae_checkpoint_epoch: int
    shadow_validation_mae: float
    early_stop_reason: str
    production_state_dict: dict[str, torch.Tensor]
    shadow_state_dict: dict[str, torch.Tensor]


class TrainingDiagnosticObserver(Protocol):
    def on_epoch(self, record: TrainingDiagnosticEpochRecord) -> None: ...

    def on_training_end(self, summary: TrainingDiagnosticSummary) -> None: ...


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


def _global_gradient_l2(parameters: list[torch.nn.Parameter]) -> float:
    squared = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().pow(2).sum()
    return float(torch.sqrt(squared).item())


def _global_parameter_l2(parameters: list[torch.nn.Parameter]) -> float:
    squared = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        squared = squared + parameter.detach().pow(2).sum()
    return float(torch.sqrt(squared).item())


def _validation_point_metrics(
    output,
    batch: dict[str, torch.Tensor],
) -> tuple[float, float]:
    selected = batch["target_masks"].bool()
    if not torch.any(selected):
        raise ValueError("diagnostic validation requires observed target values")
    error = output.value_mean[selected] - batch["target_values"][selected]
    mae = error.abs().mean()
    rmse = torch.sqrt(error.pow(2).mean())
    return float(mae.item()), float(rmse.item())


def _flow_displacement(
    output,
    batch: dict[str, torch.Tensor],
) -> tuple[float, float, float]:
    valid = batch["valid"].bool()
    if valid.shape != output.pre_event_states.shape[:2]:
        raise ValueError("diagnostic valid mask shape does not match model output")
    previous = torch.cat(
        [
            torch.zeros_like(output.post_event_states[:, :1]),
            output.post_event_states[:, :-1],
        ],
        dim=1,
    )
    displacement = torch.linalg.vector_norm(
        output.pre_event_states - previous,
        dim=-1,
    )[valid]
    if displacement.numel() == 0:
        raise ValueError("diagnostic validation requires valid sequence positions")
    return (
        float(displacement.mean().item()),
        float(displacement.median().item()),
        float(torch.quantile(displacement, 0.95).item()),
    )


def _require_finite_diagnostic(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite diagnostic value: {name}")


def _cpu_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state_dict.items()
    }


def _emit_epoch(
    diagnostics: TrainingDiagnosticObserver,
    record: TrainingDiagnosticEpochRecord,
) -> None:
    try:
        diagnostics.on_epoch(record)
    except Exception as error:
        raise RuntimeError("training diagnostic observer failed during on_epoch") from error


def _emit_summary(
    diagnostics: TrainingDiagnosticObserver,
    summary: TrainingDiagnosticSummary,
) -> None:
    try:
        diagnostics.on_training_end(summary)
    except Exception as error:
        raise RuntimeError(
            "training diagnostic observer failed during on_training_end"
        ) from error


def _fit_core(
    model: Phase05FlowJumpAdapter,
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    config: Phase05Config,
    device: torch.device,
    diagnostics: TrainingDiagnosticObserver | None = None,
) -> Phase05FlowJumpAdapter:
    if diagnostics is not None:
        if "valid" not in train:
            raise ValueError("diagnostic training requires train batch['valid']")
        if "valid" not in validation:
            raise ValueError("diagnostic training requires validation batch['valid']")

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
    best_epoch = 0
    shadow_best_mae = float("inf")
    shadow_best_epoch = 0
    shadow_best_state = deepcopy(model.state_dict()) if diagnostics is not None else None
    stale_epochs = 0
    epochs_run = 0
    early_stop_reason = "max_epochs_reached"

    for epoch_index in range(config.max_epochs):
        epoch = epoch_index + 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _core_loss(model, train, config)
        train_loss = float(loss.detach().item()) if diagnostics is not None else 0.0
        loss.backward()
        gradient_norm = (
            _global_gradient_l2(parameters) if diagnostics is not None else 0.0
        )
        optimizer.step()
        parameter_norm = (
            _global_parameter_l2(parameters) if diagnostics is not None else 0.0
        )

        model.eval()
        validation_mae = 0.0
        validation_rmse = 0.0
        flow_mean = 0.0
        flow_median = 0.0
        flow_p95 = 0.0
        with torch.no_grad():
            validation_loss = float(_core_loss(model, validation, config).item())
            if diagnostics is not None:
                diagnostic_output = _forward(model, validation)
                validation_mae, validation_rmse = _validation_point_metrics(
                    diagnostic_output,
                    validation,
                )
                flow_mean, flow_median, flow_p95 = _flow_displacement(
                    diagnostic_output,
                    validation,
                )

        should_stop = False
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            should_stop = stale_epochs >= config.patience

        if diagnostics is not None:
            for name, value in (
                ("train_core_loss", train_loss),
                ("validation_core_loss", validation_loss),
                ("validation_mae", validation_mae),
                ("validation_rmse", validation_rmse),
                ("gradient_l2_norm", gradient_norm),
                ("parameter_l2_norm", parameter_norm),
                ("mean_flow_displacement", flow_mean),
                ("median_flow_displacement", flow_median),
                ("p95_flow_displacement", flow_p95),
                ("best_validation_core_loss_so_far", best_loss),
            ):
                _require_finite_diagnostic(name, value)

            if validation_mae < shadow_best_mae:
                shadow_best_mae = validation_mae
                shadow_best_epoch = epoch
                shadow_best_state = deepcopy(model.state_dict())

            _emit_epoch(
                diagnostics,
                TrainingDiagnosticEpochRecord(
                    epoch=epoch,
                    train_core_loss=train_loss,
                    validation_core_loss=validation_loss,
                    validation_mae=validation_mae,
                    validation_rmse=validation_rmse,
                    gradient_l2_norm=gradient_norm,
                    parameter_l2_norm=parameter_norm,
                    mean_flow_displacement=flow_mean,
                    median_flow_displacement=flow_median,
                    p95_flow_displacement=flow_p95,
                    best_validation_core_loss_so_far=best_loss,
                    best_core_epoch=best_epoch,
                    shadow_best_validation_mae_so_far=shadow_best_mae,
                    shadow_best_mae_epoch=shadow_best_epoch,
                    stale_epochs=stale_epochs,
                ),
            )

        epochs_run = epoch
        if should_stop:
            early_stop_reason = "patience_exhausted"
            break

    model.load_state_dict(best_state)

    if diagnostics is not None:
        if shadow_best_state is None or best_epoch == 0 or shadow_best_epoch == 0:
            raise RuntimeError("diagnostic checkpoint selection did not initialize")
        _emit_summary(
            diagnostics,
            TrainingDiagnosticSummary(
                epochs_run=epochs_run,
                stop_epoch=epochs_run,
                selected_checkpoint_epoch=best_epoch,
                selected_validation_core_loss=best_loss,
                shadow_mae_checkpoint_epoch=shadow_best_epoch,
                shadow_validation_mae=shadow_best_mae,
                early_stop_reason=early_stop_reason,
                production_state_dict=_cpu_state_dict(best_state),
                shadow_state_dict=_cpu_state_dict(shadow_best_state),
            ),
        )
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
    requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
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
    diagnostics: TrainingDiagnosticObserver | None = None,
) -> Phase05FlowJumpAdapter:
    _fit_core(model, train, validation, config, device, diagnostics)
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

    if "latent_targets" in batch and "latent_valid" in batch:
        latent_selected = batch["latent_valid"].bool()
        metrics["latent_aligned_r2"] = aligned_latent_r2(
            batch["latent_targets"][latent_selected].detach().cpu().numpy(),
            output.post_event_states[latent_selected].detach().cpu().numpy(),
        )

    return {name: float(np.asarray(value)) for name, value in metrics.items()}


__all__ = [
    "TrainingDiagnosticEpochRecord",
    "TrainingDiagnosticObserver",
    "TrainingDiagnosticSummary",
    "evaluate_phase05_model",
    "fit_decoupled_scale_head",
    "fit_phase05_model",
]
