import math

import torch


def masked_half_mse(
    mean: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    terms = 0.5 * (target - mean).square()
    observed = mask.to(dtype=terms.dtype)
    return (terms * observed).sum() / observed.sum().clamp_min(1.0)


def masked_residual_gaussian_nll(
    log_scale: torch.Tensor,
    residual: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    detached_residual = residual.detach()
    safe_log_scale = log_scale.clamp(-7.0, 7.0)
    inverse_variance = torch.exp(-2.0 * safe_log_scale)
    terms = (
        0.5 * detached_residual.square() * inverse_variance
        + safe_log_scale
        + 0.5 * math.log(2.0 * math.pi)
    )
    observed = mask.to(dtype=terms.dtype)
    return (terms * observed).sum() / observed.sum().clamp_min(1.0)


__all__ = ["masked_half_mse", "masked_residual_gaussian_nll"]
