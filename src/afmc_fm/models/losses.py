import math

import torch
from torch.nn import functional as F


def _gaussian_terms(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    safe_log_scale = log_scale.clamp(-7.0, 7.0)
    inverse_variance = torch.exp(-2.0 * safe_log_scale)
    return 0.5 * (target - mean).square() * inverse_variance + safe_log_scale + 0.5 * math.log(2.0 * math.pi)


def gaussian_nll(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return _gaussian_terms(mean, log_scale, target).mean()


def masked_gaussian_nll(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    terms = _gaussian_terms(mean, log_scale, target)
    observed = mask.to(dtype=terms.dtype)
    return (terms * observed).sum() / observed.sum().clamp_min(1.0)


def event_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target.float())


def observation_bce(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    terms = F.binary_cross_entropy_with_logits(
        logits, target_mask.float(), reduction="none"
    )
    weights = valid.to(dtype=terms.dtype)
    if weights.ndim == terms.ndim - 1:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(terms)
    return (terms * weights).sum() / weights.sum().clamp_min(1.0)
