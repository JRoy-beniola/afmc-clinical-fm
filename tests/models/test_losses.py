import torch

from afmc_fm.models.losses import gaussian_nll, masked_gaussian_nll, observation_bce


def test_masked_gaussian_nll_ignores_unobserved_targets():
    mean = torch.tensor([[0.0, 100.0]])
    log_scale = torch.zeros_like(mean)
    target = torch.tensor([[0.0, -100.0]])
    mask = torch.tensor([[1.0, 0.0]])
    loss = masked_gaussian_nll(mean, log_scale, target, mask)
    assert loss.item() < 1.0


def test_gaussian_nll_is_finite_for_extreme_log_scale():
    loss = gaussian_nll(
        torch.zeros(2),
        torch.tensor([100.0, -100.0]),
        torch.ones(2),
    )
    assert torch.isfinite(loss)


def test_observation_bce_accepts_binary_mask_targets():
    logits = torch.zeros(2, 3, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    loss = observation_bce(logits, target)
    loss.backward()
