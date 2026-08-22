import torch

from afmc_fm.models.flow_jump import FlowJumpAdapter
from afmc_fm.models.losses import event_bce, masked_gaussian_nll


def _model():
    return FlowJumpAdapter(
        representation_dim=16,
        value_dim=3,
        event_dim=3,
        state_dim=24,
        site_dim=2,
    )


def test_elapsed_time_changes_pre_event_state():
    model = _model()
    z = torch.zeros(2, 24)
    short = model.flow(z, torch.tensor([1.0, 1.0]))
    long = model.flow(z, torch.tensor([30.0, 30.0]))
    assert not torch.allclose(short, long)


def test_event_jump_changes_state():
    model = _model()
    z = torch.zeros(2, 24)
    rep = torch.randn(2, 16)
    values = torch.zeros(2, 3)
    masks = torch.zeros(2, 3)
    event_a = torch.zeros(2, 3)
    event_b = event_a.clone()
    event_b[:, 1] = 1.0
    a = model.jump(z, rep, values, masks, event_a)
    b = model.jump(z, rep, values, masks, event_b)
    assert not torch.allclose(a, b)


def test_forward_returns_pre_and_post_event_states():
    model = _model()
    batch, steps = 3, 5
    out = model(
        representations=torch.randn(batch, steps, 16),
        values=torch.randn(batch, steps, 3),
        masks=torch.ones(batch, steps, 3),
        event_features=torch.zeros(batch, steps, 3),
        times=torch.arange(steps).float().repeat(batch, 1),
        site_context=torch.zeros(batch, steps, 2),
    )
    assert out.pre_event_states.shape == (batch, steps, 24)
    assert out.post_event_states.shape == (batch, steps, 24)
    assert out.value_mean.shape == (batch, steps, 3)
    assert out.event_logits.shape == (batch, steps)


def test_forward_and_losses_have_finite_gradients():
    model = _model()
    batch, steps = 2, 4
    out = model(
        representations=torch.randn(batch, steps, 16),
        values=torch.randn(batch, steps, 3),
        masks=torch.ones(batch, steps, 3),
        event_features=torch.zeros(batch, steps, 3),
        times=torch.arange(steps).float().repeat(batch, 1),
        site_context=torch.zeros(batch, steps, 2),
    )
    target = torch.randn(batch, steps, 3)
    event_target = torch.zeros(batch, steps)
    loss = masked_gaussian_nll(out.value_mean, out.value_log_scale, target, torch.ones_like(target))
    loss = loss + event_bce(out.event_logits, event_target)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_observation_logits_are_computed_from_pre_event_state():
    model = FlowJumpAdapter(
        representation_dim=16,
        value_dim=3,
        event_dim=3,
        state_dim=24,
        site_dim=2,
        model_observation_process=True,
    )
    pre = torch.randn(4, 24)
    site = torch.zeros(4, 2)
    logits = model.predict_observation(pre, site)
    assert logits.shape == (4, 3)


def test_observation_forward_outputs_one_logit_per_mask_value():
    model = FlowJumpAdapter(16, 3, 3, state_dim=24, site_dim=2, model_observation_process=True)
    output = model(
        representations=torch.randn(2, 4, 16),
        values=torch.randn(2, 4, 3),
        masks=torch.ones(2, 4, 3),
        event_features=torch.zeros(2, 4, 3),
        times=torch.arange(4).float().repeat(2, 1),
        site_context=torch.zeros(2, 4, 2),
    )
    assert output.observation_logits is not None
    assert output.observation_logits.shape == (2, 4, 3)
