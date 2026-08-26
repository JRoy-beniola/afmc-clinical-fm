import torch

from afmc_fm.models.flow_jump import FlowJumpAdapter
from afmc_fm.models.losses import event_bce, masked_gaussian_nll


def _model():
    return FlowJumpAdapter(
        representation_dim=16,
        value_dim=3,
        event_dim=3,
        state_dim=24,
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
    event_a = torch.zeros(2, 3)
    event_b = event_a.clone()
    event_b[:, 1] = 1.0
    a = model.jump(z, event_a)
    b = model.jump(z, event_b)
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
        model_observation_process=True,
    )
    pre = torch.randn(4, 24)
    logits = model.predict_observation(pre)
    assert logits.shape == (4, 3)


def test_observation_forward_outputs_one_logit_per_mask_value():
    model = FlowJumpAdapter(16, 3, 3, state_dim=24, model_observation_process=True)
    output = model(
        representations=torch.randn(2, 4, 16),
        values=torch.randn(2, 4, 3),
        masks=torch.ones(2, 4, 3),
        event_features=torch.zeros(2, 4, 3),
        times=torch.arange(4).float().repeat(2, 1),
    )
    assert output.observation_logits is not None
    assert output.observation_logits.shape == (2, 4, 3)


def test_opportunity_only_step_does_not_update_state_but_clinical_step_can():
    model = FlowJumpAdapter(
        16,
        3,
        3,
        state_dim=24,
        model_observation_process=True,
    )
    values = torch.zeros(1, 2, 3)
    values[:, 1] = torch.randn(1, 3)
    masks = torch.zeros(1, 2, 3)
    masks[:, 1] = 1.0
    event_features = torch.zeros(1, 2, 3)
    event_features[:, 0, 2] = 1.0
    event_features[:, 1, 0] = 1.0
    output = model(
        representations=torch.randn(1, 2, 16),
        values=values,
        masks=masks,
        event_features=event_features,
        times=torch.tensor([[0.0, 1.0]]),
        update_mask=torch.tensor([[0.0, 1.0]]),
    )

    torch.testing.assert_close(
        output.post_event_states[:, 0],
        output.pre_event_states[:, 0],
    )
    assert not torch.allclose(
        output.post_event_states[:, 1],
        output.pre_event_states[:, 1],
    )
    assert output.observation_logits is not None
    assert output.observation_logits.shape[1] == 2


def test_no_jump_retains_assimilation_but_ignores_event_semantics():
    model = FlowJumpAdapter(16, 3, 3, no_jump=True)
    values = torch.randn(1, 2, 3)
    masks = torch.ones(1, 2, 3)
    times = torch.tensor([[0.0, 1.0]])
    update_mask = torch.ones(1, 2)
    event_a = torch.zeros(1, 2, 3)
    event_b = torch.ones(1, 2, 3)

    baseline = model(
        torch.zeros(1, 2, 16),
        values,
        masks,
        event_a,
        times,
        update_mask=update_mask,
    )
    changed_events = model(
        torch.zeros(1, 2, 16),
        values,
        masks,
        event_b,
        times,
        update_mask=update_mask,
    )
    changed_representation = model(
        torch.ones(1, 2, 16),
        values,
        masks,
        event_a,
        times,
        update_mask=update_mask,
    )

    torch.testing.assert_close(baseline.value_mean, changed_events.value_mean)
    torch.testing.assert_close(baseline.event_logits, changed_events.event_logits)
    assert not torch.allclose(baseline.value_mean, changed_representation.value_mean)


def test_ablation_flags_disable_flow_jump_and_probabilistic_scale():
    no_flow = FlowJumpAdapter(16, 3, 3, no_flow=True)
    state = torch.randn(2, 24)
    assert torch.equal(no_flow.flow(state, torch.ones(2)), state)

    no_jump = FlowJumpAdapter(16, 3, 3, no_jump=True)
    jumped = no_jump.jump(state, torch.zeros(2, 3))
    assert torch.equal(jumped, state)

    deterministic = FlowJumpAdapter(16, 3, 3, no_probabilistic_scale=True)
    output = deterministic(
        torch.randn(2, 3, 16),
        torch.randn(2, 3, 3),
        torch.ones(2, 3, 3),
        torch.zeros(2, 3, 3),
        torch.arange(3).float().repeat(2, 1),
    )
    assert torch.count_nonzero(output.value_log_scale) == 0


def test_observation_model_requires_no_site_identity_input():
    model = FlowJumpAdapter(
        representation_dim=16,
        value_dim=3,
        event_dim=3,
        state_dim=24,
        model_observation_process=True,
    )
    output = model(
        representations=torch.randn(2, 4, 16),
        values=torch.randn(2, 4, 3),
        masks=torch.ones(2, 4, 3),
        event_features=torch.zeros(2, 4, 3),
        times=torch.arange(4).float().repeat(2, 1),
    )
    assert output.observation_logits is not None
    assert output.observation_logits.shape == (2, 4, 3)


def test_no_representation_ablation_ignores_representation_values():
    model = FlowJumpAdapter(16, 3, 3, no_representation=True)
    values = torch.randn(2, 4, 3)
    masks = torch.ones(2, 4, 3)
    events = torch.zeros(2, 4, 3)
    times = torch.arange(4).float().repeat(2, 1)
    zeros = model(torch.zeros(2, 4, 16), values, masks, events, times)
    random = model(torch.randn(2, 4, 16), values, masks, events, times)
    torch.testing.assert_close(zeros.value_mean, random.value_mean)
    torch.testing.assert_close(zeros.event_logits, random.event_logits)


def test_no_observation_head_ablation_removes_observation_logits():
    model = FlowJumpAdapter(
        16,
        3,
        3,
        model_observation_process=True,
        no_observation_head=True,
    )
    output = model(
        torch.randn(2, 3, 16),
        torch.randn(2, 3, 3),
        torch.ones(2, 3, 3),
        torch.zeros(2, 3, 3),
        torch.arange(3).float().repeat(2, 1),
    )
    assert output.observation_logits is None
