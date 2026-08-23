import pytest
import torch

from afmc_fm.models.flow_jump import FlowJumpAdapter
from afmc_fm.phase05.model import Phase05FlowJumpAdapter


def _model(
    *,
    flow_mode: str = "time_scaled",
    jump_mode: str = "residual",
    uncertainty_mode: str = "joint",
) -> Phase05FlowJumpAdapter:
    return Phase05FlowJumpAdapter(
        representation_dim=16,
        value_dim=3,
        event_dim=3,
        state_dim=24,
        flow_mode=flow_mode,
        jump_mode=jump_mode,
        uncertainty_mode=uncertainty_mode,
        time_scale_days=30.0,
    )


def test_time_scaled_flow_is_exact_identity_at_zero_elapsed_time():
    model = _model(flow_mode="time_scaled")
    state = torch.randn(4, 24)

    evolved = model.flow(state, torch.zeros(4))

    torch.testing.assert_close(evolved, state, rtol=0.0, atol=0.0)


def test_no_flow_is_exact_identity_for_any_elapsed_time():
    model = _model(flow_mode="none")
    state = torch.randn(4, 24)

    evolved = model.flow(state, torch.tensor([0.0, 1.0, 30.0, 300.0]))

    torch.testing.assert_close(evolved, state, rtol=0.0, atol=0.0)


def test_gated_flow_reproduces_historical_formula_when_weights_match():
    torch.manual_seed(7)
    historical = FlowJumpAdapter(16, 3, 3, state_dim=24)
    candidate = _model(flow_mode="gated")
    candidate.flow_gate.load_state_dict(historical.flow_gate.state_dict())
    candidate.flow_candidate.load_state_dict(historical.flow_candidate.state_dict())
    state = torch.randn(5, 24)
    delta_t = torch.tensor([0.0, 0.5, 1.0, 10.0, 50.0])

    torch.testing.assert_close(
        candidate.flow(state, delta_t),
        historical.flow(state, delta_t),
    )


def test_residual_jump_is_identity_when_event_is_ineligible():
    model = _model(jump_mode="residual")
    state = torch.randn(3, 24)
    events_a = torch.zeros(3, 3)
    events_b = torch.randn(3, 3)
    ineligible = torch.zeros(3)

    a = model.jump(state, events_a, ineligible)
    b = model.jump(state, events_b, ineligible)

    torch.testing.assert_close(a, state, rtol=0.0, atol=0.0)
    torch.testing.assert_close(b, state, rtol=0.0, atol=0.0)


def test_residual_jump_can_change_eligible_state_and_backpropagate():
    model = _model(jump_mode="residual")
    state = torch.randn(3, 24, requires_grad=True)
    events = torch.randn(3, 3)
    eligible = torch.ones(3)

    jumped = model.jump(state, events, eligible)
    assert not torch.allclose(jumped, state)
    jumped.square().mean().backward()

    for layer in (model.jump_gate, model.jump_candidate):
        assert layer is not None
        gradients = [parameter.grad for parameter in layer.parameters()]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_no_jump_is_exact_identity():
    model = _model(jump_mode="none")
    state = torch.randn(2, 24)

    jumped = model.jump(state, torch.randn(2, 3), torch.ones(2))

    torch.testing.assert_close(jumped, state, rtol=0.0, atol=0.0)


def test_historical_gru_jump_remains_broadly_event_conditioned():
    torch.manual_seed(11)
    historical = FlowJumpAdapter(16, 3, 3, state_dim=24)
    model = _model(jump_mode="gru")
    assert historical.jump_cell is not None
    assert model.jump_cell is not None
    model.jump_cell.load_state_dict(historical.jump_cell.state_dict())
    state = torch.randn(2, 24)
    events = torch.randn(2, 3)

    torch.testing.assert_close(
        model.jump(state, events, torch.zeros(2)),
        historical.jump(state, events),
    )


def test_phase05_assimilation_is_identical_to_historical_grucell():
    torch.manual_seed(13)
    historical = FlowJumpAdapter(16, 3, 3, state_dim=24)
    model = _model()
    model.assimilation_cell.load_state_dict(historical.assimilation_cell.state_dict())
    state = torch.randn(4, 24)
    representation = torch.randn(4, 16)
    values = torch.randn(4, 3)
    masks = torch.randint(0, 2, (4, 3)).float()

    torch.testing.assert_close(
        model.assimilate(state, representation, values, masks),
        historical.assimilate(state, representation, values, masks),
    )


def test_forward_keeps_opportunity_only_step_at_pre_event_state():
    model = _model()
    output = model(
        representations=torch.randn(1, 2, 16),
        values=torch.randn(1, 2, 3),
        masks=torch.ones(1, 2, 3),
        event_features=torch.randn(1, 2, 3),
        times=torch.tensor([[0.0, 7.0]]),
        update_mask=torch.tensor([[0.0, 1.0]]),
        jump_eligible_mask=torch.tensor([[0.0, 1.0]]),
    )

    torch.testing.assert_close(
        output.post_event_states[:, 0],
        output.pre_event_states[:, 0],
    )
    assert output.post_event_states.shape == (1, 2, 24)
    assert output.value_mean.shape == (1, 2, 3)
    assert output.event_logits.shape == (1, 2)


@pytest.mark.parametrize("uncertainty_mode", ["joint", "decoupled", "deterministic"])
def test_all_uncertainty_modes_have_finite_core_outputs(uncertainty_mode):
    model = _model(uncertainty_mode=uncertainty_mode)
    output = model(
        representations=torch.randn(2, 3, 16),
        values=torch.randn(2, 3, 3),
        masks=torch.ones(2, 3, 3),
        event_features=torch.randn(2, 3, 3),
        times=torch.tensor([[0.0, 2.0, 9.0], [0.0, 5.0, 7.0]]),
        update_mask=torch.ones(2, 3),
        jump_eligible_mask=torch.ones(2, 3),
    )

    assert torch.isfinite(output.post_event_states).all()
    assert torch.isfinite(output.value_mean).all()
    assert torch.isfinite(output.event_logits).all()
    if output.value_log_scale is not None:
        assert torch.isfinite(output.value_log_scale).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_phase05_model_cuda_forward_smoke():
    device = torch.device("cuda")
    model = _model().to(device)
    output = model(
        representations=torch.randn(2, 3, 16, device=device),
        values=torch.randn(2, 3, 3, device=device),
        masks=torch.ones(2, 3, 3, device=device),
        event_features=torch.randn(2, 3, 3, device=device),
        times=torch.tensor([[0.0, 2.0, 9.0], [0.0, 5.0, 7.0]], device=device),
        update_mask=torch.ones(2, 3, device=device),
        jump_eligible_mask=torch.ones(2, 3, device=device),
    )

    assert output.post_event_states.device.type == "cuda"
    assert torch.isfinite(output.value_mean).all()
