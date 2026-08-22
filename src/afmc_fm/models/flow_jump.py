from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FlowJumpOutput:
    pre_event_states: torch.Tensor
    post_event_states: torch.Tensor
    value_mean: torch.Tensor
    value_log_scale: torch.Tensor
    event_logits: torch.Tensor
    observation_logits: torch.Tensor | None


class FlowJumpAdapter(nn.Module):
    def __init__(
        self,
        representation_dim: int,
        value_dim: int,
        event_dim: int,
        state_dim: int = 24,
        site_dim: int = 0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.site_dim = site_dim
        flow_dim = state_dim + 1
        self.flow_gate = nn.Linear(flow_dim, state_dim)
        self.flow_candidate = nn.Linear(flow_dim, state_dim)
        jump_dim = representation_dim + 2 * value_dim + event_dim
        self.jump_cell = nn.GRUCell(jump_dim, state_dim)
        self.value_head = nn.Linear(state_dim, 2 * value_dim)
        self.event_head = nn.Linear(state_dim, 1)

    def flow(self, state: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        flow_input = torch.cat([state, torch.log1p(delta_t.clamp_min(0)).unsqueeze(-1)], dim=-1)
        gate = torch.sigmoid(self.flow_gate(flow_input))
        candidate = torch.tanh(self.flow_candidate(flow_input))
        return state + gate * candidate

    def jump(
        self,
        state: torch.Tensor,
        representation: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        event_features: torch.Tensor,
    ) -> torch.Tensor:
        jump_input = torch.cat(
            [representation, values * masks, masks, event_features], dim=-1
        )
        return self.jump_cell(jump_input, state)

    def forward(
        self,
        representations: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        event_features: torch.Tensor,
        times: torch.Tensor,
        site_context: torch.Tensor | None = None,
    ) -> FlowJumpOutput:
        del site_context
        batch, steps, _ = representations.shape
        state = representations.new_zeros((batch, self.state_dim))
        pre_states = []
        post_states = []
        means = []
        log_scales = []
        event_logits = []
        for index in range(steps):
            delta_t = times[:, index] if index == 0 else times[:, index] - times[:, index - 1]
            pre_state = self.flow(state, delta_t)
            state = self.jump(
                pre_state,
                representations[:, index],
                values[:, index],
                masks[:, index],
                event_features[:, index],
            )
            value_output = self.value_head(state)
            mean, log_scale = value_output.chunk(2, dim=-1)
            pre_states.append(pre_state)
            post_states.append(state)
            means.append(mean)
            log_scales.append(log_scale)
            event_logits.append(self.event_head(state).squeeze(-1))
        return FlowJumpOutput(
            pre_event_states=torch.stack(pre_states, dim=1),
            post_event_states=torch.stack(post_states, dim=1),
            value_mean=torch.stack(means, dim=1),
            value_log_scale=torch.stack(log_scales, dim=1),
            event_logits=torch.stack(event_logits, dim=1),
            observation_logits=None,
        )
