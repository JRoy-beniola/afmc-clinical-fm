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
        model_observation_process: bool = False,
        no_flow: bool = False,
        no_jump: bool = False,
        no_probabilistic_scale: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.site_dim = site_dim
        self.model_observation_process = model_observation_process
        self.no_flow = no_flow
        self.no_jump = no_jump
        self.no_probabilistic_scale = no_probabilistic_scale
        flow_dim = state_dim + 1
        self.flow_gate = nn.Linear(flow_dim, state_dim)
        self.flow_candidate = nn.Linear(flow_dim, state_dim)
        jump_dim = representation_dim + 2 * value_dim + event_dim
        self.jump_cell = nn.GRUCell(jump_dim, state_dim)
        self.value_head = nn.Linear(state_dim, 2 * value_dim)
        self.event_head = nn.Linear(state_dim, 1)
        self.observation_head: nn.Module | None = None
        if model_observation_process:
            self.observation_head = nn.Sequential(
                nn.Linear(state_dim + site_dim, state_dim),
                nn.Tanh(),
                nn.Linear(state_dim, value_dim),
            )

    def flow(self, state: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        if self.no_flow:
            return state
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
        if self.no_jump:
            return state
        jump_input = torch.cat(
            [representation, values * masks, masks, event_features], dim=-1
        )
        return self.jump_cell(jump_input, state)

    def predict_observation(
        self,
        pre_event_state: torch.Tensor,
        site_context: torch.Tensor,
    ) -> torch.Tensor:
        if self.observation_head is None:
            raise RuntimeError("observation-process modelling is disabled")
        features = torch.cat([pre_event_state, site_context], dim=-1)
        return self.observation_head(features)

    def forward(
        self,
        representations: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        event_features: torch.Tensor,
        times: torch.Tensor,
        site_context: torch.Tensor | None = None,
    ) -> FlowJumpOutput:
        if self.model_observation_process and site_context is None:
            raise ValueError("site_context is required for observation-process modelling")
        batch, steps, _ = representations.shape
        state = representations.new_zeros((batch, self.state_dim))
        pre_states = []
        post_states = []
        means = []
        log_scales = []
        event_logits = []
        observation_logits = []
        for index in range(steps):
            delta_t = times[:, index] if index == 0 else times[:, index] - times[:, index - 1]
            pre_state = self.flow(state, delta_t)
            if self.model_observation_process:
                assert site_context is not None
                observation_logits.append(
                    self.predict_observation(pre_state, site_context[:, index])
                )
            state = self.jump(
                pre_state,
                representations[:, index],
                values[:, index],
                masks[:, index],
                event_features[:, index],
            )
            value_output = self.value_head(state)
            mean, log_scale = value_output.chunk(2, dim=-1)
            if self.no_probabilistic_scale:
                log_scale = torch.zeros_like(log_scale)
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
            observation_logits=(
                torch.stack(observation_logits, dim=1)
                if observation_logits
                else None
            ),
        )
