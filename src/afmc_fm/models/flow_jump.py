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
        model_observation_process: bool = False,
        no_representation: bool = False,
        no_flow: bool = False,
        no_jump: bool = False,
        no_probabilistic_scale: bool = False,
        no_observation_head: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.model_observation_process = (
            model_observation_process and not no_observation_head
        )
        self.no_representation = no_representation
        self.no_flow = no_flow
        self.no_jump = no_jump
        self.no_probabilistic_scale = no_probabilistic_scale
        flow_dim = state_dim + 1
        self.flow_gate = nn.Linear(flow_dim, state_dim)
        self.flow_candidate = nn.Linear(flow_dim, state_dim)
        assimilation_dim = (
            (0 if no_representation else representation_dim) + 2 * value_dim
        )
        self.assimilation_cell = nn.GRUCell(assimilation_dim, state_dim)
        self.jump_cell = None if no_jump else nn.GRUCell(event_dim, state_dim)
        self.value_head = nn.Linear(state_dim, 2 * value_dim)
        self.event_head = nn.Linear(state_dim, 1)
        self.observation_head: nn.Module | None = None
        if self.model_observation_process:
            self.observation_head = nn.Sequential(
                nn.Linear(state_dim, state_dim),
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

    def assimilate(
        self,
        state: torch.Tensor,
        representation: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        assimilation_parts = [values * masks, masks]
        if not self.no_representation:
            assimilation_parts.insert(0, representation)
        assimilation_input = torch.cat(assimilation_parts, dim=-1)
        return self.assimilation_cell(assimilation_input, state)

    def jump(
        self,
        state: torch.Tensor,
        event_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.jump_cell is None:
            return state
        return self.jump_cell(event_features, state)

    def predict_observation(
        self,
        pre_event_state: torch.Tensor,
    ) -> torch.Tensor:
        if self.observation_head is None:
            raise RuntimeError("observation-process modelling is disabled")
        return self.observation_head(pre_event_state)

    def forward(
        self,
        representations: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        event_features: torch.Tensor,
        times: torch.Tensor,
        update_mask: torch.Tensor | None = None,
    ) -> FlowJumpOutput:
        batch, steps, _ = representations.shape
        if update_mask is None:
            update_mask = (
                (masks.abs().sum(dim=-1) > 0)
                | (event_features.abs().sum(dim=-1) > 0)
            ).to(representations.dtype)
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
                observation_logits.append(self.predict_observation(pre_state))
            assimilated_state = self.assimilate(
                pre_state,
                representations[:, index],
                values[:, index],
                masks[:, index],
            )
            updated_state = self.jump(
                assimilated_state,
                event_features[:, index],
            )
            state = torch.where(
                update_mask[:, index].bool().unsqueeze(-1),
                updated_state,
                pre_state,
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
