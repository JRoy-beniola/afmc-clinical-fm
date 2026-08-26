from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

FlowMode = Literal["gated", "time_scaled", "none"]
JumpMode = Literal["gru", "residual", "none"]
UncertaintyMode = Literal["joint", "decoupled", "deterministic"]


@dataclass(frozen=True)
class Phase05FlowJumpOutput:
    pre_event_states: torch.Tensor
    post_event_states: torch.Tensor
    value_mean: torch.Tensor
    value_log_scale: torch.Tensor | None
    event_logits: torch.Tensor


class Phase05FlowJumpAdapter(nn.Module):
    def __init__(
        self,
        representation_dim: int,
        value_dim: int,
        event_dim: int,
        *,
        state_dim: int = 24,
        flow_mode: FlowMode = "time_scaled",
        jump_mode: JumpMode = "residual",
        uncertainty_mode: UncertaintyMode = "deterministic",
        time_scale_days: float = 30.0,
    ) -> None:
        super().__init__()
        if representation_dim <= 0 or value_dim <= 0 or event_dim <= 0:
            raise ValueError("representation, value, and event dimensions must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if flow_mode not in {"gated", "time_scaled", "none"}:
            raise ValueError("unknown flow_mode")
        if jump_mode not in {"gru", "residual", "none"}:
            raise ValueError("unknown jump_mode")
        if uncertainty_mode not in {"joint", "decoupled", "deterministic"}:
            raise ValueError("unknown uncertainty_mode")
        if time_scale_days <= 0:
            raise ValueError("time_scale_days must be positive")

        self.state_dim = state_dim
        self.flow_mode = flow_mode
        self.jump_mode = jump_mode
        self.uncertainty_mode = uncertainty_mode
        self.time_scale_days = float(time_scale_days)

        self.flow_gate: nn.Linear | None = None
        self.flow_candidate: nn.Linear | None = None
        self.time_flow: nn.Linear | None = None
        if flow_mode == "gated":
            flow_dim = state_dim + 1
            self.flow_gate = nn.Linear(flow_dim, state_dim)
            self.flow_candidate = nn.Linear(flow_dim, state_dim)
        elif flow_mode == "time_scaled":
            self.time_flow = nn.Linear(state_dim, state_dim)

        assimilation_dim = representation_dim + 2 * value_dim
        self.assimilation_cell = nn.GRUCell(assimilation_dim, state_dim)

        self.jump_cell: nn.GRUCell | None = None
        self.jump_gate: nn.Linear | None = None
        self.jump_candidate: nn.Linear | None = None
        if jump_mode == "gru":
            self.jump_cell = nn.GRUCell(event_dim, state_dim)
        elif jump_mode == "residual":
            jump_input_dim = state_dim + event_dim
            self.jump_gate = nn.Linear(jump_input_dim, state_dim)
            self.jump_candidate = nn.Linear(jump_input_dim, state_dim)

        self.joint_value_head: nn.Linear | None = None
        self.mean_head: nn.Linear | None = None
        self.scale_head: nn.Linear | None = None
        if uncertainty_mode == "joint":
            self.joint_value_head = nn.Linear(state_dim, 2 * value_dim)
        else:
            self.mean_head = nn.Linear(state_dim, value_dim)
            if uncertainty_mode == "decoupled":
                self.scale_head = nn.Linear(state_dim, value_dim)
        self.event_head = nn.Linear(state_dim, 1)

    def flow(self, state: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        nonnegative_delta = delta_t.clamp_min(0.0)
        if self.flow_mode == "none":
            return state
        if self.flow_mode == "gated":
            assert self.flow_gate is not None
            assert self.flow_candidate is not None
            flow_input = torch.cat(
                [state, torch.log1p(nonnegative_delta).unsqueeze(-1)],
                dim=-1,
            )
            gate = torch.sigmoid(self.flow_gate(flow_input))
            candidate = torch.tanh(self.flow_candidate(flow_input))
            return state + gate * candidate

        assert self.time_flow is not None
        scale = nonnegative_delta / (self.time_scale_days + nonnegative_delta)
        candidate = torch.tanh(self.time_flow(state))
        return state + scale.unsqueeze(-1) * candidate

    def assimilate(
        self,
        state: torch.Tensor,
        representation: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        assimilation_input = torch.cat(
            [representation, values * masks, masks],
            dim=-1,
        )
        return self.assimilation_cell(assimilation_input, state)

    def jump(
        self,
        state: torch.Tensor,
        event_features: torch.Tensor,
        jump_eligible: torch.Tensor,
    ) -> torch.Tensor:
        if self.jump_mode == "none":
            return state
        if self.jump_mode == "gru":
            assert self.jump_cell is not None
            return self.jump_cell(event_features, state)

        assert self.jump_gate is not None
        assert self.jump_candidate is not None
        features = torch.cat([state, event_features], dim=-1)
        gate = torch.sigmoid(self.jump_gate(features))
        candidate = torch.tanh(self.jump_candidate(features))
        return state + jump_eligible.to(state.dtype).unsqueeze(-1) * gate * candidate

    def _forecast(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.uncertainty_mode == "joint":
            assert self.joint_value_head is not None
            mean, log_scale = self.joint_value_head(state).chunk(2, dim=-1)
            return mean, log_scale

        assert self.mean_head is not None
        mean = self.mean_head(state)
        if self.uncertainty_mode == "deterministic":
            return mean, None

        assert self.scale_head is not None
        log_scale = self.scale_head(state.detach())
        return mean, log_scale

    def forward(
        self,
        representations: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        event_features: torch.Tensor,
        times: torch.Tensor,
        update_mask: torch.Tensor,
        jump_eligible_mask: torch.Tensor,
    ) -> Phase05FlowJumpOutput:
        batch, steps, _ = representations.shape
        state = representations.new_zeros((batch, self.state_dim))
        pre_states: list[torch.Tensor] = []
        post_states: list[torch.Tensor] = []
        means: list[torch.Tensor] = []
        log_scales: list[torch.Tensor] = []
        event_logits: list[torch.Tensor] = []

        for index in range(steps):
            delta_t = (
                times[:, index]
                if index == 0
                else times[:, index] - times[:, index - 1]
            )
            pre_state = self.flow(state, delta_t)
            assimilated_state = self.assimilate(
                pre_state,
                representations[:, index],
                values[:, index],
                masks[:, index],
            )
            updated_state = self.jump(
                assimilated_state,
                event_features[:, index],
                jump_eligible_mask[:, index],
            )
            state = torch.where(
                update_mask[:, index].bool().unsqueeze(-1),
                updated_state,
                pre_state,
            )
            mean, log_scale = self._forecast(state)
            pre_states.append(pre_state)
            post_states.append(state)
            means.append(mean)
            if log_scale is not None:
                log_scales.append(log_scale)
            event_logits.append(self.event_head(state).squeeze(-1))

        return Phase05FlowJumpOutput(
            pre_event_states=torch.stack(pre_states, dim=1),
            post_event_states=torch.stack(post_states, dim=1),
            value_mean=torch.stack(means, dim=1),
            value_log_scale=(
                torch.stack(log_scales, dim=1)
                if len(log_scales) == steps
                else None
            ),
            event_logits=torch.stack(event_logits, dim=1),
        )


__all__ = [
    "FlowMode",
    "JumpMode",
    "Phase05FlowJumpAdapter",
    "Phase05FlowJumpOutput",
    "UncertaintyMode",
]
