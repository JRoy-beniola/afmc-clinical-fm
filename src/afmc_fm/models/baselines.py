from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from torch import nn

from afmc_fm.data.sequences import PatientSequence


class ProbeRegressor:
    def __init__(self) -> None:
        self.model = Ridge(alpha=1.0)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "ProbeRegressor":
        self.model.fit(features, targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)


class ProbeClassifier:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "ProbeClassifier":
        self.model.fit(features, targets)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]


class GradientBoostingRegressorBaseline:
    def __init__(self) -> None:
        self.model = HistGradientBoostingRegressor()

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "GradientBoostingRegressorBaseline":
        self.model.fit(features, targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)


class MLPRegressorBaseline:
    def __init__(self, seed: int = 0) -> None:
        self.model = MLPRegressor(
            hidden_layer_sizes=(32,),
            activation="tanh",
            solver="lbfgs",
            alpha=1e-3,
            max_iter=500,
            random_state=seed,
        )

    def fit(
        self, features: np.ndarray, targets: np.ndarray
    ) -> "MLPRegressorBaseline":
        self.model.fit(features, targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.size
            for parameter in (*self.model.coefs_, *self.model.intercepts_)
        )


@dataclass(frozen=True)
class GRUBaselineOutput:
    states: torch.Tensor
    value_mean: torch.Tensor
    value_log_scale: torch.Tensor
    event_logits: torch.Tensor


class GRUBaseline(nn.Module):
    def __init__(
        self,
        value_dim: int,
        event_dim: int,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        input_dim = 2 * value_dim + event_dim + 1
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.value_head = nn.Linear(hidden_size, 2 * value_dim)
        self.event_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        representations: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        event_features: torch.Tensor,
        times: torch.Tensor,
    ) -> GRUBaselineOutput:
        delta_t = torch.zeros_like(times)
        delta_t[:, 1:] = (times[:, 1:] - times[:, :-1]).clamp_min(0.0)
        inputs = torch.cat(
            [
                values * masks,
                masks,
                event_features,
                torch.log1p(delta_t).unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden, _ = self.gru(inputs)
        value_output = self.value_head(hidden)
        value_mean, value_log_scale = value_output.chunk(2, dim=-1)
        return GRUBaselineOutput(
            states=hidden,
            value_mean=value_mean,
            value_log_scale=value_log_scale,
            event_logits=self.event_head(hidden).squeeze(-1),
        )


def flatten_sequence_examples(
    sequences: Iterable[PatientSequence],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequence_list = list(sequences)
    if not sequence_list:
        raise ValueError("at least one sequence is required")
    features = np.concatenate([sequence.representations for sequence in sequence_list])
    targets = np.concatenate([sequence.target_next_values for sequence in sequence_list])
    masks = np.concatenate([sequence.target_next_masks for sequence in sequence_list])
    event_targets = np.concatenate(
        [sequence.target_event_within_horizon for sequence in sequence_list]
    )
    return features, targets, masks, event_targets
