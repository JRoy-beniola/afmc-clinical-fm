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


class TorchRidgeRegressor:
    """Torch ridge regression with an unpenalized intercept."""

    def __init__(
        self,
        alpha: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.alpha = alpha
        self.device = torch.device(device)
        self.coef_: torch.Tensor | None = None
        self.intercept_: torch.Tensor | None = None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> "TorchRidgeRegressor":
        feature_tensor = torch.as_tensor(
            features,
            dtype=torch.float64,
            device=self.device,
        )
        target_tensor = torch.as_tensor(
            targets,
            dtype=torch.float64,
            device=self.device,
        )
        feature_mean = feature_tensor.mean(dim=0)
        target_mean = target_tensor.mean()
        centered_features = feature_tensor - feature_mean
        centered_targets = target_tensor - target_mean
        regularized_gram = centered_features.T @ centered_features
        regularized_gram = regularized_gram + self.alpha * torch.eye(
            feature_tensor.shape[1],
            dtype=feature_tensor.dtype,
            device=self.device,
        )
        right_hand_side = centered_features.T @ centered_targets
        cholesky_factor = torch.linalg.cholesky(regularized_gram)
        self.coef_ = torch.cholesky_solve(
            right_hand_side.unsqueeze(-1),
            cholesky_factor,
        ).squeeze(-1)
        self.intercept_ = target_mean - feature_mean @ self.coef_
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("TorchRidgeRegressor must be fitted before prediction")
        feature_tensor = torch.as_tensor(
            features,
            dtype=torch.float64,
            device=self.device,
        )
        prediction = feature_tensor @ self.coef_ + self.intercept_
        return prediction.detach().cpu().numpy()

    def trainable_parameter_count(self) -> int:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("TorchRidgeRegressor must be fitted before counting parameters")
        return self.coef_.numel() + self.intercept_.numel()


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


class TorchMLPRegressorBaseline:
    _L2_ALPHA = 1e-3

    def __init__(
        self,
        input_dim: int,
        seed: int,
        device: torch.device | str,
        hidden_size: int = 32,
    ) -> None:
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            ).to(device=self.device, dtype=torch.float64)

    def fit(
        self, features: np.ndarray, targets: np.ndarray
    ) -> "TorchMLPRegressorBaseline":
        feature_tensor = torch.as_tensor(
            features,
            dtype=torch.float64,
            device=self.device,
        )
        target_tensor = torch.as_tensor(
            targets,
            dtype=torch.float64,
            device=self.device,
        )
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=500,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            prediction = self.model(feature_tensor).squeeze(-1)
            squared_error = torch.nn.functional.mse_loss(
                prediction, target_tensor
            )
            weight_penalty = (
                self.model[0].weight.square().sum()
                + self.model[2].weight.square().sum()
            )
            loss = 0.5 * (
                squared_error
                + self._L2_ALPHA * weight_penalty / len(feature_tensor)
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        feature_tensor = torch.as_tensor(
            features,
            dtype=torch.float64,
            device=self.device,
        )
        with torch.no_grad():
            prediction = self.model(feature_tensor).squeeze(-1)
        return prediction.detach().cpu().numpy()

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())


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