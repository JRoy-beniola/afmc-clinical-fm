import numpy as np
import torch

from afmc_fm.models.baselines import GRUBaseline, ProbeClassifier, ProbeRegressor


def test_probe_regressor_fits_small_signal():
    x = np.arange(40, dtype=float).reshape(20, 2)
    y = x[:, 0] * 0.5 - x[:, 1] * 0.1
    model = ProbeRegressor().fit(x, y)
    pred = model.predict(x)
    assert np.mean((pred - y) ** 2) < 1e-6


def test_probe_classifier_returns_probabilities():
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    model = ProbeClassifier().fit(x, y)
    prob = model.predict_proba(x)
    assert prob.shape == (4,)
    assert np.all((prob >= 0.0) & (prob <= 1.0))


def test_gru_baseline_has_finite_shapes_and_gradients():
    model = GRUBaseline(
        representation_dim=16,
        value_dim=3,
        event_dim=3,
        hidden_size=12,
    )
    batch, steps = 2, 5
    output = model(
        representations=torch.randn(batch, steps, 16),
        values=torch.randn(batch, steps, 3),
        masks=torch.ones(batch, steps, 3),
        event_features=torch.zeros(batch, steps, 3),
        times=torch.arange(steps).float().repeat(batch, 1),
    )
    assert output.value_mean.shape == (batch, steps, 3)
    assert output.value_log_scale.shape == (batch, steps, 3)
    assert output.event_logits.shape == (batch, steps)
    loss = (
        output.value_mean.square().mean()
        + output.value_log_scale.square().mean()
        + output.event_logits.square().mean()
    )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(gradient is not None for gradient in gradients)
