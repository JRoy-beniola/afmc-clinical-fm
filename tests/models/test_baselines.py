import numpy as np
import pytest
import torch

from afmc_fm.models.baselines import (
    GRUBaseline,
    MLPRegressorBaseline,
    ProbeClassifier,
    ProbeRegressor,
    TorchRidgeRegressor,
)

RIDGE_PARITY_ATOL = 1e-10
RIDGE_PARITY_RTOL = 1e-10


def _well_conditioned_ridge_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260823)
    train_features = rng.normal(size=(96, 7))
    coefficients = np.array([1.25, -0.75, 0.5, 2.0, -1.5, 0.25, 0.9])
    targets = train_features @ coefficients + 4.25 + rng.normal(scale=0.05, size=96)
    test_features = rng.normal(size=(24, 7))
    return train_features, targets, test_features


def test_probe_regressor_fits_small_signal():
    x = np.arange(40, dtype=float).reshape(20, 2)
    y = x[:, 0] * 0.5 - x[:, 1] * 0.1
    model = ProbeRegressor().fit(x, y)
    pred = model.predict(x)
    assert np.mean((pred - y) ** 2) < 1e-6


def test_torch_ridge_cpu_predictions_match_sklearn_reference():
    train_x, train_y, test_x = _well_conditioned_ridge_problem()
    reference = ProbeRegressor().fit(train_x, train_y).predict(test_x)

    prediction = TorchRidgeRegressor(alpha=1.0, device="cpu").fit(
        train_x, train_y
    ).predict(test_x)

    np.testing.assert_allclose(
        prediction,
        reference,
        atol=RIDGE_PARITY_ATOL,
        rtol=RIDGE_PARITY_RTOL,
    )


def test_torch_ridge_does_not_penalize_intercept():
    train_x, _, test_x = _well_conditioned_ridge_problem()
    constant_target = np.full(len(train_x), 7.25)

    prediction = TorchRidgeRegressor(alpha=1.0, device="cpu").fit(
        train_x, constant_target
    ).predict(test_x)

    np.testing.assert_allclose(prediction, 7.25, atol=1e-12, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_torch_ridge_cuda_predictions_match_cpu():
    train_x, train_y, test_x = _well_conditioned_ridge_problem()
    cpu_model = TorchRidgeRegressor(alpha=1.0, device="cpu").fit(train_x, train_y)
    cuda_model = TorchRidgeRegressor(alpha=1.0, device="cuda").fit(train_x, train_y)

    cpu_prediction = cpu_model.predict(test_x)
    cuda_prediction = cuda_model.predict(test_x)

    assert cuda_model.coef_.device.type == "cuda"
    np.testing.assert_allclose(
        cuda_prediction,
        cpu_prediction,
        atol=RIDGE_PARITY_ATOL,
        rtol=RIDGE_PARITY_RTOL,
    )


def test_probe_classifier_returns_probabilities():
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    model = ProbeClassifier().fit(x, y)
    prob = model.predict_proba(x)
    assert prob.shape == (4,)
    assert np.all((prob >= 0.0) & (prob <= 1.0))


def test_gru_baseline_has_finite_shapes_and_gradients():
    model = GRUBaseline(
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
    assert output.states.shape == (batch, steps, 12)
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


def test_mlp_representation_head_fits_nonlinear_signal():
    x = np.linspace(-1.0, 1.0, 80).reshape(-1, 1)
    y = x[:, 0] ** 2
    model = MLPRegressorBaseline(seed=3).fit(x, y)
    prediction = model.predict(x)
    assert np.mean((prediction - y) ** 2) < 0.03
    expected_parameters = sum(
        parameter.size
        for parameter in (*model.model.coefs_, *model.model.intercepts_)
    )
    assert model.trainable_parameter_count() == expected_parameters


def test_gru_from_scratch_is_invariant_to_representation_values():
    model = GRUBaseline(value_dim=3, event_dim=3, hidden_size=12)
    values = torch.randn(2, 4, 3)
    masks = torch.ones(2, 4, 3)
    events = torch.zeros(2, 4, 3)
    times = torch.arange(4).float().repeat(2, 1)
    zeros = model(torch.zeros(2, 4, 16), values, masks, events, times)
    random = model(torch.randn(2, 4, 16), values, masks, events, times)
    torch.testing.assert_close(zeros.value_mean, random.value_mean)
    torch.testing.assert_close(zeros.event_logits, random.event_logits)
