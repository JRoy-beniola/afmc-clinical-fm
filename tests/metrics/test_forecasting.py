import math

import numpy as np

from afmc_fm.metrics.forecasting import (
    binary_metrics,
    gaussian_forecasting_metrics,
    interval_coverage,
    regression_metrics,
)


def test_perfect_regression_has_zero_error():
    y = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


def test_interval_coverage_known_case():
    y = np.array([0.0, 2.0, 5.0, 10.0])
    lower = np.array([-1.0, 1.0, 6.0, 8.0])
    upper = np.array([1.0, 3.0, 7.0, 9.0])
    assert interval_coverage(y, lower, upper) == 0.5


def test_single_class_binary_metrics_do_not_crash():
    metrics = binary_metrics(np.zeros(4), np.full(4, 0.2))
    assert math.isnan(metrics["roc_auc"])
    assert np.isfinite(metrics["brier"])


def test_gaussian_forecasting_metrics_include_uncertainty_quality():
    truth = np.array([0.0, 1.0, -1.0])
    mean = np.zeros(3)
    log_scale = np.zeros(3)
    metrics = gaussian_forecasting_metrics(truth, mean, log_scale)
    assert set(metrics) == {"nll", "coverage_90"}
    assert np.isfinite(metrics["nll"])
    assert 0.0 <= metrics["coverage_90"] <= 1.0
