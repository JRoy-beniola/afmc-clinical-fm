import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true)
    probability = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    auc = float("nan") if np.unique(truth).size < 2 else float(roc_auc_score(truth, probability))
    return {
        "roc_auc": auc,
        "brier": float(brier_score_loss(truth, probability)),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
    }


def interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    truth = np.asarray(y_true)
    if truth.shape != np.asarray(lower).shape or truth.shape != np.asarray(upper).shape:
        raise ValueError("truth and interval bounds must have identical shapes")
    return float(np.mean((truth >= lower) & (truth <= upper)))
