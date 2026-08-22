import numpy as np
from scipy.spatial import procrustes
from sklearn.metrics import r2_score


def _validated_pair(z_true: np.ndarray, z_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(z_true, dtype=float)
    estimate = np.asarray(z_hat, dtype=float)
    if truth.ndim != 2 or estimate.ndim != 2 or truth.shape[0] != estimate.shape[0]:
        raise ValueError("latent arrays must be 2D with equal sample counts")
    return truth, estimate


def aligned_latent_r2(z_true: np.ndarray, z_hat: np.ndarray) -> float:
    truth, estimate = _validated_pair(z_true, z_hat)
    design = np.column_stack([estimate, np.ones(estimate.shape[0])])
    coefficients, *_ = np.linalg.lstsq(design, truth, rcond=None)
    aligned = design @ coefficients
    return float(r2_score(truth, aligned, multioutput="variance_weighted"))


def procrustes_latent_error(z_true: np.ndarray, z_hat: np.ndarray) -> float:
    truth, estimate = _validated_pair(z_true, z_hat)
    if truth.shape[1] != estimate.shape[1]:
        raise ValueError("Procrustes comparison requires equal latent dimensions")
    _, _, disparity = procrustes(truth, estimate)
    return float(disparity)
