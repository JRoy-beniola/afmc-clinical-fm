import numpy as np

from afmc_fm.metrics.latent import aligned_latent_r2, procrustes_latent_error


def test_linear_transform_of_truth_is_recoverable_after_alignment():
    rng = np.random.default_rng(0)
    z_true = rng.normal(size=(200, 3))
    transform = np.array([[2.0, 0.5, 0.0], [0.0, 1.5, 0.3], [0.2, 0.0, 1.0]])
    z_hat = z_true @ transform
    assert aligned_latent_r2(z_true, z_hat) > 0.99


def test_identical_latents_have_zero_procrustes_error():
    z = np.arange(30, dtype=float).reshape(10, 3)
    assert procrustes_latent_error(z, z) < 1e-12
