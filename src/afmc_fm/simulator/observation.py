import math

import numpy as np

from afmc_fm.simulator.config import OBSERVATION_REGIMES

LAB_CODES = ("LAB_FAST", "LAB_SLOW", "LAB_BURDEN")


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def observation_probability(
    regime: str,
    latent_state: np.ndarray,
    previous_observed_value: float | None,
    site_id: int,
) -> float:
    if regime not in OBSERVATION_REGIMES:
        raise ValueError(f"unsupported observation regime: {regime}")

    if regime == "mcar":
        logit = -0.2
    elif regime == "mar":
        history_value = 0.0 if previous_observed_value is None else previous_observed_value
        logit = -0.6 + 0.8 * history_value
    elif regime == "mnar":
        logit = -0.8 + 1.1 * float(latent_state[0])
    else:
        site_intercept = -1.0 + 1.2 * (site_id % 2)
        logit = site_intercept + 0.7 * float(latent_state[0])
    return float(np.clip(_sigmoid(logit), 0.02, 0.98))


def emit_labs(
    state: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float,
) -> dict[str, float]:
    return {
        "LAB_FAST": float(
            1.4 * state[0] + 0.25 * np.tanh(state[1]) + rng.normal(0, noise_scale)
        ),
        "LAB_SLOW": float(
            0.6 * state[1] ** 2
            + 0.3 * state[2 % len(state)]
            + rng.normal(0, noise_scale)
        ),
        "LAB_BURDEN": float(
            0.5 * state.mean() + 0.2 * state[0] * state[-1] + rng.normal(0, noise_scale)
        ),
    }
