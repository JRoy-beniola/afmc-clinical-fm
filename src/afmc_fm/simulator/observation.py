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
    value_code: str | None = None,
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
        site_group = site_id % 2
        site_intercept = (-0.35, -0.65)[site_group]
        variable_offsets = {
            0: {"LAB_FAST": 0.9, "LAB_SLOW": -0.7, "LAB_BURDEN": 0.1},
            1: {"LAB_FAST": -0.8, "LAB_SLOW": 0.8, "LAB_BURDEN": -0.2},
        }
        variable_offset = (
            0.0
            if value_code is None
            else variable_offsets[site_group][value_code]
        )
        logit = site_intercept + variable_offset + 0.7 * float(latent_state[0])
    return float(np.clip(_sigmoid(logit), 0.02, 0.98))


def measurement_noise_scale(
    regime: str,
    site_id: int,
    value_code: str,
    base_scale: float,
) -> float:
    if value_code not in LAB_CODES:
        raise ValueError(f"unsupported value code: {value_code}")
    if regime != "site_shift":
        return base_scale
    multipliers = {
        0: {"LAB_FAST": 0.6, "LAB_SLOW": 1.5, "LAB_BURDEN": 0.9},
        1: {"LAB_FAST": 1.6, "LAB_SLOW": 0.7, "LAB_BURDEN": 1.3},
    }
    return base_scale * multipliers[site_id % 2][value_code]


def emit_labs(
    state: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float,
    *,
    regime: str = "mnar",
    site_id: int = 0,
) -> dict[str, float]:
    noise = {
        code: measurement_noise_scale(regime, site_id, code, noise_scale)
        for code in LAB_CODES
    }
    return {
        "LAB_FAST": float(
            1.4 * state[0]
            + 0.25 * np.tanh(state[1])
            + rng.normal(0, noise["LAB_FAST"])
        ),
        "LAB_SLOW": float(
            0.6 * state[1] ** 2
            + 0.3 * state[2 % len(state)]
            + rng.normal(0, noise["LAB_SLOW"])
        ),
        "LAB_BURDEN": float(
            0.5 * state.mean()
            + 0.2 * state[0] * state[-1]
            + rng.normal(0, noise["LAB_BURDEN"])
        ),
    }
