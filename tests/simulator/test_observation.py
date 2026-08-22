import numpy as np

from afmc_fm.simulator.observation import emit_labs, observation_probability


def test_mcar_probability_does_not_depend_on_latent_state():
    p_low = observation_probability("mcar", np.array([-2.0, 0.0]), None, 0)
    p_high = observation_probability("mcar", np.array([2.0, 0.0]), None, 0)
    assert p_low == p_high


def test_mnar_probability_increases_with_latent_severity():
    p_low = observation_probability("mnar", np.array([-2.0, 0.0]), None, 0)
    p_high = observation_probability("mnar", np.array([2.0, 0.0]), None, 0)
    assert p_high > p_low


def test_site_shift_changes_measurement_probability():
    state = np.array([0.5, -0.2])
    assert observation_probability("site_shift", state, None, 0) != observation_probability(
        "site_shift", state, None, 1
    )



def test_emissions_have_three_finite_lab_channels():
    state = np.array([0.5, -0.2, 0.3, 0.1])
    labs = emit_labs(state, np.random.default_rng(1), 0.1)
    assert set(labs) == {"LAB_FAST", "LAB_SLOW", "LAB_BURDEN"}
    assert np.isfinite(list(labs.values())).all()
