from dataclasses import replace
from pathlib import Path

import pytest

_CONFIG_PATH = Path("configs/experiments/phase07.yaml")


def _config_api():
    try:
        from afmc_fm.phase07.config import Phase07Config, load_phase07_config
    except ModuleNotFoundError:
        pytest.fail("Phase 0.7 config module is not implemented")
    return Phase07Config, load_phase07_config


def test_locked_phase07_config_matches_frozen_design():
    _, load_phase07_config = _config_api()
    config = load_phase07_config(_CONFIG_PATH)

    assert config.phase05_config == "configs/experiments/phase05.yaml"
    assert config.phase07_spec == (
        "docs/superpowers/specs/2026-08-28-phase0-7-optimization-horizon-intervention-design.md"
    )
    assert config.simulator_config == "configs/simulator/full.yaml"
    assert config.world == "smooth"
    assert config.n_train == 40
    assert config.contexts == (
        (406, 506),
        (407, 507),
        (408, 508),
        (409, 509),
        (410, 510),
    )
    assert config.model_seeds == tuple(range(1101, 1111))
    assert config.flow_modes == ("none", "time_scaled")
    assert config.optimization_policies == ("standard_early_stop", "forced_horizon")
    assert config.max_epochs == 100
    assert config.patience == 12
    assert config.bootstrap_resamples == 10_000
    assert config.bootstrap_seed == 20260827
    assert config.forbidden_cohort_seeds == tuple(range(701, 711))
    assert config.forbidden_subset_seeds == tuple(range(801, 811))
    assert config.forbidden_model_seeds == tuple(range(901, 911))


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("world", "jumps", "world"),
        ("n_train", 20, "n_train"),
        ("contexts", ((406, 506),), "contexts"),
        ("model_seeds", tuple(range(1101, 1110)), "model_seeds"),
        ("flow_modes", ("none", "gated"), "flow_modes"),
        (
            "optimization_policies",
            ("standard_early_stop",),
            "optimization_policies",
        ),
        ("max_epochs", 99, "max_epochs"),
        ("patience", 11, "patience"),
        ("bootstrap_resamples", 9999, "bootstrap_resamples"),
        ("bootstrap_seed", 20260828, "bootstrap_seed"),
        ("forbidden_cohort_seeds", tuple(range(701, 710)), "forbidden_cohort_seeds"),
        ("forbidden_subset_seeds", tuple(range(801, 810)), "forbidden_subset_seeds"),
        ("forbidden_model_seeds", tuple(range(901, 910)), "forbidden_model_seeds"),
    ],
)
def test_phase07_config_rejects_frozen_protocol_drift(field_name, bad_value, message):
    _, load_phase07_config = _config_api()
    config = load_phase07_config(_CONFIG_PATH)

    with pytest.raises(ValueError, match=message):
        replace(config, **{field_name: bad_value})


def test_phase07_config_rejects_non_string_provenance_paths():
    _, load_phase07_config = _config_api()
    config = load_phase07_config(_CONFIG_PATH)

    with pytest.raises((TypeError, ValueError), match="phase07_spec"):
        replace(config, phase07_spec=Path(config.phase07_spec))


def test_phase07_config_type_is_frozen():
    Phase07Config, _ = _config_api()
    config = Phase07Config()

    with pytest.raises(AttributeError):
        config.world = "jumps"  # type: ignore[misc]
