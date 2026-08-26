from dataclasses import replace
from pathlib import Path

import pytest

from afmc_fm.phase06.config import Phase06Config, load_phase06_config

_CONFIG_PATH = Path("configs/experiments/phase06.yaml")


def test_locked_phase06_config_matches_approved_matrix():
    config = load_phase06_config(_CONFIG_PATH)

    assert config.phase05_config == "configs/experiments/phase05.yaml"
    assert config.phase06_spec == (
        "docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md"
    )
    assert config.simulator_config == "configs/simulator/full.yaml"
    assert config.world == "smooth"
    assert config.d1_train_sizes == (5, 10, 20, 40)
    assert config.d2_train_sizes == (5, 40)
    assert config.d1_flow_modes == ("none", "time_scaled")
    assert config.d2_flow_modes == ("none", "time_scaled")
    assert config.jump_mode == "none"
    assert config.uncertainty_mode == "deterministic"
    assert config.bootstrap_resamples == 10_000
    assert config.bootstrap_seed == 20260826
    assert config.forbidden_cohort_seeds == tuple(range(701, 711))
    assert config.forbidden_subset_seeds == tuple(range(801, 811))
    assert config.forbidden_model_seeds == tuple(range(901, 911))


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("world", "jumps", "world"),
        ("d1_train_sizes", (5, 10, 20), "d1_train_sizes"),
        ("d2_train_sizes", (5, 20, 40), "d2_train_sizes"),
        ("d1_flow_modes", ("none", "gated"), "d1_flow_modes"),
        ("d2_flow_modes", ("none", "gated"), "d2_flow_modes"),
        ("jump_mode", "historical_gru", "jump_mode"),
        ("uncertainty_mode", "joint", "uncertainty_mode"),
        ("bootstrap_resamples", 9999, "bootstrap_resamples"),
        ("bootstrap_seed", 20260824, "bootstrap_seed"),
        ("forbidden_cohort_seeds", tuple(range(701, 710)), "forbidden_cohort_seeds"),
        ("forbidden_subset_seeds", tuple(range(801, 810)), "forbidden_subset_seeds"),
        ("forbidden_model_seeds", tuple(range(901, 910)), "forbidden_model_seeds"),
    ],
)
def test_phase06_config_rejects_protocol_drift(field_name, bad_value, message):
    config = load_phase06_config(_CONFIG_PATH)
    values = {field_name: bad_value}

    with pytest.raises(ValueError, match=message):
        replace(config, **values)


def test_phase06_config_rejects_non_string_provenance_paths():
    config = load_phase06_config(_CONFIG_PATH)

    with pytest.raises((TypeError, ValueError), match="phase05_config"):
        replace(config, phase05_config=Path(config.phase05_config))


def test_phase06_config_type_is_frozen():
    config = Phase06Config()

    with pytest.raises(AttributeError):
        config.world = "jumps"  # type: ignore[misc]
