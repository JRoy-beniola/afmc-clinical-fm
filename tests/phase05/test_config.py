from pathlib import Path

import pytest
from afmc_fm.phase05.config import Phase05Config, SeedBundle, load_phase05_config

CONFIG_PATH = Path("configs/experiments/phase05.yaml")


def test_phase05_config_loads_exact_protocol_axes():
    config = load_phase05_config(CONFIG_PATH)

    assert config.train_sizes == (5, 10, 20, 40, 80, 100)
    assert config.primary_train_sizes == (5, 10, 20, 40)
    assert len(config.development_bundles) == 5
    assert len(config.confirmatory_bundles) == 10
    assert set(config.development_bundles).isdisjoint(config.confirmatory_bundles)
    assert config.time_scale_days == 30.0
    assert config.bootstrap_resamples == 10_000
    assert config.development_win_requirement == 4
    assert config.confirmatory_win_requirement == 8
    assert config.provisional_relative_effect == 0.02
    assert config.misspecification_tolerance == 0.05


def test_phase05_config_uses_predeclared_matched_seed_bundles():
    config = load_phase05_config(CONFIG_PATH)

    assert config.development_bundles == tuple(
        SeedBundle(400 + index, 500 + index, 600 + index)
        for index in range(1, 6)
    )
    assert config.confirmatory_bundles == tuple(
        SeedBundle(700 + index, 800 + index, 900 + index)
        for index in range(1, 11)
    )


def test_phase05_config_rejects_overlap_with_phase0_seed_bundles():
    with pytest.raises(ValueError, match="must not overlap Phase-0 seed bundles"):
        Phase05Config(
            development_bundles=(SeedBundle(101, 201, 301),) * 5,
            confirmatory_bundles=tuple(
                SeedBundle(700 + index, 800 + index, 900 + index)
                for index in range(1, 11)
            ),
        )


def test_phase05_config_rejects_development_confirmation_overlap():
    overlap = SeedBundle(401, 501, 601)
    with pytest.raises(
        ValueError,
        match="development and confirmatory seed bundles must be disjoint",
    ):
        Phase05Config(
            development_bundles=(
                overlap,
                SeedBundle(402, 502, 602),
                SeedBundle(403, 503, 603),
                SeedBundle(404, 504, 604),
                SeedBundle(405, 505, 605),
            ),
            confirmatory_bundles=(
                overlap,
                *tuple(
                    SeedBundle(701 + index, 801 + index, 901 + index)
                    for index in range(1, 10)
                ),
            ),
        )


def test_phase05_config_requires_frozen_primary_train_sizes():
    with pytest.raises(ValueError, match="primary_train_sizes must be exactly"):
        Phase05Config(primary_train_sizes=(5, 10, 20))


def test_phase05_config_rejects_invalid_thresholds_and_time_scale():
    with pytest.raises(ValueError, match="time_scale_days must be positive"):
        Phase05Config(time_scale_days=0.0)
    with pytest.raises(
        ValueError,
        match="provisional_relative_effect must be between 0 and 1",
    ):
        Phase05Config(provisional_relative_effect=1.5)
    with pytest.raises(
        ValueError,
        match="misspecification_tolerance must be between 0 and 1",
    ):
        Phase05Config(misspecification_tolerance=-0.1)
