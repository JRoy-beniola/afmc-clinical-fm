import math

import numpy as np
import pandas as pd
import pytest

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.protocol import (
    build_protocol_lock,
    estimate_phase0_relative_noise_floor,
)

_TARGET_WORLDS = ("smooth", "jumps", "informative_observation")
_PRIMARY_TRAIN_SIZES = (5, 10, 20, 40)
_MODELS = ("flow_jump", "representation_linear", "gru_from_scratch")
_PHASE0_BUNDLES = tuple(
    (100 + index, 200 + index, 300 + index) for index in range(1, 6)
)
_SPEC_COMMIT = "67c6662c64e69606bcbd3eca2bc8139548013051"
_PHASE0_EXECUTION = "d6f105eee73fcb8e9cc5987d292b1bb98a687382"


def _phase0_metrics(*, relative_step: float = 0.01) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * relative_step
    for world_index, world in enumerate(_TARGET_WORLDS):
        for n_index, n_train in enumerate(_PRIMARY_TRAIN_SIZES):
            for model_index, model in enumerate(_MODELS):
                base = 1.0 + 0.1 * world_index + 0.01 * n_index + 0.001 * model_index
                for seed_index, bundle in enumerate(_PHASE0_BUNDLES):
                    rows.append(
                        {
                            "benchmark": "low_n",
                            "site_or_shift": "all",
                            "ablation": "none",
                            "metric": "mae",
                            "world": world,
                            "n_train": n_train,
                            "model": model,
                            "cohort_seed": bundle[0],
                            "subset_seed": bundle[1],
                            "model_seed": bundle[2],
                            "value": base * (1.0 + offsets[seed_index]),
                        }
                    )
    rows.append(
        {
            "benchmark": "low_n",
            "site_or_shift": "all",
            "ablation": "none",
            "metric": "rmse",
            "world": "smooth",
            "n_train": 5,
            "model": "flow_jump",
            "cohort_seed": 101,
            "subset_seed": 201,
            "model_seed": 301,
            "value": 999.0,
        }
    )
    return pd.DataFrame(rows)


def test_phase0_noise_floor_uses_predeclared_robust_estimator():
    metrics = _phase0_metrics(relative_step=0.01)

    result = estimate_phase0_relative_noise_floor(metrics)

    expected = 1.4826 * 0.01 / math.sqrt(5)
    assert result == pytest.approx(expected)


def test_phase0_noise_floor_rejects_missing_required_group():
    metrics = _phase0_metrics()
    drop = (
        (metrics["world"] == "smooth")
        & (metrics["n_train"] == 5)
        & (metrics["model"] == "flow_jump")
        & (metrics["metric"] == "mae")
    )

    with pytest.raises(ValueError, match="missing required Phase-0 groups"):
        estimate_phase0_relative_noise_floor(metrics.loc[~drop].copy())


def test_phase0_noise_floor_rejects_group_without_five_matched_seed_rows():
    metrics = _phase0_metrics()
    target = metrics.index[
        (metrics["world"] == "smooth")
        & (metrics["n_train"] == 5)
        & (metrics["model"] == "flow_jump")
        & (metrics["metric"] == "mae")
    ]
    metrics = metrics.drop(target[0])

    with pytest.raises(ValueError, match="exactly five matched seed rows"):
        estimate_phase0_relative_noise_floor(metrics)


def test_phase0_noise_floor_rejects_duplicate_matched_seed_bundle():
    metrics = _phase0_metrics()
    target = metrics.index[
        (metrics["world"] == "smooth")
        & (metrics["n_train"] == 5)
        & (metrics["model"] == "flow_jump")
        & (metrics["metric"] == "mae")
    ]
    first = target[0]
    second = target[1]
    for column in ("cohort_seed", "subset_seed", "model_seed"):
        metrics.loc[second, column] = metrics.loc[first, column]

    with pytest.raises(ValueError, match="five distinct matched seed bundles"):
        estimate_phase0_relative_noise_floor(metrics)


def test_phase0_noise_floor_rejects_nonfinite_values():
    metrics = _phase0_metrics()
    target = metrics.index[
        (metrics["world"] == "smooth")
        & (metrics["n_train"] == 5)
        & (metrics["model"] == "flow_jump")
        & (metrics["metric"] == "mae")
    ][0]
    metrics.loc[target, "value"] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        estimate_phase0_relative_noise_floor(metrics)


def test_phase0_noise_floor_rejects_near_zero_group_median():
    metrics = _phase0_metrics()
    target = (
        (metrics["world"] == "smooth")
        & (metrics["n_train"] == 5)
        & (metrics["model"] == "flow_jump")
        & (metrics["metric"] == "mae")
    )
    metrics.loc[target, "value"] = [1e-15, 2e-15, 3e-15, 4e-15, 5e-15]

    with pytest.raises(ValueError, match="near-zero median"):
        estimate_phase0_relative_noise_floor(metrics)


def test_protocol_lock_never_lowers_the_two_percent_floor():
    config = Phase05Config()
    metrics = _phase0_metrics(relative_step=0.01)

    lock = build_protocol_lock(
        config,
        metrics,
        phase0_metrics_sha256="a" * 64,
        spec_commit=_SPEC_COMMIT,
        phase0_execution_sha=_PHASE0_EXECUTION,
    )

    expected_noise = 1.4826 * 0.01 / math.sqrt(5)
    assert lock["schema_version"] == 1
    assert lock["phase0_noise_floor"] == pytest.approx(expected_noise)
    assert lock["locked_min_relative_effect"] == pytest.approx(0.02)
    assert lock["locked_uncertainty_mae_tolerance"] == pytest.approx(0.02)
    assert lock["phase05_config_sha256"] == canonical_config_hash(config)
    assert lock["phase0_metrics_sha256"] == "a" * 64
    assert lock["spec_commit"] == _SPEC_COMMIT
    assert lock["phase0_execution_sha"] == _PHASE0_EXECUTION
    assert lock["time_scale_days"] == 30.0
    assert lock["jump_eligible_event_codes"] == ["SYNTHETIC_INTERVENTION"]
    assert lock["development_bundles"] == [
        [400 + index, 500 + index, 600 + index] for index in range(1, 6)
    ]
    assert lock["confirmatory_bundles"] == [
        [700 + index, 800 + index, 900 + index] for index in range(1, 11)
    ]


def test_protocol_lock_raises_threshold_when_phase0_noise_is_larger():
    config = Phase05Config()
    metrics = _phase0_metrics(relative_step=0.05)

    lock = build_protocol_lock(
        config,
        metrics,
        phase0_metrics_sha256="b" * 64,
        spec_commit=_SPEC_COMMIT,
        phase0_execution_sha=_PHASE0_EXECUTION,
    )

    expected_noise = 1.4826 * 0.05 / math.sqrt(5)
    assert expected_noise > 0.02
    assert lock["locked_min_relative_effect"] == pytest.approx(expected_noise)
    assert lock["locked_uncertainty_mae_tolerance"] == pytest.approx(expected_noise)


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase0_metrics_sha256", "not-a-sha"),
        ("spec_commit", "short"),
        ("phase0_execution_sha", "bad"),
    ],
)
def test_protocol_lock_rejects_invalid_provenance_sha(field, value):
    kwargs = {
        "phase0_metrics_sha256": "c" * 64,
        "spec_commit": _SPEC_COMMIT,
        "phase0_execution_sha": _PHASE0_EXECUTION,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match="40- or 64-character hexadecimal SHA"):
        build_protocol_lock(Phase05Config(), _phase0_metrics(), **kwargs)
