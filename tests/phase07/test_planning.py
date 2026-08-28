import hashlib
import json
from collections import Counter
from dataclasses import replace

import pytest

from afmc_fm.phase07.config import load_phase07_config

_CONFIG_PATH = "configs/experiments/phase07.yaml"


def _planning_api():
    try:
        from afmc_fm.phase07.planning import (
            Phase07CellSpec,
            phase07_plan_sha256,
            plan_phase07_cells,
        )
    except ModuleNotFoundError:
        pytest.fail("Phase 0.7 planning module is not implemented")
    return Phase07CellSpec, phase07_plan_sha256, plan_phase07_cells


def test_phase07_planner_materializes_exact_frozen_200_cell_matrix():
    _, _, plan_phase07_cells = _planning_api()
    config = load_phase07_config(_CONFIG_PATH)

    cells = plan_phase07_cells(config)

    assert len(cells) == 200
    assert len({cell.cell_id for cell in cells}) == 200
    assert {
        (cell.cohort_seed, cell.subset_seed, cell.model_seed)
        for cell in cells
    } == {
        (cohort_seed, subset_seed, model_seed)
        for cohort_seed, subset_seed in config.contexts
        for model_seed in config.model_seeds
    }
    pair_counts = Counter(
        (cell.cohort_seed, cell.subset_seed, cell.model_seed) for cell in cells
    )
    assert len(pair_counts) == 50
    assert set(pair_counts.values()) == {4}
    assert {cell.world for cell in cells} == {"smooth"}
    assert {cell.n_train for cell in cells} == {40}
    assert {cell.flow_mode for cell in cells} == {"none", "time_scaled"}
    assert {cell.optimization_policy for cell in cells} == {
        "standard_early_stop",
        "forced_horizon",
    }


def test_phase07_planner_has_deterministic_locked_order():
    _, _, plan_phase07_cells = _planning_api()
    cells = plan_phase07_cells(load_phase07_config(_CONFIG_PATH))

    first = cells[0]
    assert (
        first.cohort_seed,
        first.subset_seed,
        first.model_seed,
        first.flow_mode,
        first.optimization_policy,
    ) == (406, 506, 1101, "none", "standard_early_stop")

    assert (
        cells[1].flow_mode,
        cells[1].optimization_policy,
        cells[2].flow_mode,
        cells[2].optimization_policy,
        cells[3].flow_mode,
        cells[3].optimization_policy,
    ) == (
        "none",
        "forced_horizon",
        "time_scaled",
        "standard_early_stop",
        "time_scaled",
        "forced_horizon",
    )

    last = cells[-1]
    assert (
        last.cohort_seed,
        last.subset_seed,
        last.model_seed,
        last.flow_mode,
        last.optimization_policy,
    ) == (410, 510, 1110, "time_scaled", "forced_horizon")


def test_phase07_plan_sha256_is_canonical_and_deterministic():
    _, phase07_plan_sha256, plan_phase07_cells = _planning_api()
    cells = plan_phase07_cells(load_phase07_config(_CONFIG_PATH))
    payload = [
        {
            "world": cell.world,
            "cohort_seed": cell.cohort_seed,
            "subset_seed": cell.subset_seed,
            "model_seed": cell.model_seed,
            "n_train": cell.n_train,
            "flow_mode": cell.flow_mode,
            "optimization_policy": cell.optimization_policy,
        }
        for cell in cells
    ]
    expected = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    assert phase07_plan_sha256(cells) == expected
    assert phase07_plan_sha256(tuple(cells)) == expected
    assert len(expected) == 64


def test_phase07_cell_rejects_protected_or_out_of_scope_seeds():
    _, _, plan_phase07_cells = _planning_api()
    cell = plan_phase07_cells(load_phase07_config(_CONFIG_PATH))[0]

    with pytest.raises(ValueError, match="cohort"):
        replace(cell, cohort_seed=701)
    with pytest.raises(ValueError, match="subset"):
        replace(cell, subset_seed=801)
    with pytest.raises(ValueError, match="model"):
        replace(cell, model_seed=901)
    with pytest.raises(ValueError, match="context"):
        replace(cell, cohort_seed=411, subset_seed=511)
    with pytest.raises(ValueError, match="model_seed"):
        replace(cell, model_seed=1111)


def test_phase07_cell_type_is_frozen():
    _, _, plan_phase07_cells = _planning_api()
    cell = plan_phase07_cells(load_phase07_config(_CONFIG_PATH))[0]

    with pytest.raises(AttributeError):
        cell.n_train = 20  # type: ignore[misc]
