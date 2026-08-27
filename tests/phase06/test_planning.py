import importlib
from collections import Counter
from pathlib import Path

import pytest

from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase06.config import load_phase06_config

planning = importlib.import_module("afmc_fm.phase06.planning")
Phase06CellSpec = planning.Phase06CellSpec
plan_d1_cells = planning.plan_d1_cells
plan_d2a_cells = planning.plan_d2a_cells

_PHASE06_CONFIG = Path("configs/experiments/phase06.yaml")


def _phase06_config():
    return load_phase06_config(_PHASE06_CONFIG)


def _phase05_config():
    return load_phase05_config(_phase06_config().phase05_config)


def test_d1_plan_is_exactly_40_locked_cells():
    cells = plan_d1_cells(_phase06_config(), _phase05_config())

    assert len(cells) == 40
    assert len({cell.cell_id for cell in cells}) == 40
    assert {cell.stage for cell in cells} == {"d1"}
    assert {cell.world for cell in cells} == {"smooth"}
    assert {cell.n_train for cell in cells} == {5, 10, 20, 40}
    assert {cell.flow_mode for cell in cells} == {"none", "time_scaled"}
    assert {cell.jump_mode for cell in cells} == {"none"}
    assert {cell.uncertainty_mode for cell in cells} == {"deterministic"}
    assert {
        (cell.cohort_seed, cell.subset_seed, cell.model_seed) for cell in cells
    } == {
        (401, 501, 601),
        (402, 502, 602),
        (403, 503, 603),
        (404, 504, 604),
        (405, 505, 605),
    }


def test_d1_expands_each_development_bundle_over_all_n_and_flow_modes():
    cells = plan_d1_cells(_phase06_config(), _phase05_config())
    counts = Counter(
        (cell.cohort_seed, cell.subset_seed, cell.model_seed) for cell in cells
    )

    assert set(counts.values()) == {8}
    for bundle in counts:
        combinations = {
            (cell.n_train, cell.flow_mode)
            for cell in cells
            if (cell.cohort_seed, cell.subset_seed, cell.model_seed) == bundle
        }
        assert combinations == {
            (n_train, flow_mode)
            for n_train in (5, 10, 20, 40)
            for flow_mode in ("none", "time_scaled")
        }


def test_d2a_plan_is_exactly_100_locked_cells():
    cells = plan_d2a_cells(_phase06_config())

    assert len(cells) == 100
    assert len({cell.cell_id for cell in cells}) == 100
    assert {cell.stage for cell in cells} == {"d2a"}
    assert {cell.world for cell in cells} == {"smooth"}
    assert {cell.n_train for cell in cells} == {5, 40}
    assert {cell.flow_mode for cell in cells} == {"none", "time_scaled"}
    assert {cell.jump_mode for cell in cells} == {"none"}
    assert {cell.uncertainty_mode for cell in cells} == {"deterministic"}

    seed_triples = {
        (cell.cohort_seed, cell.subset_seed, cell.model_seed) for cell in cells
    }
    assert len(seed_triples) == 25
    assert all(
        sum(
            1
            for cell in cells
            if (cell.cohort_seed, cell.subset_seed, cell.model_seed) == triple
        )
        == 4
        for triple in seed_triples
    )


def test_d2a_uses_locked_latin_square_mapping():
    cells = plan_d2a_cells(_phase06_config())
    seed_triples = {
        (cell.cohort_seed, cell.subset_seed, cell.model_seed) for cell in cells
    }

    expected = {
        (401 + i, 501 + j, 601 + ((i + j) % 5))
        for i in range(5)
        for j in range(5)
    }
    assert seed_triples == expected


def test_d2a_is_strength_two_orthogonal_array():
    cells = plan_d2a_cells(_phase06_config())
    seed_triples = {
        (cell.cohort_seed, cell.subset_seed, cell.model_seed) for cell in cells
    }

    cohort_counts = Counter(cohort for cohort, _, _ in seed_triples)
    subset_counts = Counter(subset for _, subset, _ in seed_triples)
    model_counts = Counter(model for _, _, model in seed_triples)
    assert set(cohort_counts.values()) == {5}
    assert set(subset_counts.values()) == {5}
    assert set(model_counts.values()) == {5}

    cohort_subset = Counter((cohort, subset) for cohort, subset, _ in seed_triples)
    cohort_model = Counter((cohort, model) for cohort, _, model in seed_triples)
    subset_model = Counter((subset, model) for _, subset, model in seed_triples)
    assert len(cohort_subset) == 25 and set(cohort_subset.values()) == {1}
    assert len(cohort_model) == 25 and set(cohort_model.values()) == {1}
    assert len(subset_model) == 25 and set(subset_model.values()) == {1}


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("stage", "confirmation", "stage"),
        ("world", "jumps", "world"),
        ("n_train", 0, "n_train"),
        ("flow_mode", "gated", "flow_mode"),
        ("jump_mode", "historical_gru", "jump_mode"),
        ("uncertainty_mode", "joint", "uncertainty_mode"),
    ],
)
def test_cell_spec_rejects_scope_drift(field_name, bad_value, message):
    values = {
        "stage": "d1",
        "world": "smooth",
        "cohort_seed": 401,
        "subset_seed": 501,
        "model_seed": 601,
        "n_train": 5,
        "flow_mode": "none",
        "jump_mode": "none",
        "uncertainty_mode": "deterministic",
    }
    values[field_name] = bad_value

    with pytest.raises(ValueError, match=message):
        Phase06CellSpec(**values)


@pytest.mark.parametrize(
    "seed_values",
    [
        (701, 501, 601),
        (401, 801, 601),
        (401, 501, 901),
    ],
)
def test_cell_spec_applies_confirmatory_seed_firewall(seed_values):
    cohort_seed, subset_seed, model_seed = seed_values

    with pytest.raises(ValueError, match="confirmatory"):
        Phase06CellSpec(
            stage="d1",
            world="smooth",
            cohort_seed=cohort_seed,
            subset_seed=subset_seed,
            model_seed=model_seed,
            n_train=5,
            flow_mode="none",
        )


def test_cell_id_is_stable_and_complete():
    cell = Phase06CellSpec(
        stage="d2a",
        world="smooth",
        cohort_seed=401,
        subset_seed=503,
        model_seed=603,
        n_train=40,
        flow_mode="time_scaled",
    )

    assert cell.cell_id == (
        "d2a__smooth__cohort401__subset503__model603__"
        "n40__time_scaled__none__deterministic"
    )
