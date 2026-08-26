from __future__ import annotations

from afmc_fm.phase06.config import Phase06Config
from afmc_fm.phase06.planning import plan_d4b_cells


def test_d4b_planner_is_exact_frozen_100_cell_matrix() -> None:
    cells = plan_d4b_cells(Phase06Config())

    assert len(cells) == 100
    assert len({cell.cell_id for cell in cells}) == 100
    assert {cell.stage for cell in cells} == {"d4b"}
    assert {cell.world for cell in cells} == {"smooth"}
    assert {cell.n_train for cell in cells} == {40}
    assert {cell.flow_mode for cell in cells} == {"none", "time_scaled"}
    assert {cell.jump_mode for cell in cells} == {"none"}
    assert {cell.uncertainty_mode for cell in cells} == {"deterministic"}

    assert {
        (cell.cohort_seed, cell.subset_seed) for cell in cells
    } == {
        (401, 501),
        (402, 502),
        (403, 503),
        (404, 504),
        (405, 505),
    }
    assert {cell.model_seed for cell in cells} == set(range(1001, 1011))

    pairs = {}
    for cell in cells:
        key = (cell.cohort_seed, cell.subset_seed, cell.model_seed)
        pairs.setdefault(key, set()).add(cell.flow_mode)
    assert len(pairs) == 50
    assert all(modes == {"none", "time_scaled"} for modes in pairs.values())


def test_d4b_planner_never_reaches_confirmatory_seed_namespaces() -> None:
    cells = plan_d4b_cells(Phase06Config())

    assert not ({cell.cohort_seed for cell in cells} & set(range(701, 711)))
    assert not ({cell.subset_seed for cell in cells} & set(range(801, 811)))
    assert not ({cell.model_seed for cell in cells} & set(range(901, 911)))


def test_d4b_planner_does_not_reuse_d1_d2_model_seed_bank() -> None:
    cells = plan_d4b_cells(Phase06Config())

    assert not ({cell.model_seed for cell in cells} & set(range(601, 606)))
