from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path

from afmc_fm.phase06.config import load_phase06_config
from afmc_fm.phase06.planning import Phase06CellSpec
from afmc_fm.phase06.store import Phase06Store

planning = importlib.import_module("afmc_fm.phase06.planning")
_PHASE06_CONFIG = Path("configs/experiments/phase06.yaml")

# RED contract: production support is intentionally absent at this commit.


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def test_d2b_plan_is_exact_complementary_strength_two_array():
    planner = getattr(planning, "plan_d2b_cells", None)
    assert callable(planner), "plan_d2b_cells must exist"

    cells = planner(load_phase06_config(_PHASE06_CONFIG))
    assert len(cells) == 100
    assert len({cell.cell_id for cell in cells}) == 100
    assert {cell.stage for cell in cells} == {"d2b"}
    assert {cell.world for cell in cells} == {"smooth"}
    assert Counter(cell.n_train for cell in cells) == {5: 50, 40: 50}
    assert Counter(cell.flow_mode for cell in cells) == {
        "none": 50,
        "time_scaled": 50,
    }
    assert {cell.jump_mode for cell in cells} == {"none"}
    assert {cell.uncertainty_mode for cell in cells} == {"deterministic"}

    triples = {
        (cell.cohort_seed, cell.subset_seed, cell.model_seed) for cell in cells
    }
    expected = {
        (401 + i, 501 + j, 601 + ((i + 2 * j) % 5))
        for i in range(5)
        for j in range(5)
    }
    assert triples == expected
    assert all(401 <= cohort <= 405 for cohort, _, _ in triples)
    assert all(501 <= subset <= 505 for _, subset, _ in triples)
    assert all(601 <= model <= 605 for _, _, model in triples)

    cohort_subset = Counter((cohort, subset) for cohort, subset, _ in triples)
    cohort_model = Counter((cohort, model) for cohort, _, model in triples)
    subset_model = Counter((subset, model) for _, subset, model in triples)
    assert len(cohort_subset) == 25 and set(cohort_subset.values()) == {1}
    assert len(cohort_model) == 25 and set(cohort_model.values()) == {1}
    assert len(subset_model) == 25 and set(subset_model.values()) == {1}


def test_d2b_cell_spec_and_store_accept_only_locked_d2_scope(tmp_path):
    cell = Phase06CellSpec(
        stage="d2b",
        world="smooth",
        cohort_seed=401,
        subset_seed=502,
        model_seed=603,
        n_train=40,
        flow_mode="time_scaled",
    )
    assert cell.cell_id.startswith("d2b__smooth__")

    lock = {
        "schema_version": 1,
        "phase06_config_sha256": "b" * 64,
        "execution_commit": "a" * 40,
    }
    protocol_hash = hashlib.sha256(_canonical_json_bytes(lock)).hexdigest()
    store = Phase06Store(
        tmp_path,
        protocol_hash=protocol_hash,
        config_hash="b" * 64,
        execution_commit="a" * 40,
    )
    store.write_protocol_lock(lock)
    observed = store.validate_resume("d2b", expected_cell_ids={cell.cell_id})
    assert observed == frozenset()
