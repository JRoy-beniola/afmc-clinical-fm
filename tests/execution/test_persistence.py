import json
from dataclasses import replace

import pandas as pd
import pytest

from afmc_fm.execution.jobs import CellResult, ShardSpec
from afmc_fm.execution.persistence import RunIdentity, RunStore

PROTOCOL_ANCHOR = "be5a66b2e45362f60c90844e4e25673fb7bb3e21"


def _identity(
    *,
    simulator_config_hash: str = "simulator-sha256",
    experiment_config_hash: str = "experiment-sha256",
) -> RunIdentity:
    return RunIdentity(
        protocol_anchor=PROTOCOL_ANCHOR,
        simulator_config_hash=simulator_config_hash,
        experiment_config_hash=experiment_config_hash,
    )


def _cell() -> CellResult:
    shard = ShardSpec("jumps", 101, 201, 301)
    return CellResult(
        shard=shard,
        benchmark="low_n",
        n_train=5,
        model="engineered_linear",
        ablation="none",
        metrics=pd.DataFrame(
            [
                {
                    "benchmark": "low_n",
                    "world": "jumps",
                    "cohort_seed": 101,
                    "subset_seed": 201,
                    "model_seed": 301,
                    "n_train": 5,
                    "model": "engineered_linear",
                    "ablation": "none",
                    "target": "value",
                    "metric": "rmse",
                    "value": 0.125,
                },
                {
                    "benchmark": "low_n",
                    "world": "jumps",
                    "cohort_seed": 101,
                    "subset_seed": 201,
                    "model_seed": 301,
                    "n_train": 5,
                    "model": "engineered_linear",
                    "ablation": "none",
                    "target": "event",
                    "metric": "auprc",
                    "value": 0.75,
                },
            ]
        ),
    )


def test_write_cell_atomically_persists_complete_metadata_and_all_metric_rows(tmp_path):
    store = RunStore(tmp_path / "run", _identity())

    cell_id = store.write_cell(_cell())

    assert cell_id == ("low_n__jumps__cohort101__subset201__model301__n5__engineered_linear__none")
    cells_dir = tmp_path / "run" / "shards" / "jumps__cohort101__subset201__model301" / "cells"
    assert [path.name for path in cells_dir.iterdir()] == [f"{cell_id}.json"]
    payload = json.loads((cells_dir / f"{cell_id}.json").read_text())
    assert payload == {
        "cell": {
            "ablation": "none",
            "benchmark": "low_n",
            "cell_id": cell_id,
            "model": "engineered_linear",
            "n_train": 5,
        },
        "metric_rows": _cell().metrics.to_dict(orient="records"),
        "run_identity": {
            "experiment_config_hash": "experiment-sha256",
            "protocol_anchor": PROTOCOL_ANCHOR,
            "simulator_config_hash": "simulator-sha256",
        },
        "schema_version": 1,
        "shard": {
            "cohort_seed": 101,
            "model_seed": 301,
            "shard_id": "jumps__cohort101__subset201__model301",
            "subset_seed": 201,
            "world": "jumps",
        },
    }


@pytest.mark.parametrize(
    ("changed_identity", "changed_field"),
    [
        (
            RunIdentity(
                protocol_anchor="different-scientific-anchor",
                simulator_config_hash="simulator-sha256",
                experiment_config_hash="experiment-sha256",
            ),
            "protocol_anchor",
        ),
        (
            _identity(simulator_config_hash="different-simulator-hash"),
            "simulator_config_hash",
        ),
        (
            _identity(experiment_config_hash="different-experiment-hash"),
            "experiment_config_hash",
        ),
    ],
)
def test_validate_resume_hard_fails_for_incompatible_scientific_identity(
    tmp_path, changed_identity, changed_field
):
    output = tmp_path / "run"
    RunStore(output, _identity()).write_cell(_cell())

    incompatible_store = RunStore(output, changed_identity)

    with pytest.raises(ValueError, match=f"incompatible run identity.*{changed_field}"):
        incompatible_store.validate_resume()


def test_validate_resume_hard_fails_for_incompatible_execution_schema(tmp_path):
    store = RunStore(tmp_path / "run", _identity())
    cell_id = store.write_cell(_cell())
    path = tmp_path / "run" / "shards" / _cell().shard.shard_id / "cells" / f"{cell_id}.json"
    payload = json.loads(path.read_text())
    path.write_text(json.dumps({**payload, "schema_version": 999}))

    with pytest.raises(ValueError, match="incompatible execution schema"):
        store.validate_resume()


def test_validate_resume_hard_fails_when_persisted_identity_has_unknown_fields(
    tmp_path,
):
    store = RunStore(tmp_path / "run", _identity())
    cell_id = store.write_cell(_cell())
    path = tmp_path / "run" / "shards" / _cell().shard.shard_id / "cells" / f"{cell_id}.json"
    payload = json.loads(path.read_text())
    payload["run_identity"]["unknown_scientific_hash"] = "cannot-safely-ignore"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="incompatible run identity"):
        store.validate_resume()


@pytest.mark.parametrize("malformed_json", ["null", "[]", '"scalar"'])
def test_validate_resume_treats_decodable_non_object_json_as_unfinished(tmp_path, malformed_json):
    cells_dir = tmp_path / "run" / "shards" / "interrupted" / "cells"
    cells_dir.mkdir(parents=True)
    (cells_dir / "malformed.json").write_text(malformed_json)
    store = RunStore(tmp_path / "run", _identity())

    store.validate_resume()

    assert store.load_completed_cell_ids() == frozenset()


def test_load_completed_cell_ids_counts_only_final_schema_and_identity_valid_json(tmp_path):
    store = RunStore(tmp_path / "run", _identity())
    valid_cell_id = store.write_cell(_cell())
    cells_dir = tmp_path / "run" / "shards" / "jumps__cohort101__subset201__model301" / "cells"
    valid_payload = json.loads((cells_dir / f"{valid_cell_id}.json").read_text())
    (cells_dir / "interrupted.json.tmp").write_text('{"schema_version": 1')
    (cells_dir / "partial.json.partial").write_text(json.dumps(valid_payload))
    (cells_dir / "corrupt.json").write_text("not JSON")
    (cells_dir / "truncated.json").write_text('{"schema_version": 1')
    wrong_schema = {**valid_payload, "schema_version": 999}
    (cells_dir / "wrong-schema.json").write_text(json.dumps(wrong_schema))
    wrong_identity = {
        **valid_payload,
        "run_identity": {
            **valid_payload["run_identity"],
            "experiment_config_hash": "another-experiment",
        },
    }
    (cells_dir / "wrong-identity.json").write_text(json.dumps(wrong_identity))

    assert store.load_completed_cell_ids() == frozenset({valid_cell_id})


@pytest.mark.parametrize(
    "inconsistency",
    [
        "filename",
        "cell_components",
        "shard_directory",
        "shard_components",
        "missing_metric_rows",
        "metric_cell_identity",
        "metric_shard_identity",
    ],
)
def test_load_completed_cell_ids_rejects_internally_inconsistent_cells(tmp_path, inconsistency):
    store = RunStore(tmp_path / "run", _identity())
    cell_id = store.write_cell(_cell())
    shard_id = _cell().shard.shard_id
    path = tmp_path / "run" / "shards" / shard_id / "cells" / f"{cell_id}.json"
    payload = json.loads(path.read_text())

    if inconsistency == "filename":
        path = path.replace(path.with_name("different-cell-id.json"))
    elif inconsistency == "cell_components":
        payload["cell"]["n_train"] = 10
    elif inconsistency == "shard_directory":
        other_dir = tmp_path / "run" / "shards" / "different-shard" / "cells"
        other_dir.mkdir(parents=True)
        path = path.replace(other_dir / path.name)
    elif inconsistency == "shard_components":
        payload["shard"]["cohort_seed"] = 999
    elif inconsistency == "missing_metric_rows":
        payload["metric_rows"] = []
    elif inconsistency == "metric_cell_identity":
        payload["metric_rows"][0]["model"] = "different_model"
    elif inconsistency == "metric_shard_identity":
        payload["metric_rows"][0]["world"] = "smooth"

    path.write_text(json.dumps(payload))

    assert store.load_completed_cell_ids() == frozenset()


def test_write_cell_validates_against_authoritative_cellresult_shard_before_promotion(
    tmp_path,
):
    store = RunStore(tmp_path / "run", _identity())
    cell = _cell()
    misleading_metrics = cell.metrics.copy()
    misleading_metrics["world"] = "smooth"
    misleading_metrics["cohort_seed"] = 999
    inconsistent_cell = replace(cell, metrics=misleading_metrics)
    authoritative_cell_id = (
        "low_n__jumps__cohort101__subset201__model301__n5__engineered_linear__none"
    )

    with pytest.raises(ValueError, match=authoritative_cell_id):
        store.write_cell(inconsistent_cell)

    cells_dir = tmp_path / "run" / "shards" / cell.shard.shard_id / "cells"
    assert not cells_dir.exists() or list(cells_dir.iterdir()) == []


def test_write_cell_refuses_to_overwrite_an_incompatible_run(tmp_path):
    output = tmp_path / "run"
    compatible_store = RunStore(output, _identity())
    cell_id = compatible_store.write_cell(_cell())
    final_path = output / "shards" / _cell().shard.shard_id / "cells" / f"{cell_id}.json"
    original_bytes = final_path.read_bytes()
    incompatible_store = RunStore(
        output,
        _identity(experiment_config_hash="incompatible-experiment"),
    )

    with pytest.raises(ValueError, match="incompatible run identity"):
        incompatible_store.write_cell(_cell())

    assert final_path.read_bytes() == original_bytes
    assert [path.name for path in final_path.parent.iterdir()] == [final_path.name]


def test_mark_shard_complete_requires_every_expected_cell_to_exist_and_validate(
    tmp_path,
):
    output = tmp_path / "run"
    store = RunStore(output, _identity())
    first = _cell()
    first_id = store.write_cell(first)
    second_metrics = first.metrics.copy()
    second_metrics["n_train"] = 10
    second = replace(first, n_train=10, metrics=second_metrics)
    second_id = "low_n__jumps__cohort101__subset201__model301__n10__engineered_linear__none"
    marker = output / "shards" / first.shard.shard_id / "COMPLETE"

    with pytest.raises(ValueError, match="missing or invalid.*n10"):
        store.mark_shard_complete(first.shard.shard_id, {first_id, second_id})
    assert not marker.exists()

    assert store.write_cell(second) == second_id
    second_path = marker.parent / "cells" / f"{second_id}.json"
    second_path.write_text('{"schema_version": 1')
    with pytest.raises(ValueError, match="missing or invalid.*n10"):
        store.mark_shard_complete(first.shard.shard_id, {first_id, second_id})
    assert not marker.exists()

    store.write_cell(second)
    store.mark_shard_complete(first.shard.shard_id, {first_id, second_id})

    assert marker.read_text() == "COMPLETE\n"


def test_iter_metric_rows_derives_rows_only_from_valid_persisted_cells(tmp_path):
    store = RunStore(tmp_path / "run", _identity())
    cell = _cell()
    cell_id = store.write_cell(cell)
    cells_dir = tmp_path / "run" / "shards" / cell.shard.shard_id / "cells"
    valid_payload = json.loads((cells_dir / f"{cell_id}.json").read_text())
    invalid_payload = {
        **valid_payload,
        "cell": {**valid_payload["cell"], "cell_id": "invented-cell"},
        "metric_rows": [{"metric": "invented", "value": -999.0}],
    }
    (cells_dir / "invented-cell.json").write_text(json.dumps(invalid_payload))
    (cells_dir / "truncated.json").write_text('{"metric_rows": [')

    assert list(store.iter_metric_rows()) == cell.metrics.to_dict(orient="records")
