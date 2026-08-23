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
                    "seed": 201,
                    "n_train": 5,
                    "n_fit": 4,
                    "n_validation": 1,
                    "model": "engineered_linear",
                    "ablation": "none",
                    "split": "test",
                    "site_or_shift": "all",
                    "metric": "rmse",
                    "value": 0.125,
                    "trainable_parameters": 17,
                    "backend": None,
                },
                {
                    "benchmark": "low_n",
                    "world": "jumps",
                    "cohort_seed": 101,
                    "subset_seed": 201,
                    "model_seed": 301,
                    "seed": 201,
                    "n_train": 5,
                    "n_fit": 4,
                    "n_validation": 1,
                    "model": "engineered_linear",
                    "ablation": "none",
                    "split": "test",
                    "site_or_shift": "all",
                    "metric": "auprc",
                    "value": 0.75,
                    "trainable_parameters": 17,
                    "backend": None,
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


@pytest.mark.parametrize("invalid_schema", [999, True])
def test_validate_resume_hard_fails_for_incompatible_execution_schema(tmp_path, invalid_schema):
    store = RunStore(tmp_path / "run", _identity())
    cell_id = store.write_cell(_cell())
    path = tmp_path / "run" / "shards" / _cell().shard.shard_id / "cells" / f"{cell_id}.json"
    payload = json.loads(path.read_text())
    path.write_text(json.dumps({**payload, "schema_version": invalid_schema}))

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


def test_load_completed_cell_ids_counts_only_final_valid_json_and_skips_corruption(tmp_path):
    store = RunStore(tmp_path / "run", _identity())
    valid_cell_id = store.write_cell(_cell())
    cells_dir = tmp_path / "run" / "shards" / "jumps__cohort101__subset201__model301" / "cells"
    valid_payload = json.loads((cells_dir / f"{valid_cell_id}.json").read_text())
    (cells_dir / "interrupted.json.tmp").write_text('{"schema_version": 1')
    (cells_dir / "partial.json.partial").write_text(json.dumps(valid_payload))
    (cells_dir / "corrupt.json").write_text("not JSON")
    (cells_dir / "truncated.json").write_text('{"schema_version": 1')

    assert store.load_completed_cell_ids() == frozenset({valid_cell_id})


@pytest.mark.parametrize("reader", ["load", "iterate"])
@pytest.mark.parametrize("mismatch", ["schema", "identity"])
def test_public_readers_hard_fail_instead_of_filtering_incompatible_cells(
    tmp_path, reader, mismatch
):
    store = RunStore(tmp_path / "run", _identity())
    cell_id = store.write_cell(_cell())
    path = tmp_path / "run" / "shards" / _cell().shard.shard_id / "cells" / f"{cell_id}.json"
    payload = json.loads(path.read_text())
    if mismatch == "schema":
        payload["schema_version"] = 999
        message = "incompatible execution schema"
    else:
        payload["run_identity"]["experiment_config_hash"] = "different"
        message = "incompatible run identity"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        if reader == "load":
            store.load_completed_cell_ids()
        else:
            list(store.iter_metric_rows())


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


@pytest.mark.parametrize(
    ("mutation", "replacement"),
    [
        ("remove_metric", None),
        ("add_unknown_key", None),
        ("cohort_seed", 101.0),
        ("n_validation", True),
        ("trainable_parameters", True),
        ("value", "0.125"),
        ("value", float("inf")),
        ("value", float("nan")),
        ("backend", 7),
    ],
)
def test_schema_invalid_metric_rows_never_count_or_feed_iteration(tmp_path, mutation, replacement):
    store = RunStore(tmp_path / "run", _identity())
    cell_id = store.write_cell(_cell())
    path = tmp_path / "run" / "shards" / _cell().shard.shard_id / "cells" / f"{cell_id}.json"
    payload = json.loads(path.read_text())
    row = payload["metric_rows"][0]
    if mutation == "remove_metric":
        del row["metric"]
    elif mutation == "add_unknown_key":
        row["unexpected"] = "not-in-schema"
    else:
        row[mutation] = replacement
    path.write_text(json.dumps(payload))

    assert store.load_completed_cell_ids() == frozenset()
    assert list(store.iter_metric_rows()) == []


def test_write_cell_preserves_float_precision_and_normalizes_nullable_fields(
    tmp_path,
):
    store = RunStore(tmp_path / "run", _identity())
    cell = _cell()
    metrics = cell.metrics.drop(columns=["backend"]).copy()
    precise_value = 0.12345678901234568
    metrics.loc[0, "value"] = precise_value
    metrics.loc[1, "value"] = float("nan")

    cell_id = store.write_cell(replace(cell, metrics=metrics))

    path = tmp_path / "run" / "shards" / cell.shard.shard_id / "cells" / f"{cell_id}.json"
    rows = json.loads(path.read_text())["metric_rows"]
    assert rows[0]["value"] == precise_value
    assert rows[1]["value"] is None
    assert [row["backend"] for row in rows] == [None, None]


@pytest.mark.parametrize("non_finite", [float("inf"), float("-inf")])
def test_write_cell_rejects_infinite_metric_values_without_artifacts(tmp_path, non_finite):
    store = RunStore(tmp_path / "run", _identity())
    cell = _cell()
    metrics = cell.metrics.copy()
    metrics.loc[0, "value"] = non_finite

    with pytest.raises(ValueError, match="infinite metric value"):
        store.write_cell(replace(cell, metrics=metrics))

    assert not (tmp_path / "run").exists()


def test_preflight_failure_cleans_owned_sibling_temporary_file(tmp_path):
    store = RunStore(tmp_path / "run", _identity())
    cell = _cell()
    cell_id = "low_n__jumps__cohort101__subset201__model301__n5__engineered_linear__none"
    cells_dir = tmp_path / "run" / "shards" / cell.shard.shard_id / "cells"
    cells_dir.mkdir(parents=True)
    temporary_path = cells_dir / f"{cell_id}.json.tmp"
    temporary_path.write_text("interrupted prior attempt")
    metrics = cell.metrics.copy()
    metrics.loc[0, "value"] = float("inf")

    with pytest.raises(ValueError, match="infinite metric value"):
        store.write_cell(replace(cell, metrics=metrics))

    assert not temporary_path.exists()


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

    assert json.loads(marker.read_text()) == {
        "cell_ids": sorted((first_id, second_id)),
        "run_identity": {
            "experiment_config_hash": "experiment-sha256",
            "protocol_anchor": PROTOCOL_ANCHOR,
            "simulator_config_hash": "simulator-sha256",
        },
        "schema_version": 1,
        "shard_id": first.shard.shard_id,
    }


def test_mark_shard_complete_requires_exact_nonempty_expected_cell_set(tmp_path):
    store = RunStore(tmp_path / "run", _identity())
    first = _cell()
    first_id = store.write_cell(first)
    second_metrics = first.metrics.copy()
    second_metrics["n_train"] = 10
    second_id = store.write_cell(replace(first, n_train=10, metrics=second_metrics))
    marker = tmp_path / "run" / "shards" / first.shard.shard_id / "COMPLETE"

    with pytest.raises(ValueError, match="expected/completed cell set mismatch"):
        store.mark_shard_complete(first.shard.shard_id, {first_id})
    assert not marker.exists()

    with pytest.raises(ValueError, match="non-empty"):
        store.mark_shard_complete(first.shard.shard_id, set())
    assert not marker.exists()
    assert second_id != first_id


@pytest.mark.parametrize("mutation", ["truncated", "schema", "identity"])
def test_revalidation_removes_stale_complete_marker_when_a_cell_becomes_invalid(tmp_path, mutation):
    store = RunStore(tmp_path / "run", _identity())
    cell = _cell()
    cell_id = store.write_cell(cell)
    marker = tmp_path / "run" / "shards" / cell.shard.shard_id / "COMPLETE"
    store.mark_shard_complete(cell.shard.shard_id, {cell_id})
    path = marker.parent / "cells" / f"{cell_id}.json"
    if mutation == "truncated":
        path.write_text('{"schema_version": 1')
    else:
        payload = json.loads(path.read_text())
        if mutation == "schema":
            payload["schema_version"] = 999
            message = "incompatible execution schema"
        else:
            payload["run_identity"]["experiment_config_hash"] = "different"
            message = "incompatible run identity"
        path.write_text(json.dumps(payload))

    if mutation == "truncated":
        store.validate_resume()
    else:
        with pytest.raises(ValueError, match=message):
            store.validate_resume()

    assert not marker.exists()


def test_identical_duplicate_cell_write_is_idempotent_and_preserves_completion(
    tmp_path,
):
    store = RunStore(tmp_path / "run", _identity())
    cell = _cell()
    cell_id = store.write_cell(cell)
    marker = tmp_path / "run" / "shards" / cell.shard.shard_id / "COMPLETE"
    path = marker.parent / "cells" / f"{cell_id}.json"
    store.mark_shard_complete(cell.shard.shard_id, {cell_id})
    original_inode = path.stat().st_ino
    original_marker = marker.read_bytes()

    assert store.write_cell(cell) == cell_id

    assert path.stat().st_ino == original_inode
    assert marker.read_bytes() == original_marker


def test_conflicting_duplicate_cell_write_is_rejected_without_replacing_result(
    tmp_path,
):
    store = RunStore(tmp_path / "run", _identity())
    cell = _cell()
    cell_id = store.write_cell(cell)
    marker = tmp_path / "run" / "shards" / cell.shard.shard_id / "COMPLETE"
    path = marker.parent / "cells" / f"{cell_id}.json"
    store.mark_shard_complete(cell.shard.shard_id, {cell_id})
    original_bytes = path.read_bytes()
    conflicting_metrics = cell.metrics.copy()
    conflicting_metrics.loc[0, "value"] = 999.0

    with pytest.raises(ValueError, match="conflicting persisted cell"):
        store.write_cell(replace(cell, metrics=conflicting_metrics))

    assert path.read_bytes() == original_bytes
    assert marker.exists()


def test_writing_a_new_cell_invalidates_prior_shard_completion(tmp_path):
    store = RunStore(tmp_path / "run", _identity())
    first = _cell()
    first_id = store.write_cell(first)
    marker = tmp_path / "run" / "shards" / first.shard.shard_id / "COMPLETE"
    store.mark_shard_complete(first.shard.shard_id, {first_id})
    second_metrics = first.metrics.copy()
    second_metrics["n_train"] = 10

    store.write_cell(replace(first, n_train=10, metrics=second_metrics))

    assert not marker.exists()


def test_public_shard_paths_reject_traversal_without_touching_outside_marker(
    tmp_path,
):
    output = tmp_path / "run"
    outside_marker = output / "escape" / "COMPLETE"
    outside_marker.parent.mkdir(parents=True)
    outside_marker.write_text("do-not-touch")
    store = RunStore(output, _identity())

    with pytest.raises(ValueError, match="canonical safe path segment"):
        store.mark_shard_complete("../escape", {"invented-cell"})
    with pytest.raises(ValueError, match="canonical safe path segment"):
        store.load_completed_cell_ids("../escape")

    assert outside_marker.read_text() == "do-not-touch"


def test_write_cell_rejects_nonportable_shard_path_segment_before_creating_output(
    tmp_path,
):
    store = RunStore(tmp_path / "run", _identity())
    cell = _cell()
    unsafe_shard = ShardSpec("bad\\world", 101, 201, 301)
    metrics = cell.metrics.copy()
    metrics["world"] = "bad\\world"

    with pytest.raises(ValueError, match="canonical safe path segment"):
        store.write_cell(replace(cell, shard=unsafe_shard, metrics=metrics))

    assert not (tmp_path / "run").exists()


def test_global_reader_rejects_symlinked_shard_outside_run_tree(tmp_path):
    store = RunStore(tmp_path / "run", _identity())
    cell = _cell()
    store.write_cell(cell)
    shard_dir = tmp_path / "run" / "shards" / cell.shard.shard_id
    outside_dir = tmp_path / "outside" / cell.shard.shard_id
    outside_dir.parent.mkdir()
    shard_dir.replace(outside_dir)
    shard_dir.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="resolve beneath output/shards"):
        store.load_completed_cell_ids()


def test_resume_validation_does_not_unlink_marker_through_external_shard_symlink(
    tmp_path,
):
    output = tmp_path / "run"
    shard_id = _cell().shard.shard_id
    outside_dir = tmp_path / "outside" / shard_id
    outside_dir.mkdir(parents=True)
    outside_marker = outside_dir / "COMPLETE"
    outside_marker.write_text("external sentinel")
    shards_dir = output / "shards"
    shards_dir.mkdir(parents=True)
    (shards_dir / shard_id).symlink_to(outside_dir, target_is_directory=True)
    store = RunStore(output, _identity())

    with pytest.raises(ValueError, match="resolve beneath output/shards"):
        store.validate_resume()

    assert outside_marker.read_text() == "external sentinel"


def test_global_reader_rejects_noncanonical_persisted_shard_directory(tmp_path):
    store = RunStore(tmp_path / "run", _identity())
    cell = _cell()
    store.write_cell(cell)
    shard_dir = tmp_path / "run" / "shards" / cell.shard.shard_id
    shard_dir.replace(shard_dir.with_name("bad\\world"))

    with pytest.raises(ValueError, match="canonical safe path segment"):
        store.load_completed_cell_ids()


@pytest.mark.parametrize("operation", ["write", "validate", "load"])
def test_entrypoints_reject_symlinked_shards_root_outside_output(tmp_path, operation):
    output = tmp_path / "run"
    output.mkdir()
    external_shards = tmp_path / "external-shards"
    external_shards.mkdir()
    (output / "shards").symlink_to(external_shards, target_is_directory=True)
    store = RunStore(output, _identity())

    with pytest.raises(ValueError, match="shards root must resolve directly beneath output"):
        if operation == "write":
            store.write_cell(_cell())
        elif operation == "validate":
            store.validate_resume()
        else:
            store.load_completed_cell_ids()

    assert list(external_shards.iterdir()) == []


def test_resume_validation_removes_marker_with_boolean_schema_version(tmp_path):
    store = RunStore(tmp_path / "run", _identity())
    cell = _cell()
    cell_id = store.write_cell(cell)
    marker = tmp_path / "run" / "shards" / cell.shard.shard_id / "COMPLETE"
    store.mark_shard_complete(cell.shard.shard_id, {cell_id})
    payload = json.loads(marker.read_text())
    payload["schema_version"] = True
    marker.write_text(json.dumps(payload))

    store.validate_resume()

    assert not marker.exists()


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


def test_iter_metric_rows_can_scope_aggregation_to_expected_cell_ids(tmp_path):
    store = RunStore(tmp_path / "run", _identity())
    first = _cell()
    first_id = store.write_cell(first)
    second_metrics = first.metrics.copy()
    second_metrics["n_train"] = 10
    store.write_cell(replace(first, n_train=10, metrics=second_metrics))

    rows = list(store.iter_metric_rows(frozenset({first_id})))

    assert rows == first.metrics.to_dict(orient="records")
