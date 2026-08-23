import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from afmc_fm.execution.jobs import CellResult

CELL_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RunIdentity:
    protocol_anchor: str
    simulator_config_hash: str
    experiment_config_hash: str


class RunStore:
    def __init__(self, output: Path, identity: RunIdentity) -> None:
        self.output = Path(output)
        self.identity = identity

    def write_cell(self, result: CellResult) -> str:
        self.validate_resume()
        cell_id = _cell_id(result)
        cells_dir = self.output / "shards" / result.shard.shard_id / "cells"
        final_path = cells_dir / f"{cell_id}.json"
        temporary_path = final_path.with_suffix(".json.tmp")
        payload = {
            "schema_version": CELL_SCHEMA_VERSION,
            "run_identity": asdict(self.identity),
            "shard": {
                **asdict(result.shard),
                "shard_id": result.shard.shard_id,
            },
            "cell": {
                "cell_id": cell_id,
                "benchmark": result.benchmark,
                "n_train": result.n_train,
                "model": result.model,
                "ablation": result.ablation,
            },
            "metric_rows": _metric_rows(result),
        }
        expected_identity = asdict(self.identity)
        if not _is_valid_payload(payload, final_path, expected_identity):
            raise ValueError(f"invalid persisted cell payload: {cell_id}")
        cells_dir.mkdir(parents=True, exist_ok=True)
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, allow_nan=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            serialized = json.loads(temporary_path.read_text(encoding="utf-8"))
            if not _is_valid_payload(serialized, final_path, expected_identity):
                raise ValueError(f"invalid serialized cell payload: {cell_id}")
            temporary_path.replace(final_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return cell_id

    def validate_resume(self) -> None:
        expected_identity = asdict(self.identity)
        for path in self.output.glob("shards/*/cells/*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("schema_version") != CELL_SCHEMA_VERSION:
                raise ValueError(f"incompatible execution schema in persisted cell: {path}")
            actual_identity = payload.get("run_identity")
            if actual_identity == expected_identity:
                continue
            for field, expected_value in expected_identity.items():
                if (
                    not isinstance(actual_identity, dict)
                    or actual_identity.get(field) != expected_value
                ):
                    raise ValueError(f"incompatible run identity in persisted cell: {field}")
            raise ValueError("incompatible run identity in persisted cell: unknown fields")

    def load_completed_cell_ids(self, shard_id: str | None = None) -> frozenset[str]:
        if shard_id is None:
            paths = self.output.glob("shards/*/cells/*.json")
        else:
            paths = (self.output / "shards" / shard_id / "cells").glob("*.json")
        completed: set[str] = set()
        expected_identity = asdict(self.identity)
        for path in paths:
            payload = _load_valid_payload(path, expected_identity)
            if payload is not None:
                completed.add(payload["cell"]["cell_id"])
        return frozenset(completed)

    def mark_shard_complete(
        self, shard_id: str, expected_cell_ids: set[str] | frozenset[str]
    ) -> None:
        self.validate_resume()
        completed = self.load_completed_cell_ids(shard_id)
        missing_or_invalid = set(expected_cell_ids) - completed
        if missing_or_invalid:
            missing = ", ".join(sorted(missing_or_invalid))
            raise ValueError(f"missing or invalid expected cells: {missing}")
        marker = self.output / "shards" / shard_id / "COMPLETE"
        temporary_marker = marker.with_name("COMPLETE.tmp")
        try:
            with temporary_marker.open("w", encoding="utf-8") as handle:
                handle.write("COMPLETE\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary_marker.replace(marker)
        except Exception:
            temporary_marker.unlink(missing_ok=True)
            raise

    def iter_metric_rows(self) -> Iterator[dict[str, Any]]:
        expected_identity = asdict(self.identity)
        paths = sorted(self.output.glob("shards/*/cells/*.json"))
        for path in paths:
            payload = _load_valid_payload(path, expected_identity)
            if payload is not None:
                yield from payload["metric_rows"]


def _cell_id(result: CellResult) -> str:
    return _cell_id_from_components(
        result.benchmark,
        result.shard.shard_id,
        result.n_train,
        result.model,
        result.ablation,
    )


def _metric_rows(result: CellResult) -> list[dict[str, Any]]:
    return json.loads(result.metrics.to_json(orient="records", date_format="iso"))


def _load_valid_payload(path: Path, expected_identity: dict[str, str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not _is_valid_payload(payload, path, expected_identity):
        return None
    return payload


def _is_valid_payload(payload: Any, path: Path, expected_identity: dict[str, str]) -> bool:
    if not isinstance(payload, dict):
        return False
    if type(payload.get("schema_version")) is not int:
        return False
    if payload["schema_version"] != CELL_SCHEMA_VERSION:
        return False
    if payload.get("run_identity") != expected_identity:
        return False

    shard = payload.get("shard")
    cell = payload.get("cell")
    rows = payload.get("metric_rows")
    if not isinstance(shard, dict) or not isinstance(cell, dict):
        return False
    if not isinstance(rows, list) or not rows:
        return False
    string_fields = (
        shard.get("world"),
        shard.get("shard_id"),
        cell.get("benchmark"),
        cell.get("model"),
        cell.get("ablation"),
        cell.get("cell_id"),
    )
    if any(not isinstance(value, str) or not value for value in string_fields):
        return False
    integer_fields = (
        shard.get("cohort_seed"),
        shard.get("subset_seed"),
        shard.get("model_seed"),
        cell.get("n_train"),
    )
    if any(type(value) is not int for value in integer_fields):
        return False

    expected_shard_id = (
        f"{shard['world']}__cohort{shard['cohort_seed']}"
        f"__subset{shard['subset_seed']}__model{shard['model_seed']}"
    )
    if shard["shard_id"] != expected_shard_id:
        return False
    if path.parent.name != "cells" or path.parent.parent.name != expected_shard_id:
        return False
    expected_cell_id = _cell_id_from_components(
        cell["benchmark"],
        expected_shard_id,
        cell["n_train"],
        cell["model"],
        cell["ablation"],
    )
    if cell["cell_id"] != expected_cell_id or path.stem != expected_cell_id:
        return False

    expected_row_identity = {
        "benchmark": cell["benchmark"],
        "world": shard["world"],
        "cohort_seed": shard["cohort_seed"],
        "subset_seed": shard["subset_seed"],
        "model_seed": shard["model_seed"],
        "n_train": cell["n_train"],
        "model": cell["model"],
        "ablation": cell["ablation"],
    }
    return not any(
        not isinstance(row, dict)
        or any(row.get(field) != value for field, value in expected_row_identity.items())
        for row in rows
    )


def _cell_id_from_components(
    benchmark: str, shard_id: str, n_train: int, model: str, ablation: str
) -> str:
    return f"{benchmark}__{shard_id}__n{n_train}__{model}__{ablation}"
