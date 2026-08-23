import hashlib
import json
import math
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

import pandas as pd

from afmc_fm.execution.jobs import CellResult, ShardSpec, benchmark_cell_id

CELL_SCHEMA_VERSION = 1
ROOT_RUN_SCHEMA_VERSION = 1
PROTOCOL_ANCHOR = "be5a66b2e45362f60c90844e4e25673fb7bb3e21"
METRIC_ROW_KEYS = frozenset(
    {
        "ablation",
        "backend",
        "benchmark",
        "cohort_seed",
        "metric",
        "model",
        "model_seed",
        "n_fit",
        "n_train",
        "n_validation",
        "seed",
        "site_or_shift",
        "split",
        "subset_seed",
        "trainable_parameters",
        "value",
        "world",
    }
)


@dataclass(frozen=True, slots=True)
class RunIdentity:
    protocol_anchor: str
    simulator_config_hash: str
    experiment_config_hash: str


def canonical_config_bytes(config: object) -> bytes:
    """Serialize recursively normalized config content using canonical JSON.

    Native ``Path`` values are rejected because their parsing depends on the
    host. Explicit ``PurePosixPath`` and ``PureWindowsPath`` values are tagged
    with their flavour, and all path text is emitted with ``/``.
    """
    if not (
        (is_dataclass(config) and not isinstance(config, type))
        or isinstance(config, Mapping)
    ):
        raise TypeError("config must be a dataclass instance or mapping")
    return json.dumps(
        _normalize_config_value(config),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_config_hash(config: object) -> str:
    """Return a stable SHA-256 content hash for supported configurations."""
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def _normalize_config_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalize_config_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("canonical config mapping keys must be strings")
        return {
            key: _normalize_config_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Path):
        raise TypeError(
            "native Path is host-dependent; use PurePosixPath or PureWindowsPath"
        )
    if isinstance(value, PurePath):
        flavour = "windows" if isinstance(value, PureWindowsPath) else "posix"
        return {
            "__afmc_type__": "path",
            "flavour": flavour,
            "value": value.as_posix(),
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_config_value(item) for item in value]
        normalized.sort(
            key=lambda item: json.dumps(
                item,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return {
            "__afmc_type__": "frozenset" if isinstance(value, frozenset) else "set",
            "items": normalized,
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value(item) for item in value]
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical config float values must be finite")
        return value
    raise TypeError(
        f"unsupported canonical config type: {type(value).__qualname__}"
    )


class RunStore:
    def __init__(self, output: Path, identity: RunIdentity) -> None:
        self.output = Path(output)
        self.identity = identity

    def initialize_run(
        self,
        expected_by_shard: Mapping[str, frozenset[str]],
        *,
        execution_commit_sha: str,
        original_started_at: datetime,
        resume: bool,
    ) -> dict[str, Any]:
        """Create or validate the immutable root identity and exact run plan."""
        _require_commit_sha(execution_commit_sha)
        root_path = self.output / "run_record.json"
        if resume:
            return self.load_run(expected_by_shard)

        if root_path.exists():
            raise ValueError("run root record already exists")
        expected_plan = _canonical_expected_plan(expected_by_shard)
        record = {
            "schema_version": ROOT_RUN_SCHEMA_VERSION,
            "run_identity": asdict(self.identity),
            "execution_commit_sha": execution_commit_sha,
            "expected_shards": expected_plan,
            "original_started_at": original_started_at.isoformat(),
            "completed_invocation_count": 0,
            "cumulative_wall_time_seconds": 0.0,
            "last_invocation": None,
        }
        self.output.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(root_path, record)
        return record

    def load_run(
        self,
        expected_by_shard: Mapping[str, frozenset[str]],
    ) -> dict[str, Any]:
        """Load and validate the immutable scientific identity and exact plan."""
        root_path = self.output / "run_record.json"
        if not root_path.is_file():
            raise ValueError("missing run root record for resume")
        record = _load_root_record(root_path)
        if record["schema_version"] != ROOT_RUN_SCHEMA_VERSION:
            raise ValueError("incompatible run root schema")
        if record["run_identity"] != asdict(self.identity):
            raise ValueError("incompatible run identity in root record")
        if record["expected_shards"] != _canonical_expected_plan(expected_by_shard):
            raise ValueError("incompatible run plan in root record")
        return record

    def complete_invocation(
        self,
        expected_by_shard: Mapping[str, frozenset[str]],
        *,
        execution_commit_sha: str,
        invocation_started_at: datetime,
        invocation_ended_at: datetime,
        invocation_wall_time_seconds: float,
        terminal_state: str,
    ) -> dict[str, Any]:
        """Atomically add one quiescent invocation to cumulative run timing."""
        if invocation_wall_time_seconds < 0:
            raise ValueError("invocation wall time must be non-negative")
        if invocation_ended_at < invocation_started_at:
            raise ValueError("invocation end must not precede start")
        if terminal_state not in {"completed", "completed_with_failures", "fail_fast"}:
            raise ValueError("unknown invocation terminal state")
        record = self.initialize_run(
            expected_by_shard,
            execution_commit_sha=execution_commit_sha,
            original_started_at=invocation_started_at,
            resume=True,
        )
        updated = {
            **record,
            "completed_invocation_count": record["completed_invocation_count"] + 1,
            "cumulative_wall_time_seconds": (
                float(record["cumulative_wall_time_seconds"])
                + invocation_wall_time_seconds
            ),
            "last_invocation": {
                "ended_at": invocation_ended_at.isoformat(),
                "started_at": invocation_started_at.isoformat(),
                "terminal_state": terminal_state,
                "wall_time_seconds": invocation_wall_time_seconds,
            },
        }
        _write_json_atomic(self.output / "run_record.json", updated)
        return updated

    def write_cell(self, result: CellResult) -> str:
        shard_dir = self._shard_dir(result.shard.shard_id)
        cell_id = result.cell_id
        _require_safe_segment(cell_id, "cell ID")
        cells_dir = shard_dir / "cells"
        final_path = cells_dir / f"{cell_id}.json"
        temporary_path = final_path.with_suffix(".json.tmp")
        expected_identity = asdict(self.identity)
        try:
            temporary_path.unlink(missing_ok=True)
            self._validated_completed_cell_ids_by_shard()
            payload = {
                "schema_version": CELL_SCHEMA_VERSION,
                "run_identity": expected_identity,
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
            if not _is_valid_payload(payload, final_path, expected_identity):
                raise ValueError(f"invalid persisted cell payload: {cell_id}")
            if final_path.exists():
                existing = _load_valid_payload(final_path, expected_identity)
                if existing == payload:
                    return cell_id
                if existing is not None:
                    raise ValueError(f"conflicting persisted cell: {cell_id}")
            (shard_dir / "COMPLETE").unlink(missing_ok=True)
            cells_dir.mkdir(parents=True, exist_ok=True)
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
        """Validate cells and reconcile markers at a quiescent run boundary."""
        self._validated_completed_cell_ids_by_shard(reconcile_markers=True)

    def _validated_completed_cell_ids_by_shard(
        self,
        *,
        reconcile_markers: bool = False,
    ) -> dict[str, frozenset[str]]:
        self._shards_root()
        expected_identity = asdict(self.identity)
        completed: dict[str, set[str]] = {}
        for path in self.output.glob("shards/*/cells/*.json"):
            self._require_contained(path)
            _require_safe_segment(path.parent.parent.name, "persisted shard ID")
            _require_safe_segment(path.stem, "persisted cell ID")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if (
                type(payload.get("schema_version")) is not int
                or payload["schema_version"] != CELL_SCHEMA_VERSION
            ):
                if reconcile_markers:
                    self._invalidate_marker_for_cell_path(path)
                raise ValueError(f"incompatible execution schema in persisted cell: {path}")
            actual_identity = payload.get("run_identity")
            if actual_identity != expected_identity:
                for field, expected_value in expected_identity.items():
                    if (
                        not isinstance(actual_identity, dict)
                        or actual_identity.get(field) != expected_value
                    ):
                        if reconcile_markers:
                            self._invalidate_marker_for_cell_path(path)
                        raise ValueError(
                            f"incompatible run identity in persisted cell: {field}"
                        )
                if reconcile_markers:
                    self._invalidate_marker_for_cell_path(path)
                raise ValueError(
                    "incompatible run identity in persisted cell: unknown fields"
                )
            if _is_valid_payload(payload, path, expected_identity):
                shard_id = payload["shard"]["shard_id"]
                completed.setdefault(shard_id, set()).add(payload["cell"]["cell_id"])
        snapshot = {
            shard_id: frozenset(cell_ids)
            for shard_id, cell_ids in completed.items()
        }
        if reconcile_markers:
            self._remove_stale_complete_markers(snapshot)
        return snapshot

    def load_completed_cell_ids(self, shard_id: str | None = None) -> frozenset[str]:
        if shard_id is not None:
            self._shard_dir(shard_id)
        snapshot = self._validated_completed_cell_ids_by_shard()
        if shard_id is not None:
            return snapshot.get(shard_id, frozenset())
        return frozenset().union(*snapshot.values()) if snapshot else frozenset()

    def load_completed_cell_ids_by_shard(self) -> dict[str, frozenset[str]]:
        return self._validated_completed_cell_ids_by_shard()

    def mark_shard_complete(
        self, shard_id: str, expected_cell_ids: set[str] | frozenset[str]
    ) -> None:
        shard_dir = self._shard_dir(shard_id)
        completed_by_shard = self._validated_completed_cell_ids_by_shard()
        marker = shard_dir / "COMPLETE"
        if not isinstance(expected_cell_ids, (set, frozenset)) or not expected_cell_ids:
            marker.unlink(missing_ok=True)
            raise ValueError("expected cell IDs must be a non-empty set")
        for cell_id in expected_cell_ids:
            _require_safe_segment(cell_id, "expected cell ID")
        completed = completed_by_shard.get(shard_id, frozenset())
        expected = frozenset(expected_cell_ids)
        if expected != completed:
            marker.unlink(missing_ok=True)
            missing_or_invalid = expected - completed
            if missing_or_invalid:
                missing = ", ".join(sorted(missing_or_invalid))
                raise ValueError(f"missing or invalid expected cells: {missing}")
            raise ValueError("expected/completed cell set mismatch")
        temporary_marker = marker.with_name("COMPLETE.tmp")
        payload = {
            "schema_version": CELL_SCHEMA_VERSION,
            "run_identity": asdict(self.identity),
            "shard_id": shard_id,
            "cell_ids": sorted(expected),
        }
        try:
            with temporary_marker.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, allow_nan=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary_marker.replace(marker)
        except Exception:
            temporary_marker.unlink(missing_ok=True)
            raise

    def _remove_stale_complete_markers(
        self,
        completed_by_shard: dict[str, frozenset[str]],
    ) -> None:
        expected_identity = asdict(self.identity)
        for marker in self.output.glob("shards/*/COMPLETE"):
            self._require_contained(marker)
            shard_id = marker.parent.name
            _require_safe_segment(shard_id, "persisted shard ID")
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                marker.unlink(missing_ok=True)
                continue
            valid_shape = (
                isinstance(payload, dict)
                and set(payload) == {"cell_ids", "run_identity", "schema_version", "shard_id"}
                and type(payload["schema_version"]) is int
                and payload["schema_version"] == CELL_SCHEMA_VERSION
                and payload["run_identity"] == expected_identity
                and payload["shard_id"] == shard_id
                and isinstance(payload["cell_ids"], list)
                and bool(payload["cell_ids"])
                and all(type(cell_id) is str for cell_id in payload["cell_ids"])
                and len(payload["cell_ids"]) == len(set(payload["cell_ids"]))
            )
            marker_cells = frozenset(payload["cell_ids"]) if valid_shape else frozenset()
            completed = completed_by_shard.get(shard_id, frozenset())
            if not valid_shape or marker_cells != completed:
                marker.unlink(missing_ok=True)

    def _shard_dir(self, shard_id: str) -> Path:
        _require_safe_segment(shard_id, "shard ID")
        shards_root = self._shards_root()
        shard_dir = (shards_root / shard_id).resolve()
        if shard_dir.parent != shards_root:
            raise ValueError("shard ID must resolve beneath output/shards")
        return shard_dir

    def _shards_root(self) -> Path:
        resolved_output = self.output.resolve()
        shards_root = (self.output / "shards").resolve()
        if shards_root.parent != resolved_output:
            raise ValueError("shards root must resolve directly beneath output")
        return shards_root

    def _require_contained(self, path: Path) -> None:
        shards_root = self._shards_root()
        try:
            path.resolve().relative_to(shards_root)
        except ValueError as exc:
            raise ValueError("persisted path must resolve beneath output/shards") from exc

    def _invalidate_marker_for_cell_path(self, path: Path) -> None:
        marker = path.parent.parent / "COMPLETE"
        self._require_contained(marker)
        marker.unlink(missing_ok=True)

    def iter_metric_rows(
        self,
        expected_cell_ids: frozenset[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        self._validated_completed_cell_ids_by_shard()
        expected_identity = asdict(self.identity)
        paths = sorted(self.output.glob("shards/*/cells/*.json"))
        for path in paths:
            payload = _load_valid_payload(path, expected_identity)
            if payload is not None and (
                expected_cell_ids is None
                or payload["cell"]["cell_id"] in expected_cell_ids
            ):
                yield from payload["metric_rows"]


def _metric_rows(result: CellResult) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for raw_row in result.metrics.to_dict(orient="records"):
        row = {field: _native_scalar(value) for field, value in raw_row.items()}
        row.setdefault("backend", None)
        value = row.get("value")
        if value is None or pd.isna(value):
            row["value"] = None
        elif type(value) in {int, float}:
            value = float(value)
            if math.isinf(value):
                raise ValueError("infinite metric value cannot be persisted")
            row["value"] = value
        backend = row.get("backend")
        if backend is not None and pd.isna(backend):
            row["backend"] = None
        normalized_rows.append(row)
    return normalized_rows


def _native_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


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

    expected_shard = ShardSpec(
        shard["world"],
        shard["cohort_seed"],
        shard["subset_seed"],
        shard["model_seed"],
    )
    expected_shard_id = expected_shard.shard_id
    if shard["shard_id"] != expected_shard_id:
        return False
    if path.parent.name != "cells" or path.parent.parent.name != expected_shard_id:
        return False
    expected_cell_id = benchmark_cell_id(
        cell["benchmark"],
        expected_shard,
        cell["n_train"],
        cell["model"],
        cell["ablation"],
    )
    if cell["cell_id"] != expected_cell_id or path.stem != expected_cell_id:
        return False

    expected_row_identity: dict[str, str | int] = {
        "benchmark": cell["benchmark"],
        "world": shard["world"],
        "cohort_seed": shard["cohort_seed"],
        "subset_seed": shard["subset_seed"],
        "model_seed": shard["model_seed"],
        "n_train": cell["n_train"],
        "model": cell["model"],
        "ablation": cell["ablation"],
    }
    return all(_is_valid_metric_row(row, expected_row_identity) for row in rows)


def _is_valid_metric_row(row: Any, expected_identity: dict[str, str | int]) -> bool:
    if not isinstance(row, dict) or set(row) != METRIC_ROW_KEYS:
        return False
    string_fields = (
        "ablation",
        "benchmark",
        "metric",
        "model",
        "site_or_shift",
        "split",
        "world",
    )
    if any(type(row[field]) is not str or not row[field] for field in string_fields):
        return False
    integer_fields = (
        "cohort_seed",
        "model_seed",
        "n_fit",
        "n_train",
        "n_validation",
        "seed",
        "subset_seed",
        "trainable_parameters",
    )
    if any(type(row[field]) is not int for field in integer_fields):
        return False
    if type(row["value"]) is not float and row["value"] is not None:
        return False
    if type(row["value"]) is float and not math.isfinite(row["value"]):
        return False
    if type(row["backend"]) is not str and row["backend"] is not None:
        return False
    if row["backend"] == "":
        return False
    if row["seed"] != row["subset_seed"]:
        return False
    return all(
        type(row[field]) is type(expected) and row[field] == expected
        for field, expected in expected_identity.items()
    )


def _require_safe_segment(value: Any, label: str) -> None:
    if (
        type(value) is not str
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
        or value in {".", ".."}
    ):
        raise ValueError(f"{label} must be one canonical safe path segment")


def _canonical_expected_plan(
    expected_by_shard: Mapping[str, frozenset[str]],
) -> list[dict[str, object]]:
    if not expected_by_shard:
        raise ValueError("expected run plan must contain at least one shard")
    plan: list[dict[str, object]] = []
    for shard_id, cell_ids in expected_by_shard.items():
        _require_safe_segment(shard_id, "expected shard ID")
        if not isinstance(cell_ids, frozenset) or not cell_ids:
            raise ValueError("expected run-plan cell IDs must be non-empty frozensets")
        for cell_id in cell_ids:
            _require_safe_segment(cell_id, "expected cell ID")
        plan.append({"shard_id": shard_id, "cell_ids": sorted(cell_ids)})
    return sorted(plan, key=lambda item: str(item["shard_id"]))


def _load_root_record(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid run root record") from error
    expected_keys = {
        "completed_invocation_count",
        "cumulative_wall_time_seconds",
        "execution_commit_sha",
        "expected_shards",
        "last_invocation",
        "original_started_at",
        "run_identity",
        "schema_version",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError("invalid run root record")
    if type(record["schema_version"]) is not int:
        raise ValueError("invalid run root record")
    if not isinstance(record["run_identity"], dict):
        raise TypeError("invalid run root record")
    if not isinstance(record["expected_shards"], list):
        raise TypeError("invalid run root record")
    if type(record["completed_invocation_count"]) is not int:
        raise ValueError("invalid run root record")
    if type(record["cumulative_wall_time_seconds"]) not in {int, float}:
        raise ValueError("invalid run root record")
    if (
        record["completed_invocation_count"] < 0
        or not math.isfinite(float(record["cumulative_wall_time_seconds"]))
        or record["cumulative_wall_time_seconds"] < 0
    ):
        raise ValueError("invalid run root record")
    if record["last_invocation"] is not None and not isinstance(
        record["last_invocation"], dict
    ):
        raise ValueError("invalid run root record")
    if isinstance(record["last_invocation"], dict) and set(
        record["last_invocation"]
    ) != {"ended_at", "started_at", "terminal_state", "wall_time_seconds"}:
        raise ValueError("invalid run root record")
    if not isinstance(record["original_started_at"], str):
        raise TypeError("invalid run root record")
    try:
        original_start = datetime.fromisoformat(record["original_started_at"])
    except ValueError as error:
        raise ValueError("invalid run root record") from error
    if original_start.tzinfo is None:
        raise ValueError("invalid run root record")
    _require_commit_sha(record["execution_commit_sha"])
    return record


def _require_commit_sha(value: object) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("execution commit must be a full lowercase Git SHA")


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
