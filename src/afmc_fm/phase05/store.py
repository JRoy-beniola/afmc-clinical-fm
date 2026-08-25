import csv
import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import load_phase05_config

PHASE05_STORE_SCHEMA_VERSION = 1
_CELL_SCHEMA_VERSION = 1
_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SAFE_ARTIFACT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_CONFIRMATORY_STAGES = frozenset({"confirmation", "robustness"})
_SPEC_RELATIVE_PATH = (
    Path("docs")
    / "superpowers"
    / "specs"
    / "2026-08-24-phase0-5-mechanistic-redesign-design.md"
)
_CONFIG_RELATIVE_PATH = Path("configs") / "experiments" / "phase05.yaml"


@dataclass(frozen=True, slots=True)
class Phase05CellResult:
    stage: str
    world: str
    cohort_seed: int
    subset_seed: int
    model_seed: int
    n_train: int
    model: str
    variant: str
    metrics: pd.DataFrame
    frozen_candidate_hash: str | None = None

    @property
    def seed_bundle(self) -> tuple[int, int, int]:
        return (self.cohort_seed, self.subset_seed, self.model_seed)

    @property
    def cell_id(self) -> str:
        return (
            f"{self.stage}__{self.world}__cohort{self.cohort_seed}__"
            f"subset{self.subset_seed}__model{self.model_seed}__n{self.n_train}__"
            f"{self.model}__{self.variant}"
        )


class Phase05Store:
    def __init__(
        self,
        output: Path,
        protocol_hash: str,
        *,
        spec_hash: str | None = None,
        config_hash: str | None = None,
    ) -> None:
        resolved_spec_hash = _default_spec_hash() if spec_hash is None else spec_hash
        resolved_config_hash = (
            _default_config_hash() if config_hash is None else config_hash
        )
        self.output = Path(output)
        self.protocol_hash = _require_sha256(protocol_hash, "protocol_hash")
        self.spec_hash = _require_sha256(resolved_spec_hash, "spec_hash")
        self.config_hash = _require_sha256(resolved_config_hash, "config_hash")

    @property
    def identity(self) -> dict[str, object]:
        return {
            "schema_version": PHASE05_STORE_SCHEMA_VERSION,
            "spec_sha256": self.spec_hash,
            "config_sha256": self.config_hash,
            "protocol_lock_sha256": self.protocol_hash,
        }

    def write_protocol_lock(self, lock: dict[str, object]) -> str:
        self._require_preconfirmation_mutation()
        normalized = _normalize_json(lock)
        if not isinstance(normalized, dict):
            raise TypeError("protocol lock must be a JSON object")
        data = _canonical_json_bytes(normalized)
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != self.protocol_hash:
            raise ValueError("protocol lock hash does not match store identity")
        config_hash = normalized.get("phase05_config_sha256")
        if config_hash is not None and config_hash != self.config_hash:
            raise ValueError("protocol lock config hash does not match store identity")

        path = self.output / "protocol_lock.json"
        if path.exists():
            if path.read_bytes() == data:
                return actual_hash
            raise ValueError("conflicting protocol lock")
        _atomic_write_bytes(path, data)
        return actual_hash

    def replace_development_artifact(self, name: str, payload: bytes) -> None:
        self._require_preconfirmation_mutation()
        _require_artifact_name(name)
        if not isinstance(payload, bytes):
            raise TypeError("development artifact payload must be bytes")
        path = self.output / "development" / name
        if path.exists() and path.read_bytes() == payload:
            return
        _atomic_write_bytes(
            path,
            payload,
            validator=_development_artifact_validator(name),
        )

    def write_frozen_candidate(self, candidate: dict[str, object]) -> str:
        self._require_preconfirmation_mutation()
        self._validate_protocol_identity()
        normalized = _normalize_json(candidate)
        if not isinstance(normalized, dict):
            raise TypeError("frozen candidate must be a JSON object")
        data = _canonical_json_bytes(normalized)
        candidate_hash = hashlib.sha256(data).hexdigest()
        path = self.output / "frozen_candidate.json"
        if path.exists():
            if path.read_bytes() == data:
                return candidate_hash
            raise ValueError("conflicting frozen candidate")
        _atomic_write_bytes(path, data)
        return candidate_hash

    def mark_confirmation_started(self) -> str:
        protocol_path = self.output / "protocol_lock.json"
        if not protocol_path.is_file():
            raise RuntimeError("protocol_lock.json is required before confirmation")
        candidate_path = self.output / "frozen_candidate.json"
        if not candidate_path.is_file():
            raise RuntimeError("frozen_candidate.json is required before confirmation")
        self._validate_protocol_identity()
        candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        path = self._confirmation_marker_path()
        if path.exists():
            self._validate_confirmation_marker(candidate_hash)
            return candidate_hash
        payload = {
            "identity": self.identity,
            "frozen_candidate_sha256": candidate_hash,
            "started_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write_bytes(path, _canonical_json_bytes(payload))
        return candidate_hash

    def write_cell(self, result: Phase05CellResult) -> str:
        self._validate_protocol_identity()
        self._validate_cell_result(result)
        if self._confirmation_marker_path().exists():
            if result.stage not in _CONFIRMATORY_STAGES:
                raise RuntimeError("confirmation has started; development cells are immutable")
        elif result.stage in _CONFIRMATORY_STAGES:
            raise RuntimeError("confirmation must be started before confirmatory cell writes")

        if result.stage in _CONFIRMATORY_STAGES:
            current_hash = self._current_frozen_candidate_hash()
            if result.frozen_candidate_hash != current_hash:
                raise ValueError("frozen candidate hash does not match persisted candidate")

        payload = self._cell_payload(result)
        stage_dir = self._stage_dir(result.stage)
        cells_dir = stage_dir / "cells"
        final_path = cells_dir / f"{result.cell_id}.json"
        serialized = _canonical_json_bytes(payload)
        if final_path.exists():
            existing = self._read_cell_payload(final_path, result.stage)
            if existing == payload:
                return result.cell_id
            raise ValueError(f"conflicting persisted cell: {result.cell_id}")

        (stage_dir / "COMPLETE").unlink(missing_ok=True)
        _atomic_write_bytes(final_path, serialized)
        persisted = self._read_cell_payload(final_path, result.stage)
        if persisted != payload:
            final_path.unlink(missing_ok=True)
            raise ValueError(f"invalid persisted cell payload: {result.cell_id}")
        return result.cell_id

    def validate_resume(
        self,
        stage: str,
        *,
        expected_cell_ids: set[str] | frozenset[str],
        expected_seed_bundles: set[tuple[int, int, int]]
        | frozenset[tuple[int, int, int]],
        frozen_candidate_hash: str | None = None,
    ) -> frozenset[str]:
        self._validate_protocol_identity()
        _require_safe_segment(stage, "stage")
        expected_cells = _validate_expected_cell_ids(expected_cell_ids)
        expected_bundles = _validate_seed_bundles(expected_seed_bundles)

        if stage in _CONFIRMATORY_STAGES:
            if not self._confirmation_marker_path().is_file():
                raise ValueError("confirmation start marker is missing")
            if frozen_candidate_hash is None:
                raise ValueError("frozen candidate hash is required for confirmatory resume")

        if frozen_candidate_hash is not None:
            _require_sha256(frozen_candidate_hash, "frozen_candidate_hash")
            actual_candidate_hash = self._current_frozen_candidate_hash()
            if actual_candidate_hash != frozen_candidate_hash:
                raise ValueError("frozen candidate hash does not match persisted candidate")
            self._validate_confirmation_marker(actual_candidate_hash)

        cells = self._load_stage_cells(stage)
        observed = frozenset(cells)
        unexpected = observed - expected_cells
        if unexpected:
            raise ValueError(
                "unexpected persisted cell IDs: " + ", ".join(sorted(unexpected))
            )

        for cell_id, payload in cells.items():
            bundle = _payload_seed_bundle(payload)
            if bundle not in expected_bundles:
                raise ValueError(f"unexpected seed bundle in persisted cell: {cell_id}")
            if frozen_candidate_hash is not None:
                cell_hash = payload["cell"]["frozen_candidate_hash"]
                if cell_hash != frozen_candidate_hash:
                    raise ValueError(
                        f"frozen candidate hash mismatch in persisted cell: {cell_id}"
                    )

        marker = self._stage_dir(stage) / "COMPLETE"
        if marker.exists():
            marker_payload = _load_json_object(marker)
            expected_marker = {
                "identity": self.identity,
                "stage": stage,
                "cell_ids": sorted(expected_cells),
            }
            if marker_payload != expected_marker or observed != expected_cells:
                raise ValueError("complete marker does not match exact expected cell set")
        return observed

    def mark_stage_complete(
        self,
        stage: str,
        expected_cell_ids: set[str] | frozenset[str],
    ) -> None:
        self._validate_protocol_identity()
        _require_safe_segment(stage, "stage")
        if self._confirmation_marker_path().exists() and stage not in _CONFIRMATORY_STAGES:
            raise RuntimeError("confirmation has started; development artifacts are immutable")
        expected = _validate_expected_cell_ids(expected_cell_ids)
        if not expected:
            raise ValueError("expected cell IDs must be non-empty")
        observed = frozenset(self._load_stage_cells(stage))
        if observed != expected:
            missing = expected - observed
            unexpected = observed - expected
            if missing:
                raise ValueError(
                    "missing expected cells: " + ", ".join(sorted(missing))
                )
            raise ValueError(
                "unexpected persisted cell IDs: " + ", ".join(sorted(unexpected))
            )
        payload = {
            "identity": self.identity,
            "stage": stage,
            "cell_ids": sorted(expected),
        }
        _atomic_write_bytes(
            self._stage_dir(stage) / "COMPLETE",
            _canonical_json_bytes(payload),
        )

    def _cell_payload(self, result: Phase05CellResult) -> dict[str, object]:
        rows = _normalize_json(result.metrics.to_dict(orient="records"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("cell metrics must contain at least one row")
        return {
            "schema_version": _CELL_SCHEMA_VERSION,
            "identity": self.identity,
            "stage": result.stage,
            "seed_bundle": {
                "cohort_seed": result.cohort_seed,
                "subset_seed": result.subset_seed,
                "model_seed": result.model_seed,
            },
            "cell": {
                "cell_id": result.cell_id,
                "n_train": result.n_train,
                "model": result.model,
                "variant": result.variant,
                "frozen_candidate_hash": result.frozen_candidate_hash,
            },
            "world": result.world,
            "metric_rows": rows,
        }

    def _validate_cell_result(self, result: Phase05CellResult) -> None:
        for name, value in (
            ("stage", result.stage),
            ("world", result.world),
            ("model", result.model),
            ("variant", result.variant),
        ):
            _require_safe_segment(value, name)
        for name, value in (
            ("cohort_seed", result.cohort_seed),
            ("subset_seed", result.subset_seed),
            ("model_seed", result.model_seed),
            ("n_train", result.n_train),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(result.metrics, pd.DataFrame):
            raise TypeError("metrics must be a pandas DataFrame")
        if result.frozen_candidate_hash is not None:
            _require_sha256(result.frozen_candidate_hash, "frozen_candidate_hash")

    def _load_stage_cells(self, stage: str) -> dict[str, dict[str, Any]]:
        _require_safe_segment(stage, "stage")
        cells_dir = self._stage_dir(stage) / "cells"
        if not cells_dir.exists():
            return {}
        cells: dict[str, dict[str, Any]] = {}
        for path in sorted(cells_dir.glob("*.json")):
            payload = self._read_cell_payload(path, stage)
            cell_id = payload["cell"]["cell_id"]
            if cell_id in cells:
                raise ValueError(f"duplicate persisted cell ID: {cell_id}")
            cells[cell_id] = payload
        return cells

    def _read_cell_payload(self, path: Path, stage: str) -> dict[str, Any]:
        payload = _load_json_object(path)
        if payload.get("schema_version") != _CELL_SCHEMA_VERSION:
            raise ValueError(f"incompatible Phase-0.5 cell schema: {path}")
        if payload.get("identity") != self.identity:
            raise ValueError(f"protocol identity mismatch in persisted cell: {path}")
        if payload.get("stage") != stage:
            raise ValueError(f"stage mismatch in persisted cell: {path}")
        cell = payload.get("cell")
        seed_bundle = payload.get("seed_bundle")
        if not isinstance(cell, dict) or not isinstance(seed_bundle, dict):
            raise TypeError(f"invalid persisted cell structure: {path}")
        required_cell = {
            "cell_id",
            "n_train",
            "model",
            "variant",
            "frozen_candidate_hash",
        }
        if set(cell) != required_cell:
            raise ValueError(f"invalid persisted cell metadata: {path}")
        if path.stem != cell["cell_id"]:
            raise ValueError(f"persisted cell filename mismatch: {path}")
        expected_id = _cell_id_from_payload(payload)
        if expected_id != cell["cell_id"]:
            raise ValueError(f"persisted cell identity mismatch: {path}")
        rows = payload.get("metric_rows")
        if not isinstance(rows, list) or not rows or not all(
            isinstance(row, dict) for row in rows
        ):
            raise ValueError(f"invalid persisted metric rows: {path}")
        return payload

    def _validate_protocol_identity(self) -> None:
        path = self.output / "protocol_lock.json"
        if not path.is_file():
            raise ValueError("protocol identity is missing protocol_lock.json")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.protocol_hash:
            raise ValueError("protocol identity mismatch: protocol lock hash")
        payload = _load_json_object(path)
        config_hash = payload.get("phase05_config_sha256")
        if config_hash is not None and config_hash != self.config_hash:
            raise ValueError("protocol identity mismatch: Phase-0.5 config hash")

    def _validate_confirmation_marker(self, candidate_hash: str) -> None:
        path = self._confirmation_marker_path()
        if not path.is_file():
            return
        payload = _load_json_object(path)
        if set(payload) != {"identity", "frozen_candidate_sha256", "started_at"}:
            raise ValueError("confirmation marker protocol identity mismatch")
        if payload.get("identity") != self.identity:
            raise ValueError("confirmation marker protocol identity mismatch")
        if payload.get("frozen_candidate_sha256") != candidate_hash:
            raise ValueError("confirmation marker protocol identity mismatch")
        started_at = payload.get("started_at")
        if not isinstance(started_at, str):
            raise ValueError("confirmation marker start timestamp is invalid")
        try:
            parsed = datetime.fromisoformat(started_at)
        except ValueError as error:
            raise ValueError("confirmation marker start timestamp is invalid") from error
        offset = parsed.utcoffset()
        if parsed.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError("confirmation marker start timestamp must be UTC")

    def _current_frozen_candidate_hash(self) -> str:
        path = self.output / "frozen_candidate.json"
        if not path.is_file():
            raise ValueError("frozen candidate hash unavailable: frozen_candidate.json missing")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _require_preconfirmation_mutation(self) -> None:
        if self._confirmation_marker_path().exists():
            raise RuntimeError("confirmation has started; frozen artifacts are immutable")

    def _confirmation_marker_path(self) -> Path:
        return self.output / "confirmation" / "STARTED"

    def _stage_dir(self, stage: str) -> Path:
        _require_safe_segment(stage, "stage")
        return self.output / "stages" / stage


def _cell_id_from_payload(payload: dict[str, Any]) -> str:
    bundle = payload.get("seed_bundle")
    cell = payload.get("cell")
    stage = payload.get("stage")
    world = payload.get("world")
    if not isinstance(bundle, dict) or not isinstance(cell, dict):
        raise TypeError("invalid persisted cell structure")
    try:
        cohort_seed = bundle["cohort_seed"]
        subset_seed = bundle["subset_seed"]
        model_seed = bundle["model_seed"]
        n_train = cell["n_train"]
        model = cell["model"]
        variant = cell["variant"]
    except KeyError as exc:
        raise ValueError("invalid persisted cell structure") from exc
    for name, value in (
        ("stage", stage),
        ("world", world),
        ("model", model),
        ("variant", variant),
    ):
        _require_safe_segment(value, name)
    for name, value in (
        ("cohort_seed", cohort_seed),
        ("subset_seed", subset_seed),
        ("model_seed", model_seed),
        ("n_train", n_train),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return (
        f"{stage}__{world}__cohort{cohort_seed}__subset{subset_seed}__"
        f"model{model_seed}__n{n_train}__{model}__{variant}"
    )


def _payload_seed_bundle(payload: dict[str, Any]) -> tuple[int, int, int]:
    bundle = payload["seed_bundle"]
    values = (
        bundle.get("cohort_seed"),
        bundle.get("subset_seed"),
        bundle.get("model_seed"),
    )
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("invalid seed bundle in persisted cell")
    return values


def _validate_expected_cell_ids(
    values: set[str] | frozenset[str],
) -> frozenset[str]:
    if not isinstance(values, (set, frozenset)):
        raise TypeError("expected cell IDs must be a set")
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("expected cell IDs must be non-empty strings")
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("unsafe expected cell ID")
    return frozenset(values)


def _validate_seed_bundles(
    values: set[tuple[int, int, int]] | frozenset[tuple[int, int, int]],
) -> frozenset[tuple[int, int, int]]:
    if not isinstance(values, (set, frozenset)):
        raise TypeError("expected seed bundles must be a set")
    normalized: set[tuple[int, int, int]] = set()
    for bundle in values:
        if not isinstance(bundle, tuple) or len(bundle) != 3:
            raise ValueError("seed bundles must be three-integer tuples")
        if any(type(value) is not int or value <= 0 for value in bundle):
            raise ValueError("seed bundles must contain positive integers")
        normalized.add(bundle)
    return frozenset(normalized)


def _require_safe_segment(value: object, name: str) -> str:
    if not isinstance(value, str) or _SAFE_SEGMENT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe non-empty path segment")
    return value


def _require_artifact_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or _SAFE_ARTIFACT_RE.fullmatch(value) is None
        or value in {".", ".."}
        or ".." in value
    ):
        raise ValueError("development artifact name must be a safe filename")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _normalize_json(value: object) -> object:
    if isinstance(value, np.generic):
        return _normalize_json(value.item())
    if value is pd.NA:
        return None
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError("non-finite values cannot be persisted")
        return value
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError("persisted JSON object keys must be strings")
        return {key: _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise TypeError(f"unsupported persisted JSON value: {type(value).__qualname__}")


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return payload


def _development_artifact_validator(name: str) -> Callable[[bytes], None] | None:
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return _validate_development_csv
    if suffix == ".json":
        return _validate_development_json
    return None


def _validate_development_csv(data: bytes) -> None:
    try:
        text = data.decode("utf-8")
        rows = list(csv.reader(text.splitlines(), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("invalid development CSV") from exc
    if not rows or not rows[0]:
        raise ValueError("invalid development CSV")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("invalid development CSV")


def _validate_development_json(data: bytes) -> None:
    try:
        json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid development JSON") from exc


def _atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    validator: Callable[[bytes], None] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.unlink(missing_ok=True)
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        persisted = temporary.read_bytes()
        if persisted != data:
            raise OSError(f"temporary artifact validation failed: {path}")
        if validator is not None:
            validator(persisted)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_spec_hash() -> str:
    path = _repo_root() / _SPEC_RELATIVE_PATH
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"approved Phase-0.5 spec is unavailable: {path}") from exc
    return hashlib.sha256(payload).hexdigest()


def _default_config_hash() -> str:
    path = _repo_root() / _CONFIG_RELATIVE_PATH
    try:
        config = load_phase05_config(path)
        return canonical_config_hash(config)
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"approved Phase-0.5 config is unavailable: {path}") from exc


__all__ = ["PHASE05_STORE_SCHEMA_VERSION", "Phase05CellResult", "Phase05Store"]
