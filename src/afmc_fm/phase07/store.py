from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from afmc_fm.phase07.planning import Phase07CellSpec

PHASE07_STORE_SCHEMA_VERSION = 1
_CELL_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_IDENTITY_SHA_KEYS = (
    "phase07_spec_sha256",
    "phase07_config_sha256",
    "phase05_config_sha256",
    "simulator_config_sha256",
    "protocol_lock_sha256",
    "phase07_plan_sha256",
)
_REQUIRED_BUNDLE_FILES = frozenset(
    {
        "cell.json",
        "metrics.csv",
        "training_trace.csv",
        "summary.json",
        "production_checkpoint.pt",
        "shadow_mae_checkpoint.pt",
        "artifact_manifest.json",
        "COMPLETE",
    }
)
_EVIDENCE_FILES = (
    "cell.json",
    "metrics.csv",
    "training_trace.csv",
    "summary.json",
    "production_checkpoint.pt",
    "shadow_mae_checkpoint.pt",
)


class Phase07Store:
    def __init__(
        self,
        output: str | Path,
        *,
        identity: Mapping[str, object],
    ) -> None:
        self.output = Path(output)
        self.identity = _validate_identity(identity)
        self._expected_root_bytes: dict[str, bytes] | None = None

    @property
    def stage_dir(self) -> Path:
        return self.output / "stages" / "phase07"

    @property
    def cells_dir(self) -> Path:
        return self.stage_dir / "cells"

    @property
    def staging_dir(self) -> Path:
        return self.stage_dir / ".tmp"

    def initialize(
        self,
        manifest: Mapping[str, object],
        protocol_lock: Mapping[str, object],
        plan_payload: Mapping[str, object],
        *,
        resume: bool,
    ) -> None:
        if type(resume) is not bool:
            raise TypeError("resume must be a bool")

        normalized_manifest = _normalize_json_mapping(manifest, "execution manifest")
        normalized_protocol = _normalize_json_mapping(protocol_lock, "protocol lock")
        normalized_plan = _normalize_json_mapping(plan_payload, "plan payload")
        self._validate_root_payloads(
            normalized_manifest,
            normalized_protocol,
            normalized_plan,
        )
        expected = {
            "execution_manifest.json": _canonical_json_bytes(normalized_manifest),
            "protocol_lock.json": _canonical_json_bytes(normalized_protocol),
            "plan.json": _canonical_json_bytes(normalized_plan),
        }

        if resume:
            if not self.output.is_dir():
                raise ValueError("Phase 0.7 resume output root does not exist")
            for name, data in expected.items():
                path = self.output / name
                if not path.is_file() or path.read_bytes() != data:
                    raise ValueError(f"Phase 0.7 persisted run identity mismatch: {name}")
            self._expected_root_bytes = expected
            self._validate_root_identity()
            self._discard_abandoned_staging()
        else:
            if self.output.exists():
                if not self.output.is_dir():
                    raise ValueError("Phase 0.7 output root must be a directory")
                if any(self.output.iterdir()):
                    raise ValueError("Phase 0.7 output root must be empty without resume")
            self.output.mkdir(parents=True, exist_ok=True)
            for name, data in expected.items():
                _atomic_write_bytes(self.output / name, data)
            self._expected_root_bytes = expected
            self._validate_root_identity()

        self.cells_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def write_cell_bundle(
        self,
        cell: Phase07CellSpec,
        *,
        metrics: pd.DataFrame,
        trace: pd.DataFrame,
        summary: Mapping[str, object],
        production_state_dict: Mapping[str, torch.Tensor],
        shadow_state_dict: Mapping[str, torch.Tensor],
    ) -> str:
        self._validate_root_identity()
        _require_cell(cell)
        metric_frame = _validate_metrics(metrics, cell)
        trace_frame = _validate_trace(trace)
        summary_payload = _validate_summary(summary, cell)
        production_state = _clone_state_dict(
            production_state_dict,
            "production_state_dict",
        )
        shadow_state = _clone_state_dict(
            shadow_state_dict,
            "shadow_state_dict",
        )

        final_dir = self.cells_dir / cell.cell_id
        if final_dir.exists():
            if not final_dir.is_dir():
                raise ValueError(f"conflicting authoritative cell path: {cell.cell_id}")
            self._validate_bundle_dir(final_dir, cell, require_complete=True)
            if self._existing_bundle_matches(
                final_dir,
                cell,
                metrics=metric_frame,
                trace=trace_frame,
                summary=summary_payload,
                production_state_dict=production_state,
                shadow_state_dict=shadow_state,
            ):
                return cell.cell_id
            raise ValueError(f"conflicting authoritative cell bundle: {cell.cell_id}")

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        staging = self.staging_dir / f"{cell.cell_id}.{uuid.uuid4().hex}"
        staging.mkdir(parents=False, exist_ok=False)
        try:
            _write_bytes_durable(
                staging / "cell.json",
                _canonical_json_bytes(
                    {
                        "schema_version": _CELL_SCHEMA_VERSION,
                        "identity": self.identity,
                        "cell": _cell_metadata(cell),
                    }
                ),
            )
            _write_bytes_durable(
                staging / "metrics.csv",
                _frame_csv_bytes(metric_frame),
            )
            _write_bytes_durable(
                staging / "training_trace.csv",
                _frame_csv_bytes(trace_frame),
            )
            _write_bytes_durable(
                staging / "summary.json",
                _canonical_json_bytes(summary_payload),
            )
            _write_checkpoint_durable(
                staging / "production_checkpoint.pt",
                production_state,
            )
            _write_checkpoint_durable(
                staging / "shadow_mae_checkpoint.pt",
                shadow_state,
            )

            artifact_hashes = {
                name: _sha256((staging / name).read_bytes()) for name in _EVIDENCE_FILES
            }
            _write_bytes_durable(
                staging / "artifact_manifest.json",
                _canonical_json_bytes(
                    {
                        "schema_version": _CELL_SCHEMA_VERSION,
                        "identity": self.identity,
                        "cell_id": cell.cell_id,
                        "artifact_sha256": artifact_hashes,
                    }
                ),
            )
            self._validate_bundle_dir(staging, cell, require_complete=False)
            _write_bytes_durable(
                staging / "COMPLETE",
                _canonical_json_bytes(
                    {
                        "schema_version": _CELL_SCHEMA_VERSION,
                        "phase": "phase07",
                        "identity": self.identity,
                        "cell_id": cell.cell_id,
                    }
                ),
            )
            self._validate_bundle_dir(staging, cell, require_complete=True)
            _fsync_dir(staging)
            os.replace(staging, final_dir)
            _fsync_dir(self.cells_dir)
            self._validate_bundle_dir(final_dir, cell, require_complete=True)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise
        return cell.cell_id

    def validate_resume(
        self,
        expected_cells: Iterable[Phase07CellSpec],
    ) -> frozenset[str]:
        self._validate_root_identity()
        expected = _expected_cells_by_id(expected_cells)
        observed = self._observed_cell_ids()
        unexpected = observed - set(expected)
        if unexpected:
            raise ValueError(
                "unexpected persisted cell IDs: " + ", ".join(sorted(unexpected))
            )
        for cell_id in sorted(observed):
            self._validate_bundle_dir(
                self.cells_dir / cell_id,
                expected[cell_id],
                require_complete=True,
            )

        marker = self.stage_dir / "COMPLETE"
        if marker.exists():
            if not marker.is_file():
                raise ValueError("conflicting Phase 0.7 completion marker")
            payload = _load_json_object(marker)
            expected_marker = self._completion_payload(expected)
            if payload != expected_marker or observed != set(expected):
                raise ValueError("Phase 0.7 completion marker does not match exact expected cell set")
        return frozenset(observed)

    def load_metrics(
        self,
        expected_cells: Iterable[Phase07CellSpec],
    ) -> pd.DataFrame:
        expected = _expected_cells_by_id(expected_cells)
        observed = self.validate_resume(expected.values())
        missing = set(expected) - set(observed)
        if missing:
            raise ValueError("missing expected cells: " + ", ".join(sorted(missing)))
        frames = [
            pd.read_csv(self.cells_dir / cell_id / "metrics.csv")
            for cell_id in sorted(expected)
        ]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    def mark_complete(
        self,
        expected_cells: Iterable[Phase07CellSpec],
    ) -> None:
        expected = _expected_cells_by_id(expected_cells)
        if not expected:
            raise ValueError("expected cells must not be empty")
        observed = self.validate_resume(expected.values())
        missing = set(expected) - set(observed)
        if missing:
            raise ValueError("missing expected cells: " + ", ".join(sorted(missing)))
        unexpected = set(observed) - set(expected)
        if unexpected:
            raise ValueError(
                "unexpected persisted cell IDs: " + ", ".join(sorted(unexpected))
            )

        payload = self._completion_payload(expected)
        data = _canonical_json_bytes(payload)
        marker = self.stage_dir / "COMPLETE"
        if marker.exists():
            if marker.is_file() and marker.read_bytes() == data:
                return
            raise ValueError("conflicting Phase 0.7 completion marker")
        _atomic_write_bytes(marker, data)
        _fsync_dir(self.stage_dir)
        self.require_complete(expected.values())

    def require_complete(
        self,
        expected_cells: Iterable[Phase07CellSpec],
    ) -> None:
        expected = _expected_cells_by_id(expected_cells)
        observed = self.validate_resume(expected.values())
        marker = self.stage_dir / "COMPLETE"
        if not marker.is_file():
            raise ValueError("Phase 0.7 official evidence is missing COMPLETE")
        if observed != frozenset(expected):
            raise ValueError("Phase 0.7 official evidence is not the exact completed cell set")
        if _load_json_object(marker) != self._completion_payload(expected):
            raise ValueError("Phase 0.7 completion marker identity mismatch")

    def _validate_root_payloads(
        self,
        manifest: dict[str, object],
        protocol_lock: dict[str, object],
        plan_payload: dict[str, object],
    ) -> None:
        if manifest.get("phase") != "phase07":
            raise ValueError("Phase 0.7 execution manifest identity mismatch")
        for key, expected in self.identity.items():
            if manifest.get(key) != expected:
                raise ValueError(f"Phase 0.7 execution manifest identity mismatch: {key}")

        protocol_bytes = _canonical_json_bytes(protocol_lock)
        if _sha256(protocol_bytes) != self.identity["protocol_lock_sha256"]:
            raise ValueError("Phase 0.7 protocol identity mismatch: protocol_lock_sha256")
        for key in (
            "execution_commit",
            "phase07_spec_sha256",
            "phase07_config_sha256",
            "expected_cell_count",
        ):
            if protocol_lock.get(key) != self.identity[key]:
                raise ValueError(f"Phase 0.7 protocol identity mismatch: {key}")

        if plan_payload.get("phase") != "phase07":
            raise ValueError("Phase 0.7 plan identity mismatch: phase")
        if plan_payload.get("phase07_plan_sha256") != self.identity["phase07_plan_sha256"]:
            raise ValueError("Phase 0.7 plan identity mismatch: phase07_plan_sha256")
        if plan_payload.get("expected_cell_count") != self.identity["expected_cell_count"]:
            raise ValueError("Phase 0.7 plan identity mismatch: expected_cell_count")
        cells = plan_payload.get("cells")
        if not isinstance(cells, list):
            raise TypeError("Phase 0.7 plan cells must be a list")

    def _validate_root_identity(self) -> None:
        if self._expected_root_bytes is None:
            raise ValueError("Phase 0.7 store is not initialized")
        for name, expected in self._expected_root_bytes.items():
            path = self.output / name
            if not path.is_file() or path.read_bytes() != expected:
                raise ValueError(f"Phase 0.7 persisted run identity mismatch: {name}")

        manifest = _load_json_object(self.output / "execution_manifest.json")
        for key, expected in self.identity.items():
            if manifest.get(key) != expected:
                raise ValueError(f"Phase 0.7 persisted run identity mismatch: {key}")
        protocol = _load_json_object(self.output / "protocol_lock.json")
        if _sha256(_canonical_json_bytes(protocol)) != self.identity["protocol_lock_sha256"]:
            raise ValueError("Phase 0.7 persisted protocol identity mismatch")

    def _discard_abandoned_staging(self) -> None:
        if not self.staging_dir.exists():
            return
        if not self.staging_dir.is_dir():
            raise ValueError("Phase 0.7 staging root is not a directory")
        for path in self.staging_dir.iterdir():
            if not path.is_dir():
                raise ValueError(f"unexpected Phase 0.7 staging artifact: {path}")
            shutil.rmtree(path)
        _fsync_dir(self.staging_dir)

    def _observed_cell_ids(self) -> set[str]:
        if not self.cells_dir.exists():
            return set()
        if not self.cells_dir.is_dir():
            raise ValueError("Phase 0.7 authoritative cells path is not a directory")
        observed: set[str] = set()
        for path in self.cells_dir.iterdir():
            if not path.is_dir():
                raise ValueError(f"unexpected authoritative Phase 0.7 artifact: {path}")
            if not path.name:
                raise ValueError("invalid authoritative Phase 0.7 cell ID")
            observed.add(path.name)
        return observed

    def _validate_bundle_dir(
        self,
        directory: Path,
        cell: Phase07CellSpec,
        *,
        require_complete: bool,
    ) -> None:
        if not directory.is_dir():
            raise ValueError(f"missing authoritative cell bundle: {cell.cell_id}")
        observed_files = {path.name for path in directory.iterdir() if path.is_file()}
        observed_dirs = [path.name for path in directory.iterdir() if path.is_dir()]
        if observed_dirs:
            raise ValueError(f"unexpected nested cell artifacts: {cell.cell_id}")
        required = set(_REQUIRED_BUNDLE_FILES)
        if not require_complete:
            required.remove("COMPLETE")
        missing = required - observed_files
        if missing:
            raise ValueError(
                f"incomplete persisted cell bundle: {cell.cell_id}: "
                + ", ".join(sorted(missing))
            )
        allowed = required if not require_complete else set(_REQUIRED_BUNDLE_FILES)
        extras = observed_files - allowed
        if extras:
            raise ValueError(
                f"unexpected persisted cell artifacts: {cell.cell_id}: "
                + ", ".join(sorted(extras))
            )

        cell_payload = _load_json_object(directory / "cell.json")
        expected_cell_payload = {
            "schema_version": _CELL_SCHEMA_VERSION,
            "identity": self.identity,
            "cell": _cell_metadata(cell),
        }
        if cell_payload != expected_cell_payload:
            raise ValueError(f"persisted cell identity mismatch: {cell.cell_id}")

        artifact_manifest = _load_json_object(directory / "artifact_manifest.json")
        if artifact_manifest.get("schema_version") != _CELL_SCHEMA_VERSION:
            raise ValueError(f"invalid artifact manifest schema: {cell.cell_id}")
        if artifact_manifest.get("identity") != self.identity:
            raise ValueError(f"artifact manifest identity mismatch: {cell.cell_id}")
        if artifact_manifest.get("cell_id") != cell.cell_id:
            raise ValueError(f"artifact manifest cell mismatch: {cell.cell_id}")
        hashes = artifact_manifest.get("artifact_sha256")
        if not isinstance(hashes, dict):
            raise TypeError(f"invalid artifact hash manifest: {cell.cell_id}")
        expected_hashes = {
            name: _sha256((directory / name).read_bytes()) for name in _EVIDENCE_FILES
        }
        if hashes != expected_hashes:
            raise ValueError(f"artifact hash mismatch: {cell.cell_id}")

        metrics = pd.read_csv(directory / "metrics.csv")
        _validate_metrics(metrics, cell)
        trace = pd.read_csv(directory / "training_trace.csv")
        _validate_trace(trace)
        summary = _load_json_object(directory / "summary.json")
        _validate_summary(summary, cell)
        _load_checkpoint(directory / "production_checkpoint.pt")
        _load_checkpoint(directory / "shadow_mae_checkpoint.pt")

        if require_complete:
            complete = _load_json_object(directory / "COMPLETE")
            expected_complete = {
                "schema_version": _CELL_SCHEMA_VERSION,
                "phase": "phase07",
                "identity": self.identity,
                "cell_id": cell.cell_id,
            }
            if complete != expected_complete:
                raise ValueError(f"cell completion marker mismatch: {cell.cell_id}")

    def _existing_bundle_matches(
        self,
        directory: Path,
        cell: Phase07CellSpec,
        *,
        metrics: pd.DataFrame,
        trace: pd.DataFrame,
        summary: dict[str, object],
        production_state_dict: Mapping[str, torch.Tensor],
        shadow_state_dict: Mapping[str, torch.Tensor],
    ) -> bool:
        if (directory / "metrics.csv").read_bytes() != _frame_csv_bytes(metrics):
            return False
        if (directory / "training_trace.csv").read_bytes() != _frame_csv_bytes(trace):
            return False
        if (directory / "summary.json").read_bytes() != _canonical_json_bytes(summary):
            return False
        if _load_json_object(directory / "cell.json").get("cell") != _cell_metadata(cell):
            return False
        return _state_dict_equal(
            _load_checkpoint(directory / "production_checkpoint.pt"),
            production_state_dict,
        ) and _state_dict_equal(
            _load_checkpoint(directory / "shadow_mae_checkpoint.pt"),
            shadow_state_dict,
        )

    def _completion_payload(
        self,
        expected: Mapping[str, Phase07CellSpec],
    ) -> dict[str, object]:
        return {
            "schema_version": PHASE07_STORE_SCHEMA_VERSION,
            "phase": "phase07",
            "identity": self.identity,
            "completed_cell_count": len(expected),
            "cell_ids": sorted(expected),
        }


def _validate_identity(identity: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(identity, Mapping):
        raise TypeError("Phase 0.7 store identity must be a mapping")
    required = {
        "execution_commit",
        *_IDENTITY_SHA_KEYS,
        "expected_cell_count",
    }
    if set(identity) != required:
        raise ValueError("Phase 0.7 store identity has unexpected or missing fields")
    execution_commit = identity["execution_commit"]
    if not isinstance(execution_commit, str) or _COMMIT_RE.fullmatch(execution_commit) is None:
        raise ValueError("execution_commit must be a 40-character hexadecimal SHA")
    normalized: dict[str, object] = {"execution_commit": execution_commit.lower()}
    for key in _IDENTITY_SHA_KEYS:
        value = identity[key]
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{key} must be a 64-character hexadecimal SHA-256")
        normalized[key] = value.lower()
    expected_count = identity["expected_cell_count"]
    if type(expected_count) is not int or expected_count != 200:
        raise ValueError("expected_cell_count must be exactly 200")
    normalized["expected_cell_count"] = expected_count
    return normalized


def _require_cell(cell: object) -> Phase07CellSpec:
    if not isinstance(cell, Phase07CellSpec):
        raise TypeError("cell must be a Phase07CellSpec")
    return cell


def _expected_cells_by_id(
    cells: Iterable[Phase07CellSpec],
) -> dict[str, Phase07CellSpec]:
    try:
        planned = tuple(cells)
    except TypeError as error:
        raise TypeError("expected cells must be iterable") from error
    expected: dict[str, Phase07CellSpec] = {}
    for cell in planned:
        _require_cell(cell)
        if cell.cell_id in expected:
            raise ValueError(f"duplicate expected cell ID: {cell.cell_id}")
        expected[cell.cell_id] = cell
    return expected


def _cell_metadata(cell: Phase07CellSpec) -> dict[str, object]:
    return {
        "world": cell.world,
        "cohort_seed": cell.cohort_seed,
        "subset_seed": cell.subset_seed,
        "model_seed": cell.model_seed,
        "n_train": cell.n_train,
        "flow_mode": cell.flow_mode,
        "optimization_policy": cell.optimization_policy,
        "cell_id": cell.cell_id,
    }


def _validate_metrics(metrics: pd.DataFrame, cell: Phase07CellSpec) -> pd.DataFrame:
    if not isinstance(metrics, pd.DataFrame):
        raise TypeError("metrics must be a pandas DataFrame")
    if metrics.empty:
        raise ValueError("metrics must contain at least one row")
    required = {
        "stage",
        "world",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "n_train",
        "variant",
        "optimization_policy",
        "split",
        "metric",
        "value",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError("metrics are missing columns: " + ", ".join(sorted(missing)))
    expected_variant = f"{cell.flow_mode}__none__deterministic"
    expected_values = {
        "stage": "phase07",
        "world": cell.world,
        "cohort_seed": cell.cohort_seed,
        "subset_seed": cell.subset_seed,
        "model_seed": cell.model_seed,
        "n_train": cell.n_train,
        "variant": expected_variant,
        "optimization_policy": cell.optimization_policy,
    }
    for column, expected in expected_values.items():
        if not (metrics[column] == expected).all():
            raise ValueError(f"metrics cell identity mismatch: {column}")
    numeric = pd.to_numeric(metrics["value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("metrics contain non-finite values")
    return metrics.copy()


def _validate_trace(trace: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(trace, pd.DataFrame):
        raise TypeError("trace must be a pandas DataFrame")
    if trace.empty:
        raise ValueError("trace must contain at least one row")
    if not len(trace.columns) or any(not isinstance(column, str) or not column for column in trace.columns):
        raise ValueError("trace columns must be non-empty strings")
    return trace.copy()


def _validate_summary(
    summary: Mapping[str, object],
    cell: Phase07CellSpec,
) -> dict[str, object]:
    normalized = _normalize_json_mapping(summary, "summary")
    if normalized.get("optimization_policy") != cell.optimization_policy:
        raise ValueError("summary optimization policy does not match cell")
    if "would_patience_exhaust_epoch" not in normalized:
        raise ValueError("summary is missing would_patience_exhaust_epoch")
    return normalized


def _clone_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    label: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"{label} must be a non-empty state dict")
    cloned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label} values must be tensors")
        cloned[key] = value.detach().cpu().clone()
    return cloned


def _state_dict_equal(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> bool:
    if set(left) != set(right):
        return False
    return all(
        isinstance(right[key], torch.Tensor)
        and left[key].dtype == right[key].dtype
        and left[key].shape == right[key].shape
        and torch.equal(left[key], right[key].detach().cpu())
        for key in left
    )


def _load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"invalid checkpoint artifact: {path}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"invalid checkpoint state dict: {path}")
    if any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in payload.items()
    ):
        raise ValueError(f"invalid checkpoint state dict: {path}")
    return payload


def _normalize_json_mapping(
    value: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    normalized = _normalize_json(dict(value))
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must normalize to a JSON object")
    return normalized


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
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("persisted JSON object keys must be strings")
        return {key: _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise TypeError(f"unsupported persisted JSON value: {type(value).__qualname__}")


def _frame_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


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
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return payload


def _write_checkpoint_durable(
    path: Path,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict), path)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    _load_checkpoint(path)


def _write_bytes_durable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != data:
        raise OSError(f"durable artifact validation failed: {path}")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_bytes_durable(temporary, data)
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = ["PHASE07_STORE_SCHEMA_VERSION", "Phase07Store"]
