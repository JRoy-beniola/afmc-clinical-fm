from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from afmc_fm.phase06.planning import Phase06CellSpec

PHASE06_STORE_SCHEMA_VERSION = 1
_CELL_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_ALLOWED_STAGES = frozenset({"d1", "d2a"})
_RESERVED_SUMMARY_KEYS = frozenset({"artifact_sha256", "identity"})


class Phase06Store:
    def __init__(
        self,
        output: str | Path,
        *,
        protocol_hash: str,
        config_hash: str,
        execution_commit: str,
    ) -> None:
        self.output = Path(output)
        self.protocol_hash = _require_sha256(protocol_hash, "protocol_hash")
        self.config_hash = _require_sha256(config_hash, "config_hash")
        self.execution_commit = _require_commit(execution_commit)

    @property
    def identity(self) -> dict[str, str]:
        return {
            "protocol_lock_sha256": self.protocol_hash,
            "phase06_config_sha256": self.config_hash,
            "execution_commit": self.execution_commit,
        }

    def write_protocol_lock(self, lock: dict[str, object]) -> str:
        normalized = _normalize_json(lock)
        if not isinstance(normalized, dict):
            raise TypeError("protocol lock must be a JSON object")
        data = _canonical_json_bytes(normalized)
        actual_hash = _sha256(data)
        if actual_hash != self.protocol_hash:
            raise ValueError("protocol lock hash does not match store identity")
        if normalized.get("phase06_config_sha256") != self.config_hash:
            raise ValueError("protocol lock config hash does not match store identity")
        if normalized.get("execution_commit") != self.execution_commit:
            raise ValueError("protocol lock execution commit does not match store identity")

        path = self.output / "protocol_lock.json"
        if path.exists():
            if path.is_file() and path.read_bytes() == data:
                return actual_hash
            raise ValueError("conflicting protocol lock")
        _atomic_write_bytes(path, data)
        return actual_hash

    def write_cell_bundle(
        self,
        cell: Phase06CellSpec,
        *,
        metrics: pd.DataFrame,
        trace: pd.DataFrame,
        summary: Mapping[str, object],
        production_state_dict: Mapping[str, torch.Tensor],
        shadow_state_dict: Mapping[str, torch.Tensor],
    ) -> str:
        self._validate_protocol_identity()
        _validate_cell(cell)
        metric_rows = _frame_records(metrics, "metrics")
        trace_frame = _validate_trace(trace)
        summary_payload = _validate_summary(summary)
        production_state = _clone_state_dict(production_state_dict, "production_state_dict")
        shadow_state = _clone_state_dict(shadow_state_dict, "shadow_state_dict")

        paths = self._bundle_paths(cell)
        existing = {name: path.exists() for name, path in paths.items()}
        if any(existing.values()):
            if not all(existing.values()):
                raise ValueError(f"conflicting persisted cell bundle: {cell.cell_id}")
            if self._existing_bundle_matches(
                cell,
                metric_rows=metric_rows,
                trace=trace_frame,
                summary=summary_payload,
                production_state_dict=production_state,
                shadow_state_dict=shadow_state,
            ):
                return cell.cell_id
            raise ValueError(f"conflicting persisted cell bundle: {cell.cell_id}")

        cell_payload = {
            "schema_version": _CELL_SCHEMA_VERSION,
            "identity": self.identity,
            "cell": _cell_metadata(cell),
            "metric_rows": metric_rows,
        }
        cell_bytes = _canonical_json_bytes(cell_payload)
        trace_bytes = _frame_csv_bytes(trace_frame)

        checkpoint_temps: dict[str, Path] = {}
        byte_temps: dict[str, Path] = {}
        committed: list[Path] = []
        try:
            checkpoint_temps["production_checkpoint"] = _write_checkpoint_temp(
                paths["production_checkpoint"], production_state
            )
            checkpoint_temps["shadow_mae_checkpoint"] = _write_checkpoint_temp(
                paths["shadow_mae_checkpoint"], shadow_state
            )
            production_hash = _sha256(
                checkpoint_temps["production_checkpoint"].read_bytes()
            )
            shadow_hash = _sha256(
                checkpoint_temps["shadow_mae_checkpoint"].read_bytes()
            )
            manifest = dict(summary_payload)
            manifest["artifact_sha256"] = {
                "cell_metrics": _sha256(cell_bytes),
                "training_trace": _sha256(trace_bytes),
                "production_checkpoint": production_hash,
                "shadow_mae_checkpoint": shadow_hash,
            }
            manifest["identity"] = self.identity
            summary_bytes = _canonical_json_bytes(manifest)

            for name, data in (
                ("cell_metrics", cell_bytes),
                ("training_trace", trace_bytes),
                ("summary", summary_bytes),
            ):
                byte_temps[name] = _write_bytes_temp(paths[name], data)

            (self._stage_dir(cell.stage) / "COMPLETE").unlink(missing_ok=True)
            for name in (
                "cell_metrics",
                "training_trace",
                "production_checkpoint",
                "shadow_mae_checkpoint",
                "summary",
            ):
                temporary = (
                    checkpoint_temps[name]
                    if name in checkpoint_temps
                    else byte_temps[name]
                )
                os.replace(temporary, paths[name])
                committed.append(paths[name])

            self._validate_persisted_bundle(cell)
        except Exception:
            for temporary in (*checkpoint_temps.values(), *byte_temps.values()):
                temporary.unlink(missing_ok=True)
            for path in committed:
                path.unlink(missing_ok=True)
            raise
        return cell.cell_id

    def validate_resume(
        self,
        stage: str,
        *,
        expected_cell_ids: set[str] | frozenset[str],
    ) -> frozenset[str]:
        self._validate_protocol_identity()
        stage = _require_stage(stage)
        expected = _validate_expected_cell_ids(expected_cell_ids)
        observed = self._observed_cell_ids(stage)
        unexpected = observed - expected
        if unexpected:
            raise ValueError(
                "unexpected persisted cell IDs: " + ", ".join(sorted(unexpected))
            )

        for cell_id in sorted(observed):
            payload = self._read_cell_payload(stage, cell_id)
            cell = _cell_from_payload(payload)
            self._validate_persisted_bundle(cell)

        marker = self._stage_dir(stage) / "COMPLETE"
        if marker.exists():
            marker_payload = _load_json_object(marker)
            expected_marker = {
                "schema_version": PHASE06_STORE_SCHEMA_VERSION,
                "identity": self.identity,
                "stage": stage,
                "cell_ids": sorted(expected),
            }
            if marker_payload != expected_marker or observed != expected:
                raise ValueError("complete marker does not match exact expected cell set")
        return observed

    def mark_stage_complete(
        self,
        stage: str,
        expected_cell_ids: set[str] | frozenset[str],
    ) -> None:
        stage = _require_stage(stage)
        expected = _validate_expected_cell_ids(expected_cell_ids)
        if not expected:
            raise ValueError("expected cell IDs must be non-empty")
        observed = self.validate_resume(stage, expected_cell_ids=expected)
        if observed != expected:
            missing = expected - observed
            if missing:
                raise ValueError("missing expected cells: " + ", ".join(sorted(missing)))
            unexpected = observed - expected
            raise ValueError(
                "unexpected persisted cell IDs: " + ", ".join(sorted(unexpected))
            )
        payload = {
            "schema_version": PHASE06_STORE_SCHEMA_VERSION,
            "identity": self.identity,
            "stage": stage,
            "cell_ids": sorted(expected),
        }
        data = _canonical_json_bytes(payload)
        marker = self._stage_dir(stage) / "COMPLETE"
        if marker.exists():
            if marker.is_file() and marker.read_bytes() == data:
                return
            raise ValueError("conflicting stage completion marker")
        _atomic_write_bytes(marker, data)

    def load_stage_metrics(self, stage: str) -> pd.DataFrame:
        self._validate_protocol_identity()
        stage = _require_stage(stage)
        frames: list[pd.DataFrame] = []
        for cell_id in sorted(self._observed_cell_ids(stage)):
            payload = self._read_cell_payload(stage, cell_id)
            cell = _cell_from_payload(payload)
            self._validate_persisted_bundle(cell)
            rows = payload.get("metric_rows")
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"invalid persisted metric rows: {cell_id}")
            frames.append(pd.DataFrame(rows))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    def _existing_bundle_matches(
        self,
        cell: Phase06CellSpec,
        *,
        metric_rows: list[dict[str, object]],
        trace: pd.DataFrame,
        summary: dict[str, object],
        production_state_dict: dict[str, torch.Tensor],
        shadow_state_dict: dict[str, torch.Tensor],
    ) -> bool:
        try:
            self._validate_persisted_bundle(cell)
            paths = self._bundle_paths(cell)
            cell_payload = _load_json_object(paths["cell_metrics"])
            expected_cell_payload = {
                "schema_version": _CELL_SCHEMA_VERSION,
                "identity": self.identity,
                "cell": _cell_metadata(cell),
                "metric_rows": metric_rows,
            }
            if cell_payload != expected_cell_payload:
                return False
            if paths["training_trace"].read_bytes() != _frame_csv_bytes(trace):
                return False
            persisted_summary = _load_json_object(paths["summary"])
            semantic_summary = {
                key: value
                for key, value in persisted_summary.items()
                if key not in _RESERVED_SUMMARY_KEYS
            }
            if semantic_summary != summary:
                return False
            persisted_production = _load_checkpoint(paths["production_checkpoint"])
            persisted_shadow = _load_checkpoint(paths["shadow_mae_checkpoint"])
            return _state_dict_equal(persisted_production, production_state_dict) and _state_dict_equal(
                persisted_shadow, shadow_state_dict
            )
        except (OSError, TypeError, ValueError, RuntimeError):
            return False

    def _validate_persisted_bundle(self, cell: Phase06CellSpec) -> None:
        paths = self._bundle_paths(cell)
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise ValueError(
                f"incomplete persisted cell bundle: {cell.cell_id}: " + ", ".join(missing)
            )
        cell_payload = _load_json_object(paths["cell_metrics"])
        if cell_payload.get("schema_version") != _CELL_SCHEMA_VERSION:
            raise ValueError(f"incompatible Phase 0.6 cell schema: {cell.cell_id}")
        if cell_payload.get("identity") != self.identity:
            raise ValueError(f"protocol identity mismatch in persisted cell: {cell.cell_id}")
        if cell_payload.get("cell") != _cell_metadata(cell):
            raise ValueError(f"persisted cell identity mismatch: {cell.cell_id}")
        rows = cell_payload.get("metric_rows")
        if not isinstance(rows, list) or not rows or not all(
            isinstance(row, dict) for row in rows
        ):
            raise ValueError(f"invalid persisted metric rows: {cell.cell_id}")

        manifest = _load_json_object(paths["summary"])
        if manifest.get("identity") != self.identity:
            raise ValueError(f"protocol identity mismatch in persisted summary: {cell.cell_id}")
        hashes = manifest.get("artifact_sha256")
        if not isinstance(hashes, dict):
            raise ValueError(f"invalid artifact hash manifest: {cell.cell_id}")
        expected_hashes = {
            "cell_metrics": _sha256(paths["cell_metrics"].read_bytes()),
            "training_trace": _sha256(paths["training_trace"].read_bytes()),
            "production_checkpoint": _sha256(paths["production_checkpoint"].read_bytes()),
            "shadow_mae_checkpoint": _sha256(paths["shadow_mae_checkpoint"].read_bytes()),
        }
        if hashes != expected_hashes:
            raise ValueError(f"artifact hash mismatch: {cell.cell_id}")
        _load_checkpoint(paths["production_checkpoint"])
        _load_checkpoint(paths["shadow_mae_checkpoint"])

    def _validate_protocol_identity(self) -> None:
        path = self.output / "protocol_lock.json"
        if not path.is_file():
            raise ValueError("protocol identity is missing protocol_lock.json")
        data = path.read_bytes()
        if _sha256(data) != self.protocol_hash:
            raise ValueError("protocol identity mismatch: protocol lock hash")
        payload = _load_json_object(path)
        if payload.get("phase06_config_sha256") != self.config_hash:
            raise ValueError("protocol identity mismatch: Phase 0.6 config hash")
        if payload.get("execution_commit") != self.execution_commit:
            raise ValueError("protocol identity mismatch: execution commit")

    def _read_cell_payload(self, stage: str, cell_id: str) -> dict[str, Any]:
        path = self._stage_dir(stage) / "cells" / f"{cell_id}.json"
        payload = _load_json_object(path)
        cell = payload.get("cell")
        if not isinstance(cell, dict) or cell.get("cell_id") != cell_id:
            raise ValueError(f"persisted cell filename mismatch: {path}")
        return payload

    def _observed_cell_ids(self, stage: str) -> frozenset[str]:
        cells_dir = self._stage_dir(stage) / "cells"
        if not cells_dir.exists():
            return frozenset()
        observed: set[str] = set()
        for path in sorted(cells_dir.glob("*.json")):
            payload = _load_json_object(path)
            cell = payload.get("cell")
            if not isinstance(cell, dict):
                raise ValueError(f"invalid persisted cell structure: {path}")
            cell_id = cell.get("cell_id")
            if not isinstance(cell_id, str) or not cell_id:
                raise ValueError(f"invalid persisted cell ID: {path}")
            if path.stem != cell_id:
                raise ValueError(f"persisted cell filename mismatch: {path}")
            if cell_id in observed:
                raise ValueError(f"duplicate persisted cell ID: {cell_id}")
            observed.add(cell_id)
        return frozenset(observed)

    def _bundle_paths(self, cell: Phase06CellSpec) -> dict[str, Path]:
        stage = self._stage_dir(cell.stage)
        return {
            "cell_metrics": stage / "cells" / f"{cell.cell_id}.json",
            "training_trace": stage / "traces" / f"{cell.cell_id}.csv",
            "summary": stage / "summaries" / f"{cell.cell_id}.json",
            "production_checkpoint": stage
            / "checkpoints"
            / f"{cell.cell_id}__production.pt",
            "shadow_mae_checkpoint": stage
            / "checkpoints"
            / f"{cell.cell_id}__shadow_mae.pt",
        }

    def _stage_dir(self, stage: str) -> Path:
        return self.output / "stages" / _require_stage(stage)


def _validate_cell(cell: Phase06CellSpec) -> None:
    if not isinstance(cell, Phase06CellSpec):
        raise TypeError("cell must be a Phase06CellSpec")


def _cell_metadata(cell: Phase06CellSpec) -> dict[str, object]:
    return {
        "cell_id": cell.cell_id,
        "stage": cell.stage,
        "world": cell.world,
        "cohort_seed": cell.cohort_seed,
        "subset_seed": cell.subset_seed,
        "model_seed": cell.model_seed,
        "n_train": cell.n_train,
        "flow_mode": cell.flow_mode,
        "jump_mode": cell.jump_mode,
        "uncertainty_mode": cell.uncertainty_mode,
    }


def _cell_from_payload(payload: dict[str, Any]) -> Phase06CellSpec:
    cell = payload.get("cell")
    if not isinstance(cell, dict):
        raise ValueError("invalid persisted cell structure")
    try:
        result = Phase06CellSpec(
            stage=cell["stage"],
            world=cell["world"],
            cohort_seed=cell["cohort_seed"],
            subset_seed=cell["subset_seed"],
            model_seed=cell["model_seed"],
            n_train=cell["n_train"],
            flow_mode=cell["flow_mode"],
            jump_mode=cell["jump_mode"],
            uncertainty_mode=cell["uncertainty_mode"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid persisted cell metadata") from error
    if cell != _cell_metadata(result):
        raise ValueError("invalid persisted cell metadata")
    return result


def _frame_records(frame: pd.DataFrame, label: str) -> list[dict[str, object]]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    if frame.empty:
        raise ValueError(f"{label} must contain at least one row")
    normalized = _normalize_json(frame.to_dict(orient="records"))
    if not isinstance(normalized, list) or not all(isinstance(row, dict) for row in normalized):
        raise TypeError(f"{label} could not be normalized")
    return normalized


def _validate_trace(frame: pd.DataFrame) -> pd.DataFrame:
    _frame_records(frame, "trace")
    return frame.copy()


def _validate_summary(summary: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(summary, Mapping):
        raise TypeError("summary must be a mapping")
    if any(key in summary for key in _RESERVED_SUMMARY_KEYS):
        raise ValueError("summary contains reserved persistence keys")
    normalized = _normalize_json(dict(summary))
    if not isinstance(normalized, dict):
        raise TypeError("summary must normalize to a JSON object")
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
        left[key].dtype == right[key].dtype
        and left[key].shape == right[key].shape
        and torch.equal(left[key], right[key])
        for key in left
    )


def _load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"invalid checkpoint artifact: {path}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"invalid checkpoint state dict: {path}")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in payload.items()):
        raise ValueError(f"invalid checkpoint state dict: {path}")
    return payload


def _frame_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _write_checkpoint_temp(
    final_path: Path,
    state_dict: Mapping[str, torch.Tensor],
) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(final_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(dict(state_dict), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _load_checkpoint(temporary)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_bytes_temp(final_path: Path, data: bytes) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(final_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.read_bytes() != data:
            raise OSError(f"temporary artifact validation failed: {final_path}")
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = _write_bytes_temp(path, data)
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


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


def _require_stage(stage: object) -> str:
    if not isinstance(stage, str) or stage not in _ALLOWED_STAGES:
        raise ValueError("stage must be d1 or d2a")
    return stage


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _require_commit(value: object) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError("execution_commit must be a 40- or 64-character hexadecimal SHA")
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
    if isinstance(value, Mapping):
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
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return payload


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = ["PHASE06_STORE_SCHEMA_VERSION", "Phase06Store"]
