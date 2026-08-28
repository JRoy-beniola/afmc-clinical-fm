from __future__ import annotations

import json
import os
import platform
import sys
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import torch

_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_metadata() -> dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_name: str | None = None
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    data = _canonical_json_bytes(payload)
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Phase 0.7 execution provenance is invalid") from error
    if not isinstance(payload, dict):
        raise TypeError("Phase 0.7 execution provenance must be a JSON object")
    return payload


def _normalize_identity(identity: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(identity, Mapping):
        raise TypeError("identity must be a mapping")
    try:
        normalized = json.loads(
            json.dumps(dict(identity), allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise TypeError("identity must be JSON serializable") from error
    if not isinstance(normalized, dict) or not normalized:
        raise ValueError("identity must not be empty")
    return normalized


class Phase07ExecutionProvenance:
    def __init__(
        self,
        output: str | Path,
        *,
        identity: Mapping[str, object],
        device: str,
        planned_cell_count: int,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if type(planned_cell_count) is not int or planned_cell_count <= 0:
            raise ValueError("planned_cell_count must be a positive integer")
        self.output = Path(output)
        self.path = self.output / "execution_provenance.json"
        self.identity = _normalize_identity(identity)
        self.device = device
        self.planned_cell_count = planned_cell_count
        self._active_index: int | None = None
        self._active_started_monotonic: float | None = None

    def _base_payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "phase": "phase07",
            "identity": self.identity,
            "planned_cell_count": self.planned_cell_count,
            "invocations": [],
        }

    def _load_or_create(self) -> dict[str, object]:
        if not self.path.exists():
            return self._base_payload()
        payload = _load_json_object(self.path)
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Phase 0.7 execution provenance schema mismatch")
        if payload.get("phase") != "phase07":
            raise ValueError("Phase 0.7 execution provenance phase mismatch")
        if payload.get("identity") != self.identity:
            raise ValueError("Phase 0.7 execution provenance identity mismatch")
        if payload.get("planned_cell_count") != self.planned_cell_count:
            raise ValueError("Phase 0.7 execution provenance planned-cell mismatch")
        invocations = payload.get("invocations")
        if not isinstance(invocations, list):
            raise TypeError("Phase 0.7 execution provenance invocations must be a list")
        return payload

    def _require_active(self) -> tuple[dict[str, object], dict[str, object]]:
        if self._active_index is None or self._active_started_monotonic is None:
            raise ValueError("Phase 0.7 execution provenance has no active invocation")
        payload = self._load_or_create()
        invocations = payload["invocations"]
        if not isinstance(invocations, list) or self._active_index >= len(invocations):
            raise ValueError("Phase 0.7 execution provenance active invocation is missing")
        invocation = invocations[self._active_index]
        if not isinstance(invocation, dict) or invocation.get("status") != "running":
            raise ValueError("Phase 0.7 execution provenance active invocation is invalid")
        return payload, invocation

    def start(self, *, completed_before: int) -> None:
        if type(completed_before) is not int or not 0 <= completed_before <= self.planned_cell_count:
            raise ValueError("completed_before must be within the planned cell count")
        if self._active_index is not None:
            raise ValueError("Phase 0.7 execution provenance invocation is already active")
        payload = self._load_or_create()
        invocations = payload["invocations"]
        if not isinstance(invocations, list):
            raise TypeError("Phase 0.7 execution provenance invocations must be a list")
        invocation = {
            "status": "running",
            "device": self.device,
            "runtime": _runtime_metadata(),
            "completed_before_count": completed_before,
            "completed_after_count": completed_before,
            "last_attempted_cell_id": None,
            "failure": None,
            "started_at_utc": _utc_now(),
            "ended_at_utc": None,
            "wall_time_seconds": None,
        }
        invocations.append(invocation)
        self._active_index = len(invocations) - 1
        self._active_started_monotonic = time.monotonic()
        _atomic_write_json(self.path, payload)

    def attempt(self, cell_id: str) -> None:
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("cell_id must be a non-empty string")
        payload, invocation = self._require_active()
        invocation["last_attempted_cell_id"] = cell_id
        _atomic_write_json(self.path, payload)

    def fail(self, *, completed_after: int, error: BaseException) -> None:
        if type(completed_after) is not int or not 0 <= completed_after <= self.planned_cell_count:
            raise ValueError("completed_after must be within the planned cell count")
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception")
        payload, invocation = self._require_active()
        invocation["status"] = "failed"
        invocation["completed_after_count"] = completed_after
        invocation["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        self._finish_invocation(invocation)
        _atomic_write_json(self.path, payload)
        self._clear_active()

    def finish(self, *, completed_after: int) -> None:
        if type(completed_after) is not int or not 0 <= completed_after <= self.planned_cell_count:
            raise ValueError("completed_after must be within the planned cell count")
        payload, invocation = self._require_active()
        invocation["status"] = "complete"
        invocation["completed_after_count"] = completed_after
        invocation["failure"] = None
        self._finish_invocation(invocation)
        _atomic_write_json(self.path, payload)
        self._clear_active()

    def _finish_invocation(self, invocation: dict[str, object]) -> None:
        if self._active_started_monotonic is None:
            raise ValueError("Phase 0.7 execution provenance start time is missing")
        invocation["ended_at_utc"] = _utc_now()
        invocation["wall_time_seconds"] = max(
            0.0,
            time.monotonic() - self._active_started_monotonic,
        )

    def _clear_active(self) -> None:
        self._active_index = None
        self._active_started_monotonic = None


__all__ = ["Phase07ExecutionProvenance"]
