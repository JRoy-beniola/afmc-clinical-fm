from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from afmc_fm.phase05.store import Phase05Store

_PROVENANCE_SCHEMA_VERSION = 1
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def failure_payload(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error),
    }


def record_execution_invocation(
    store: Phase05Store,
    *,
    implementation_sha: str,
    simulator_config_sha256: str | None,
    invocation: dict[str, object],
) -> None:
    if _SHA40_RE.fullmatch(implementation_sha) is None:
        raise ValueError("implementation_sha must be a 40-character hexadecimal SHA")
    if (
        simulator_config_sha256 is not None
        and _SHA256_RE.fullmatch(simulator_config_sha256) is None
    ):
        raise ValueError(
            "simulator_config_sha256 must be a 64-character hexadecimal SHA-256"
        )

    path = store.output / "execution_provenance.json"
    if path.exists():
        payload = _load_provenance(path)
        if payload.get("identity") != store.identity:
            raise ValueError("execution provenance protocol identity mismatch")
        if payload.get("implementation_sha") != implementation_sha:
            raise ValueError("execution provenance implementation SHA mismatch")
        if payload.get("simulator_config_sha256") != simulator_config_sha256:
            raise ValueError("execution provenance simulator config hash mismatch")
        invocations = payload.get("invocations")
        if not isinstance(invocations, list):
            raise TypeError("execution provenance invocations must be a list")
    else:
        payload = {
            "schema_version": _PROVENANCE_SCHEMA_VERSION,
            "identity": store.identity,
            "implementation_sha": implementation_sha,
            "simulator_config_sha256": simulator_config_sha256,
            "invocations": [],
        }
        invocations = payload["invocations"]

    normalized = _normalize_json(invocation)
    if not isinstance(normalized, dict):
        raise TypeError("execution invocation must be a JSON object")
    normalized["invocation_number"] = len(invocations) + 1
    invocations.append(normalized)
    _atomic_json(path, payload)


def _load_provenance(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid Phase-0.5 execution provenance") from error
    if not isinstance(payload, dict):
        raise TypeError("Phase-0.5 execution provenance must contain an object")
    if payload.get("schema_version") != _PROVENANCE_SCHEMA_VERSION:
        raise ValueError("incompatible Phase-0.5 execution provenance schema")
    return payload


def _normalize_json(value: object) -> object:
    if isinstance(value, str):
        return str(value)
    if value is None or type(value) in {bool, int, float}:
        return value
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError("execution provenance keys must be strings")
        return {key: _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise TypeError(
        f"unsupported execution provenance value: {type(value).__qualname__}"
    )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    data = (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.unlink(missing_ok=True)
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.read_bytes() != data:
            raise OSError("temporary execution provenance validation failed")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = ["failure_payload", "record_execution_invocation"]
