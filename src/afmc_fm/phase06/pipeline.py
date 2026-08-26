from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase06.config import Phase06Config
from afmc_fm.phase06.planning import Phase06CellSpec, plan_d1_cells, plan_d2a_cells
from afmc_fm.phase06.store import Phase06Store

_RESERVED_SUMMARY_KEYS = frozenset({"artifact_sha256", "identity"})
_ANALYSIS_SUFFIXES = frozenset({".csv", ".json"})
_PARENT_PHASE06_EXECUTION_SHA = "1718402df1d6ef344168677e6d26ea664708e1bc"
_PARENT_PHASE06_PROTOCOL_SHA256 = (
    "c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51"
)


def open_bound_store(output: str | Path) -> Phase06Store:
    output = Path(output)
    path = output / "protocol_lock.json"
    if not path.is_file():
        raise ValueError("protocol identity is missing protocol_lock.json")
    data = path.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("protocol lock is not valid JSON") from error
    if not isinstance(payload, dict):
        raise TypeError("protocol lock must contain a JSON object")
    return Phase06Store(
        output,
        protocol_hash=hashlib.sha256(data).hexdigest(),
        config_hash=payload.get("phase06_config_sha256"),
        execution_commit=payload.get("execution_commit"),
    )


def _load_json_object_bytes(data: bytes, label: str) -> dict[str, object]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _require_parent_complete_marker(
    output: Path,
    *,
    stage: str,
    cells: Sequence[Phase06CellSpec],
    identity: dict[str, str],
) -> None:
    marker = output / "stages" / stage / "COMPLETE"
    if not marker.is_file():
        raise ValueError(f"parent {stage} stage is not complete")
    try:
        payload = _load_json_object_bytes(marker.read_bytes(), f"parent {stage} completion marker")
    except OSError as error:
        raise ValueError(f"parent {stage} stage is not complete") from error
    expected = {
        "schema_version": 1,
        "identity": identity,
        "stage": stage,
        "cell_ids": sorted(cell.cell_id for cell in cells),
    }
    if payload != expected:
        raise ValueError(f"parent {stage} stage is not complete")


def load_phase06_d2b_parent_evidence(
    output: str | Path,
    *,
    config: Phase06Config,
    phase05_config: Phase05Config,
) -> dict[str, object]:
    output = Path(output)
    protocol_path = output / "protocol_lock.json"
    if not protocol_path.is_file():
        raise ValueError("parent protocol lock is missing")
    try:
        protocol_bytes = protocol_path.read_bytes()
    except OSError as error:
        raise ValueError("parent protocol lock cannot be read") from error
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    if protocol_hash != _PARENT_PHASE06_PROTOCOL_SHA256:
        raise ValueError("parent protocol bytes do not match the frozen Phase 0.6 parent")
    protocol = _load_json_object_bytes(protocol_bytes, "parent protocol lock")

    expected_config_hash = canonical_config_hash(config)
    expected_phase05_hash = canonical_config_hash(phase05_config)
    phase06_spec_path = Path(config.phase06_spec)
    if not phase06_spec_path.is_file():
        raise ValueError("parent Phase 0.6 spec cannot be verified")
    expected_spec_hash = hashlib.sha256(phase06_spec_path.read_bytes()).hexdigest()
    expected_forbidden = {
        "cohort": list(config.forbidden_cohort_seeds),
        "subset": list(config.forbidden_subset_seeds),
        "model": list(config.forbidden_model_seeds),
    }

    if protocol.get("execution_commit") != _PARENT_PHASE06_EXECUTION_SHA:
        raise ValueError("parent protocol execution identity does not match frozen parent")
    if protocol.get("phase06_config_sha256") != expected_config_hash:
        raise ValueError("parent protocol Phase 0.6 config identity mismatch")
    if protocol.get("phase05_config_sha256") != expected_phase05_hash:
        raise ValueError("parent protocol Phase 0.5 config identity mismatch")
    if protocol.get("phase06_spec_sha256") != expected_spec_hash:
        raise ValueError("parent protocol Phase 0.6 spec identity mismatch")
    if protocol.get("forbidden_seed_sets") != expected_forbidden:
        raise ValueError("parent protocol forbidden seed identity mismatch")

    identity = {
        "protocol_lock_sha256": protocol_hash,
        "phase06_config_sha256": expected_config_hash,
        "execution_commit": _PARENT_PHASE06_EXECUTION_SHA,
    }
    _require_parent_complete_marker(
        output,
        stage="d1",
        cells=plan_d1_cells(config, phase05_config),
        identity=identity,
    )
    _require_parent_complete_marker(
        output,
        stage="d2a",
        cells=plan_d2a_cells(config),
        identity=identity,
    )

    d3_path = output / "analysis" / "phase06_d3_adjudication.json"
    if not d3_path.is_file():
        raise ValueError("parent D3 adjudication is missing")
    try:
        d3_bytes = d3_path.read_bytes()
        d3 = _load_json_object_bytes(d3_bytes, "parent D3 adjudication")
    except OSError as error:
        raise ValueError("parent D3 adjudication cannot be read") from error
    if d3.get("next_required_stage") != "D2B":
        raise ValueError("parent D3 next_required_stage must be D2B")

    return {
        "parent_execution_sha": _PARENT_PHASE06_EXECUTION_SHA,
        "parent_protocol_lock_sha256": protocol_hash,
        "parent_d3_sha256": hashlib.sha256(d3_bytes).hexdigest(),
        "parent_d3_next_required_stage": "D2B",
        "parent_phase06_config_sha256": expected_config_hash,
        "parent_phase06_spec_sha256": expected_spec_hash,
        "parent_forbidden_seed_sets": expected_forbidden,
    }


def require_completed_stage(
    store: Phase06Store,
    cells: Sequence[Phase06CellSpec],
) -> frozenset[str]:
    planned = tuple(cells)
    if not planned:
        raise ValueError("completed-stage validation requires a non-empty plan")
    stages = {cell.stage for cell in planned}
    if len(stages) != 1:
        raise ValueError("completed-stage validation requires one stage")
    stage = next(iter(stages))
    expected = frozenset(cell.cell_id for cell in planned)
    if len(expected) != len(planned):
        raise ValueError("completed-stage validation received duplicate cell IDs")

    marker = store.output / "stages" / stage / "COMPLETE"
    if not marker.is_file():
        raise ValueError(f"{stage} stage is not complete")
    observed = store.validate_resume(stage, expected_cell_ids=expected)
    if observed != expected:
        raise ValueError(f"{stage} stage is not complete")
    return observed


def _cell_identity(cell: Phase06CellSpec) -> dict[str, object]:
    return {
        "stage": cell.stage,
        "world": cell.world,
        "cohort_seed": cell.cohort_seed,
        "subset_seed": cell.subset_seed,
        "model_seed": cell.model_seed,
        "n_train": cell.n_train,
        "variant": f"{cell.flow_mode}__{cell.jump_mode}__{cell.uncertainty_mode}",
    }


def load_stage_summaries(
    store: Phase06Store,
    cells: Sequence[Phase06CellSpec],
) -> pd.DataFrame:
    planned = tuple(cells)
    rows: list[dict[str, object]] = []
    for cell in planned:
        path = store.output / "stages" / cell.stage / "summaries" / f"{cell.cell_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid persisted summary: {cell.cell_id}") from error
        if not isinstance(payload, dict):
            raise TypeError(f"persisted summary must be an object: {cell.cell_id}")
        semantic = {
            key: value for key, value in payload.items() if key not in _RESERVED_SUMMARY_KEYS
        }
        identity = _cell_identity(cell)
        overlap = set(identity) & set(semantic)
        if overlap:
            raise ValueError(
                f"persisted summary shadows cell identity: {cell.cell_id}: "
                + ", ".join(sorted(overlap))
            )
        rows.append({**identity, **semantic})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows)


def load_stage_traces(
    store: Phase06Store,
    cells: Sequence[Phase06CellSpec],
) -> pd.DataFrame:
    planned = tuple(cells)
    frames: list[pd.DataFrame] = []
    for cell in planned:
        path = store.output / "stages" / cell.stage / "traces" / f"{cell.cell_id}.csv"
        try:
            trace = pd.read_csv(path)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
            raise ValueError(f"invalid persisted trace: {cell.cell_id}") from error
        if trace.empty:
            raise ValueError(f"persisted trace is empty: {cell.cell_id}")
        identity = _cell_identity(cell)
        overlap = set(identity) & set(trace.columns)
        if overlap:
            raise ValueError(
                f"persisted trace shadows cell identity: {cell.cell_id}: "
                + ", ".join(sorted(overlap))
            )
        prefix = pd.DataFrame(
            {key: [value] * len(trace) for key, value in identity.items()},
            index=trace.index,
        )
        frames.append(pd.concat([prefix, trace], axis=1))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _analysis_path(store: Phase06Store, name: str, suffix: str) -> Path:
    if suffix not in _ANALYSIS_SUFFIXES:
        raise ValueError("analysis suffix must be .csv or .json")
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or not name.endswith(suffix)
    ):
        raise ValueError("unsafe analysis artifact name")
    return store.output / "analysis" / name


def _write_analysis_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        if path.is_file() and path.read_bytes() == data:
            return
        raise ValueError(f"conflicting analysis artifact: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.read_bytes() != data:
            raise OSError(f"temporary analysis validation failed: {path.name}")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_analysis_csv(store: Phase06Store, name: str, frame: pd.DataFrame) -> Path:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("analysis CSV must contain a non-empty DataFrame")
    path = _analysis_path(store, name, ".csv")
    data = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    _write_analysis_bytes(path, data)
    return path


def write_analysis_json(store: Phase06Store, name: str, payload: object) -> Path:
    path = _analysis_path(store, name, ".json")
    try:
        data = (
            json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("analysis JSON is not canonically serializable") from error
    _write_analysis_bytes(path, data)
    return path


__all__ = [
    "load_phase06_d2b_parent_evidence",
    "load_stage_summaries",
    "load_stage_traces",
    "open_bound_store",
    "require_completed_stage",
    "write_analysis_csv",
    "write_analysis_json",
]
