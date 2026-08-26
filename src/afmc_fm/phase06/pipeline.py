from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase06.analysis import analyze_d2a
from afmc_fm.phase06.config import Phase06Config
from afmc_fm.phase06.d2b import adjudicate_d2b, analyze_d2b
from afmc_fm.phase06.planning import (
    Phase06CellSpec,
    plan_d1_cells,
    plan_d2a_cells,
    plan_d2b_cells,
)
from afmc_fm.phase06.store import Phase06Store

_RESERVED_SUMMARY_KEYS = frozenset({"artifact_sha256", "identity"})
_ANALYSIS_SUFFIXES = frozenset({".csv", ".json"})
_PARENT_PHASE06_EXECUTION_SHA = "1718402df1d6ef344168677e6d26ea664708e1bc"
_PARENT_PHASE06_PROTOCOL_SHA256 = (
    "c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51"
)
_PARENT_PHASE06_D3_SHA256 = (
    "6b9fffed7503fae6beeac2314238ae3d10ffdebfd27ea71ad952ab1f87916460"
)
_D2B_PHASE06_EXECUTION_SHA = "516c9e3c0e965582fa5cce976e9d8ebf32ea8404"
_D2B_PROTOCOL_CANONICAL_SHA256 = (
    "33f7cb1f6e71560f547a746cb7f5eb41130f794ab1e3a6fb1928c40e05b29f12"
)
_D2B_ADJUDICATION_CANONICAL_SHA256 = (
    "ea28fd4d5f6490a10fad20d5d3f3e76a1de08bf6b1be3b9805c9cf6c519e845f"
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


def _canonical_payload_hash(payload: object) -> str:
    data = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


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


def _sha256_file(path: Path, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"parent D3 input evidence is missing: {label}")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"parent D3 input evidence cannot be read: {label}") from error


def _canonical_json_hash(path: Path, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"parent D3 input evidence is missing: {label}")
    try:
        payload = _load_json_object_bytes(path.read_bytes(), label)
    except OSError as error:
        raise ValueError(f"parent D3 input evidence cannot be read: {label}") from error
    return _canonical_payload_hash(payload)


def _expected_parent_d3_input_hashes(
    store: Phase06Store,
    d1_cells: Sequence[Phase06CellSpec],
) -> dict[str, str]:
    analysis = store.output / "analysis"
    traces = load_stage_traces(store, d1_cells)
    trace_bytes = traces.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return {
        "d1_table": _sha256_file(
            analysis / "phase06_d1_reproduction.csv", "D1 reproduction table"
        ),
        "d1_decision": _canonical_json_hash(
            analysis / "phase06_d1_classification.json", "D1 classification"
        ),
        "d2_variance_components": _sha256_file(
            analysis / "phase06_d2_variance_components.csv", "D2 variance components"
        ),
        "d2_bootstrap_diagnostics": _sha256_file(
            analysis / "phase06_d2_bootstrap_diagnostics.csv", "D2 bootstrap diagnostics"
        ),
        "d2_n_shift_rows": _sha256_file(
            analysis / "phase06_d2_n_shift.csv", "D2 N-shift rows"
        ),
        "d2_n_shift_summary": _canonical_json_hash(
            analysis / "phase06_d2_n_shift_summary.json", "D2 N-shift summary"
        ),
        "d1_traces": hashlib.sha256(trace_bytes).hexdigest(),
    }


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
    d1_cells = plan_d1_cells(config, phase05_config)
    d2a_cells = plan_d2a_cells(config)
    _require_parent_complete_marker(
        output,
        stage="d1",
        cells=d1_cells,
        identity=identity,
    )
    _require_parent_complete_marker(
        output,
        stage="d2a",
        cells=d2a_cells,
        identity=identity,
    )

    parent_store = open_bound_store(output)
    require_completed_stage(parent_store, d1_cells)
    require_completed_stage(parent_store, d2a_cells)

    d3_path = output / "analysis" / "phase06_d3_adjudication.json"
    if not d3_path.is_file():
        raise ValueError("parent D3 adjudication is missing")
    try:
        d3_bytes = d3_path.read_bytes()
        d3 = _load_json_object_bytes(d3_bytes, "parent D3 adjudication")
    except OSError as error:
        raise ValueError("parent D3 adjudication cannot be read") from error
    d3_hash = hashlib.sha256(d3_bytes).hexdigest()
    if d3_hash != _PARENT_PHASE06_D3_SHA256:
        raise ValueError("parent D3 SHA-256 does not match frozen parent")
    if d3.get("next_required_stage") != "D2B":
        raise ValueError("parent D3 next_required_stage must be D2B")
    expected_input_hashes = _expected_parent_d3_input_hashes(parent_store, d1_cells)
    if d3.get("input_artifact_hashes") != expected_input_hashes:
        raise ValueError("parent D3 input_artifact_hashes do not match parent evidence")

    return {
        "parent_execution_sha": _PARENT_PHASE06_EXECUTION_SHA,
        "parent_protocol_lock_sha256": protocol_hash,
        "parent_d3_sha256": d3_hash,
        "parent_d3_next_required_stage": "D2B",
        "parent_phase06_config_sha256": expected_config_hash,
        "parent_phase06_spec_sha256": expected_spec_hash,
        "parent_forbidden_seed_sets": expected_forbidden,
    }


def load_phase06_d4b_parent_evidence(
    core_parent_output: str | Path,
    d2b_parent_output: str | Path,
    *,
    config: Phase06Config,
    phase05_config: Phase05Config,
) -> dict[str, object]:
    core_parent_output = Path(core_parent_output)
    d2b_parent_output = Path(d2b_parent_output)
    core_evidence = load_phase06_d2b_parent_evidence(
        core_parent_output,
        config=config,
        phase05_config=phase05_config,
    )

    protocol_path = d2b_parent_output / "protocol_lock.json"
    if not protocol_path.is_file():
        raise ValueError("D2-B parent protocol lock is missing")
    try:
        protocol = _load_json_object_bytes(
            protocol_path.read_bytes(), "D2-B parent protocol lock"
        )
    except OSError as error:
        raise ValueError("D2-B parent protocol lock cannot be read") from error
    protocol_canonical_hash = _canonical_payload_hash(protocol)
    if protocol_canonical_hash != _D2B_PROTOCOL_CANONICAL_SHA256:
        raise ValueError("D2-B parent protocol identity does not match frozen evidence")

    expected_config_hash = canonical_config_hash(config)
    expected_phase05_hash = canonical_config_hash(phase05_config)
    phase06_spec_path = Path(config.phase06_spec)
    if not phase06_spec_path.is_file():
        raise ValueError("D2-B parent Phase 0.6 spec cannot be verified")
    expected_spec_hash = hashlib.sha256(phase06_spec_path.read_bytes()).hexdigest()
    expected_forbidden = {
        "cohort": list(config.forbidden_cohort_seeds),
        "subset": list(config.forbidden_subset_seeds),
        "model": list(config.forbidden_model_seeds),
    }
    expected_protocol_fields = {
        "schema_version": 2,
        "execution_commit": _D2B_PHASE06_EXECUTION_SHA,
        "phase06_config_sha256": expected_config_hash,
        "phase05_config_sha256": expected_phase05_hash,
        "phase06_spec_sha256": expected_spec_hash,
        "forbidden_seed_sets": expected_forbidden,
        "parent_execution_sha": core_evidence["parent_execution_sha"],
        "parent_protocol_lock_sha256": core_evidence["parent_protocol_lock_sha256"],
        "parent_d3_sha256": core_evidence["parent_d3_sha256"],
        "parent_d3_next_required_stage": "D2B",
        "d2b_mapping": "model_index=(cohort_index+2*subset_index)%5",
    }
    for key, expected in expected_protocol_fields.items():
        if protocol.get(key) != expected:
            raise ValueError(f"D2-B parent protocol linkage drift: {key}")

    d2b_store = open_bound_store(d2b_parent_output)
    d2b_cells = plan_d2b_cells(config)
    require_completed_stage(d2b_store, d2b_cells)

    adjudication_path = d2b_parent_output / "analysis" / "phase06_d2b_adjudication.json"
    if not adjudication_path.is_file():
        raise ValueError("D2-B parent adjudication is missing")
    try:
        persisted_adjudication = _load_json_object_bytes(
            adjudication_path.read_bytes(), "D2-B parent adjudication"
        )
    except OSError as error:
        raise ValueError("D2-B parent adjudication cannot be read") from error
    adjudication_hash = _canonical_payload_hash(persisted_adjudication)
    if adjudication_hash != _D2B_ADJUDICATION_CANONICAL_SHA256:
        raise ValueError("D2-B adjudication identity does not match frozen evidence")

    expected_decision = {
        "complementary_array_evidence": "sufficient",
        "dominant_factors": {
            "d2a": {"40": "model", "5": "none"},
            "d2b": {"40": "model", "5": "none"},
        },
        "next_required_stage": "D4_OPTIMIZATION",
        "remaining_parent_escalations": ["D4_OPTIMIZATION", "D4_CAPACITY_TIME"],
    }
    for key, expected in expected_decision.items():
        if persisted_adjudication.get(key) != expected:
            raise ValueError(f"D2-B adjudication decision drift: {key}")

    core_store = open_bound_store(core_parent_output)
    d3_path = core_parent_output / "analysis" / "phase06_d3_adjudication.json"
    if not d3_path.is_file():
        raise ValueError("core parent D3 adjudication is missing")
    try:
        core_d3 = _load_json_object_bytes(d3_path.read_bytes(), "core parent D3 adjudication")
    except OSError as error:
        raise ValueError("core parent D3 adjudication cannot be read") from error

    d2a_result = analyze_d2a(core_store.load_stage_metrics("d2a"), config)
    d2b_result = analyze_d2b(d2b_store.load_stage_metrics("d2b"), config)
    recomputed_adjudication = adjudicate_d2b(core_d3, d2a_result, d2b_result)
    if recomputed_adjudication != persisted_adjudication:
        raise ValueError("persisted D2-B adjudication does not match recomputed evidence")

    return {
        "core_parent_execution_sha": _PARENT_PHASE06_EXECUTION_SHA,
        "core_parent_protocol_lock_sha256": _PARENT_PHASE06_PROTOCOL_SHA256,
        "core_parent_d3_sha256": _PARENT_PHASE06_D3_SHA256,
        "d2b_parent_execution_sha": _D2B_PHASE06_EXECUTION_SHA,
        "d2b_parent_protocol_canonical_sha256": protocol_canonical_hash,
        "d2b_parent_adjudication_canonical_sha256": adjudication_hash,
        "d2b_next_required_stage": "D4_OPTIMIZATION",
        "phase06_config_sha256": expected_config_hash,
        "phase06_spec_sha256": expected_spec_hash,
        "forbidden_seed_sets": expected_forbidden,
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
    "load_phase06_d4b_parent_evidence",
    "load_stage_summaries",
    "load_stage_traces",
    "open_bound_store",
    "require_completed_stage",
    "write_analysis_csv",
    "write_analysis_json",
]
