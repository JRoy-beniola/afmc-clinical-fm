from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .integrity import CheckResult
from .models import PhaseDefinition
from .report_source import ReportSnapshot

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ReportSourceManifest:
    schema_version: int
    phase_id: str
    reference_report: Path
    reference_report_sha256: str
    report_source: Path
    report_source_sha256: str
    extraction_snapshot: Path
    extractor: str
    extractor_schema_version: int
    block_count: int
    paragraph_count: int
    table_count: int
    image_count: int
    audit_status: str
    known_limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("reference_report", "report_source", "extraction_snapshot"):
            payload[key] = getattr(self, key).as_posix()
        payload["known_limitations"] = list(self.known_limitations)
        return payload


def build_manifest(
    phase: PhaseDefinition,
    snapshot: ReportSnapshot,
    source_path: Path,
    source_bytes: bytes,
    *,
    audit_status: str = "pending",
    known_limitations: tuple[str, ...] = (),
) -> ReportSourceManifest:
    if phase.official_report is None:
        raise ValueError(f"phase {phase.phase_id} has no official report")
    source_path = Path(source_path)
    return ReportSourceManifest(
        schema_version=1,
        phase_id=phase.phase_id,
        reference_report=phase.official_report,
        reference_report_sha256=snapshot.reference_sha256,
        report_source=source_path,
        report_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        extraction_snapshot=source_path.with_name("extraction.json"),
        extractor="afmc_fm.reproducibility.report_source",
        extractor_schema_version=1,
        block_count=len(snapshot.blocks),
        paragraph_count=snapshot.paragraph_count,
        table_count=snapshot.table_count,
        image_count=snapshot.image_count,
        audit_status=audit_status,
        known_limitations=tuple(known_limitations),
    )


def dump_manifest(manifest: ReportSourceManifest) -> str:
    return yaml.safe_dump(
        manifest.to_dict(),
        sort_keys=False,
        allow_unicode=True,
    )


def load_manifest(path: Path) -> ReportSourceManifest:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("report manifest must be a mapping")
    try:
        return ReportSourceManifest(
            schema_version=int(payload["schema_version"]),
            phase_id=str(payload["phase_id"]),
            reference_report=Path(payload["reference_report"]),
            reference_report_sha256=str(payload["reference_report_sha256"]),
            report_source=Path(payload["report_source"]),
            report_source_sha256=str(payload["report_source_sha256"]),
            extraction_snapshot=Path(payload["extraction_snapshot"]),
            extractor=str(payload["extractor"]),
            extractor_schema_version=int(payload["extractor_schema_version"]),
            block_count=int(payload["block_count"]),
            paragraph_count=int(payload["paragraph_count"]),
            table_count=int(payload["table_count"]),
            image_count=int(payload["image_count"]),
            audit_status=str(payload["audit_status"]),
            known_limitations=tuple(str(item) for item in payload.get("known_limitations", [])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid report manifest schema") from exc


def _safe_repository_path(path: Path) -> bool:
    return not path.is_absolute() and ".." not in path.parts


def _path_hash_check(
    root: Path,
    path: Path,
    expected: str,
    *,
    label: str,
) -> CheckResult:
    if not _safe_repository_path(path):
        return CheckResult(
            code=f"{label}_unsafe_path",
            ok=False,
            subject=path.as_posix(),
            detail="path must remain within the repository",
        )
    target = root / path
    if not target.is_file():
        return CheckResult(
            code=f"{label}_missing",
            ok=False,
            subject=path.as_posix(),
            detail=f"{label.replace('_', ' ')} is missing",
        )
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    matches = actual == expected
    return CheckResult(
        code=f"{label}_sha256_match" if matches else f"{label}_sha256_mismatch",
        ok=matches,
        subject=path.as_posix(),
        detail=f"sha256={actual}" if matches else f"expected {expected}; observed {actual}",
    )


def validate_manifest(
    root: Path,
    manifest: ReportSourceManifest,
) -> tuple[CheckResult, ...]:
    root = Path(root)
    checks: list[CheckResult] = []

    for label, digest in (
        ("reference_report", manifest.reference_report_sha256),
        ("report_source", manifest.report_source_sha256),
    ):
        valid = _SHA256_RE.fullmatch(digest) is not None
        checks.append(
            CheckResult(
                code=f"{label}_sha256_valid" if valid else f"{label}_invalid_sha256",
                ok=valid,
                subject=label,
                detail="valid lowercase SHA-256" if valid else "expected 64 lowercase hex characters",
            )
        )

    if _SHA256_RE.fullmatch(manifest.reference_report_sha256):
        checks.append(
            _path_hash_check(
                root,
                manifest.reference_report,
                manifest.reference_report_sha256,
                label="reference_report",
            )
        )
    if _SHA256_RE.fullmatch(manifest.report_source_sha256):
        checks.append(
            _path_hash_check(
                root,
                manifest.report_source,
                manifest.report_source_sha256,
                label="report_source",
            )
        )

    snapshot_safe = _safe_repository_path(manifest.extraction_snapshot)
    snapshot_present = snapshot_safe and (root / manifest.extraction_snapshot).is_file()
    checks.append(
        CheckResult(
            code=(
                "extraction_snapshot_present"
                if snapshot_present
                else "extraction_snapshot_missing"
                if snapshot_safe
                else "extraction_snapshot_unsafe_path"
            ),
            ok=snapshot_present,
            subject=manifest.extraction_snapshot.as_posix(),
            detail="extraction snapshot present" if snapshot_present else "extraction snapshot unavailable",
        )
    )

    schema_ok = manifest.schema_version == 1 and manifest.extractor_schema_version == 1
    checks.append(
        CheckResult(
            code="report_manifest_schema_valid" if schema_ok else "report_manifest_schema_invalid",
            ok=schema_ok,
            subject=manifest.phase_id,
            detail="report-source schema version 1" if schema_ok else "unsupported report-source schema",
        )
    )
    return tuple(checks)
