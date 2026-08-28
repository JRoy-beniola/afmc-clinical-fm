from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from .bootstrap import _atomic_write
from .integrity import CheckResult
from .models import PhaseDefinition
from .report_manifest import dump_manifest, load_manifest
from .report_source import extract_docx, render_markdown


@dataclass(frozen=True)
class ReportAudit:
    phase_id: str
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def _check(code: str, ok: bool, subject: str, detail: str) -> CheckResult:
    return CheckResult(code=code, ok=ok, subject=subject, detail=detail)


def _normalize(value: str) -> str:
    translated = (
        value.replace("→", "->")
        .replace("—", "-")
        .replace("–", "-")
        .replace("−", "-")
    )
    return " ".join(translated.casefold().split())


def _snapshot_bytes(snapshot: object) -> bytes:
    payload = snapshot.to_json_dict()  # type: ignore[attr-defined]
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def audit_report_source(
    root: Path,
    phase: PhaseDefinition,
    destination: Path,
    *,
    accept: bool = False,
) -> ReportAudit:
    """Recompute and audit a recovered report-source bundle against immutable evidence."""

    repository = Path(root).resolve()
    bundle = Path(destination).resolve()
    manifest_path = bundle / "report.yaml"
    source_path = bundle / "report-source.md"
    extraction_path = bundle / "extraction.json"

    required = (manifest_path, source_path, extraction_path)
    missing = tuple(path for path in required if not path.is_file())
    if missing:
        checks = tuple(
            _check("missing_path", False, str(path), "required report-source bundle file is missing")
            for path in missing
        )
        return ReportAudit(phase_id=phase.phase_id, checks=checks)

    manifest = load_manifest(manifest_path)
    checks: list[CheckResult] = []

    phase_match = manifest.phase_id == phase.phase_id
    checks.append(
        _check(
            "phase_id_match" if phase_match else "phase_id_mismatch",
            phase_match,
            manifest.phase_id,
            f"expected phase {phase.phase_id}",
        )
    )

    if phase.official_report is None:
        checks.append(
            _check("reference_report_missing", False, phase.phase_id, "phase has no official report")
        )
        return ReportAudit(phase_id=phase.phase_id, checks=tuple(checks))

    reference = repository / phase.official_report
    if not reference.is_file():
        checks.append(
            _check(
                "reference_report_missing",
                False,
                phase.official_report.as_posix(),
                "official report is missing",
            )
        )
        return ReportAudit(phase_id=phase.phase_id, checks=tuple(checks))

    snapshot = extract_docx(reference)
    expected_source = render_markdown(snapshot).encode("utf-8")
    expected_extraction = _snapshot_bytes(snapshot)
    observed_source = source_path.read_bytes()
    observed_extraction = extraction_path.read_bytes()

    reference_path_match = manifest.reference_report == phase.official_report
    checks.append(
        _check(
            "reference_path_match" if reference_path_match else "reference_path_mismatch",
            reference_path_match,
            manifest.reference_report.as_posix(),
            f"expected {phase.official_report.as_posix()}",
        )
    )

    reference_hash_match = manifest.reference_report_sha256 == snapshot.reference_sha256
    checks.append(
        _check(
            "reference_sha256_match" if reference_hash_match else "reference_sha256_mismatch",
            reference_hash_match,
            phase.official_report.as_posix(),
            f"sha256={snapshot.reference_sha256}",
        )
    )

    source_match = observed_source == expected_source
    checks.append(
        _check(
            "report_source_match" if source_match else "report_source_mismatch",
            source_match,
            str(source_path),
            "canonical source is a byte-identical deterministic rerender",
        )
    )

    source_hash = hashlib.sha256(observed_source).hexdigest()
    source_hash_match = manifest.report_source_sha256 == source_hash
    checks.append(
        _check(
            "report_source_sha256_match" if source_hash_match else "report_source_sha256_mismatch",
            source_hash_match,
            str(source_path),
            f"sha256={source_hash}",
        )
    )

    extraction_match = observed_extraction == expected_extraction
    checks.append(
        _check(
            "extraction_snapshot_match" if extraction_match else "extraction_snapshot_mismatch",
            extraction_match,
            str(extraction_path),
            "extraction snapshot matches a fresh deterministic extraction",
        )
    )

    counts_match = (
        manifest.block_count == len(snapshot.blocks)
        and manifest.paragraph_count == snapshot.paragraph_count
        and manifest.table_count == snapshot.table_count
        and manifest.image_count == snapshot.image_count
    )
    checks.append(
        _check(
            "extraction_counts_match" if counts_match else "extraction_counts_mismatch",
            counts_match,
            phase.phase_id,
            (
                f"blocks={len(snapshot.blocks)} paragraphs={snapshot.paragraph_count} "
                f"tables={snapshot.table_count} images={snapshot.image_count}"
            ),
        )
    )

    canonical_root = Path("docs/reproducibility") / phase.phase_id
    canonical_source = canonical_root / "report-source.md"
    canonical_snapshot = canonical_root / "extraction.json"
    canonical_paths_match = (
        manifest.report_source == canonical_source
        and manifest.extraction_snapshot == canonical_snapshot
    )
    checks.append(
        _check(
            "canonical_paths_match" if canonical_paths_match else "canonical_paths_mismatch",
            canonical_paths_match,
            phase.phase_id,
            f"expected source namespace {canonical_root.as_posix()}",
        )
    )

    decision_text = ""
    decision_path = repository / phase.decision_record
    if decision_path.is_file():
        decision_text = decision_path.read_text(encoding="utf-8")
    expected_classification = _normalize(phase.expected_classification)
    source_text = observed_source.decode("utf-8", errors="replace")
    classification_bound = (
        expected_classification in _normalize(source_text)
        or expected_classification in _normalize(decision_text)
    )
    checks.append(
        _check(
            "classification_bound" if classification_bound else "classification_unbound",
            classification_bound,
            phase.decision_record.as_posix(),
            f"expected frozen classification: {phase.expected_classification}",
        )
    )

    result = ReportAudit(phase_id=phase.phase_id, checks=tuple(checks))
    if accept and result.ok:
        accepted = replace(manifest, audit_status="accepted")
        _atomic_write(manifest_path, dump_manifest(accepted).encode("utf-8"))
    return result
