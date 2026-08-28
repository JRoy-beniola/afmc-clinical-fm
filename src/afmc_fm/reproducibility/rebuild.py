from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactResult, materialize_artifact
from .document_compare import DocumentComparison, compare_documents
from .documents import build_report_docx
from .rebuild_models import RebuildManifest, load_rebuild_manifest, validate_rebuild_destination
from .registry import get_phase
from .report_manifest import ReportSourceManifest, load_manifest, validate_manifest
from .verify import verify_phase


@dataclass(frozen=True)
class RebuildReport:
    phase_id: str
    destination: Path
    artifacts: tuple[ArtifactResult, ...]
    candidate_report: Path
    rebuild_report: Path
    structural_comparison: DocumentComparison
    source_sha256: str
    reference_sha256: str
    historical_archive_before: str
    historical_archive_after: str
    historical_archive_unchanged: bool

    @property
    def generated_count(self) -> int:
        return sum(result.generated for result in self.artifacts)

    @property
    def reference_copy_count(self) -> int:
        return sum(result.mode == "reference-copy" for result in self.artifacts)

    @property
    def ok(self) -> bool:
        return self.historical_archive_unchanged and self.structural_comparison.ok


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
        digest.update(b"\0")
    return digest.hexdigest()


def _require_phase_verification(root: Path, phase_id: str) -> None:
    phase = get_phase(phase_id)
    report = verify_phase(root, phase)
    if report.ok:
        return
    failed = ", ".join(check.code for check in report.checks if not check.ok)
    raise ValueError(f"phase {phase_id} failed read-only verification: {failed}")


def _load_accepted_report_manifest(root: Path, phase_id: str) -> ReportSourceManifest:
    phase = get_phase(phase_id)
    path = root / "docs/reproducibility" / phase_id / "report.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"report-source manifest is missing: {path.relative_to(root)}")
    manifest = load_manifest(path)
    if manifest.phase_id != phase_id:
        raise ValueError("report-source manifest phase_id does not match requested phase")
    if manifest.audit_status != "accepted":
        raise ValueError(f"phase {phase_id} report source must have audit_status=accepted")
    if phase.official_report is None or manifest.reference_report != phase.official_report:
        raise ValueError("report-source manifest reference_report does not match phase registry")
    if phase.report_source is None or manifest.report_source != phase.report_source:
        raise ValueError("report-source manifest report_source does not match phase registry")

    checks = validate_manifest(root, manifest)
    failed = tuple(check for check in checks if not check.ok)
    if failed:
        codes = ", ".join(check.code for check in failed)
        raise ValueError(f"phase {phase_id} report-source manifest failed validation: {codes}")
    return manifest


def _load_aligned_rebuild_manifest(
    root: Path,
    phase_id: str,
    report_manifest: ReportSourceManifest,
) -> RebuildManifest:
    path = root / "docs/reproducibility" / phase_id / "rebuild.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"rebuild manifest is missing: {path.relative_to(root)}")
    manifest = load_rebuild_manifest(path)
    if manifest.phase_id != phase_id:
        raise ValueError("rebuild manifest phase_id does not match requested phase")
    if manifest.report_source != report_manifest.report_source:
        raise ValueError("rebuild manifest report_source does not match accepted R2 source")
    if manifest.reference_report != report_manifest.reference_report:
        raise ValueError("rebuild manifest reference_report does not match accepted R2 reference")
    return manifest


def _safe_destination_child(destination: Path, relative: Path) -> Path:
    root = destination.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise ValueError("rebuild output escapes destination")
    return candidate


def _reset_destination(destination: Path) -> None:
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    elif destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)


def _artifact_payload(
    manifest: RebuildManifest,
    results: tuple[ArtifactResult, ...],
    destination: Path,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for declaration, result in zip(manifest.artifacts, results, strict=True):
        payload.append(
            {
                "kind": declaration.kind,
                "source": declaration.source.as_posix(),
                "output": result.output.relative_to(destination).as_posix(),
                "mode": result.mode,
                "generator": declaration.generator,
                "generated": result.generated,
                "sha256": result.sha256,
            }
        )
    return payload


def _write_report_json(report: RebuildReport, manifest: RebuildManifest) -> None:
    payload = {
        "schema_version": 1,
        "phase_id": report.phase_id,
        "destination": report.destination.as_posix(),
        "source_sha256": report.source_sha256,
        "reference_sha256": report.reference_sha256,
        "candidate_report": report.candidate_report.relative_to(report.destination).as_posix(),
        "artifact_counts": {
            "generated": report.generated_count,
            "reference_copy": report.reference_copy_count,
        },
        "artifacts": _artifact_payload(manifest, report.artifacts, report.destination),
        "structural_comparison": report.structural_comparison.to_json_dict(),
        "historical_archive_before": report.historical_archive_before,
        "historical_archive_after": report.historical_archive_after,
        "historical_archive_unchanged": report.historical_archive_unchanged,
        "ok": report.ok,
    }
    report.rebuild_report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rebuild_phase(
    root: Path,
    phase_id: str,
    destination: Path | None = None,
) -> RebuildReport:
    """Rebuild one supported historical report without training or archive mutation."""

    repository = Path(root).resolve()
    phase = get_phase(phase_id)
    if not phase.rebuild_supported:
        raise ValueError(f"phase {phase_id} does not support deterministic rebuild")

    _require_phase_verification(repository, phase_id)
    report_manifest = _load_accepted_report_manifest(repository, phase_id)
    rebuild_manifest = _load_aligned_rebuild_manifest(repository, phase_id, report_manifest)
    target = validate_rebuild_destination(repository, phase_id, destination)

    historical_root = repository / phase.official_evidence_root
    historical_before = _tree_digest(historical_root)

    _reset_destination(target)
    artifact_results = tuple(
        materialize_artifact(repository, target, declaration)
        for declaration in rebuild_manifest.artifacts
    )

    candidate_report = _safe_destination_child(target, rebuild_manifest.document_output)
    build_report_docx(
        source=repository / rebuild_manifest.report_source,
        reference=repository / rebuild_manifest.reference_report,
        output=candidate_report,
    )
    comparison = compare_documents(
        repository / rebuild_manifest.reference_report,
        candidate_report,
    )

    historical_after = _tree_digest(historical_root)
    unchanged = historical_before == historical_after
    if not unchanged:
        raise RuntimeError(f"rebuild modified historical archive for phase {phase_id}")

    report_path = target / "rebuild-report.json"
    report = RebuildReport(
        phase_id=phase_id,
        destination=target,
        artifacts=artifact_results,
        candidate_report=candidate_report,
        rebuild_report=report_path,
        structural_comparison=comparison,
        source_sha256=report_manifest.report_source_sha256,
        reference_sha256=report_manifest.reference_report_sha256,
        historical_archive_before=historical_before,
        historical_archive_after=historical_after,
        historical_archive_unchanged=unchanged,
    )
    _write_report_json(report, rebuild_manifest)
    return report
