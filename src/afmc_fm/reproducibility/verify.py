from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .integrity import CheckResult, verify_manifest
from .models import PhaseDefinition
from .registry import iter_phases

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class VerificationReport:
    phase_id: str
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def _normalize_classification(value: str) -> str:
    translated = (
        value.replace("→", "->")
        .replace("—", "-")
        .replace("–", "-")
        .replace("−", "-")
    )
    return " ".join(translated.casefold().split())


def _path_check(root: Path, path: Path, *, label: str) -> CheckResult:
    exists = (root / path).exists()
    return CheckResult(
        code="path_present" if exists else "missing_path",
        ok=exists,
        subject=path.as_posix(),
        detail=f"{label} {'present' if exists else 'missing'}",
    )


def _sha_check(value: str | None, *, label: str) -> CheckResult | None:
    if value is None:
        return None
    valid = _GIT_SHA_RE.fullmatch(value) is not None
    return CheckResult(
        code="git_sha_valid" if valid else "invalid_sha",
        ok=valid,
        subject=label,
        detail=(
            f"{label} is a 40-character lowercase Git SHA"
            if valid
            else f"{label} must be a 40-character lowercase hexadecimal Git SHA"
        ),
    )


def _classification_check(root: Path, phase: PhaseDefinition) -> CheckResult:
    path = root / phase.decision_record
    if not path.is_file():
        return CheckResult(
            code="classification_unavailable",
            ok=False,
            subject=phase.decision_record.as_posix(),
            detail="decision record is missing",
        )

    observed = _normalize_classification(path.read_text(encoding="utf-8"))
    expected = _normalize_classification(phase.expected_classification)
    matches = expected in observed
    return CheckResult(
        code="classification_match" if matches else "classification_mismatch",
        ok=matches,
        subject=phase.decision_record.as_posix(),
        detail=(
            f"frozen classification found: {phase.expected_classification}"
            if matches
            else f"frozen classification not found: {phase.expected_classification}"
        ),
    )


def verify_phase(root: Path, phase: PhaseDefinition) -> VerificationReport:
    """Verify one frozen phase using read-only repository evidence."""

    root = Path(root)
    checks: list[CheckResult] = []

    evidence_parts = phase.official_evidence_root.parts
    in_reproduction_output = evidence_parts[:2] == ("outputs", "reproduction")
    checks.append(
        CheckResult(
            code=(
                "official_evidence_in_reproduction_output"
                if in_reproduction_output
                else "official_evidence_location_valid"
            ),
            ok=not in_reproduction_output,
            subject=phase.official_evidence_root.as_posix(),
            detail=(
                "outputs/reproduction cannot be treated as official historical evidence"
                if in_reproduction_output
                else "official evidence root is outside outputs/reproduction"
            ),
        )
    )

    implementation_check = _sha_check(
        phase.implementation_sha,
        label="implementation_sha",
    )
    if implementation_check is not None:
        checks.append(implementation_check)
    execution_check = _sha_check(phase.execution_sha, label="execution_sha")
    if execution_check is not None:
        checks.append(execution_check)

    paths: list[tuple[Path, str]] = [
        (phase.official_evidence_root, "official evidence root"),
        (phase.decision_record, "decision record"),
    ]
    if phase.official_report is not None:
        paths.append((phase.official_report, "official report"))
    paths.extend((path, "protocol/config/spec") for path in phase.protocol_paths)
    paths.extend((path, "raw evidence") for path in phase.raw_evidence_paths)
    paths.extend((path, "derived table") for path in phase.derived_table_paths)
    paths.extend((path, "figure") for path in phase.figure_paths)
    if phase.report_source is not None:
        paths.append((phase.report_source, "report source"))

    checks.extend(_path_check(root, path, label=label) for path, label in paths)
    checks.append(_classification_check(root, phase))

    for manifest in phase.manifests:
        checks.extend(verify_manifest(root, phase, manifest))

    return VerificationReport(phase_id=phase.phase_id, checks=tuple(checks))


def verify_all(root: Path) -> tuple[VerificationReport, ...]:
    return tuple(verify_phase(root, phase) for phase in iter_phases())
