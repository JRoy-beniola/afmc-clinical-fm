from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .models import ManifestSpec, PhaseDefinition

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CheckResult:
    code: str
    ok: bool
    subject: str
    detail: str


def _manifest_base(root: Path, phase: PhaseDefinition, spec: ManifestSpec) -> Path:
    if spec.base == "repository":
        return root
    if spec.base == "evidence_root":
        return root / phase.official_evidence_root
    if spec.base == "manifest_parent":
        return (root / spec.path).parent
    raise ValueError(f"unsupported manifest base: {spec.base}")


def _safe_manifest_path(value: str) -> PurePosixPath | None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _declared_unavailable(spec: ManifestSpec, path: PurePosixPath) -> str | None:
    for pattern in spec.unavailable_patterns:
        if path.match(pattern):
            return pattern
    return None


def verify_manifest(
    root: Path,
    phase: PhaseDefinition,
    spec: ManifestSpec,
) -> tuple[CheckResult, ...]:
    """Verify one checksum manifest without mutating the repository."""

    manifest_path = root / spec.path
    if not manifest_path.is_file():
        return (
            CheckResult(
                code="missing_path",
                ok=False,
                subject=spec.path.as_posix(),
                detail="checksum manifest is missing",
            ),
        )

    base = _manifest_base(root, phase, spec)
    checks: list[CheckResult] = []
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            checks.append(
                CheckResult(
                    code="invalid_manifest_line",
                    ok=False,
                    subject=f"{spec.path.as_posix()}:{line_number}",
                    detail="expected SHA-256 digest and relative path",
                )
            )
            continue

        digest, path_text = parts
        path_text = path_text.strip().removeprefix("*")

        if _SHA256_RE.fullmatch(digest) is None:
            checks.append(
                CheckResult(
                    code="invalid_sha",
                    ok=False,
                    subject=f"{spec.path.as_posix()}:{line_number}",
                    detail="manifest digest must be 64 lowercase hexadecimal characters",
                )
            )
            continue

        relative_path = _safe_manifest_path(path_text)
        if relative_path is None:
            checks.append(
                CheckResult(
                    code="unsafe_manifest_path",
                    ok=False,
                    subject=f"{spec.path.as_posix()}:{line_number}",
                    detail=f"manifest path escapes its declared base: {path_text}",
                )
            )
            continue

        target = base.joinpath(*relative_path.parts)
        subject = target.relative_to(root).as_posix() if target.is_relative_to(root) else str(target)
        if not target.is_file():
            unavailable_pattern = _declared_unavailable(spec, relative_path)
            if unavailable_pattern is not None:
                checks.append(
                    CheckResult(
                        code="manifest_target_unavailable",
                        ok=True,
                        subject=subject,
                        detail=(
                            "manifest target is unavailable in this repository checkout "
                            f"under declared pattern {unavailable_pattern}; digest not verified"
                        ),
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        code="missing_path",
                        ok=False,
                        subject=subject,
                        detail="manifest target is missing",
                    )
                )
            continue

        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        matches = actual == digest
        checks.append(
            CheckResult(
                code="sha256_match" if matches else "sha256_mismatch",
                ok=matches,
                subject=subject,
                detail=(
                    f"sha256={actual}"
                    if matches
                    else f"expected {digest}; observed {actual}"
                ),
            )
        )

    return tuple(checks)
