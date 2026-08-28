from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .models import PhaseDefinition
from .registry import get_phase
from .report_manifest import build_manifest, dump_manifest
from .report_source import extract_docx, render_markdown

_OUTPUT_NAMES = ("report-source.md", "report.yaml", "extraction.json")


@dataclass(frozen=True)
class BootstrapResult:
    phase_id: str
    destination: Path
    report_source: Path
    report_manifest: Path
    extraction_snapshot: Path


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validated_destination(root: Path, destination: Path) -> Path:
    repository = Path(root).resolve()
    candidate = Path(destination)
    if not candidate.is_absolute():
        candidate = repository / candidate
    candidate = candidate.resolve()

    historical = (repository / "docs/results").resolve()
    reproduction = (repository / "outputs/reproduction").resolve()
    if _within(candidate, historical):
        raise ValueError("bootstrap destination may not be historical evidence")
    if _within(candidate, reproduction):
        raise ValueError("bootstrap destination may not be reproduction output")
    return candidate


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _snapshot_bytes(snapshot: object) -> bytes:
    payload = snapshot.to_json_dict()  # type: ignore[attr-defined]
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def bootstrap_phase(
    *,
    root: Path,
    phase: PhaseDefinition,
    destination: Path,
    replace: bool = False,
) -> BootstrapResult:
    """Recover deterministic documentary source without modifying historical evidence."""

    repository = Path(root).resolve()
    target = _validated_destination(repository, destination)
    canonical_root = (repository / "docs/reproducibility").resolve()
    if replace and not _within(target, canonical_root):
        raise ValueError("replace is permitted only beneath docs/reproducibility")

    if phase.official_report is None:
        raise ValueError(f"phase {phase.phase_id} has no official report")
    if phase.official_report.is_absolute() or ".." in phase.official_report.parts:
        raise ValueError("official report must be a repository-relative path")
    reference = repository / phase.official_report
    if not reference.is_file():
        raise FileNotFoundError(f"official report is missing: {phase.official_report}")

    targets = {name: target / name for name in _OUTPUT_NAMES}
    existing = tuple(path for path in targets.values() if path.exists())
    if existing and not replace:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"bootstrap output already exists: {names}")

    snapshot = extract_docx(reference)
    source_bytes = render_markdown(snapshot).encode("utf-8")
    canonical_source = (
        Path("docs/reproducibility") / phase.phase_id / "report-source.md"
    )
    manifest = build_manifest(
        phase,
        snapshot,
        canonical_source,
        source_bytes,
        audit_status="pending",
    )
    payloads = {
        "report-source.md": source_bytes,
        "report.yaml": dump_manifest(manifest).encode("utf-8"),
        "extraction.json": _snapshot_bytes(snapshot),
    }

    target.mkdir(parents=True, exist_ok=True)
    for name in _OUTPUT_NAMES:
        _atomic_write(targets[name], payloads[name])

    return BootstrapResult(
        phase_id=phase.phase_id,
        destination=target,
        report_source=targets["report-source.md"],
        report_manifest=targets["report.yaml"],
        extraction_snapshot=targets["extraction.json"],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bootstrap_reports.py",
        description="Recover canonical source from an immutable historical AFMC report.",
    )
    parser.add_argument("phase", choices=("phase0", "phase05", "phase06"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = bootstrap_phase(
        root=args.root,
        phase=get_phase(args.phase),
        destination=args.destination,
        replace=args.replace,
    )
    print(f"{result.phase_id}: {result.destination}")
    return 0
