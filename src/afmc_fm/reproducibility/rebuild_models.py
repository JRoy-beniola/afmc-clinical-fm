from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml

ArtifactMode = Literal["generate", "reference-copy", "source-render"]
_ALLOWED_MODES = {"generate", "reference-copy", "source-render"}
_MANIFEST_KEYS = {
    "schema_version",
    "phase_id",
    "report_source",
    "reference_report",
    "artifacts",
    "document_output",
}
_ARTIFACT_KEYS = {"kind", "source", "output", "mode", "generator"}


@dataclass(frozen=True)
class ArtifactDeclaration:
    kind: str
    source: Path
    output: Path
    mode: ArtifactMode
    generator: str | None = None


@dataclass(frozen=True)
class RebuildManifest:
    phase_id: str
    report_source: Path
    reference_report: Path
    artifacts: tuple[ArtifactDeclaration, ...]
    document_output: Path


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping")
    return value


def _relative_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path")
    return Path(*path.parts)


def _artifact(value: object, *, index: int) -> ArtifactDeclaration:
    data = _mapping(value, label=f"artifacts[{index}]")
    unknown = set(data) - _ARTIFACT_KEYS
    if unknown:
        raise ValueError(f"artifacts[{index}] has unknown keys: {sorted(unknown)}")

    required = {"kind", "source", "output", "mode"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"artifacts[{index}] is missing keys: {sorted(missing)}")

    kind = data["kind"]
    mode = data["mode"]
    generator = data.get("generator")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"artifacts[{index}].kind must be a non-empty string")
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"artifacts[{index}].mode must be one of {sorted(_ALLOWED_MODES)}")
    if mode == "generate":
        if not isinstance(generator, str) or not generator:
            raise ValueError(f"artifacts[{index}] generate mode requires a generator")
    elif generator is not None:
        raise ValueError(f"artifacts[{index}] {mode} mode forbids a generator")

    return ArtifactDeclaration(
        kind=kind,
        source=_relative_path(data["source"], label=f"artifacts[{index}].source"),
        output=_relative_path(data["output"], label=f"artifacts[{index}].output"),
        mode=mode,
        generator=generator,
    )


def load_rebuild_manifest(path: Path) -> RebuildManifest:
    """Load a strict, path-safe deterministic rebuild manifest."""

    manifest_path = Path(path)
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"unable to load rebuild manifest: {manifest_path}") from exc

    data = _mapping(raw, label="rebuild manifest")
    unknown = set(data) - _MANIFEST_KEYS
    if unknown:
        raise ValueError(f"rebuild manifest has unknown keys: {sorted(unknown)}")
    missing = _MANIFEST_KEYS - set(data)
    if missing:
        raise ValueError(f"rebuild manifest is missing keys: {sorted(missing)}")
    if data["schema_version"] != 1:
        raise ValueError("rebuild manifest schema_version must be 1")

    phase_id = data["phase_id"]
    if not isinstance(phase_id, str) or not phase_id or "/" in phase_id or "\\" in phase_id:
        raise ValueError("phase_id must be a simple non-empty identifier")

    artifacts_raw = data["artifacts"]
    if not isinstance(artifacts_raw, list):
        raise TypeError("artifacts must be a list")

    return RebuildManifest(
        phase_id=phase_id,
        report_source=_relative_path(data["report_source"], label="report_source"),
        reference_report=_relative_path(data["reference_report"], label="reference_report"),
        artifacts=tuple(_artifact(item, index=index) for index, item in enumerate(artifacts_raw)),
        document_output=_relative_path(data["document_output"], label="document_output"),
    )


def validate_rebuild_destination(
    root: Path,
    phase_id: str,
    destination: Path | None,
) -> Path:
    """Resolve a rebuild destination and enforce the reproduction-tree firewall."""

    repository = Path(root).resolve()
    if not phase_id or "/" in phase_id or "\\" in phase_id or phase_id in {".", ".."}:
        raise ValueError("phase_id must be a simple non-empty identifier")

    reproduction_root = (repository / "outputs/reproduction").resolve()
    if reproduction_root != repository and not reproduction_root.is_relative_to(repository):
        raise ValueError("outputs/reproduction must resolve within the repository")

    allowed_root = (repository / "outputs/reproduction" / phase_id / "rebuild").resolve()
    if allowed_root != reproduction_root and not allowed_root.is_relative_to(reproduction_root):
        expected = Path("outputs/reproduction") / phase_id / "rebuild"
        raise ValueError(f"rebuild destination must be beneath {expected.as_posix()}")

    if destination is None:
        return allowed_root

    candidate = Path(destination)
    if not candidate.is_absolute():
        candidate = repository / candidate
    candidate = candidate.resolve()
    if candidate != allowed_root and not candidate.is_relative_to(allowed_root):
        expected = Path("outputs/reproduction") / phase_id / "rebuild"
        raise ValueError(f"rebuild destination must be beneath {expected.as_posix()}")
    return candidate
