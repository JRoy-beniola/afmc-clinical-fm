from __future__ import annotations

import hashlib
import importlib
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .rebuild_models import ArtifactDeclaration

Generator = Callable[[Path, Path, Path], None]


@dataclass(frozen=True)
class ArtifactResult:
    output: Path
    mode: str
    sha256: str
    generated: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def _restore(root: Path, snapshot: dict[str, bytes]) -> None:
    if root.exists():
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file() and path.relative_to(root).as_posix() not in snapshot:
                path.unlink()
    for relative, payload in snapshot.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def _load_generator(entry_point: str) -> Generator:
    module_name, separator, attribute = entry_point.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("generator entry point must use 'module:function' syntax")
    module = importlib.import_module(module_name)
    generator = getattr(module, attribute, None)
    if not callable(generator):
        raise TypeError(f"generator is not callable: {entry_point}")
    return generator


def _changed_paths(before: dict[str, bytes], after: dict[str, bytes]) -> set[str]:
    names = set(before) | set(after)
    return {name for name in names if before.get(name) != after.get(name)}


def _safe_output(destination: Path, relative: Path) -> Path:
    base = destination.resolve()
    output = (base / relative).resolve()
    if output != base and not output.is_relative_to(base):
        raise ValueError("artifact output escapes rebuild destination")
    return output


def materialize_artifact(
    root: Path,
    destination: Path,
    declaration: ArtifactDeclaration,
) -> ArtifactResult:
    """Materialize one declared artifact without mutating historical evidence."""

    repository = Path(root).resolve()
    destination_path = Path(destination).resolve()
    source = (repository / declaration.source).resolve()
    if source != repository and not source.is_relative_to(repository):
        raise ValueError("artifact source escapes repository")
    if not source.is_file():
        raise FileNotFoundError(source)

    output = _safe_output(destination_path, declaration.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if declaration.mode == "reference-copy":
        shutil.copy2(source, output)
        return ArtifactResult(
            output=output,
            mode=declaration.mode,
            sha256=_sha256(output),
            generated=False,
        )

    if declaration.mode != "generate":
        raise ValueError(f"artifact mode is not materializable by this adapter: {declaration.mode}")
    if declaration.generator is None:
        raise ValueError("generate mode requires a generator")

    generator = _load_generator(declaration.generator)
    historical_root = repository / "docs/results"
    historical_before = _snapshot(historical_root)
    repository_before = _snapshot(repository)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")

    try:
        generator(repository, source, temporary)
        if not temporary.is_file():
            raise RuntimeError("generator did not create its declared output")

        historical_after = _snapshot(historical_root)
        if historical_after != historical_before:
            _restore(historical_root, historical_before)
            raise RuntimeError("generator modified historical archive")

        repository_after = _snapshot(repository)
        allowed = temporary.relative_to(repository).as_posix()
        changes = _changed_paths(repository_before, repository_after) - {allowed}
        if changes:
            raise RuntimeError(
                "generator wrote outside declared output: " + ", ".join(sorted(changes))
            )

        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    return ArtifactResult(
        output=output,
        mode=declaration.mode,
        sha256=_sha256(output),
        generated=True,
    )
