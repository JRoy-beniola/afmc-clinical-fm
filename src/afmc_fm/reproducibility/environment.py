from __future__ import annotations

import importlib.metadata
import platform as platform_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .integrity import verify_manifest
from .models import EnvironmentStatus, PhaseDefinition

_PROJECT_METADATA_NAMES = (
    "pyproject.toml",
    "requirements.txt",
    "requirements.lock",
    "poetry.lock",
    "uv.lock",
    "Pipfile.lock",
    "environment.yml",
    "environment.yaml",
    "conda-lock.yml",
)
_RUNTIME_KEYS = {
    "python_version",
    "os",
    "platform",
    "cuda_runtime",
    "gpu_name",
    "torch_version",
    "library_versions",
}


@dataclass(frozen=True)
class EnvironmentRecord:
    """Evidence-backed description of a historical or reproduction environment."""

    status: EnvironmentStatus
    python_version: str | None
    platform: str | None
    packages: tuple[tuple[str, str], ...]
    torch: str | None
    cuda: str | None
    gpu: str | None
    source_paths: tuple[Path, ...]
    limitations: tuple[str, ...]


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _runtime_mappings(value: object) -> tuple[dict[str, Any], ...]:
    """Find explicitly recorded runtime mappings without treating arbitrary JSON as environment data."""

    if not isinstance(value, dict):
        return ()

    mappings: list[dict[str, Any]] = []
    if _RUNTIME_KEYS.intersection(value):
        mappings.append(value)

    runtime = value.get("runtime_metadata")
    if isinstance(runtime, dict):
        mappings.append(runtime)

    invocations = value.get("invocations")
    if isinstance(invocations, list):
        for invocation in invocations:
            if not isinstance(invocation, dict):
                continue
            runtime = invocation.get("runtime_metadata")
            if isinstance(runtime, dict):
                mappings.append(runtime)

    return tuple(mappings)


def _load_runtime_metadata(path: Path) -> tuple[dict[str, Any], ...]:
    if path.suffix.casefold() != ".json" or not path.is_file():
        return ()
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ()
    return _runtime_mappings(payload)


def _merge_runtime_metadata(
    mappings: tuple[dict[str, Any], ...],
) -> tuple[str | None, str | None, dict[str, str], str | None, str | None, str | None]:
    python_version: str | None = None
    platform_name: str | None = None
    packages: dict[str, str] = {}
    torch_version: str | None = None
    cuda_runtime: str | None = None
    gpu_name: str | None = None

    for item in mappings:
        python_version = python_version or _string(item.get("python_version"))
        platform_name = platform_name or _string(item.get("os")) or _string(item.get("platform"))
        cuda_runtime = cuda_runtime or _string(item.get("cuda_runtime"))
        gpu_name = gpu_name or _string(item.get("gpu_name"))
        torch_version = torch_version or _string(item.get("torch_version"))

        libraries = item.get("library_versions")
        if isinstance(libraries, dict):
            for name, version in libraries.items():
                if isinstance(name, str) and isinstance(version, str) and version:
                    packages.setdefault(name, version)
            torch_version = torch_version or packages.get("torch")

    return python_version, platform_name, packages, torch_version, cuda_runtime, gpu_name


def _registered_environment_candidates(phase: PhaseDefinition) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for path in (*phase.raw_evidence_paths, *phase.protocol_paths):
        name = path.name.casefold()
        if path.suffix.casefold() == ".lock" or "environment" in name or "container" in name:
            if path not in candidates:
                candidates.append(path)
    return tuple(candidates)


def _verified_exact_environment_paths(root: Path, phase: PhaseDefinition) -> tuple[Path, ...]:
    if phase.environment_status != "exact":
        return ()

    candidates = _registered_environment_candidates(phase)
    if not candidates or not phase.manifests:
        return ()

    verified_subjects: set[str] = set()
    for manifest in phase.manifests:
        for check in verify_manifest(root, phase, manifest):
            if check.ok and check.code == "sha256_match":
                verified_subjects.add(check.subject)

    return tuple(
        path
        for path in candidates
        if (root / path).is_file() and path.as_posix() in verified_subjects
    )


def _project_metadata_paths(root: Path) -> tuple[Path, ...]:
    return tuple(Path(name) for name in _PROJECT_METADATA_NAMES if (root / name).is_file())


def _simple_lock_metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return values
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        key = key.strip().casefold()
        value = value.strip()
        if key and value:
            values.setdefault(key, value)
    return values


def classify_historical_environment(root: Path, phase: PhaseDefinition) -> EnvironmentRecord:
    """Classify historical environment evidence without writing or inferring exactness."""

    repository = Path(root)
    source_paths: list[Path] = []
    runtime_mappings: list[dict[str, Any]] = []

    for relative in phase.raw_evidence_paths:
        mappings = _load_runtime_metadata(repository / relative)
        if mappings:
            source_paths.append(relative)
            runtime_mappings.extend(mappings)

    project_metadata = _project_metadata_paths(repository)
    source_paths.extend(path for path in project_metadata if path not in source_paths)

    candidates = _registered_environment_candidates(phase)
    existing_candidates = tuple(path for path in candidates if (repository / path).is_file())
    source_paths.extend(path for path in existing_candidates if path not in source_paths)

    exact_paths = _verified_exact_environment_paths(repository, phase)
    python_version, platform_name, packages, torch_version, cuda_runtime, gpu_name = (
        _merge_runtime_metadata(tuple(runtime_mappings))
    )

    if exact_paths:
        lock_values = _simple_lock_metadata(repository / exact_paths[0])
        python_version = python_version or lock_values.get("python") or lock_values.get(
            "python_version"
        )
        torch_version = torch_version or lock_values.get("torch") or lock_values.get(
            "torch_version"
        )
        cuda_runtime = cuda_runtime or lock_values.get("cuda") or lock_values.get("cuda_runtime")
        gpu_name = gpu_name or lock_values.get("gpu") or lock_values.get("gpu_name")
        platform_name = platform_name or lock_values.get("platform") or lock_values.get("os")
        return EnvironmentRecord(
            status="exact",
            python_version=python_version,
            platform=platform_name,
            packages=tuple(sorted(packages.items())),
            torch=torch_version,
            cuda=cuda_runtime,
            gpu=gpu_name,
            source_paths=tuple(source_paths),
            limitations=(),
        )

    limitations: list[str] = []
    if runtime_mappings:
        limitations.append(
            "archived runtime metadata is reconstruction evidence, not an exact immutable environment lock"
        )
    if project_metadata:
        limitations.append(
            "project/dependency metadata supports reconstruction only and cannot establish exact historical identity"
        )
    if phase.environment_status == "exact":
        limitations.append(
            "registered exact environment status could not be validated by a digest-verified environment lock/container"
        )

    has_reconstruction_evidence = bool(runtime_mappings or project_metadata or existing_candidates)
    status: EnvironmentStatus = "reconstructed" if has_reconstruction_evidence else "unknown"
    return EnvironmentRecord(
        status=status,
        python_version=python_version,
        platform=platform_name,
        packages=tuple(sorted(packages.items())),
        torch=torch_version,
        cuda=cuda_runtime,
        gpu=gpu_name,
        source_paths=tuple(source_paths),
        limitations=tuple(limitations),
    )


def _installed_packages() -> tuple[tuple[str, str], ...]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if isinstance(name, str) and name and isinstance(version, str) and version:
            packages.setdefault(name, version)
    return tuple(sorted(packages.items(), key=lambda item: item[0].casefold()))


def capture_current_environment() -> EnvironmentRecord:
    """Capture the environment performing a new reproduction, never historical identity."""

    packages = _installed_packages()
    package_map = dict(packages)
    torch_version = package_map.get("torch")
    cuda_runtime: str | None = None
    gpu_name: str | None = None

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        torch_version = str(torch.__version__)
        cuda_runtime = _string(torch.version.cuda)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)

    return EnvironmentRecord(
        status="reconstructed",
        python_version=platform_module.python_version(),
        platform=platform_module.platform(),
        packages=packages,
        torch=torch_version,
        cuda=cuda_runtime,
        gpu=gpu_name,
        source_paths=(),
        limitations=(
            "current reproduction environment captured now; it is not evidence of the historical execution environment",
        ),
    )
