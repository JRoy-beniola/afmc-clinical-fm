from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .integrity import CheckResult
from .models import PhaseDefinition

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PARENT_PLACEHOLDER_RE = re.compile(r"\{parent:([A-Za-z0-9_.-]+)\}")
_ALLOWED_TOP_LEVEL = {
    "schema_version",
    "phase_id",
    "supported",
    "blocked_reason",
    "implementation_sha",
    "command",
    "environment",
    "required_paths",
    "required_parent_bindings",
    "seed_policy",
    "comparison_policy",
}
_ALLOWED_PARENT_KEYS = {"name", "environment_variable", "source", "sha256"}
_ALLOWED_SEED_KEYS = {
    "mode",
    "protected_confirmatory",
    "historical_exact_only",
    "recorded_seeds",
}


@dataclass(frozen=True)
class ParentBinding:
    name: str
    environment_variable: str
    source: Path
    sha256: str | None


@dataclass(frozen=True)
class SeedPolicy:
    mode: str
    protected_confirmatory: bool
    historical_exact_only: bool
    recorded_seeds: tuple[int, ...]


@dataclass(frozen=True)
class RerunSpec:
    phase_id: str
    supported: bool
    blocked_reason: str | None
    implementation_sha: str
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    required_paths: tuple[Path, ...]
    required_parent_bindings: tuple[ParentBinding, ...]
    seed_policy: SeedPolicy
    comparison_policy: dict[str, object]


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping")
    return value


def _safe_relative_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path")
    return Path(*path.parts)


def _string_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings")
    return tuple(value)


def _environment(value: object) -> tuple[tuple[str, str], ...]:
    data = _mapping(value, label="environment")
    if not all(isinstance(item, str) for item in data.values()):
        raise ValueError("environment values must be strings")
    return tuple(data.items())


def _parent_binding(value: object, *, index: int) -> ParentBinding:
    data = _mapping(value, label=f"required_parent_bindings[{index}]")
    unknown = set(data) - _ALLOWED_PARENT_KEYS
    if unknown:
        raise ValueError(
            f"required_parent_bindings[{index}] has unknown keys: {sorted(unknown)}"
        )
    missing = {"name", "environment_variable", "source", "sha256"} - set(data)
    if missing:
        raise ValueError(
            f"required_parent_bindings[{index}] is missing keys: {sorted(missing)}"
        )

    name = data["name"]
    variable = data["environment_variable"]
    digest = data["sha256"]
    if not isinstance(name, str) or not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"required_parent_bindings[{index}].name is invalid")
    if not isinstance(variable, str) or not variable:
        raise ValueError(
            f"required_parent_bindings[{index}].environment_variable must be non-empty"
        )
    if digest is not None and (not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None):
        raise ValueError(f"required_parent_bindings[{index}].sha256 must be null or SHA-256")

    return ParentBinding(
        name=name,
        environment_variable=variable,
        source=_safe_relative_path(
            data["source"], label=f"required_parent_bindings[{index}].source"
        ),
        sha256=digest,
    )


def _seed_policy(value: object) -> SeedPolicy:
    data = _mapping(value, label="seed_policy")
    unknown = set(data) - _ALLOWED_SEED_KEYS
    if unknown:
        raise ValueError(f"seed_policy has unknown keys: {sorted(unknown)}")
    missing = _ALLOWED_SEED_KEYS - set(data)
    if missing:
        raise ValueError(f"seed_policy is missing keys: {sorted(missing)}")

    mode = data["mode"]
    protected = data["protected_confirmatory"]
    exact_only = data["historical_exact_only"]
    seeds = data["recorded_seeds"]
    if not isinstance(mode, str) or not mode:
        raise ValueError("seed_policy.mode must be a non-empty string")
    if not isinstance(protected, bool) or not isinstance(exact_only, bool):
        raise TypeError("seed_policy boolean fields must be booleans")
    if not isinstance(seeds, list) or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise ValueError("seed_policy.recorded_seeds must be a list of integers")

    return SeedPolicy(
        mode=mode,
        protected_confirmatory=protected,
        historical_exact_only=exact_only,
        recorded_seeds=tuple(seeds),
    )


def load_rerun_spec(path: Path) -> RerunSpec:
    """Load one strict, path-safe historical rerun specification."""

    spec_path = Path(path)
    try:
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"unable to load rerun specification: {spec_path}") from exc

    data = _mapping(raw, label="rerun specification")
    unknown = set(data) - _ALLOWED_TOP_LEVEL
    if unknown:
        raise ValueError(f"rerun specification has unknown keys: {sorted(unknown)}")
    missing = _ALLOWED_TOP_LEVEL - set(data)
    if missing:
        raise ValueError(f"rerun specification is missing keys: {sorted(missing)}")
    if data["schema_version"] != 1:
        raise ValueError("rerun specification schema_version must be 1")

    phase_id = data["phase_id"]
    supported = data["supported"]
    blocked_reason = data["blocked_reason"]
    implementation_sha = data["implementation_sha"]
    if not isinstance(phase_id, str) or not phase_id or "/" in phase_id or "\\" in phase_id:
        raise ValueError("phase_id must be a simple non-empty identifier")
    if not isinstance(supported, bool):
        raise TypeError("supported must be a boolean")
    if blocked_reason is not None and not isinstance(blocked_reason, str):
        raise ValueError("blocked_reason must be null or a string")
    if not isinstance(implementation_sha, str) or not implementation_sha:
        raise ValueError("implementation_sha must be a non-empty string")

    required_paths_raw = data["required_paths"]
    if not isinstance(required_paths_raw, list):
        raise TypeError("required_paths must be a list")
    required_paths = tuple(
        _safe_relative_path(item, label=f"required_paths[{index}]")
        for index, item in enumerate(required_paths_raw)
    )

    parents_raw = data["required_parent_bindings"]
    if not isinstance(parents_raw, list):
        raise TypeError("required_parent_bindings must be a list")
    parents = tuple(
        _parent_binding(item, index=index) for index, item in enumerate(parents_raw)
    )
    if len({binding.name for binding in parents}) != len(parents):
        raise ValueError("required_parent_bindings names must be unique")

    comparison_policy = _mapping(data["comparison_policy"], label="comparison_policy")
    return RerunSpec(
        phase_id=phase_id,
        supported=supported,
        blocked_reason=blocked_reason,
        implementation_sha=implementation_sha,
        command=_string_list(data["command"], label="command"),
        environment=_environment(data["environment"]),
        required_paths=required_paths,
        required_parent_bindings=parents,
        seed_policy=_seed_policy(data["seed_policy"]),
        comparison_policy=dict(comparison_policy),
    )


def _check(code: str, ok: bool, subject: str, detail: str) -> CheckResult:
    return CheckResult(code=code, ok=ok, subject=subject, detail=detail)


def _looks_like_official_output(value: str) -> bool:
    normalized = value.replace("\\", "/").casefold()
    return bool(re.search(r"(?:^|/)docs/results(?:/|$)", normalized))


def _parent_placeholders(spec: RerunSpec) -> set[str]:
    values = [*spec.command, *(value for _, value in spec.environment)]
    return {
        match.group(1)
        for value in values
        for match in _PARENT_PLACEHOLDER_RE.finditer(value)
    }


def _parent_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_rerun_spec(
    root: Path,
    phase: PhaseDefinition,
    spec: RerunSpec,
) -> tuple[CheckResult, ...]:
    """Validate a rerun declaration without executing anything."""

    repository = Path(root)
    checks: list[CheckResult] = []

    checks.append(
        _check(
            "phase_binding",
            spec.phase_id == phase.phase_id,
            spec.phase_id,
            "rerun phase matches registry" if spec.phase_id == phase.phase_id else "rerun phase does not match registry",
        )
    )

    implementation_ok = (
        _SHA_RE.fullmatch(spec.implementation_sha) is not None
        and spec.implementation_sha == phase.implementation_sha
    )
    checks.append(
        _check(
            "implementation_binding",
            implementation_ok,
            spec.implementation_sha,
            "implementation SHA matches historical registry"
            if implementation_ok
            else f"implementation must equal registered historical SHA {phase.implementation_sha}",
        )
    )

    unsupported_reason_ok = spec.supported or bool(
        spec.blocked_reason and spec.blocked_reason.strip()
    )
    checks.append(
        _check(
            "unsupported_reason",
            unsupported_reason_ok,
            spec.phase_id,
            "supported rerun or concrete blocker is recorded"
            if unsupported_reason_ok
            else "unsupported rerun requires a concrete blocked_reason",
        )
    )

    command_ok = bool(spec.command)
    checks.append(
        _check(
            "command_required",
            command_ok,
            spec.phase_id,
            "rerun command is declared"
            if command_ok
            else "supported rerun requires a non-empty command",
        )
    )

    for path in spec.required_paths:
        exists = (repository / path).exists()
        checks.append(
            _check(
                "required_path",
                exists,
                path.as_posix(),
                "required rerun path exists" if exists else "required rerun path is missing",
            )
        )

    official_routes = [value for value in spec.command if _looks_like_official_output(value)]
    official_routes.extend(
        value for _, value in spec.environment if _looks_like_official_output(value)
    )
    checks.append(
        _check(
            "official_output_forbidden",
            not official_routes,
            spec.phase_id,
            "rerun declaration does not target official evidence"
            if not official_routes
            else f"rerun declaration targets docs/results/: {official_routes[0]}",
        )
    )

    seed_mode_ok = spec.seed_policy.mode.casefold() != "unrestricted"
    checks.append(
        _check(
            "seed_policy_forbidden",
            seed_mode_ok,
            spec.seed_policy.mode,
            "seed policy is constrained"
            if seed_mode_ok
            else "unrestricted seed policy is forbidden",
        )
    )

    protected_ok = (
        not spec.seed_policy.protected_confirmatory
        or (
            spec.seed_policy.historical_exact_only
            and bool(spec.seed_policy.recorded_seeds)
        )
    )
    checks.append(
        _check(
            "protected_seed_policy",
            protected_ok,
            spec.phase_id,
            "protected seed policy is historical-exact and recorded"
            if protected_ok and spec.seed_policy.protected_confirmatory
            else (
                "protected seed namespace is not requested"
                if protected_ok
                else "protected confirmatory seeds require historical_exact_only and a recorded historical seed set"
            ),
        )
    )

    bindings = {binding.name: binding for binding in spec.required_parent_bindings}
    placeholders = _parent_placeholders(spec)
    missing_bindings = sorted(placeholders - set(bindings))
    checks.append(
        _check(
            "missing_parent_binding",
            not missing_bindings,
            spec.phase_id,
            "all parent placeholders have declared bindings"
            if not missing_bindings
            else f"undeclared parent placeholders: {', '.join(missing_bindings)}",
        )
    )

    for binding in spec.required_parent_bindings:
        source = repository / binding.source
        exists = source.exists()
        checks.append(
            _check(
                "missing_parent_source",
                exists,
                binding.source.as_posix(),
                "parent source exists"
                if exists
                else f"parent source is missing: {binding.source.as_posix()}",
            )
        )
        if not spec.supported:
            continue

        has_digest = binding.sha256 is not None
        checks.append(
            _check(
                "missing_parent_hash",
                has_digest,
                binding.source.as_posix(),
                "parent binding is hash-bound"
                if has_digest
                else "supported rerun requires a SHA-256 parent binding",
            )
        )
        if exists and has_digest:
            observed = _parent_digest(source)
            digest_ok = observed == binding.sha256
            checks.append(
                _check(
                    "parent_hash_mismatch",
                    digest_ok,
                    binding.source.as_posix(),
                    "parent source SHA-256 matches"
                    if digest_ok
                    else f"expected {binding.sha256}, observed {observed or '<non-file>'}",
                )
            )

    return tuple(checks)
