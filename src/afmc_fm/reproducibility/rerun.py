from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .environment import (
    EnvironmentRecord,
    capture_current_environment,
    classify_historical_environment,
)
from .registry import get_phase, iter_phases
from .rerun_models import RerunSpec, load_rerun_spec, validate_rerun_spec
from .worktree import historical_worktree, run_in_worktree

RerunStatus = Literal[
    "READY",
    "BLOCKED_MISSING_HISTORY",
    "BLOCKED_MISSING_ENVIRONMENT",
    "BLOCKED_MISSING_PARENT",
    "BLOCKED_POLICY",
    "UNSUPPORTED",
]
ExecutionStatus = Literal["SUCCEEDED", "FAILED"]

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PARENT_PLACEHOLDER_RE = re.compile(r"\{parent:([A-Za-z0-9_.-]+)\}")
_PARENT_FAILURE_CODES = {
    "missing_parent_binding",
    "missing_parent_source",
    "missing_parent_hash",
    "parent_hash_mismatch",
}


@dataclass(frozen=True)
class RerunPlan:
    phase_id: str
    status: RerunStatus
    reasons: tuple[str, ...]
    remediation: tuple[str, ...]
    implementation_sha: str
    destination: Path
    spec_path: Path
    spec: RerunSpec
    historical_environment: EnvironmentRecord

    @property
    def ready(self) -> bool:
        return self.status == "READY"


@dataclass(frozen=True)
class RerunExecution:
    phase_id: str
    status: ExecutionStatus
    returncode: int
    implementation_sha: str
    environment: EnvironmentRecord
    destination: Path
    execution_record: Path
    inventory_path: Path
    log_path: Path
    historical_archive_unchanged: bool

    @property
    def ok(self) -> bool:
        return (
            self.status == "SUCCEEDED"
            and self.returncode == 0
            and self.historical_archive_unchanged
        )


def _run_id(value: str | None) -> str:
    candidate = value or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if _RUN_ID_RE.fullmatch(candidate) is None or candidate in {".", ".."}:
        raise ValueError("run_id must be a safe filesystem identifier")
    return candidate


def _spec_path(phase_id: str) -> Path:
    return Path("docs/reproducibility") / phase_id / "rerun.yaml"


def _rerun_destination(repository: Path, phase_id: str, run_id: str | None) -> Path:
    reproduction_root = (repository / "outputs/reproduction").resolve()
    if reproduction_root != repository and not reproduction_root.is_relative_to(repository):
        raise ValueError("outputs/reproduction must resolve within the repository")

    rerun_root = (repository / "outputs/reproduction" / phase_id / "rerun").resolve()
    expected = Path("outputs/reproduction") / phase_id / "rerun"
    if rerun_root != reproduction_root and not rerun_root.is_relative_to(reproduction_root):
        raise ValueError(f"rerun destination must be beneath {expected.as_posix()}")

    destination = (rerun_root / _run_id(run_id)).resolve()
    if destination != rerun_root and not destination.is_relative_to(rerun_root):
        raise ValueError(f"rerun destination must be beneath {expected.as_posix()}")
    return destination


def _commit_available(repository: Path, sha: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _path_available_at_commit(repository: Path, sha: str, path: Path) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}:{path.as_posix()}"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _plan(
    *,
    phase_id: str,
    status: RerunStatus,
    reasons: tuple[str, ...],
    remediation: tuple[str, ...],
    implementation_sha: str,
    destination: Path,
    spec_path: Path,
    spec: RerunSpec,
    historical_environment: EnvironmentRecord,
) -> RerunPlan:
    return RerunPlan(
        phase_id=phase_id,
        status=status,
        reasons=reasons,
        remediation=remediation,
        implementation_sha=implementation_sha,
        destination=destination,
        spec_path=spec_path,
        spec=spec,
        historical_environment=historical_environment,
    )


def plan_rerun(
    root: Path,
    phase_id: str,
    *,
    run_id: str | None = None,
) -> RerunPlan:
    """Plan one historical rerun without creating output or launching a process."""

    repository = Path(root).resolve()
    phase = get_phase(phase_id)
    relative_spec = _spec_path(phase_id)
    specification = load_rerun_spec(repository / relative_spec)
    destination = _rerun_destination(repository, phase_id, run_id)
    historical_environment = classify_historical_environment(repository, phase)

    if not phase.rerun_supported or not specification.supported:
        reasons = tuple(
            reason
            for reason in (
                "registry marks historical rerun unsupported"
                if not phase.rerun_supported
                else None,
                specification.blocked_reason,
            )
            if reason
        )
        return _plan(
            phase_id=phase_id,
            status="UNSUPPORTED",
            reasons=reasons or ("historical rerun is unsupported",),
            remediation=("satisfy and independently audit the recorded historical prerequisites",),
            implementation_sha=specification.implementation_sha,
            destination=destination,
            spec_path=relative_spec,
            spec=specification,
            historical_environment=historical_environment,
        )

    checks = validate_rerun_spec(repository, phase, specification)
    failed = tuple(check for check in checks if not check.ok)
    if failed:
        parent_failures = tuple(
            check for check in failed if check.code in _PARENT_FAILURE_CODES
        )
        status: RerunStatus = (
            "BLOCKED_MISSING_PARENT" if parent_failures else "BLOCKED_POLICY"
        )
        return _plan(
            phase_id=phase_id,
            status=status,
            reasons=tuple(f"{check.code}: {check.detail}" for check in failed),
            remediation=("repair the declared rerun specification or immutable prerequisites",),
            implementation_sha=specification.implementation_sha,
            destination=destination,
            spec_path=relative_spec,
            spec=specification,
            historical_environment=historical_environment,
        )

    if not _commit_available(repository, specification.implementation_sha):
        return _plan(
            phase_id=phase_id,
            status="BLOCKED_MISSING_HISTORY",
            reasons=(
                f"historical implementation commit is unavailable: {specification.implementation_sha}",
            ),
            remediation=("restore or fetch the exact registered historical commit",),
            implementation_sha=specification.implementation_sha,
            destination=destination,
            spec_path=relative_spec,
            spec=specification,
            historical_environment=historical_environment,
        )

    missing_historical_paths = tuple(
        path
        for path in specification.required_paths
        if not _path_available_at_commit(
            repository,
            specification.implementation_sha,
            path,
        )
    )
    if missing_historical_paths:
        return _plan(
            phase_id=phase_id,
            status="BLOCKED_POLICY",
            reasons=tuple(
                "historical_required_path_missing: " + path.as_posix()
                for path in missing_historical_paths
            ),
            remediation=(
                "bind required execution paths to files present at the historical implementation commit",
            ),
            implementation_sha=specification.implementation_sha,
            destination=destination,
            spec_path=relative_spec,
            spec=specification,
            historical_environment=historical_environment,
        )

    if historical_environment.status == "unknown":
        return _plan(
            phase_id=phase_id,
            status="BLOCKED_MISSING_ENVIRONMENT",
            reasons=("insufficient historical environment evidence for a controlled rerun",),
            remediation=("provide auditable reconstruction evidence without claiming exactness",),
            implementation_sha=specification.implementation_sha,
            destination=destination,
            spec_path=relative_spec,
            spec=specification,
            historical_environment=historical_environment,
        )

    if destination.exists():
        return _plan(
            phase_id=phase_id,
            status="BLOCKED_POLICY",
            reasons=(f"isolated rerun destination already exists: {destination}",),
            remediation=("choose a fresh run_id; existing reproduction output is never overwritten",),
            implementation_sha=specification.implementation_sha,
            destination=destination,
            spec_path=relative_spec,
            spec=specification,
            historical_environment=historical_environment,
        )

    return _plan(
        phase_id=phase_id,
        status="READY",
        reasons=(),
        remediation=(),
        implementation_sha=specification.implementation_sha,
        destination=destination,
        spec_path=relative_spec,
        spec=specification,
        historical_environment=historical_environment,
    )


def _tree_snapshot(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _historical_snapshots(root: Path) -> dict[str, dict[str, str]]:
    snapshots: dict[str, dict[str, str]] = {}
    for phase in iter_phases():
        evidence_root = root / phase.official_evidence_root
        if evidence_root.is_dir():
            snapshots[phase.phase_id] = _tree_snapshot(evidence_root)
    return snapshots


def _parent_paths(root: Path, specification: RerunSpec) -> dict[str, Path]:
    return {
        binding.name: (root / binding.source).resolve()
        for binding in specification.required_parent_bindings
    }


def _render_value(
    value: str,
    *,
    destination: Path,
    worktree: Path,
    parents: dict[str, Path],
) -> str:
    rendered = value.replace("{output}", str(destination)).replace(
        "{worktree}", str(worktree)
    )
    for name, path in parents.items():
        rendered = rendered.replace(f"{{parent:{name}}}", str(path))
    if _PARENT_PLACEHOLDER_RE.search(rendered):
        raise ValueError(f"unresolved parent placeholder in rerun value: {value}")
    return rendered


def _environment_payload(record: EnvironmentRecord) -> dict[str, object]:
    return {
        "status": record.status,
        "python_version": record.python_version,
        "platform": record.platform,
        "packages": {name: version for name, version in record.packages},
        "torch": record.torch,
        "cuda": record.cuda,
        "gpu": record.gpu,
        "source_paths": [path.as_posix() for path in record.source_paths],
        "limitations": list(record.limitations),
    }


def _output_inventory(destination: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.name in {"execution.json", "output-inventory.json"}:
            continue
        data = path.read_bytes()
        inventory.append(
            {
                "path": path.relative_to(destination).as_posix(),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return inventory


def execute_rerun(root: Path, plan: RerunPlan) -> RerunExecution:
    """Execute one READY rerun in an exact historical detached worktree."""

    if not plan.ready:
        raise ValueError(f"rerun plan is not READY: {plan.status}")

    repository = Path(root).resolve()
    fresh_plan = plan_rerun(repository, plan.phase_id, run_id=plan.destination.name)
    if not fresh_plan.ready:
        raise ValueError(
            "rerun prerequisites changed after planning: "
            f"{fresh_plan.status}: {'; '.join(fresh_plan.reasons)}"
        )

    before = _historical_snapshots(repository)
    destination = fresh_plan.destination
    destination.mkdir(parents=True, exist_ok=False)
    log_path = destination / "logs/command.log"
    parents = _parent_paths(repository, fresh_plan.spec)
    started_at = datetime.now(UTC).isoformat()

    with historical_worktree(repository, fresh_plan.implementation_sha) as worktree:
        environment_record = capture_current_environment()
        environment = dict(os.environ)
        environment.update(
            {
                key: _render_value(
                    value,
                    destination=destination,
                    worktree=worktree,
                    parents=parents,
                )
                for key, value in fresh_plan.spec.environment
            }
        )
        command = tuple(
            _render_value(
                value,
                destination=destination,
                worktree=worktree,
                parents=parents,
            )
            for value in fresh_plan.spec.command
        )
        completed = run_in_worktree(worktree, command, environment, log_path)

    unchanged = before == _historical_snapshots(repository)
    inventory_path = destination / "output-inventory.json"
    inventory_path.write_text(
        json.dumps(_output_inventory(destination), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status: ExecutionStatus = (
        "SUCCEEDED" if completed.returncode == 0 and unchanged else "FAILED"
    )
    execution_record = destination / "execution.json"
    execution_record.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase_id": fresh_plan.phase_id,
                "status": status,
                "implementation_sha": fresh_plan.implementation_sha,
                "command": list(command),
                "returncode": completed.returncode,
                "started_at": started_at,
                "finished_at": datetime.now(UTC).isoformat(),
                "destination": str(destination),
                "environment": _environment_payload(environment_record),
                "historical_environment_status": fresh_plan.historical_environment.status,
                "historical_archive_unchanged": unchanged,
                "inventory": str(inventory_path),
                "log": str(log_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return RerunExecution(
        phase_id=fresh_plan.phase_id,
        status=status,
        returncode=completed.returncode,
        implementation_sha=fresh_plan.implementation_sha,
        environment=environment_record,
        destination=destination,
        execution_record=execution_record,
        inventory_path=inventory_path,
        log_path=log_path,
        historical_archive_unchanged=unchanged,
    )
