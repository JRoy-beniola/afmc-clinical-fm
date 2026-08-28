from __future__ import annotations

import re
import shlex
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )


def _require_repository(repository: Path) -> Path:
    candidate = Path(repository).resolve()
    if not candidate.is_dir():
        raise ValueError(f"Git repository does not exist: {candidate}")

    probe = _git(candidate, "rev-parse", "--show-toplevel")
    if probe.returncode != 0:
        raise ValueError(f"not a Git repository: {candidate}")
    return candidate


def _require_historical_commit(repository: Path, sha: str) -> None:
    if _SHA_RE.fullmatch(sha) is None:
        raise ValueError("historical commit must be a 40-character lowercase hexadecimal SHA")

    probe = _git(repository, "cat-file", "-e", f"{sha}^{{commit}}")
    if probe.returncode != 0:
        raise ValueError(f"historical commit is not available in this repository: {sha}")


@contextmanager
def historical_worktree(repository: Path, sha: str) -> Iterator[Path]:
    """Yield a detached temporary worktree at exactly the requested historical commit."""

    repo = _require_repository(repository)
    _require_historical_commit(repo, sha)

    with tempfile.TemporaryDirectory(prefix="afmc-repro-worktree-") as temporary_parent:
        worktree = Path(temporary_parent) / "worktree"
        added = False
        try:
            created = _git(repo, "worktree", "add", "--detach", str(worktree), sha)
            if created.returncode != 0:
                detail = created.stderr.strip() or created.stdout.strip() or "unknown Git error"
                raise RuntimeError(f"unable to create historical worktree: {detail}")
            added = True

            resolved = _git(worktree, "rev-parse", "HEAD")
            resolved_sha = resolved.stdout.strip() if resolved.returncode == 0 else ""
            if resolved.returncode != 0 or resolved_sha != sha:
                raise RuntimeError(
                    "historical worktree HEAD mismatch: "
                    f"requested {sha}, observed {resolved_sha or '<unresolved>'}"
                )

            yield worktree
        finally:
            if added:
                removed = _git(repo, "worktree", "remove", "--force", str(worktree))
                if removed.returncode != 0:
                    detail = removed.stderr.strip() or removed.stdout.strip() or "unknown Git error"
                    raise RuntimeError(f"unable to remove temporary historical worktree: {detail}")


def run_in_worktree(
    worktree: Path,
    command: tuple[str, ...],
    env: Mapping[str, str],
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Run one declared command in a worktree and persist captured stdout/stderr."""

    directory = Path(worktree)
    if not directory.is_dir():
        raise ValueError(f"worktree does not exist: {directory}")
    if not command:
        raise ValueError("worktree command must not be empty")

    completed = subprocess.run(
        list(command),
        cwd=directory,
        env=dict(env),
        check=False,
        capture_output=True,
        text=True,
    )

    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "command: "
        + shlex.join(command)
        + "\n"
        + f"returncode: {completed.returncode}\n"
        + "\n[stdout]\n"
        + completed.stdout
        + "\n[stderr]\n"
        + completed.stderr,
        encoding="utf-8",
    )
    return completed
