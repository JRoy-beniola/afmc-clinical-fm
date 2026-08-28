import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

worktree_module = importlib.import_module("afmc_fm.reproducibility.worktree")
historical_worktree = worktree_module.historical_worktree
run_in_worktree = worktree_module.run_in_worktree


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _fixture_repository(tmp_path: Path) -> tuple[Path, str, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "reproducibility@example.invalid")
    _git(repository, "config", "user.name", "Reproducibility Test")

    marker = repository / "marker.txt"
    marker.write_text("HISTORICAL\n", encoding="utf-8")
    _git(repository, "add", "marker.txt")
    _git(repository, "commit", "-m", "historical")
    historical_sha = _git(repository, "rev-parse", "HEAD")

    marker.write_text("CURRENT\n", encoding="utf-8")
    _git(repository, "add", "marker.txt")
    _git(repository, "commit", "-m", "current")
    current_sha = _git(repository, "rev-parse", "HEAD")
    return repository, historical_sha, current_sha


def test_historical_worktree_checks_out_requested_commit_and_cleans_up(tmp_path: Path):
    repository, historical_sha, current_sha = _fixture_repository(tmp_path)
    created_path: Path | None = None

    with historical_worktree(repository, historical_sha) as worktree:
        created_path = worktree
        assert worktree != repository
        assert _git(worktree, "rev-parse", "HEAD") == historical_sha
        assert (worktree / "marker.txt").read_text(encoding="utf-8") == "HISTORICAL\n"
        assert _git(repository, "rev-parse", "HEAD") == current_sha
        assert (repository / "marker.txt").read_text(encoding="utf-8") == "CURRENT\n"

    assert created_path is not None
    assert not created_path.exists()
    assert _git(repository, "rev-parse", "HEAD") == current_sha


def test_missing_historical_sha_fails_without_current_head_fallback(tmp_path: Path):
    repository, _, current_sha = _fixture_repository(tmp_path)
    missing_sha = "f" * 40

    with pytest.raises(ValueError, match="historical commit"):
        with historical_worktree(repository, missing_sha):
            pytest.fail("missing historical SHA must not yield a worktree")

    assert _git(repository, "rev-parse", "HEAD") == current_sha
    assert (repository / "marker.txt").read_text(encoding="utf-8") == "CURRENT\n"


def test_historical_worktree_does_not_remove_existing_user_worktree(tmp_path: Path):
    repository, historical_sha, _ = _fixture_repository(tmp_path)
    user_worktree = tmp_path / "user-worktree"
    _git(repository, "worktree", "add", "--detach", str(user_worktree), "HEAD")

    try:
        with historical_worktree(repository, historical_sha) as worktree:
            assert worktree != user_worktree
            assert user_worktree.exists()
        assert user_worktree.exists()
        assert (user_worktree / "marker.txt").is_file()
    finally:
        _git(repository, "worktree", "remove", "--force", str(user_worktree))


def test_run_in_worktree_captures_output_and_environment(tmp_path: Path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    log_path = tmp_path / "logs" / "command.log"
    environment = dict(os.environ)
    environment["AFMC_REPRO_TEST"] = "bound-value"

    completed = run_in_worktree(
        worktree,
        (
            sys.executable,
            "-c",
            "import os, sys; print(os.environ['AFMC_REPRO_TEST']); print('stderr-line', file=sys.stderr)",
        ),
        environment,
        log_path,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "bound-value"
    assert completed.stderr.strip() == "stderr-line"
    log = log_path.read_text(encoding="utf-8")
    assert "bound-value" in log
    assert "stderr-line" in log
