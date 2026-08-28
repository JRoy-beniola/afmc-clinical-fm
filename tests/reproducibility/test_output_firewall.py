from pathlib import Path

import pytest

from afmc_fm.reproducibility.rerun import plan_rerun

from .test_rerun import _fixture_repository


def test_rerun_plan_rejects_symlink_escape(tmp_path: Path, monkeypatch):
    repository, _, _ = _fixture_repository(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    reproduction_root = repository / "outputs/reproduction"
    reproduction_root.mkdir(parents=True)
    (reproduction_root / "fixture").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outputs/reproduction"):
        plan_rerun(repository, "fixture", run_id="symlink-escape")

    assert list(outside.iterdir()) == []
