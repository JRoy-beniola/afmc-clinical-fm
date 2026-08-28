import hashlib
from pathlib import Path

import pytest

from afmc_fm.reproducibility.rerun import plan_rerun
from afmc_fm.reproducibility.verify import verify_phase

from .test_rerun import _fixture_repository
from .test_verify import fixture_phase, write_valid_fixture


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


def test_manifest_rejects_symlink_escape_from_declared_base(tmp_path: Path):
    write_valid_fixture(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("external evidence\n", encoding="utf-8")
    evidence_root = tmp_path / "docs/results/fixture"
    (evidence_root / "linked.csv").symlink_to(outside)
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    (evidence_root / "MANIFEST.sha256").write_text(
        f"{digest}  linked.csv\n",
        encoding="utf-8",
    )

    report = verify_phase(tmp_path, fixture_phase())

    assert any(
        check.code == "unsafe_manifest_path" and "linked.csv" in check.detail
        for check in report.checks
    )


def test_manifest_rejects_windows_style_parent_traversal(tmp_path: Path):
    write_valid_fixture(tmp_path)
    manifest = tmp_path / "docs/results/fixture/MANIFEST.sha256"
    manifest.write_text("0" * 64 + "  ..\\outside.csv\n", encoding="utf-8")

    report = verify_phase(tmp_path, fixture_phase())

    assert any(check.code == "unsafe_manifest_path" for check in report.checks)
