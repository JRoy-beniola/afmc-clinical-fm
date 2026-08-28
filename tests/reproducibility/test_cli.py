from pathlib import Path
from types import SimpleNamespace

import pytest

from afmc_fm.reproducibility import cli


def _fake_report(root: Path, phase_id: str, *, ok: bool = True):
    destination = root / "outputs/reproduction" / phase_id / "rebuild"
    return SimpleNamespace(
        phase_id=phase_id,
        destination=destination,
        generated_count=1,
        reference_copy_count=2,
        candidate_report=destination / "report/candidate.docx",
        structural_comparison=SimpleNamespace(ok=ok),
        ok=ok,
    )


def test_status_lists_all_phases(capsys):
    assert cli.main(["status"]) == 0
    output = capsys.readouterr().out
    for phase in ("phase0", "phase05", "phase06", "phase06-posthoc"):
        assert phase in output


def test_unknown_verify_phase_is_argparse_error():
    with pytest.raises(SystemExit) as exc:
        cli.main(["verify", "phase07"])
    assert exc.value.code == 2


def test_verify_all_aggregates_failures(tmp_path, capsys):
    assert cli.main(["verify", "all"], root=tmp_path) == 1
    output = capsys.readouterr().out
    for phase in ("phase0", "phase05", "phase06", "phase06-posthoc"):
        assert f"FAIL {phase}" in output


def test_rebuild_single_reports_provenance_counts_and_candidate(tmp_path, capsys, monkeypatch):
    calls = []

    def fake_rebuild(root, phase_id, destination=None):
        calls.append((Path(root), phase_id, destination))
        return _fake_report(Path(root), phase_id)

    monkeypatch.setattr(cli, "rebuild_phase", fake_rebuild)

    assert cli.main(["rebuild", "phase0"], root=tmp_path) == 0
    output = capsys.readouterr().out
    assert "PASS phase0" in output
    assert "generated=1" in output
    assert "reference-copy=2" in output
    assert "structural=PASS" in output
    assert "report/candidate.docx" in output
    assert calls == [(tmp_path, "phase0", None)]


def test_rebuild_single_accepts_custom_destination(tmp_path, monkeypatch):
    calls = []
    destination = tmp_path / "outputs/reproduction/phase0/rebuild/custom"

    def fake_rebuild(root, phase_id, destination=None):
        calls.append((Path(root), phase_id, destination))
        return _fake_report(Path(root), phase_id)

    monkeypatch.setattr(cli, "rebuild_phase", fake_rebuild)

    assert (
        cli.main(
            ["rebuild", "phase0", "--destination", str(destination)],
            root=tmp_path,
        )
        == 0
    )
    assert calls == [(tmp_path, "phase0", destination)]


def test_rebuild_unsupported_single_phase_is_nonzero(capsys):
    assert cli.main(["rebuild", "phase06-posthoc"]) == 1
    assert "UNSUPPORTED phase06-posthoc" in capsys.readouterr().out


def test_rebuild_all_runs_supported_and_reports_unsupported(tmp_path, capsys, monkeypatch):
    called = []

    def fake_rebuild(root, phase_id, destination=None):
        called.append(phase_id)
        return _fake_report(Path(root), phase_id)

    monkeypatch.setattr(cli, "rebuild_phase", fake_rebuild)

    assert cli.main(["rebuild", "all"], root=tmp_path) == 0
    output = capsys.readouterr().out
    assert called == ["phase0", "phase05", "phase06"]
    for phase_id in called:
        assert f"PASS {phase_id}" in output
    assert "UNSUPPORTED phase06-posthoc" in output


def test_rebuild_all_rejects_custom_destination(tmp_path):
    destination = tmp_path / "outputs/reproduction/custom"
    with pytest.raises(SystemExit) as exc:
        cli.main(["rebuild", "all", "--destination", str(destination)], root=tmp_path)
    assert exc.value.code == 2


def test_failed_rebuild_returns_nonzero(tmp_path, capsys, monkeypatch):
    def fake_rebuild(root, phase_id, destination=None):
        return _fake_report(Path(root), phase_id, ok=False)

    monkeypatch.setattr(cli, "rebuild_phase", fake_rebuild)

    assert cli.main(["rebuild", "phase0"], root=tmp_path) == 1
    output = capsys.readouterr().out
    assert "FAIL phase0" in output
    assert "structural=FAIL" in output


def test_unknown_rebuild_phase_is_argparse_error():
    with pytest.raises(SystemExit) as exc:
        cli.main(["rebuild", "phase07"])
    assert exc.value.code == 2
