from pathlib import Path

from afmc_fm.cli import main


def test_phase05_report_dispatches_read_only_reporting(tmp_path: Path, monkeypatch):
    calls: list[Path] = []

    def fake_report(output: str | Path):
        calls.append(Path(output))
        return {}

    monkeypatch.setattr(
        "afmc_fm.phase05.reporting.write_phase05_report_artifacts",
        fake_report,
    )

    assert main(["phase05", "report", "--output", str(tmp_path)]) == 0
    assert calls == [tmp_path]
