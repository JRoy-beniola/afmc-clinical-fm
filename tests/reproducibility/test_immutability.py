import hashlib
from pathlib import Path

from afmc_fm.reproducibility.cli import main
from afmc_fm.reproducibility.registry import get_phase, iter_phases
from afmc_fm.reproducibility.report_audit import audit_report_source


def tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def historical_snapshots(repository: Path) -> dict[str, dict[str, str]]:
    return {
        phase.phase_id: tree_snapshot(repository / phase.official_evidence_root)
        for phase in iter_phases()
    }


def test_verify_all_preserves_historical_archives(capsys):
    repository = Path(".")
    before = historical_snapshots(repository)

    exit_code = main(["verify", "all"], root=repository)
    assert exit_code in {0, 1}
    output = capsys.readouterr().out
    for phase in iter_phases():
        assert phase.phase_id in output

    assert historical_snapshots(repository) == before
    assert not (repository / "outputs/reproduction").exists()


def test_report_source_audit_preserves_historical_archives():
    repository = Path(".")
    before = historical_snapshots(repository)

    for phase_id in ("phase0", "phase05", "phase06"):
        phase = get_phase(phase_id)
        bundle = repository / "docs/reproducibility" / phase_id
        report = audit_report_source(repository, phase, bundle)
        assert report.ok, [
            (check.code, check.detail) for check in report.checks if not check.ok
        ]

    assert historical_snapshots(repository) == before
    assert not (repository / "outputs/reproduction").exists()
