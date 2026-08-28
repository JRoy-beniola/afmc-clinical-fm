import hashlib
import shutil
from pathlib import Path

from afmc_fm.reproducibility.cli import main
from afmc_fm.reproducibility.rebuild import rebuild_phase
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


def test_supported_rebuilds_are_isolated_and_preserve_all_historical_archives(tmp_path: Path):
    repository = Path.cwd()
    reproduction_root = (repository / "outputs/reproduction").resolve()
    before = historical_snapshots(repository)
    destinations: list[Path] = []

    try:
        supported = tuple(phase for phase in iter_phases() if phase.rebuild_supported)
        assert tuple(phase.phase_id for phase in supported) == ("phase0", "phase05", "phase06")

        for phase in supported:
            destination = (
                reproduction_root
                / phase.phase_id
                / "rebuild"
                / f"immutability-{tmp_path.name}"
            )
            destinations.append(destination)
            report = rebuild_phase(repository, phase.phase_id, destination)

            assert report.ok, report.structural_comparison.to_json_dict()
            produced = (
                report.destination,
                report.candidate_report,
                report.rebuild_report,
                *(artifact.output for artifact in report.artifacts),
            )
            assert all(
                path.resolve() == reproduction_root
                or path.resolve().is_relative_to(reproduction_root)
                for path in produced
            )

        assert historical_snapshots(repository) == before
    finally:
        for destination in destinations:
            if destination.exists():
                shutil.rmtree(destination)
