import hashlib
import shutil
from pathlib import Path

from afmc_fm.reproducibility.cli import main
from afmc_fm.reproducibility.environment import classify_historical_environment
from afmc_fm.reproducibility.rebuild import rebuild_phase
from afmc_fm.reproducibility.registry import get_phase, iter_phases
from afmc_fm.reproducibility.report_audit import audit_report_source
from afmc_fm.reproducibility.rerun import execute_rerun, plan_rerun
from tests.reproducibility.test_rerun import _fixture_repository


def tree_snapshot(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
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


def output_snapshot(repository: Path) -> dict[str, str]:
    return tree_snapshot(repository / "outputs/reproduction")


def test_verify_all_preserves_historical_archives(capsys):
    repository = Path(".")
    before = historical_snapshots(repository)
    before_output = output_snapshot(repository)

    exit_code = main(["verify", "all"], root=repository)
    assert exit_code in {0, 1}
    output = capsys.readouterr().out
    for phase in iter_phases():
        assert phase.phase_id in output

    assert historical_snapshots(repository) == before
    assert output_snapshot(repository) == before_output


def test_report_source_audit_preserves_historical_archives():
    repository = Path(".")
    before = historical_snapshots(repository)
    before_output = output_snapshot(repository)

    for phase_id in ("phase0", "phase05", "phase06"):
        phase = get_phase(phase_id)
        bundle = repository / "docs/reproducibility" / phase_id
        report = audit_report_source(repository, phase, bundle)
        assert report.ok, [
            (check.code, check.detail) for check in report.checks if not check.ok
        ]

    assert historical_snapshots(repository) == before
    assert output_snapshot(repository) == before_output


def test_supported_rebuilds_are_isolated_and_preserve_all_historical_archives(tmp_path: Path):
    repository = Path.cwd()
    reproduction_root = (repository / "outputs/reproduction").resolve()
    before = historical_snapshots(repository)
    before_output = output_snapshot(repository)
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

    assert output_snapshot(repository) == before_output


def test_status_exposes_result_kind_and_historical_rerun_readiness(capsys):
    assert main(["status"], root=Path(".")) == 0
    output = capsys.readouterr().out

    for phase in iter_phases():
        phase_line = next(
            line for line in output.splitlines() if line.startswith(f"{phase.phase_id}:")
        )
        assert f"result_kind={phase.result_kind}" in phase_line
        assert f"rerun={'yes' if phase.rerun_supported else 'no'}" in phase_line
        assert "rerun_readiness=UNSUPPORTED" in phase_line
        assert f"decision: {phase.expected_classification}" in output


def test_rerun_planning_environment_classification_and_fixture_execution_preserve_archives(
    tmp_path: Path,
    monkeypatch,
):
    repository = Path.cwd()
    before = historical_snapshots(repository)
    before_output = output_snapshot(repository)

    for phase in iter_phases():
        record = classify_historical_environment(repository, phase)
        assert record.status in {"exact", "reconstructed", "unknown"}

    for phase_id in ("phase0", "phase05", "phase06"):
        plan = plan_rerun(repository, phase_id, run_id=f"closure-{phase_id}")
        assert plan.status == "UNSUPPORTED"
        assert not plan.destination.exists()

    fixture_repository, _, _ = _fixture_repository(tmp_path, monkeypatch)
    fixture_plan = plan_rerun(fixture_repository, "fixture", run_id="closure-smoke")
    assert fixture_plan.status == "READY", fixture_plan.reasons
    fixture_execution = execute_rerun(fixture_repository, fixture_plan)
    assert fixture_execution.ok
    assert fixture_execution.historical_archive_unchanged

    assert historical_snapshots(repository) == before
    assert output_snapshot(repository) == before_output
