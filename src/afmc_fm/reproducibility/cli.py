from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .rebuild import RebuildReport, rebuild_phase
from .registry import PHASES, get_phase, iter_phases
from .verify import VerificationReport, verify_all, verify_phase


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afmc-reproduce",
        description="Read-only verification and isolated rebuilds for frozen AFMC research phases.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show reproducibility status for all frozen phases.")

    verify_parser = commands.add_parser(
        "verify",
        help="Verify frozen evidence and integrity manifests without modifying them.",
    )
    verify_parser.add_argument("phase", choices=(*PHASES, "all"))

    rebuild_parser = commands.add_parser(
        "rebuild",
        help="Rebuild supported documentary artifacts under outputs/reproduction/.",
    )
    rebuild_parser.add_argument("phase", choices=(*PHASES, "all"))
    rebuild_parser.add_argument(
        "--destination",
        type=Path,
        help="Custom isolated reproduction destination for a single phase.",
    )
    return parser


def _presence(root: Path, paths: tuple[Path, ...]) -> str:
    if not paths:
        return "none"
    return "present" if all((root / path).exists() for path in paths) else "missing"


def _print_status(root: Path) -> None:
    for phase in iter_phases():
        archive = "present" if (root / phase.official_evidence_root).exists() else "missing"
        manifests = _presence(root, phase.manifest_paths)
        if phase.report_source is None:
            report_source = "unavailable"
        else:
            report_source = "present" if (root / phase.report_source).exists() else "missing"
        print(
            f"{phase.phase_id}: archive={archive} manifests={manifests} "
            f"report_source={report_source} environment={phase.environment_status} "
            f"rebuild={'yes' if phase.rebuild_supported else 'no'} "
            f"rerun={'yes' if phase.rerun_supported else 'no'}"
        )
        print(f"  decision: {phase.expected_classification}")


def _print_report(report: VerificationReport) -> bool:
    failures = tuple(check for check in report.checks if not check.ok)
    if not failures:
        print(f"PASS {report.phase_id} ({len(report.checks)} checks)")
        return True

    print(f"FAIL {report.phase_id} ({len(failures)}/{len(report.checks)} checks failed)")
    for check in failures:
        print(f"  {check.code}: {check.subject} - {check.detail}")
    return False


def _print_rebuild_report(report: RebuildReport) -> bool:
    status = "PASS" if report.ok else "FAIL"
    structural = "PASS" if report.structural_comparison.ok else "FAIL"
    print(
        f"{status} {report.phase_id}: generated={report.generated_count} "
        f"reference-copy={report.reference_copy_count} structural={structural} "
        f"candidate={report.candidate_report}"
    )
    return report.ok


def _run_rebuild(
    root: Path,
    phase_id: str,
    destination: Path | None,
) -> bool:
    phase = get_phase(phase_id)
    if not phase.rebuild_supported:
        print(f"UNSUPPORTED {phase_id}: deterministic rebuild is not registered")
        return False
    try:
        report = rebuild_phase(root, phase_id, destination)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FAIL {phase_id}: {exc}")
        return False
    return _print_rebuild_report(report)


def main(
    argv: Sequence[str] | None = None,
    *,
    root: str | Path = Path("."),
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    root_path = Path(root)

    if args.command == "status":
        _print_status(root_path)
        return 0

    if args.command == "rebuild":
        destination = args.destination
        if args.phase == "all":
            if destination is not None:
                parser.error("--destination is only valid when rebuilding a single phase")
            outcomes: list[bool] = []
            for phase in iter_phases():
                if not phase.rebuild_supported:
                    print(
                        f"UNSUPPORTED {phase.phase_id}: deterministic rebuild is not registered"
                    )
                    continue
                outcomes.append(_run_rebuild(root_path, phase.phase_id, None))
            return 0 if all(outcomes) else 1
        return 0 if _run_rebuild(root_path, args.phase, destination) else 1

    if args.phase == "all":
        reports = verify_all(root_path)
    else:
        reports = (verify_phase(root_path, get_phase(args.phase)),)

    outcomes = tuple(_print_report(report) for report in reports)
    return 0 if all(outcomes) else 1
