from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .registry import PHASES, get_phase, iter_phases
from .verify import VerificationReport, verify_all, verify_phase


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afmc-reproduce",
        description="Read-only reproducibility inspection for frozen AFMC research phases.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show reproducibility status for all frozen phases.")

    verify_parser = commands.add_parser(
        "verify",
        help="Verify frozen evidence and integrity manifests without modifying them.",
    )
    verify_parser.add_argument("phase", choices=(*PHASES, "all"))
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

    if args.phase == "all":
        reports = verify_all(root_path)
    else:
        reports = (verify_phase(root_path, get_phase(args.phase)),)

    outcomes = tuple(_print_report(report) for report in reports)
    return 0 if all(outcomes) else 1
