from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .comparison import ReproductionComparison
from .rebuild import RebuildReport, rebuild_phase
from .registry import PHASES, get_phase, iter_phases
from .rerun import RerunExecution, RerunPlan, execute_rerun, plan_rerun
from .verify import VerificationReport, verify_all, verify_phase


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afmc-reproduce",
        description=(
            "Verification, isolated rebuilds, and guarded historical rerun planning "
            "for frozen AFMC research phases."
        ),
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

    rerun_parser = commands.add_parser(
        "rerun",
        help="Plan one historical rerun; add --execute only after reviewing readiness.",
    )
    rerun_parser.add_argument("phase", choices=tuple(PHASES))
    rerun_parser.add_argument(
        "--run-id",
        help="Safe identifier for the isolated rerun output directory.",
    )
    rerun_parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly acknowledge and execute a READY historical rerun plan.",
    )
    return parser


def _presence(root: Path, paths: tuple[Path, ...]) -> str:
    if not paths:
        return "none"
    return "present" if all((root / path).exists() for path in paths) else "missing"


def _rerun_readiness(root: Path, phase_id: str) -> str:
    phase = get_phase(phase_id)
    if not phase.rerun_supported:
        return "UNSUPPORTED"
    try:
        return plan_rerun(root, phase_id, run_id="status").status
    except (FileNotFoundError, RuntimeError, ValueError):
        return "UNAVAILABLE"


def _print_status(root: Path) -> None:
    for phase in iter_phases():
        archive = "present" if (root / phase.official_evidence_root).exists() else "missing"
        manifests = _presence(root, phase.manifest_paths)
        if phase.report_source is None:
            report_source = "unavailable"
        else:
            report_source = "present" if (root / phase.report_source).exists() else "missing"
        readiness = _rerun_readiness(root, phase.phase_id)
        print(
            f"{phase.phase_id}: archive={archive} manifests={manifests} "
            f"report_source={report_source} environment={phase.environment_status} "
            f"result_kind={phase.result_kind} "
            f"rebuild={'yes' if phase.rebuild_supported else 'no'} "
            f"rerun={'yes' if phase.rerun_supported else 'no'} "
            f"rerun_readiness={readiness}"
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


def _print_rerun_plan(plan: RerunPlan, *, execution_requested: bool) -> None:
    execution = "requested" if execution_requested else "not-requested"
    print(
        f"{plan.status} {plan.phase_id}: sha={plan.implementation_sha} "
        f"environment={plan.historical_environment.status} output={plan.destination} "
        f"execution={execution}"
    )
    for reason in plan.reasons:
        print(f"  reason: {reason}")
    for remediation in plan.remediation:
        print(f"  remediation: {remediation}")


def _comparison_for_execution(
    root: Path,
    plan: RerunPlan,
    execution: RerunExecution,
) -> ReproductionComparison | None:
    """Return a comparison only when a concrete phase comparison binding exists.

    R4 deliberately does not infer table/file bindings from prose-only historical
    comparison notes. Phase-specific machine-readable policies can enable this
    later without weakening the execution acknowledgement boundary.
    """

    del root, plan, execution
    return None


def _run_rerun(
    root: Path,
    phase_id: str,
    *,
    run_id: str | None,
    execute: bool,
) -> bool:
    try:
        plan = plan_rerun(root, phase_id, run_id=run_id)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FAIL {phase_id}: {exc}")
        return False

    if not execute:
        _print_rerun_plan(plan, execution_requested=False)
        return plan.ready

    if not plan.ready:
        _print_rerun_plan(plan, execution_requested=True)
        return False

    try:
        execution = execute_rerun(root, plan)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FAIL {phase_id}: {exc}")
        return False

    comparison = _comparison_for_execution(root, plan, execution)
    comparison_label = comparison.verdict if comparison is not None else "unavailable"
    status = "PASS" if execution.ok else "FAIL"
    print(
        f"{status} {phase_id}: sha={execution.implementation_sha} "
        f"environment={execution.environment.status} output={execution.destination} "
        f"comparison={comparison_label}"
    )

    comparison_ok = comparison is None or comparison.verdict != "FAILED_REPRODUCTION"
    return execution.ok and comparison_ok


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

    if args.command == "rerun":
        return (
            0
            if _run_rerun(
                root_path,
                args.phase,
                run_id=args.run_id,
                execute=args.execute,
            )
            else 1
        )

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
