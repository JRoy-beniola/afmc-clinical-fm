#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def expected_cells(stage: str) -> int:
    if stage == "d1":
        return 40
    if stage == "d2a":
        return 100
    raise ValueError(f"unsupported Phase 0.6 stage: {stage}")


def expected_n_counts(stage: str) -> dict[int, int]:
    if stage == "d1":
        return {5: 10, 10: 10, 20: 10, 40: 10}
    if stage == "d2a":
        return {5: 50, 40: 50}
    raise ValueError(f"unsupported Phase 0.6 stage: {stage}")


def expected_flow_counts(stage: str) -> dict[str, int]:
    if stage == "d1":
        return {"none": 20, "time_scaled": 20}
    if stage == "d2a":
        return {"none": 50, "time_scaled": 50}
    raise ValueError(f"unsupported Phase 0.6 stage: {stage}")


def collect_progress(output: Path, stage: str) -> dict[str, Any]:
    expected_cells(stage)
    cells_dir = Path(output) / "stages" / stage / "cells"
    by_n: Counter[int] = Counter()
    by_flow: Counter[str] = Counter()
    done = 0
    invalid = 0

    if cells_dir.is_dir():
        for path in sorted(cells_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                cell = payload["cell"]
                if not isinstance(cell, dict):
                    raise TypeError("cell must be an object")
                cell_id = cell["cell_id"]
                n_train = cell["n_train"]
                flow_mode = cell["flow_mode"]
                if not isinstance(cell_id, str) or not cell_id:
                    raise ValueError("invalid cell_id")
                if path.stem != cell_id:
                    raise ValueError("cell filename mismatch")
                if not isinstance(n_train, int):
                    raise TypeError("n_train must be int")
                if not isinstance(flow_mode, str):
                    raise TypeError("flow_mode must be str")
                if n_train not in expected_n_counts(stage):
                    raise ValueError("unexpected N")
                if flow_mode not in expected_flow_counts(stage):
                    raise ValueError("unexpected flow mode")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                invalid += 1
                continue
            done += 1
            by_n[n_train] += 1
            by_flow[flow_mode] += 1

    return {
        "done": done,
        "invalid": invalid,
        "by_n": dict(sorted(by_n.items())),
        "by_flow": dict(sorted(by_flow.items())),
    }


def _clear() -> None:
    sys.stdout.write("\033[2J\033[H")


def _bar(done: int, total: int, width: int = 42) -> str:
    if total <= 0:
        return " " * width
    ratio = max(0.0, min(1.0, done / total))
    filled = round(width * ratio)
    return f"{GREEN}{'■' * filled}{DIM}{'□' * (width - filled)}{RESET}"


def _read_text(path: Path, default: str) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return default


def _pid_alive(path: Path) -> tuple[bool, str]:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False, "-"
    try:
        os.kill(pid, 0)
        return True, str(pid)
    except OSError:
        return False, str(pid)


def _started_epoch(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _fmt_elapsed(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _gpu_line() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        name, util, used, total, temp, power = [part.strip() for part in raw.split(",", 5)]
        return (
            f"{name}   Util {util}%   VRAM {used}/{total} MiB   "
            f"Temp {temp}°C   Power {power} W"
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return "GPU telemetry unavailable"


def _analysis_state(output: Path, stage: str) -> str:
    analysis = output / "analysis"
    if stage == "d1":
        path = analysis / "phase06_d1_classification.json"
        if not path.is_file():
            return "not yet written"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return "INVALID classification artifact"
        classification = payload.get("classification", payload.get("phenomenon_reproduction"))
        return str(classification) if classification is not None else "written"

    path = analysis / "phase06_d2_n_shift_summary.json"
    if not path.is_file():
        return "not yet written"
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "INVALID D2-A summary"
    return "written"


def _print_breakdown(
    title: str,
    observed: dict[Any, int],
    expected: dict[Any, int],
) -> None:
    print(f"{BOLD}{title}{RESET}")
    for key, total in expected.items():
        done = observed.get(key, 0)
        pct = 100.0 * done / total if total else 0.0
        print(f"  {key!s:12} {_bar(done, total, 22)} {done:3}/{total:<3} {pct:6.2f}%")
    print()


def run_dashboard(stage: str, repo: Path, control: Path, output: Path) -> None:
    total = expected_cells(stage)
    status_path = control / f"{stage}.status"
    pid_path = control / f"{stage}.pid"
    start_path = control / f"{stage}.started"
    exit_path = control / f"{stage}.exit"

    last_count: int | None = None
    last_time: float | None = None
    recent_rate = 0.0

    try:
        while True:
            now = time.time()
            progress = collect_progress(output, stage)
            done = int(progress["done"])
            invalid = int(progress["invalid"])

            if last_count is not None and last_time is not None and now > last_time:
                delta = done - last_count
                dt = now - last_time
                if delta > 0:
                    recent_rate = delta / dt * 60.0
                elif dt >= 3:
                    recent_rate *= 0.92
            last_count, last_time = done, now

            start = _started_epoch(start_path)
            elapsed = now - start if start is not None else 0.0
            avg_rate = done / elapsed * 60.0 if done and elapsed > 0 else 0.0
            rate_for_eta = recent_rate if recent_rate > 0.05 else avg_rate
            remaining = max(0, total - done)
            eta = remaining / rate_for_eta * 60.0 if rate_for_eta > 0 else None

            status = _read_text(status_path, "NOT_STARTED")
            exit_code = _read_text(exit_path, "-")
            alive, pid = _pid_alive(pid_path)
            complete = (output / "stages" / stage / "COMPLETE").is_file()
            head = _git_head(repo)
            width = max(76, min(112, shutil.get_terminal_size((100, 30)).columns))

            _clear()
            print(f"{CYAN}{'=' * width}{RESET}")
            title = f"AFMC PHASE 0.6 {stage.upper()} DIAGNOSTIC EXECUTION"
            print(f"{CYAN}{title.center(width)}{RESET}")
            print(f"{CYAN}{'=' * width}{RESET}")
            print()
            print(f"{BOLD}Execution{RESET}")
            print(f"  HEAD      {head}")
            print(f"  Output    {output}")
            print(
                f"  Status    {status}    PID {pid}    "
                f"{'ALIVE' if alive else 'NOT RUNNING'}    Exit {exit_code}"
            )
            print(f"  Elapsed   {_fmt_elapsed(elapsed)}")
            print()

            pct = 100.0 * done / total
            print(f"{BOLD}Overall progress{RESET}")
            print(f"  {_bar(done, total)}  {pct:6.2f}%")
            print(f"  Cells {GREEN}{done}{RESET}/{total}    Remaining {remaining}")
            print(
                f"  Throughput  average {avg_rate:6.2f} cells/min   "
                f"recent {recent_rate:6.2f} cells/min"
            )
            print(f"  ETA {'-' if eta is None else _fmt_elapsed(eta)}")
            if invalid:
                print(f"  {RED}{BOLD}Invalid/corrupt cell JSON files: {invalid}{RESET}")
            print()

            _print_breakdown(
                "Progress by training N",
                progress["by_n"],
                expected_n_counts(stage),
            )
            _print_breakdown(
                "Progress by flow mode",
                progress["by_flow"],
                expected_flow_counts(stage),
            )

            print(f"{BOLD}Analysis / stage boundary{RESET}")
            print(f"  COMPLETE marker  {'YES' if complete else 'NO'}")
            print(f"  Analysis         {_analysis_state(output, stage)}")
            print()
            print(f"{BOLD}GPU{RESET}")
            print(f"  {_gpu_line()}")
            print()

            if invalid:
                print(f"{RED}{BOLD}CORRUPTION VISIBLE — inspect artifacts before any resume.{RESET}")
            elif status == "FAILED":
                print(f"{RED}{BOLD}RUN FAILED — preserve outputs and diagnose before resume.{RESET}")
            elif complete and stage == "d1":
                print(f"{GREEN}{BOLD}D1 COMPLETE — HARD STOP BEFORE D2-A.{RESET}")
            elif complete:
                print(f"{GREEN}{BOLD}D2-A COMPLETE — HARD STOP BEFORE ADJUDICATION.{RESET}")
            else:
                print(f"{DIM}Ctrl+C exits this monitor only. The tmux runner continues.{RESET}")

            sys.stdout.flush()
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nMonitor exited. If the tmux runner is alive, the experiment continues.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor a Phase 0.6 diagnostic stage")
    parser.add_argument("stage", choices=("d1", "d2a"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    home = Path.home()
    repo = Path(os.environ.get("REPO", home / "afmc-clinical-fm"))
    control = Path(os.environ.get("CONTROL", home / "phase06-control"))
    head = _git_head(repo)
    output = Path(
        os.environ.get(
            "PHASE06_OUTPUT",
            repo / f"outputs/phase06_{head}",
        )
    )
    run_dashboard(args.stage, repo, control, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
