#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

HOME = Path.home()
REPO = Path(os.environ.get("REPO", HOME / "afmc-clinical-fm"))
CONTROL = Path(os.environ.get("CONTROL", HOME / "phase05-control"))
SHA = "50a94c06bc1c419ca55738f15f074cc06ccc3f36"
OUTPUT = Path(os.environ.get("PHASE05_OUTPUT", REPO / f"outputs/phase05_official_{SHA}"))

STATUS_FILE = CONTROL / "stage1.status"
PID_FILE = CONTROL / "stage1.pid"
START_FILE = CONTROL / "stage1.started"
EXIT_FILE = CONTROL / "stage1.exit"

STAGES = [
    ("flow", 60),
    ("jump", 60),
    ("uncertainty", 180),
    ("timing_audit", 120),
]
TOTAL = sum(n for _, n in STAGES)

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
WHITE = "\033[97m"

def clear():
    sys.stdout.write("\033[2J\033[H")

def bar(done: int, total: int, width: int = 46) -> str:
    if total <= 0:
        return " " * width
    ratio = max(0.0, min(1.0, done / total))
    filled = int(round(width * ratio))
    return f"{GREEN}{'■' * filled}{DIM}{'□' * (width - filled)}{RESET}"

def count_cells(stage: str) -> int:
    d = OUTPUT / "stages" / stage / "cells"
    return len(list(d.glob("*.json"))) if d.is_dir() else 0

def is_complete(stage: str) -> bool:
    return (OUTPUT / "stages" / stage / "COMPLETE").is_file()

def read_status() -> str:
    try:
        return STATUS_FILE.read_text().strip()
    except OSError:
        return "NOT_STARTED"

def pid_alive() -> tuple[bool, str]:
    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception:
        return False, "-"
    try:
        os.kill(pid, 0)
        return True, str(pid)
    except OSError:
        return False, str(pid)

def started_epoch() -> float | None:
    try:
        return float(START_FILE.read_text().strip())
    except Exception:
        return None

def gate_selection(name: str) -> tuple[str, str]:
    path = OUTPUT / "development" / f"{name}_gate.csv"
    if not path.is_file():
        return "-", "-"
    try:
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    except Exception:
        return "INVALID", "INVALID"

    selected = None
    selected_passed = None
    for row in rows:
        value = str(row.get("selected", "")).strip().lower()
        if value in {"true", "1"}:
            selected = row.get("candidate", "?")
            selected_passed = str(row.get("passed", "")).strip().lower()

    if selected is None:
        # A flow/jump gate can legitimately contain no selected mechanism if it failed.
        passed_any = any(str(r.get("passed", "")).strip().lower() in {"true", "1"} for r in rows)
        return ("-", "PASS" if passed_any else "FAIL")

    if name == "uncertainty":
        return str(selected), "SELECTED"
    return str(selected), ("PASS" if selected_passed in {"true", "1"} else "FAIL")

def gpu_line() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
        name, util, used, total, temp, power = [x.strip() for x in raw.split(",", 5)]
        return (
            f"{name}   Util {util}%   VRAM {used}/{total} MiB   "
            f"Temp {temp}°C   Power {power} W"
        )
    except Exception:
        return "GPU telemetry unavailable"

def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:
        return "unknown"

def fmt_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))

def candidate_summary() -> str:
    p = OUTPUT / "frozen_candidate.json"
    if not p.is_file():
        return "-"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return (
            f"flow={d.get('flow_mode')}  "
            f"jump={d.get('jump_mode')}  "
            f"uncertainty={d.get('uncertainty_mode')}  "
            f"strict_history={d.get('strict_history')}"
        )
    except Exception:
        return "INVALID frozen_candidate.json"

def stage_state(stage: str, done: int) -> str:
    if is_complete(stage):
        return f"{GREEN}COMPLETE{RESET}"
    if done > 0:
        return f"{YELLOW}RUNNING{RESET}"
    prior_names = [s for s, _ in STAGES[: [x[0] for x in STAGES].index(stage)]]
    if all(is_complete(p) for p in prior_names):
        return "WAITING"
    return f"{DIM}LOCKED{RESET}"

last_count = None
last_time = None
recent_rate = 0.0

try:
    while True:
        now = time.time()
        counts = {stage: count_cells(stage) for stage, _ in STAGES}
        done = sum(counts.values())

        if last_count is not None and last_time is not None and now > last_time:
            delta = done - last_count
            dt = now - last_time
            if delta > 0:
                recent_rate = delta / dt * 60.0
            elif dt >= 3:
                recent_rate *= 0.92
        last_count, last_time = done, now

        start = started_epoch()
        elapsed = (now - start) if start else 0.0
        avg_rate = (done / elapsed * 60.0) if done and elapsed > 0 else 0.0
        rate_for_eta = recent_rate if recent_rate > 0.05 else avg_rate
        remaining = max(0, TOTAL - done)
        eta = (remaining / rate_for_eta * 60.0) if rate_for_eta > 0 else None

        status = read_status()
        alive, pid = pid_alive()
        head = git_head()
        width = max(72, min(110, shutil.get_terminal_size((100, 30)).columns))

        clear()
        print(f"{CYAN}{'=' * width}{RESET}")
        title = "AFMC PHASE-0.5 OFFICIAL STAGE-I"
        print(f"{CYAN}{title.center(width)}{RESET}")
        print(f"{CYAN}{'=' * width}{RESET}")
        print()
        print(f"{BOLD}Execution{RESET}")
        print(f"  HEAD      {head}")
        print(f"  Output    {OUTPUT}")
        print(f"  Status    {status}    PID {pid}    {'ALIVE' if alive else 'NOT RUNNING'}")
        print(f"  Elapsed   {fmt_elapsed(elapsed)}")
        print()

        pct = 100.0 * done / TOTAL if TOTAL else 0.0
        print(f"{BOLD}Overall Stage-I progress{RESET}")
        print(f"  {bar(done, TOTAL)}  {pct:6.2f}%")
        print(f"  Cells {GREEN}{done}{RESET}/{TOTAL}    Remaining {remaining}")
        print(f"  Throughput  average {avg_rate:6.2f} cells/min   recent {recent_rate:6.2f} cells/min")
        print(f"  ETA {'-' if eta is None else fmt_elapsed(eta)}")
        print()

        print(f"{BOLD}Stage progress{RESET}")
        for stage, expected in STAGES:
            c = counts[stage]
            pct_s = 100.0 * c / expected
            print(
                f"  {stage:15} {bar(c, expected, 28)} "
                f"{c:3}/{expected:<3} {pct_s:6.2f}%  {stage_state(stage, c)}"
            )
        print()

        flow, flow_gate = gate_selection("flow")
        jump, jump_gate = gate_selection("jump")
        unc, unc_gate = gate_selection("uncertainty")
        print(f"{BOLD}Selections / gates{RESET}")
        print(f"  Flow         {flow:15} {flow_gate}")
        print(f"  Jump         {jump:15} {jump_gate}")
        print(f"  Uncertainty  {unc:15} {unc_gate}")
        print()

        print(f"{BOLD}GPU{RESET}")
        print(f"  {gpu_line()}")
        print()

        frozen = (OUTPUT / "frozen_candidate.json").is_file()
        confirm_started = (OUTPUT / "confirmation" / "STARTED").is_file()
        if confirm_started:
            print(f"{RED}{BOLD}WARNING: confirmation/STARTED exists. Stage-III boundary has been crossed.{RESET}")
        elif frozen:
            print(f"{GREEN}{BOLD}STAGE I COMPLETE / STAGE II FROZEN — HARD STOP BEFORE CONFIRMATION{RESET}")
            print(f"  {candidate_summary()}")
        elif status == "FAILED":
            print(f"{RED}{BOLD}RUN FAILED — preserve outputs; diagnose before using resume.{RESET}")
        else:
            print(f"{DIM}Ctrl+C exits this monitor only. The runner continues inside tmux.{RESET}")

        sys.stdout.flush()
        time.sleep(3)

except KeyboardInterrupt:
    print("\nMonitor exited. If the tmux runner is alive, the experiment continues.")
