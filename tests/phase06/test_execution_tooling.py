from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "execution" / "phase06" / "run_stage.sh"
STARTER = ROOT / "tools" / "execution" / "phase06" / "start_stage.sh"
README = ROOT / "tools" / "execution" / "phase06" / "README.txt"
MONITOR = ROOT / "tools" / "monitoring" / "phase06" / "monitor_stage.py"
VALIDATION = (
    ROOT
    / "docs"
    / "superpowers"
    / "validation"
    / "2026-08-26-phase0-6-core-validation.md"
)


def _load_monitor_module():
    spec = importlib.util.spec_from_file_location("phase06_monitor_stage", MONITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_execution_kit_scripts_exist_and_have_valid_shell_syntax():
    for path in (RUNNER, STARTER):
        assert path.is_file(), path
        subprocess.run(
            ["bash", "-n", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_runner_locks_stage_scope_validation_and_d2_prerequisites():
    text = RUNNER.read_text(encoding="utf-8")

    assert '[[ "$STAGE" != "d1" && "$STAGE" != "d2a" ]]' in text
    assert "READY FOR D1 EXECUTION" in text
    assert 'git diff --quiet "$VALIDATED_SHA" "$HEAD" -- src configs pyproject.toml' in text
    assert '"$PY" -m afmc_fm.phase06.cli "$STAGE"' in text
    assert "--device cuda" in text
    assert 'stages/d1/COMPLETE' in text
    assert 'analysis/phase06_d1_classification.json' in text
    assert 'afmc_fm.phase06.cli adjudicate' not in text
    assert '"d2b"' not in text


def test_runner_validation_parser_matches_committed_task12_record():
    runner = RUNNER.read_text(encoding="utf-8")
    validation = VALIDATION.read_text(encoding="utf-8")

    assert re.search(
        r"(?im)implementation status:.*READY FOR D1 EXECUTION",
        validation,
    )
    match = re.search(
        r"(?im)validated implementation SHA:\s*`?([0-9a-f]{40})`?",
        validation,
    )
    assert match is not None

    assert "implementation status:.*READY FOR D1 EXECUTION" in runner
    assert "Validated implementation SHA|branch/head SHA" in runner


def test_tmux_starter_creates_runner_and_monitor_windows_only():
    text = STARTER.read_text(encoding="utf-8")

    assert 'SESSION="${PHASE06_TMUX_SESSION:-phase06-$STAGE}"' in text
    assert 'tmux new-session -d -s "$SESSION" -n runner' in text
    assert 'tmux new-window -t "$SESSION" -n monitor' in text
    assert 'monitor_stage.py' in text
    assert 'tmux attach -t "$SESSION"' in text
    assert "adjudicate" not in text


def test_monitor_counts_persisted_cells_by_n_and_flow(tmp_path):
    module = _load_monitor_module()
    cells_dir = tmp_path / "stages" / "d1" / "cells"
    cells_dir.mkdir(parents=True)
    payloads = [
        {"cell": {"cell_id": "a", "n_train": 5, "flow_mode": "none"}},
        {"cell": {"cell_id": "b", "n_train": 5, "flow_mode": "time_scaled"}},
        {"cell": {"cell_id": "c", "n_train": 40, "flow_mode": "none"}},
    ]
    for payload in payloads:
        cell_id = payload["cell"]["cell_id"]
        (cells_dir / f"{cell_id}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    progress = module.collect_progress(tmp_path, "d1")

    assert module.expected_cells("d1") == 40
    assert module.expected_cells("d2a") == 100
    assert module.expected_n_counts("d1") == {5: 10, 10: 10, 20: 10, 40: 10}
    assert module.expected_n_counts("d2a") == {5: 50, 40: 50}
    assert module.expected_flow_counts("d1") == {"none": 20, "time_scaled": 20}
    assert module.expected_flow_counts("d2a") == {"none": 50, "time_scaled": 50}
    assert progress["done"] == 3
    assert progress["invalid"] == 0
    assert progress["by_n"] == {5: 2, 40: 1}
    assert progress["by_flow"] == {"none": 2, "time_scaled": 1}


def test_monitor_treats_invalid_cell_json_as_visible_corruption(tmp_path):
    module = _load_monitor_module()
    cells_dir = tmp_path / "stages" / "d2a" / "cells"
    cells_dir.mkdir(parents=True)
    (cells_dir / "bad.json").write_text("not-json", encoding="utf-8")

    progress = module.collect_progress(tmp_path, "d2a")

    assert progress["done"] == 0
    assert progress["invalid"] == 1


def test_readme_documents_install_tmux_detach_and_both_stage_launches():
    text = README.read_text(encoding="utf-8")

    assert "chmod +x ~/phase06-control/run_stage.sh" in text
    assert "chmod +x ~/phase06-control/start_stage.sh" in text
    assert "chmod +x ~/phase06-control/monitor_stage.py" in text
    assert "~/phase06-control/start_stage.sh d1 fresh" in text
    assert "~/phase06-control/start_stage.sh d2a fresh" in text
    assert "Ctrl-b d" in text
    assert "tmux attach -t phase06-d1" in text
    assert "does not run adjudicate" in text
