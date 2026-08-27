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
CORE_VALIDATION = (
    ROOT
    / "docs"
    / "superpowers"
    / "validation"
    / "2026-08-26-phase0-6-core-validation.md"
)
D2B_VALIDATION_NAME = "2026-08-26-phase0-6-d2b-validation.md"
D4B_VALIDATION_NAME = "2026-08-27-phase0-6-d4b-validation.md"


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


def test_runner_preserves_core_d1_d2a_validation_boundary():
    text = RUNNER.read_text(encoding="utf-8")

    assert "READY FOR D1 EXECUTION" in text
    assert 'git diff --quiet "$VALIDATED_SHA" "$HEAD" -- src configs pyproject.toml' in text
    assert "--device cuda" in text
    assert 'stages/d1/COMPLETE' in text
    assert 'analysis/phase06_d1_classification.json' in text
    assert 'afmc_fm.phase06.cli adjudicate' not in text


def test_runner_d2b_requires_separate_validation_parent_and_child_output():
    text = RUNNER.read_text(encoding="utf-8")

    assert '"d2b"' in text
    assert D2B_VALIDATION_NAME in text
    assert "READY FOR D2-B EXECUTION" in text
    assert "PHASE06_PARENT_OUTPUT" in text
    assert 'outputs/phase06_d2b_${HEAD}' in text
    assert '--parent-output "$PARENT_OUTPUT"' in text
    assert '--output "$OUTPUT"' in text
    assert 'stages/d2b/COMPLETE' in text
    assert 'analysis/phase06_d2b_n_shift_summary.json' in text
    assert '"$PY" -m afmc_fm.phase06.cli d2b' in text
    assert '"$PY" -m afmc_fm.phase06.cli adjudicate-d2b' not in text
    assert "full-factorial" not in text.lower()


def test_runner_d4b_requires_readiness_two_parents_and_hard_stop():
    text = RUNNER.read_text(encoding="utf-8")

    assert '"d4b"' in text
    assert D4B_VALIDATION_NAME in text
    assert "READY FOR D4-B EXECUTION" in text
    assert "PHASE06_CORE_PARENT_OUTPUT" in text
    assert "PHASE06_D2B_PARENT_OUTPUT" in text
    assert 'outputs/phase06_d4b_${HEAD}' in text
    assert 'docs/superpowers/specs/2026-08-27-phase0-6-d4b-execution-addendum.md' in text
    assert 'tools/execution/phase06/run_stage.sh' in text
    assert 'tools/execution/phase06/start_stage.sh' in text
    assert 'tools/monitoring/phase06/monitor_stage.py' in text
    assert '--core-parent-output "$CORE_PARENT_OUTPUT"' in text
    assert '--d2b-parent-output "$D2B_PARENT_OUTPUT"' in text
    assert '--output "$OUTPUT"' in text
    assert '--device cuda' in text
    assert 'stages/d4b/COMPLETE' in text
    assert 'analysis/phase06_d4b_bootstrap_diagnostics.json' in text
    assert '"$PY" -m afmc_fm.phase06.cli d4b' in text
    assert '"$PY" -m afmc_fm.phase06.cli adjudicate-d4b' not in text
    assert "d4-capacity-time" not in text.lower()
    assert "full-factorial" not in text.lower()
    assert "core_parent_protocol_sha256=" in text
    assert "d2b_parent_protocol_sha256=" in text
    assert "d2b_parent_adjudication_sha256=" in text
    assert "d4b_addendum_sha256=" in text


def test_runner_d4b_enforces_pairwise_non_nested_roots():
    text = RUNNER.read_text(encoding="utf-8")

    assert 'CORE_PARENT_OUTPUT="$(realpath -e "$PHASE06_CORE_PARENT_OUTPUT")"' in text
    assert 'D2B_PARENT_OUTPUT="$(realpath -e "$PHASE06_D2B_PARENT_OUTPUT")"' in text
    assert 'CHILD_CANONICAL="$(realpath -m "$OUTPUT")"' in text
    assert "D4-B output roots must be distinct and non-nested" in text
    assert '"$CORE_PARENT_OUTPUT" == "$D2B_PARENT_OUTPUT"' in text
    assert '"$CHILD_CANONICAL" == "$CORE_PARENT_OUTPUT/"*' in text
    assert '"$CHILD_CANONICAL" == "$D2B_PARENT_OUTPUT/"*' in text


def test_runner_validation_parser_matches_committed_task12_record():
    runner = RUNNER.read_text(encoding="utf-8")
    validation = CORE_VALIDATION.read_text(encoding="utf-8")

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


def test_tmux_starter_accepts_d2b_and_creates_runner_and_monitor_only():
    text = STARTER.read_text(encoding="utf-8")

    assert '"d2b"' in text
    assert "PHASE06_PARENT_OUTPUT" in text
    assert 'SESSION="${PHASE06_TMUX_SESSION:-phase06-$STAGE}"' in text
    assert 'tmux new-session -d -s "$SESSION" -n runner' in text
    assert 'tmux new-window -t "$SESSION" -n monitor' in text
    assert 'monitor_stage.py' in text
    assert 'tmux attach -t "$SESSION"' in text
    assert "adjudicate" not in text


def test_tmux_starter_accepts_d4b_and_requires_both_parent_bindings():
    text = STARTER.read_text(encoding="utf-8")

    assert '"d4b"' in text
    assert "PHASE06_CORE_PARENT_OUTPUT" in text
    assert "PHASE06_D2B_PARENT_OUTPUT" in text
    assert 'SESSION="${PHASE06_TMUX_SESSION:-phase06-$STAGE}"' in text
    assert 'tmux new-session -d -s "$SESSION" -n runner' in text
    assert 'tmux new-window -t "$SESSION" -n monitor' in text
    assert "adjudicate-d4b" not in text


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
    assert module.expected_cells("d2b") == 100
    assert module.expected_cells("d4b") == 100
    assert module.expected_n_counts("d1") == {5: 10, 10: 10, 20: 10, 40: 10}
    assert module.expected_n_counts("d2a") == {5: 50, 40: 50}
    assert module.expected_n_counts("d2b") == {5: 50, 40: 50}
    assert module.expected_n_counts("d4b") == {40: 100}
    assert module.expected_flow_counts("d1") == {"none": 20, "time_scaled": 20}
    assert module.expected_flow_counts("d2a") == {"none": 50, "time_scaled": 50}
    assert module.expected_flow_counts("d2b") == {"none": 50, "time_scaled": 50}
    assert module.expected_flow_counts("d4b") == {"none": 50, "time_scaled": 50}
    assert progress["done"] == 3
    assert progress["invalid"] == 0
    assert progress["by_n"] == {5: 2, 40: 1}
    assert progress["by_flow"] == {"none": 2, "time_scaled": 1}


def test_monitor_d2b_uses_child_analysis_artifact_and_accepts_stage(tmp_path):
    module = _load_monitor_module()
    analysis = tmp_path / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "phase06_d2b_n_shift_summary.json").write_text(
        "{}\n", encoding="utf-8"
    )

    assert module._analysis_state(tmp_path, "d2b") == "written"
    parsed = module._parse_args(["d2b"])
    assert parsed.stage == "d2b"


def test_monitor_d4b_uses_bootstrap_artifact_and_accepts_stage(tmp_path):
    module = _load_monitor_module()
    analysis = tmp_path / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "phase06_d4b_bootstrap_diagnostics.json").write_text(
        "{}\n", encoding="utf-8"
    )

    assert module._analysis_state(tmp_path, "d4b") == "written"
    parsed = module._parse_args(["d4b"])
    assert parsed.stage == "d4b"


def test_monitor_treats_invalid_cell_json_as_visible_corruption(tmp_path):
    module = _load_monitor_module()
    cells_dir = tmp_path / "stages" / "d2b" / "cells"
    cells_dir.mkdir(parents=True)
    (cells_dir / "bad.json").write_text("not-json", encoding="utf-8")

    progress = module.collect_progress(tmp_path, "d2b")

    assert progress["done"] == 0
    assert progress["invalid"] == 1


def test_readme_documents_d2b_parent_binding_and_cuda_launch():
    text = README.read_text(encoding="utf-8")

    assert "chmod +x ~/phase06-control/run_stage.sh" in text
    assert "chmod +x ~/phase06-control/start_stage.sh" in text
    assert "chmod +x ~/phase06-control/monitor_stage.py" in text
    assert "~/phase06-control/start_stage.sh d1 fresh" in text
    assert "~/phase06-control/start_stage.sh d2a fresh" in text
    assert "PHASE06_PARENT_OUTPUT" in text
    assert "~/phase06-control/start_stage.sh d2b fresh" in text
    assert "phase06-d2b" in text
    assert "Ctrl-b d" in text
    assert "does not run adjudicate-d2b" in text
    assert "does not authorize D4" in text


def test_readme_documents_d4b_two_parent_cuda_launch_and_stop():
    text = README.read_text(encoding="utf-8")

    assert "PHASE06_CORE_PARENT_OUTPUT" in text
    assert "PHASE06_D2B_PARENT_OUTPUT" in text
    assert "~/phase06-control/start_stage.sh d4b fresh" in text
    assert "phase06-d4b" in text
    assert "does not run adjudicate-d4b" in text
    assert "D4_CAPACITY_TIME" in text
    assert "does not authorize D4-D" in text
