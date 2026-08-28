import hashlib
import importlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from afmc_fm.reproducibility.models import PhaseDefinition
from afmc_fm.reproducibility.registry import PHASES

rerun_module = importlib.import_module("afmc_fm.reproducibility.rerun")
plan_rerun = rerun_module.plan_rerun
execute_rerun = rerun_module.execute_rerun


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _fixture_repository(tmp_path: Path, monkeypatch) -> tuple[Path, str, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "reproducibility@example.invalid")
    _git(repository, "config", "user.name", "Reproducibility Test")

    archive = repository / "docs/results/fixture"
    archive.mkdir(parents=True)
    (archive / "decision.md").write_text("FIXTURE COMPLETE\n", encoding="utf-8")
    (archive / "evidence.txt").write_text("immutable evidence\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'fixture-reproduction'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )

    script = repository / "run.py"
    script.write_text(
        """from pathlib import Path
import os
import subprocess

output = Path(os.environ["AFMC_REPRO_OUTPUT"])
output.mkdir(parents=True, exist_ok=True)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
(output / "result.txt").write_text(f"HISTORICAL\\n{head}\\n", encoding="utf-8")
print("fixture historical execution")
""",
        encoding="utf-8",
    )
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "historical fixture")
    historical_sha = _git(repository, "rev-parse", "HEAD")

    script.write_text(
        "raise RuntimeError('CURRENT HEAD MUST NOT EXECUTE')\n",
        encoding="utf-8",
    )

    specification = repository / "docs/reproducibility/fixture/rerun.yaml"
    specification.parent.mkdir(parents=True)
    specification.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "phase_id": "fixture",
                "supported": True,
                "blocked_reason": None,
                "implementation_sha": historical_sha,
                "command": ["python", "run.py"],
                "environment": {"AFMC_REPRO_OUTPUT": "{output}/scientific"},
                "required_paths": ["run.py"],
                "required_parent_bindings": [],
                "seed_policy": {
                    "mode": "historical_exact",
                    "protected_confirmatory": False,
                    "historical_exact_only": True,
                    "recorded_seeds": [101],
                },
                "comparison_policy": {"decision": "FIXTURE COMPLETE"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "current fixture")
    current_sha = _git(repository, "rev-parse", "HEAD")

    phase = PhaseDefinition(
        phase_id="fixture",
        official_evidence_root=Path("docs/results/fixture"),
        official_report=None,
        decision_record=Path("docs/results/fixture/decision.md"),
        expected_classification="FIXTURE COMPLETE",
        result_kind="historical",
        implementation_sha=historical_sha,
        execution_sha=historical_sha,
        protocol_paths=(),
        manifests=(),
        raw_evidence_paths=(),
        derived_table_paths=(),
        figure_paths=(),
        report_source=None,
        environment_status="unknown",
        rebuild_supported=False,
        rerun_supported=True,
    )
    monkeypatch.setitem(PHASES, "fixture", phase)
    return repository, historical_sha, current_sha


def _update_spec(repository: Path, mutate) -> None:
    path = repository / "docs/reproducibility/fixture/rerun.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_rerun_plan_and_execution_use_historical_commit_and_isolated_output(
    tmp_path: Path,
    monkeypatch,
):
    repository, historical_sha, current_sha = _fixture_repository(tmp_path, monkeypatch)
    archive = repository / "docs/results/fixture"
    before = _tree_snapshot(archive)

    plan = plan_rerun(repository, "fixture", run_id="smoke")

    expected_output = (repository / "outputs/reproduction/fixture/rerun/smoke").resolve()
    assert plan.status == "READY", plan.reasons
    assert plan.implementation_sha == historical_sha
    assert plan.destination == expected_output
    assert not expected_output.exists()

    execution = execute_rerun(repository, plan)

    assert execution.ok
    assert execution.returncode == 0
    assert execution.implementation_sha == historical_sha
    assert execution.environment.status == "reconstructed"
    assert execution.destination == expected_output
    assert execution.execution_record.is_file()
    assert execution.inventory_path.is_file()
    assert execution.log_path.is_file()
    assert execution.historical_archive_unchanged

    result = (expected_output / "scientific/result.txt").read_text(encoding="utf-8")
    assert result == f"HISTORICAL\n{historical_sha}\n"

    payload = json.loads(execution.execution_record.read_text(encoding="utf-8"))
    assert payload["phase_id"] == "fixture"
    assert payload["implementation_sha"] == historical_sha
    assert payload["returncode"] == 0
    assert payload["historical_archive_unchanged"] is True
    assert payload["environment"]["status"] == "reconstructed"

    assert before == _tree_snapshot(archive)
    assert _git(repository, "rev-parse", "HEAD") == current_sha
    assert (repository / "run.py").read_text(encoding="utf-8") == (
        "raise RuntimeError('CURRENT HEAD MUST NOT EXECUTE')\n"
    )


def test_rerun_plan_rejects_required_path_absent_from_historical_commit(
    tmp_path: Path,
    monkeypatch,
):
    repository, _, _ = _fixture_repository(tmp_path, monkeypatch)
    (repository / "late_only.py").write_text("print('current only')\n", encoding="utf-8")
    _update_spec(
        repository,
        lambda payload: payload.update(required_paths=["run.py", "late_only.py"]),
    )

    plan = plan_rerun(repository, "fixture", run_id="historical-path-check")

    assert plan.status == "BLOCKED_POLICY"
    assert any("late_only.py" in reason for reason in plan.reasons)
    assert not plan.destination.exists()


def test_rerun_plan_blocks_missing_historical_commit(tmp_path: Path, monkeypatch):
    repository, _, _ = _fixture_repository(tmp_path, monkeypatch)
    missing_sha = "f" * 40
    monkeypatch.setitem(
        PHASES,
        "fixture",
        replace(
            PHASES["fixture"],
            implementation_sha=missing_sha,
            execution_sha=missing_sha,
        ),
    )
    _update_spec(
        repository,
        lambda payload: payload.update(implementation_sha=missing_sha),
    )

    plan = plan_rerun(repository, "fixture", run_id="missing-history")

    assert plan.status == "BLOCKED_MISSING_HISTORY"
    assert missing_sha in " ".join(plan.reasons)
    assert not plan.destination.exists()


def test_rerun_plan_blocks_missing_environment_evidence(tmp_path: Path, monkeypatch):
    repository, _, _ = _fixture_repository(tmp_path, monkeypatch)
    (repository / "pyproject.toml").unlink()

    plan = plan_rerun(repository, "fixture", run_id="missing-environment")

    assert plan.status == "BLOCKED_MISSING_ENVIRONMENT"
    assert not plan.destination.exists()


def test_rerun_plan_blocks_missing_parent_binding_source(tmp_path: Path, monkeypatch):
    repository, _, _ = _fixture_repository(tmp_path, monkeypatch)

    def mutate(payload):
        payload["environment"]["FIXTURE_PARENT"] = "{parent:core}"
        payload["required_parent_bindings"] = [
            {
                "name": "core",
                "environment_variable": "FIXTURE_PARENT",
                "source": "parents/missing.json",
                "sha256": "0" * 64,
            }
        ]

    _update_spec(repository, mutate)

    plan = plan_rerun(repository, "fixture", run_id="missing-parent")

    assert plan.status == "BLOCKED_MISSING_PARENT"
    assert any("parents/missing.json" in reason for reason in plan.reasons)
    assert not plan.destination.exists()


def test_rerun_plan_blocks_forbidden_seed_policy_and_execution_refuses(
    tmp_path: Path,
    monkeypatch,
):
    repository, _, _ = _fixture_repository(tmp_path, monkeypatch)

    def mutate(payload):
        payload["seed_policy"]["mode"] = "unrestricted"

    _update_spec(repository, mutate)

    plan = plan_rerun(repository, "fixture", run_id="blocked-policy")

    assert plan.status == "BLOCKED_POLICY"
    assert not plan.destination.exists()
    with pytest.raises(ValueError, match="not READY"):
        execute_rerun(repository, plan)
    assert not plan.destination.exists()


def test_rerun_plan_rejects_empty_command(tmp_path: Path, monkeypatch):
    repository, _, _ = _fixture_repository(tmp_path, monkeypatch)
    _update_spec(repository, lambda payload: payload.update(command=[]))

    plan = plan_rerun(repository, "fixture", run_id="empty-command")

    assert plan.status == "BLOCKED_POLICY"
    assert any("command" in reason.casefold() for reason in plan.reasons)
    assert not plan.destination.exists()


def test_rerun_nonzero_exit_is_failed_reproduction_with_provenance(
    tmp_path: Path,
    monkeypatch,
):
    repository, historical_sha, _ = _fixture_repository(tmp_path, monkeypatch)
    archive = repository / "docs/results/fixture"
    before = _tree_snapshot(archive)
    _update_spec(
        repository,
        lambda payload: payload.update(
            command=["python", "-c", "import sys; sys.exit(7)"]
        ),
    )

    plan = plan_rerun(repository, "fixture", run_id="nonzero")
    assert plan.status == "READY", plan.reasons

    execution = execute_rerun(repository, plan)

    assert not execution.ok
    assert execution.status == "FAILED"
    assert execution.returncode == 7
    assert execution.implementation_sha == historical_sha
    assert execution.historical_archive_unchanged
    assert execution.execution_record.is_file()
    assert execution.inventory_path.is_file()
    assert execution.log_path.is_file()
    payload = json.loads(execution.execution_record.read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["returncode"] == 7
    assert payload["historical_archive_unchanged"] is True
    assert before == _tree_snapshot(archive)


@pytest.mark.parametrize("phase_id", ["phase0", "phase05", "phase06"])
def test_real_historical_reruns_remain_explicitly_unsupported(phase_id: str):
    plan = plan_rerun(Path("."), phase_id, run_id="policy-check")

    assert plan.status == "UNSUPPORTED"
    assert plan.reasons
    assert not plan.destination.exists()
