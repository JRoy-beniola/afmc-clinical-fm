import importlib
from pathlib import Path

import pytest
import yaml

from afmc_fm.reproducibility.models import PhaseDefinition

rerun_models = importlib.import_module("afmc_fm.reproducibility.rerun_models")
load_rerun_spec = rerun_models.load_rerun_spec
validate_rerun_spec = rerun_models.validate_rerun_spec


def _phase() -> PhaseDefinition:
    return PhaseDefinition(
        phase_id="fixture",
        official_evidence_root=Path("docs/results/fixture"),
        official_report=None,
        decision_record=Path("docs/results/fixture/decision.md"),
        expected_classification="FIXTURE",
        result_kind="historical",
        implementation_sha="a" * 40,
        execution_sha="a" * 40,
        protocol_paths=(),
        manifests=(),
        raw_evidence_paths=(),
        derived_table_paths=(),
        figure_paths=(),
        report_source=None,
        environment_status="unknown",
        rebuild_supported=False,
        rerun_supported=False,
    )


def _payload(**updates):
    payload = {
        "schema_version": 1,
        "phase_id": "fixture",
        "supported": True,
        "blocked_reason": None,
        "implementation_sha": "a" * 40,
        "command": ["python", "run.py", "--output", "{output}"],
        "environment": {"AFMC_REPRO_OUTPUT": "{output}"},
        "required_paths": ["run.py"],
        "required_parent_bindings": [],
        "seed_policy": {
            "mode": "historical_exact",
            "protected_confirmatory": False,
            "historical_exact_only": True,
            "recorded_seeds": [101, 102],
        },
        "comparison_policy": {"decision": "FIXTURE"},
    }
    payload.update(updates)
    return payload


def _write_spec(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "rerun.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _failed_codes(checks) -> set[str]:
    return {check.code for check in checks if not check.ok}


def test_load_valid_rerun_spec_and_validate_fixture(tmp_path: Path):
    (tmp_path / "run.py").write_text("print('fixture')\n", encoding="utf-8")
    spec = load_rerun_spec(_write_spec(tmp_path, _payload()))

    checks = validate_rerun_spec(tmp_path, _phase(), spec)

    assert all(check.ok for check in checks), [
        (check.code, check.subject, check.detail) for check in checks if not check.ok
    ]
    assert spec.command[-1] == "{output}"
    assert dict(spec.environment)["AFMC_REPRO_OUTPUT"] == "{output}"
    assert spec.seed_policy.recorded_seeds == (101, 102)


@pytest.mark.parametrize("implementation_sha", ["HEAD", "main", "b" * 40])
def test_symbolic_or_mismatched_implementation_is_rejected(tmp_path: Path, implementation_sha: str):
    (tmp_path / "run.py").write_text("print('fixture')\n", encoding="utf-8")
    spec = load_rerun_spec(
        _write_spec(tmp_path, _payload(implementation_sha=implementation_sha))
    )

    assert "implementation_binding" in _failed_codes(
        validate_rerun_spec(tmp_path, _phase(), spec)
    )


def test_command_cannot_write_into_official_results_tree(tmp_path: Path):
    (tmp_path / "run.py").write_text("print('fixture')\n", encoding="utf-8")
    spec = load_rerun_spec(
        _write_spec(
            tmp_path,
            _payload(command=["python", "run.py", "--output", "docs/results/fixture/rerun"]),
        )
    )

    assert "official_output_forbidden" in _failed_codes(
        validate_rerun_spec(tmp_path, _phase(), spec)
    )


def test_environment_cannot_bind_output_into_official_results_tree(tmp_path: Path):
    (tmp_path / "run.py").write_text("print('fixture')\n", encoding="utf-8")
    spec = load_rerun_spec(
        _write_spec(
            tmp_path,
            _payload(environment={"AFMC_REPRO_OUTPUT": "docs/results/fixture/rerun"}),
        )
    )

    assert "official_output_forbidden" in _failed_codes(
        validate_rerun_spec(tmp_path, _phase(), spec)
    )


def test_unrestricted_seed_policy_is_rejected(tmp_path: Path):
    (tmp_path / "run.py").write_text("print('fixture')\n", encoding="utf-8")
    seed_policy = _payload()["seed_policy"] | {"mode": "unrestricted"}
    spec = load_rerun_spec(_write_spec(tmp_path, _payload(seed_policy=seed_policy)))

    assert "seed_policy_forbidden" in _failed_codes(
        validate_rerun_spec(tmp_path, _phase(), spec)
    )


@pytest.mark.parametrize(
    "seed_policy",
    [
        {
            "mode": "historical_exact",
            "protected_confirmatory": True,
            "historical_exact_only": False,
            "recorded_seeds": [701],
        },
        {
            "mode": "historical_exact",
            "protected_confirmatory": True,
            "historical_exact_only": True,
            "recorded_seeds": [],
        },
    ],
)
def test_protected_seed_namespace_requires_exact_historical_record(seed_policy, tmp_path: Path):
    (tmp_path / "run.py").write_text("print('fixture')\n", encoding="utf-8")
    spec = load_rerun_spec(_write_spec(tmp_path, _payload(seed_policy=seed_policy)))

    assert "protected_seed_policy" in _failed_codes(
        validate_rerun_spec(tmp_path, _phase(), spec)
    )


def test_declared_parent_placeholder_requires_immutable_binding(tmp_path: Path):
    (tmp_path / "run.py").write_text("print('fixture')\n", encoding="utf-8")
    spec = load_rerun_spec(
        _write_spec(
            tmp_path,
            _payload(
                command=["python", "run.py", "--parent", "{parent:core}"],
                environment={"CORE_PARENT": "{parent:core}"},
            ),
        )
    )

    assert "missing_parent_binding" in _failed_codes(
        validate_rerun_spec(tmp_path, _phase(), spec)
    )


def test_parent_binding_requires_existing_hash_bound_source(tmp_path: Path):
    (tmp_path / "run.py").write_text("print('fixture')\n", encoding="utf-8")
    binding = {
        "name": "core",
        "environment_variable": "CORE_PARENT",
        "source": "parents/core",
        "sha256": "0" * 64,
    }
    spec = load_rerun_spec(
        _write_spec(
            tmp_path,
            _payload(
                command=["python", "run.py", "--parent", "{parent:core}"],
                required_parent_bindings=[binding],
            ),
        )
    )

    assert "missing_parent_source" in _failed_codes(
        validate_rerun_spec(tmp_path, _phase(), spec)
    )


def test_unsupported_spec_requires_concrete_reason(tmp_path: Path):
    spec = load_rerun_spec(
        _write_spec(
            tmp_path,
            _payload(supported=False, blocked_reason=None, command=[], required_paths=[]),
        )
    )

    assert "unsupported_reason" in _failed_codes(
        validate_rerun_spec(tmp_path, _phase(), spec)
    )


@pytest.mark.parametrize("phase_id", ["phase0", "phase05", "phase06"])
def test_real_historical_specs_are_conservative_and_never_authorize_new_experiments(phase_id: str):
    spec = load_rerun_spec(Path("docs/reproducibility") / phase_id / "rerun.yaml")

    assert spec.phase_id == phase_id
    assert not spec.supported
    assert spec.blocked_reason
    flattened = " ".join(spec.command).casefold()
    assert "phase07" not in flattened
    assert "confirm" not in flattened
    assert "capacity" not in flattened
    assert "d4-d" not in flattened
    assert spec.seed_policy.mode != "unrestricted"
