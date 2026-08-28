from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path

import pytest

from afmc_fm.phase07.config import load_phase07_config

_CONFIG_PATH = Path("configs/experiments/phase07.yaml")


def _cli_api():
    try:
        return importlib.import_module("afmc_fm.phase07.cli")
    except ModuleNotFoundError as error:
        raise AssertionError("Phase 0.7 CLI module is not implemented") from error


def test_parser_exposes_manifest_smoke_and_authorization_gated_official_commands(tmp_path):
    module = _cli_api()
    parser = module.build_parser()

    manifest = parser.parse_args(["manifest", "--output", str(tmp_path / "manifest.json")])
    assert manifest.command == "manifest"

    cpu = parser.parse_args(["smoke", "--device", "cpu"])
    cuda = parser.parse_args(["smoke", "--device", "cuda"])
    assert cpu.device == "cpu"
    assert cuda.device == "cuda"

    official = parser.parse_args(
        [
            "official",
            "--authorization",
            str(tmp_path / "authorization.json"),
            "--output",
            str(tmp_path / "official"),
        ]
    )
    assert official.command == "official"
    assert official.device == "cuda"
    assert official.resume is False

    resumed = parser.parse_args(
        [
            "official",
            "--authorization",
            str(tmp_path / "authorization.json"),
            "--output",
            str(tmp_path / "official"),
            "--resume",
        ]
    )
    assert resumed.resume is True

    analyze = parser.parse_args(
        [
            "analyze",
            "--output",
            str(tmp_path / "official"),
        ]
    )
    assert analyze.command == "analyze"
    assert analyze.output == str(tmp_path / "official")


def test_manifest_command_writes_identity_only_and_no_cell_outputs(tmp_path):
    module = _cli_api()
    output = tmp_path / "phase07-manifest.json"

    assert module.main(["manifest", "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["phase"] == "phase07"
    assert payload["expected_cell_count"] == 200
    assert payload["execution_commit"]
    assert not (tmp_path / "cells").exists()


def test_official_cli_fails_closed_before_execution_when_authorization_is_missing(
    tmp_path,
    monkeypatch,
):
    module = _cli_api()
    called = False

    def forbidden_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("official runner must not be entered before authorization validation")

    monkeypatch.setattr(module, "execute_authorized_phase07", forbidden_run)

    with pytest.raises(ValueError, match="authorization"):
        module.main(
            [
                "official",
                "--authorization",
                str(tmp_path / "missing.json"),
                "--output",
                str(tmp_path / "official"),
                "--device",
                "cpu",
            ]
        )

    assert called is False


def test_official_cli_loads_dependencies_once_and_passes_same_objects_to_execution(
    tmp_path,
    monkeypatch,
):
    module = _cli_api()
    config = load_phase07_config(_CONFIG_PATH)
    phase05_config = module.load_phase05_config(config.phase05_config)
    simulator_config = module._load_simulator_config(config.simulator_config)
    execution_sha = "a" * 40
    loads = {"phase05": 0, "simulator": 0}
    manifest = {
        "schema_version": 1,
        "phase": "phase07",
        "execution_commit": execution_sha,
    }

    def load_phase05_once(_path):
        loads["phase05"] += 1
        return phase05_config

    def load_simulator_once(_path):
        loads["simulator"] += 1
        return simulator_config

    def build_manifest(
        observed_config,
        *,
        execution_commit,
        phase07_spec_path,
        phase05_config: object,
        simulator_config: object,
    ):
        assert observed_config is config
        assert execution_commit == execution_sha
        assert phase07_spec_path == config.phase07_spec
        assert phase05_config is globals_phase05
        assert simulator_config is globals_simulator
        return manifest

    globals_phase05 = phase05_config
    globals_simulator = simulator_config

    def execute(**kwargs):
        assert kwargs["phase05_config"] is phase05_config
        assert kwargs["simulator_config"] is simulator_config
        return ()

    monkeypatch.setattr(module, "load_phase07_config", lambda _path: config)
    monkeypatch.setattr(module, "load_phase05_config", load_phase05_once)
    monkeypatch.setattr(module, "_load_simulator_config", load_simulator_once)
    monkeypatch.setattr(module, "execution_commit_sha", lambda: execution_sha)
    monkeypatch.setattr(module, "build_phase07_execution_manifest", build_manifest)
    monkeypatch.setattr(module, "require_phase07_official_authorization", lambda *_args: {})
    monkeypatch.setattr(module, "execute_authorized_phase07", execute)

    assert (
        module.main(
            [
                "official",
                "--authorization",
                str(tmp_path / "authorization.json"),
                "--output",
                str(tmp_path / "official"),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert loads == {"phase05": 1, "simulator": 1}


def test_execute_authorized_phase07_uses_preloaded_dependencies_without_reloading(
    tmp_path,
    monkeypatch,
):
    module = _cli_api()
    config = load_phase07_config(_CONFIG_PATH)
    phase05_config = module.load_phase05_config(config.phase05_config)
    simulator_config = module._load_simulator_config(config.simulator_config)
    execution_sha = "a" * 40
    manifest = module.build_phase07_execution_manifest(
        config,
        execution_commit=execution_sha,
        phase07_spec_path=config.phase07_spec,
    )

    monkeypatch.setattr(module, "require_clean_phase07_checkout", lambda: None)
    monkeypatch.setattr(module, "execution_commit_sha", lambda: execution_sha)
    monkeypatch.setattr(module, "require_phase07_official_authorization", lambda *_args: {})

    def forbidden_reload(*_args, **_kwargs):
        raise AssertionError("authorized dependency configs must not be reloaded")

    def stop_at_store(**_kwargs):
        raise RuntimeError("store boundary reached")

    monkeypatch.setattr(module, "load_phase05_config", forbidden_reload)
    monkeypatch.setattr(module, "_load_simulator_config", forbidden_reload)
    monkeypatch.setattr(module, "_initialize_store", stop_at_store)

    with pytest.raises(RuntimeError, match="store boundary reached"):
        module.execute_authorized_phase07(
            config=config,
            phase05_config=phase05_config,
            simulator_config=simulator_config,
            manifest=manifest,
            authorization_path=tmp_path / "authorization.json",
            output=tmp_path / "official",
            device="cpu",
            resume=False,
        )


def test_resume_validates_persisted_identity_before_cuda_resolution(tmp_path, monkeypatch):
    module = _cli_api()
    config = load_phase07_config(_CONFIG_PATH)
    execution_sha = "a" * 40
    manifest = module.build_phase07_execution_manifest(
        config,
        execution_commit=execution_sha,
        phase07_spec_path=config.phase07_spec,
    )
    cuda_checked = False

    monkeypatch.setattr(module, "require_clean_phase07_checkout", lambda: None)
    monkeypatch.setattr(module, "execution_commit_sha", lambda: execution_sha)
    monkeypatch.setattr(module, "require_phase07_official_authorization", lambda *_args: {})

    def reject_persisted_identity(**_kwargs):
        raise ValueError("persisted identity mismatch")

    def forbidden_cuda_query():
        nonlocal cuda_checked
        cuda_checked = True
        raise AssertionError("CUDA must not be queried before persisted identity validation")

    monkeypatch.setattr(module, "_initialize_store", reject_persisted_identity)
    monkeypatch.setattr(module.torch.cuda, "is_available", forbidden_cuda_query)

    with pytest.raises(ValueError, match="persisted identity mismatch"):
        module.execute_authorized_phase07(
            config=config,
            manifest=manifest,
            authorization_path=tmp_path / "authorization.json",
            output=tmp_path / "official",
            device="cuda",
            resume=True,
        )

    assert cuda_checked is False


def test_pyproject_registers_phase07_console_script():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["afmc-phase07"] == "afmc_fm.phase07.cli:main"
