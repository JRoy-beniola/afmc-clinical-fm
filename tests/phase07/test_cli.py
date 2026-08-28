from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path

import pytest


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


def test_pyproject_registers_phase07_console_script():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["afmc-phase07"] == "afmc_fm.phase07.cli:main"
