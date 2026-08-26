import importlib
import tomllib
from pathlib import Path

import pytest

phase06_cli = importlib.import_module("afmc_fm.phase06.cli")


def test_d1_binds_protocol_before_planning_and_passes_locked_40_cells(
    tmp_path, monkeypatch
):
    original_planner = phase06_cli.plan_d1_cells
    observed: dict[str, object] = {}

    def plan_after_binding(config, phase05_config):
        assert (tmp_path / "protocol_lock.json").is_file()
        cells = original_planner(config, phase05_config)
        observed["planned"] = cells
        return cells

    def capture_run(cells, store, phase05_config, simulator_config, device, resume):
        observed.update(
            cells=cells,
            store=store,
            phase05_config=phase05_config,
            simulator_config=simulator_config,
            device=device,
            resume=resume,
        )
        return {"stage": "d1"}

    monkeypatch.setattr(phase06_cli, "plan_d1_cells", plan_after_binding)
    monkeypatch.setattr(phase06_cli, "run_phase06_stage", capture_run)

    result = phase06_cli.main(
        [
            "d1",
            "--config",
            "configs/experiments/phase06.yaml",
            "--output",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    cells = observed["cells"]
    assert len(cells) == 40
    assert len({cell.cell_id for cell in cells}) == 40
    assert {cell.stage for cell in cells} == {"d1"}
    assert observed["planned"] == cells
    assert observed["device"] == "cpu"
    assert observed["resume"] is False
    store = observed["store"]
    assert store.output == tmp_path
    assert len(store.protocol_hash) == 64
    assert len(store.config_hash) == 64
    assert (tmp_path / "protocol_lock.json").is_file()


def test_d2a_binds_protocol_before_planning_and_passes_locked_100_cells(
    tmp_path, monkeypatch
):
    original_planner = phase06_cli.plan_d2a_cells
    observed: dict[str, object] = {}

    def plan_after_binding(config):
        assert (tmp_path / "protocol_lock.json").is_file()
        cells = original_planner(config)
        observed["planned"] = cells
        return cells

    def capture_run(cells, store, phase05_config, simulator_config, device, resume):
        observed.update(cells=cells, device=device, resume=resume)
        return {"stage": "d2a"}

    monkeypatch.setattr(phase06_cli, "plan_d2a_cells", plan_after_binding)
    monkeypatch.setattr(phase06_cli, "run_phase06_stage", capture_run)

    result = phase06_cli.main(
        [
            "d2a",
            "--config",
            "configs/experiments/phase06.yaml",
            "--output",
            str(tmp_path),
            "--device",
            "cpu",
            "--resume",
        ]
    )

    assert result == 0
    cells = observed["cells"]
    assert len(cells) == 100
    assert len({cell.cell_id for cell in cells}) == 100
    assert {cell.stage for cell in cells} == {"d2a"}
    assert observed["planned"] == cells
    assert observed["device"] == "cpu"
    assert observed["resume"] is True


def test_conflicting_output_protocol_fails_before_planning(tmp_path, monkeypatch):
    (tmp_path / "protocol_lock.json").write_text("{}\n", encoding="utf-8")
    planned = False

    def forbidden_planner(*_args, **_kwargs):
        nonlocal planned
        planned = True
        raise AssertionError("planner must not run before protocol identity is bound")

    monkeypatch.setattr(phase06_cli, "plan_d1_cells", forbidden_planner)

    with pytest.raises(ValueError, match="conflicting protocol lock"):
        phase06_cli.main(
            [
                "d1",
                "--config",
                "configs/experiments/phase06.yaml",
                "--output",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )

    assert planned is False


def test_parser_exposes_only_d1_d2a_and_adjudicate_scientific_commands(tmp_path):
    parser = phase06_cli.build_parser()
    adjudicate = parser.parse_args(["adjudicate", "--output", str(tmp_path)])
    assert adjudicate.command == "adjudicate"

    for forbidden in ("confirmation", "robustness", "d4"):
        with pytest.raises(SystemExit):
            parser.parse_args([forbidden])


def test_pyproject_registers_dedicated_phase06_console_script():
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["scripts"]["afmc-phase06"] == "afmc_fm.phase06.cli:main"
