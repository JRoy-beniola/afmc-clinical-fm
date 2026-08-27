import importlib
import tomllib
from pathlib import Path
from types import SimpleNamespace

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

    with pytest.raises(ValueError, match="d1 stage is not complete"):
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
    assert not (tmp_path / "analysis" / "phase06_d1_reproduction.csv").exists()


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

    with pytest.raises(ValueError, match="d2a stage is not complete"):
        phase06_cli.main(
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

    cells = observed["cells"]
    assert len(cells) == 100
    assert len({cell.cell_id for cell in cells}) == 100
    assert {cell.stage for cell in cells} == {"d2a"}
    assert observed["planned"] == cells
    assert observed["device"] == "cpu"
    assert observed["resume"] is True
    assert not (tmp_path / "analysis" / "phase06_d2_effects.csv").exists()


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


def test_adjudicate_requires_completed_hash_valid_stages(tmp_path):
    config = phase06_cli.load_phase06_config("configs/experiments/phase06.yaml")
    phase05_config = phase06_cli.load_phase05_config(config.phase05_config)
    phase06_cli._bind_store(config, phase05_config, tmp_path)

    with pytest.raises(ValueError, match="d1 stage is not complete"):
        phase06_cli.main(["adjudicate", "--output", str(tmp_path)])

    assert not (tmp_path / "analysis" / "phase06_d3_adjudication.json").exists()


def test_parser_exposes_only_authorized_phase06_commands(tmp_path):
    parser = phase06_cli.build_parser()
    adjudicate = parser.parse_args(["adjudicate", "--output", str(tmp_path)])
    assert adjudicate.command == "adjudicate"

    d2b = parser.parse_args(
        [
            "d2b",
            "--config",
            "configs/experiments/phase06.yaml",
            "--parent-output",
            str(tmp_path / "parent"),
            "--output",
            str(tmp_path / "child"),
        ]
    )
    assert d2b.command == "d2b"
    assert Path(d2b.parent_output) == tmp_path / "parent"

    d2b_adjudicate = parser.parse_args(
        [
            "adjudicate-d2b",
            "--parent-output",
            str(tmp_path / "parent"),
            "--output",
            str(tmp_path / "child"),
        ]
    )
    assert d2b_adjudicate.command == "adjudicate-d2b"

    d4b = parser.parse_args(
        [
            "d4b",
            "--config",
            "configs/experiments/phase06.yaml",
            "--core-parent-output",
            str(tmp_path / "core"),
            "--d2b-parent-output",
            str(tmp_path / "d2b"),
            "--output",
            str(tmp_path / "d4b"),
        ]
    )
    assert d4b.command == "d4b"
    assert d4b.device == "cuda"

    d4b_adjudicate = parser.parse_args(
        [
            "adjudicate-d4b",
            "--core-parent-output",
            str(tmp_path / "core"),
            "--d2b-parent-output",
            str(tmp_path / "d2b"),
            "--output",
            str(tmp_path / "d4b"),
        ]
    )
    assert d4b_adjudicate.command == "adjudicate-d4b"

    for forbidden in (
        "confirmation",
        "robustness",
        "d4",
        "d4-capacity-time",
        "full-factorial",
        "full-factorial-addendum",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([forbidden])


def test_d2b_passes_exact_child_plan_only_after_parent_binding(tmp_path, monkeypatch):
    config = phase06_cli.load_phase06_config("configs/experiments/phase06.yaml")
    phase05_config = phase06_cli.load_phase05_config(config.phase05_config)
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    store = SimpleNamespace(output=child)
    observed: dict[str, object] = {}

    def fake_load(args):
        assert Path(args.parent_output) == parent
        assert Path(args.output) == child
        observed["bound_before_plan"] = True
        return config, phase05_config, store

    def capture_run(cells, received_store, *_args, device, resume, **_kwargs):
        assert observed.get("bound_before_plan") is True
        observed.update(cells=tuple(cells), store=received_store, device=device, resume=resume)
        return {"stage": "d2b"}

    def capture_analysis(received_store, cells, received_config):
        observed.update(analysis_store=received_store, analysis_cells=tuple(cells))
        assert received_config is config
        return object()

    monkeypatch.setattr(phase06_cli, "_load_d2b_inputs", fake_load)
    monkeypatch.setattr(phase06_cli, "run_phase06_stage", capture_run)
    monkeypatch.setattr(phase06_cli, "_persist_d2b_analysis", capture_analysis)

    assert (
        phase06_cli.main(
            [
                "d2b",
                "--config",
                "configs/experiments/phase06.yaml",
                "--parent-output",
                str(parent),
                "--output",
                str(child),
                "--device",
                "cuda",
                "--resume",
            ]
        )
        == 0
    )

    cells = observed["cells"]
    assert len(cells) == 100
    assert len({cell.cell_id for cell in cells}) == 100
    assert {cell.stage for cell in cells} == {"d2b"}
    assert observed["store"] is store
    assert observed["analysis_store"] is store
    assert observed["analysis_cells"] == cells
    assert observed["device"] == "cuda"
    assert observed["resume"] is True


def test_d2b_refuses_same_parent_and_child_output(tmp_path):
    with pytest.raises(ValueError, match="parent.*child.*distinct"):
        phase06_cli.main(
            [
                "d2b",
                "--config",
                "configs/experiments/phase06.yaml",
                "--parent-output",
                str(tmp_path),
                "--output",
                str(tmp_path),
                "--device",
                "cuda",
            ]
        )


def test_d4b_passes_exact_plan_only_after_two_parent_binding(tmp_path, monkeypatch):
    config = phase06_cli.load_phase06_config("configs/experiments/phase06.yaml")
    phase05_config = phase06_cli.load_phase05_config(config.phase05_config)
    core = tmp_path / "core"
    d2b = tmp_path / "d2b"
    child = tmp_path / "d4b"
    store = SimpleNamespace(output=child)
    observed: dict[str, object] = {}

    def fake_load(args):
        assert Path(args.core_parent_output) == core
        assert Path(args.d2b_parent_output) == d2b
        assert Path(args.output) == child
        observed["bound_before_plan"] = True
        return config, phase05_config, store

    def capture_run(cells, received_store, *_args, device, resume, **_kwargs):
        assert observed.get("bound_before_plan") is True
        observed.update(cells=tuple(cells), store=received_store, device=device, resume=resume)
        return {"stage": "d4b"}

    def capture_analysis(received_store, cells):
        observed.update(analysis_store=received_store, analysis_cells=tuple(cells))
        return object()

    monkeypatch.setattr(phase06_cli, "_load_d4b_inputs", fake_load)
    monkeypatch.setattr(phase06_cli, "run_phase06_stage", capture_run)
    monkeypatch.setattr(phase06_cli, "_persist_d4b_analysis", capture_analysis)

    assert (
        phase06_cli.main(
            [
                "d4b",
                "--config",
                "configs/experiments/phase06.yaml",
                "--core-parent-output",
                str(core),
                "--d2b-parent-output",
                str(d2b),
                "--output",
                str(child),
                "--device",
                "cuda",
                "--resume",
            ]
        )
        == 0
    )
    cells = observed["cells"]
    assert len(cells) == 100
    assert len({cell.cell_id for cell in cells}) == 100
    assert {cell.stage for cell in cells} == {"d4b"}
    assert {cell.n_train for cell in cells} == {40}
    assert observed["store"] is store
    assert observed["analysis_store"] is store
    assert observed["analysis_cells"] == cells
    assert observed["device"] == "cuda"
    assert observed["resume"] is True


def test_public_d4b_cli_is_cuda_only(tmp_path):
    with pytest.raises(SystemExit):
        phase06_cli.build_parser().parse_args(
            [
                "d4b",
                "--config",
                "configs/experiments/phase06.yaml",
                "--core-parent-output",
                str(tmp_path / "core"),
                "--d2b-parent-output",
                str(tmp_path / "d2b"),
                "--output",
                str(tmp_path / "child"),
                "--device",
                "cpu",
            ]
        )


def test_d4b_rejects_equal_or_nested_roots_before_parent_loading(tmp_path, monkeypatch):
    def forbidden_loader(*_args, **_kwargs):
        raise AssertionError("parent loader must not run for invalid root topology")

    monkeypatch.setattr(phase06_cli, "load_phase06_d4b_parent_evidence", forbidden_loader)
    cases = [
        (tmp_path / "same", tmp_path / "d2b", tmp_path / "same"),
        (tmp_path / "core", tmp_path / "core" / "d2b", tmp_path / "child"),
        (tmp_path / "core", tmp_path / "d2b", tmp_path / "d2b" / "child"),
        (tmp_path / "child" / "core", tmp_path / "d2b", tmp_path / "child"),
    ]
    for core, d2b, child in cases:
        with pytest.raises(ValueError, match="distinct and non-nested"):
            phase06_cli.main(
                [
                    "d4b",
                    "--config",
                    "configs/experiments/phase06.yaml",
                    "--core-parent-output",
                    str(core),
                    "--d2b-parent-output",
                    str(d2b),
                    "--output",
                    str(child),
                    "--device",
                    "cuda",
                ]
            )


def test_pyproject_registers_dedicated_phase06_console_script():
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["scripts"]["afmc-phase06"] == "afmc_fm.phase06.cli:main"
