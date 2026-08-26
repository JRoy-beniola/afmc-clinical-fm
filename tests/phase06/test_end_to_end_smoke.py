from __future__ import annotations

import hashlib
import importlib
import json

import pandas as pd
import torch

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase06.planning import plan_d1_cells, plan_d2a_cells
from afmc_fm.phase06.protocol import build_phase06_protocol_lock
from afmc_fm.phase06.store import Phase06Store

phase06_cli = importlib.import_module("afmc_fm.phase06.cli")

_PARENT_EXECUTION_SHA = "1718402df1d6ef344168677e6d26ea664708e1bc"
_PARENT_PROTOCOL_SHA256 = (
    "c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51"
)
_CHILD_EXECUTION_SHA = "f" * 40


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _fake_metrics(cell) -> pd.DataFrame:
    if cell.stage == "d1":
        delta_mae = {5: -0.01, 10: -0.01, 20: -0.005, 40: 0.03}[cell.n_train]
    elif cell.n_train == 5:
        delta_mae = -0.01 + 0.005 * (cell.model_seed - 603)
    else:
        delta_mae = 0.03 + 0.005 * (cell.cohort_seed - 403)

    control_mae = 0.50
    mae = control_mae if cell.flow_mode == "none" else control_mae - delta_mae
    control_r2 = 0.40
    delta_r2 = -0.01 if cell.n_train == 40 else 0.0
    latent_r2 = control_r2 if cell.flow_mode == "none" else control_r2 + delta_r2
    variant = f"{cell.flow_mode}__none__deterministic"
    common = {
        "stage": cell.stage,
        "world": cell.world,
        "cohort_seed": cell.cohort_seed,
        "subset_seed": cell.subset_seed,
        "model_seed": cell.model_seed,
        "n_train": cell.n_train,
        "variant": variant,
        "split": "test",
    }
    return pd.DataFrame(
        [
            {**common, "metric": "mae", "value": mae},
            {**common, "metric": "latent_aligned_r2", "value": latent_r2},
        ]
    )


def _fake_trace() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"epoch": 1, "validation_mae": 0.40},
            {"epoch": 2, "validation_mae": 0.50},
            {"epoch": 3, "validation_mae": 0.50},
        ]
    )


def _fake_summary() -> dict[str, object]:
    return {
        "epochs_run": 3,
        "stop_epoch": 3,
        "selected_checkpoint_epoch": 2,
        "selected_validation_core_loss": 0.60,
        "shadow_mae_checkpoint_epoch": 1,
        "shadow_validation_mae": 0.40,
        "early_stop_reason": "max_epochs_reached",
    }


def _fake_stage_run(cells, store, *_args, **_kwargs):
    planned = tuple(cells)
    state = {"weight": torch.tensor([1.0])}
    for cell in planned:
        store.write_cell_bundle(
            cell,
            metrics=_fake_metrics(cell),
            trace=_fake_trace(),
            summary=_fake_summary(),
            production_state_dict=state,
            shadow_state_dict=state,
        )
    expected = frozenset(cell.cell_id for cell in planned)
    store.mark_stage_complete(planned[0].stage, expected)
    return {
        "stage": planned[0].stage,
        "planned_cells": len(planned),
        "completed_after": len(planned),
    }


def _write_frozen_parent(root):
    config = phase06_cli.load_phase06_config("configs/experiments/phase06.yaml")
    phase05_config = phase06_cli.load_phase05_config(config.phase05_config)
    lock = build_phase06_protocol_lock(
        config,
        phase05_config,
        execution_commit=_PARENT_EXECUTION_SHA,
        phase06_spec_path=config.phase06_spec,
        phase05_protocol_path="docs/results/phase05/raw/official_output/protocol_lock.json",
    )
    lock_bytes = _canonical_json_bytes(lock)
    assert hashlib.sha256(lock_bytes).hexdigest() == _PARENT_PROTOCOL_SHA256
    store = Phase06Store(
        root,
        protocol_hash=_PARENT_PROTOCOL_SHA256,
        config_hash=canonical_config_hash(config),
        execution_commit=_PARENT_EXECUTION_SHA,
    )
    store.write_protocol_lock(lock)

    d1_cells = plan_d1_cells(config, phase05_config)
    d1_marker = root / "stages" / "d1" / "COMPLETE"
    d1_marker.parent.mkdir(parents=True, exist_ok=True)
    d1_marker.write_bytes(
        _canonical_json_bytes(
            {
                "schema_version": 1,
                "identity": store.identity,
                "stage": "d1",
                "cell_ids": sorted(cell.cell_id for cell in d1_cells),
            }
        )
    )
    _fake_stage_run(plan_d2a_cells(config), store)

    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "phase06_d3_adjudication.json").write_bytes(
        _canonical_json_bytes(
            {
                "phenomenon_reproduction": "reproduced",
                "interaction_ambiguity": True,
                "next_required_stage": "D2B",
                "triggered_escalations": [
                    "D2B",
                    "D4_OPTIMIZATION",
                    "D4_CAPACITY_TIME",
                ],
                "input_artifact_hashes": {},
            }
        )
    )
    return store


def _snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_fake_complete_d1_d2_pipeline_persists_analysis_and_adjudicates(
    tmp_path, monkeypatch, capsys
):
    calls: list[str] = []

    def capture_stage_run(cells, store, *args, **kwargs):
        calls.append(next(iter(cells)).stage)
        return _fake_stage_run(cells, store, *args, **kwargs)

    monkeypatch.setattr(phase06_cli, "run_phase06_stage", capture_stage_run)

    config = "configs/experiments/phase06.yaml"
    assert (
        phase06_cli.main(
            ["d1", "--config", config, "--output", str(tmp_path), "--device", "cpu"]
        )
        == 0
    )
    assert (tmp_path / "stages" / "d1" / "COMPLETE").is_file()
    assert (tmp_path / "analysis" / "phase06_d1_reproduction.csv").is_file()
    assert (tmp_path / "analysis" / "phase06_d1_classification.json").is_file()

    assert (
        phase06_cli.main(
            ["d2a", "--config", config, "--output", str(tmp_path), "--device", "cpu"]
        )
        == 0
    )
    assert (tmp_path / "stages" / "d2a" / "COMPLETE").is_file()
    for name in (
        "phase06_d2_effects.csv",
        "phase06_d2_factor_level_effects.csv",
        "phase06_d2_variance_components.csv",
        "phase06_d2_bootstrap_diagnostics.csv",
        "phase06_d2_n_shift.csv",
        "phase06_d2_n_shift_summary.json",
    ):
        assert (tmp_path / "analysis" / name).is_file(), name

    before_adjudicate = list(calls)
    assert phase06_cli.main(["adjudicate", "--output", str(tmp_path)]) == 0
    assert calls == before_adjudicate
    assert (tmp_path / "analysis" / "phase06_d3_adjudication.json").is_file()
    assert capsys.readouterr().out.strip() == "D4_CAPACITY_TIME"


def test_fake_d2b_child_pipeline_preserves_parent_and_adjudicates_explicitly(
    tmp_path, monkeypatch, capsys
):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    _write_frozen_parent(parent)
    parent_before = _snapshot(parent)
    calls: list[str] = []

    def capture_stage_run(cells, store, *args, **kwargs):
        calls.append(next(iter(cells)).stage)
        return _fake_stage_run(cells, store, *args, **kwargs)

    monkeypatch.setattr(phase06_cli, "execution_commit_sha", lambda: _CHILD_EXECUTION_SHA)
    monkeypatch.setattr(phase06_cli, "run_phase06_stage", capture_stage_run)

    config = "configs/experiments/phase06.yaml"
    assert (
        phase06_cli.main(
            [
                "d2b",
                "--config",
                config,
                "--parent-output",
                str(parent),
                "--output",
                str(child),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert calls == ["d2b"]
    assert (child / "protocol_lock.json").is_file()
    assert (child / "stages" / "d2b" / "COMPLETE").is_file()
    for name in (
        "phase06_d2b_effects.csv",
        "phase06_d2b_factor_level_effects.csv",
        "phase06_d2b_variance_components.csv",
        "phase06_d2b_bootstrap_diagnostics.csv",
        "phase06_d2b_n_shift.csv",
        "phase06_d2b_n_shift_summary.json",
    ):
        assert (child / "analysis" / name).is_file(), name
    assert not (child / "analysis" / "phase06_d2b_adjudication.json").exists()
    assert _snapshot(parent) == parent_before

    before_adjudicate = list(calls)
    assert (
        phase06_cli.main(
            [
                "adjudicate-d2b",
                "--parent-output",
                str(parent),
                "--output",
                str(child),
            ]
        )
        == 0
    )
    assert calls == before_adjudicate
    assert (child / "analysis" / "phase06_d2b_overlap_rerun.json").is_file()
    assert (child / "analysis" / "phase06_d2b_adjudication.json").is_file()
    assert _snapshot(parent) == parent_before
    assert capsys.readouterr().out.strip() == "D4_OPTIMIZATION"
