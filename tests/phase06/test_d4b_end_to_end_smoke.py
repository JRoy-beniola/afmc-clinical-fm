from __future__ import annotations

import hashlib
import importlib

import pandas as pd
import torch

import afmc_fm.phase06.protocol as protocol_module
from afmc_fm.execution.persistence import canonical_config_hash

phase06_cli = importlib.import_module("afmc_fm.phase06.cli")

_D4B_CHILD_EXECUTION_SHA = "e" * 40


def _parent_evidence() -> dict[str, object]:
    config = phase06_cli.load_phase06_config("configs/experiments/phase06.yaml")
    return {
        "core_parent_execution_sha": protocol_module._PARENT_PHASE06_EXECUTION_SHA,
        "core_parent_protocol_lock_sha256": protocol_module._PARENT_PHASE06_PROTOCOL_SHA256,
        "core_parent_d3_sha256": protocol_module._PARENT_PHASE06_D3_SHA256,
        "d2b_parent_execution_sha": protocol_module._D2B_PHASE06_EXECUTION_SHA,
        "d2b_parent_protocol_canonical_sha256": (
            protocol_module._D2B_PROTOCOL_CANONICAL_SHA256
        ),
        "d2b_parent_adjudication_canonical_sha256": (
            protocol_module._D2B_ADJUDICATION_CANONICAL_SHA256
        ),
        "d2b_next_required_stage": "D4_OPTIMIZATION",
        "phase06_config_sha256": canonical_config_hash(config),
        "phase06_spec_sha256": hashlib.sha256(
            config.phase06_spec.read_bytes()
        ).hexdigest(),
        "forbidden_seed_sets": {
            "cohort": list(config.forbidden_cohort_seeds),
            "subset": list(config.forbidden_subset_seeds),
            "model": list(config.forbidden_model_seeds),
        },
    }


def _fake_metrics(cell) -> pd.DataFrame:
    context_delta = 0.02 + 0.005 * (cell.cohort_seed - 401)
    control_mae = 0.50
    mae = control_mae if cell.flow_mode == "none" else control_mae - context_delta
    return pd.DataFrame(
        [
            {
                "stage": cell.stage,
                "world": cell.world,
                "cohort_seed": cell.cohort_seed,
                "subset_seed": cell.subset_seed,
                "model_seed": cell.model_seed,
                "n_train": cell.n_train,
                "variant": f"{cell.flow_mode}__none__deterministic",
                "split": "test",
                "metric": "mae",
                "value": mae,
            }
        ]
    )


def _fake_trace(cell) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "epoch": 1,
                "gradient_l2_norm": 1.0 + (cell.model_seed - 1000) / 100.0,
                "mean_flow_displacement": 0.0 if cell.flow_mode == "none" else 0.1,
            },
            {
                "epoch": 2,
                "gradient_l2_norm": 0.9 + (cell.model_seed - 1000) / 100.0,
                "mean_flow_displacement": 0.0 if cell.flow_mode == "none" else 0.11,
            },
        ]
    )


def _fake_summary(cell) -> dict[str, object]:
    selected = 1 + ((cell.model_seed + cell.cohort_seed) % 2)
    return {
        "epochs_run": 2,
        "stop_epoch": 2,
        "selected_checkpoint_epoch": selected,
        "selected_validation_core_loss": 0.5,
        "shadow_mae_checkpoint_epoch": 1,
        "shadow_validation_mae": 0.4,
        "early_stop_reason": "max_epochs_reached",
    }


def _fake_stage_run(cells, store, *_args, **_kwargs):
    planned = tuple(cells)
    state = {"weight": torch.tensor([1.0])}
    for cell in planned:
        store.write_cell_bundle(
            cell,
            metrics=_fake_metrics(cell),
            trace=_fake_trace(cell),
            summary=_fake_summary(cell),
            production_state_dict=state,
            shadow_state_dict=state,
        )
    expected = frozenset(cell.cell_id for cell in planned)
    store.mark_stage_complete("d4b", expected)
    return {
        "stage": "d4b",
        "planned_cells": len(planned),
        "completed_after": len(planned),
    }


def _snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_fake_d4b_child_executes_100_cells_then_adjudicates_only_explicitly(
    tmp_path, monkeypatch, capsys
):
    core_parent = tmp_path / "core-parent"
    d2b_parent = tmp_path / "d2b-parent"
    child = tmp_path / "d4b-child"
    core_parent.mkdir()
    d2b_parent.mkdir()
    (core_parent / "sentinel.txt").write_text("immutable-core\n", encoding="utf-8")
    (d2b_parent / "sentinel.txt").write_text("immutable-d2b\n", encoding="utf-8")
    core_before = _snapshot(core_parent)
    d2b_before = _snapshot(d2b_parent)
    evidence = _parent_evidence()
    calls: list[str] = []

    def capture_stage_run(cells, store, *args, **kwargs):
        calls.append(next(iter(cells)).stage)
        return _fake_stage_run(cells, store, *args, **kwargs)

    monkeypatch.setattr(phase06_cli, "execution_commit_sha", lambda: _D4B_CHILD_EXECUTION_SHA)
    monkeypatch.setattr(phase06_cli, "run_phase06_stage", capture_stage_run)
    monkeypatch.setattr(
        phase06_cli,
        "load_phase06_d4b_parent_evidence",
        lambda *_args, **_kwargs: dict(evidence),
    )

    config = "configs/experiments/phase06.yaml"
    assert (
        phase06_cli.main(
            [
                "d4b",
                "--config",
                config,
                "--core-parent-output",
                str(core_parent),
                "--d2b-parent-output",
                str(d2b_parent),
                "--output",
                str(child),
                "--device",
                "cuda",
            ]
        )
        == 0
    )

    assert calls == ["d4b"]
    assert (child / "protocol_lock.json").is_file()
    assert (child / "stages" / "d4b" / "COMPLETE").is_file()
    assert len(list((child / "stages" / "d4b" / "cells").glob("*.json"))) == 100
    for name in (
        "phase06_d4b_effects.csv",
        "phase06_d4b_model_seed_summary.csv",
        "phase06_d4b_context_summary.csv",
        "phase06_d4b_bootstrap_diagnostics.json",
        "phase06_d4b_optimization_dispersion.csv",
    ):
        assert (child / "analysis" / name).is_file(), name
    assert not (child / "analysis" / "phase06_d4b_adjudication.json").exists()
    assert _snapshot(core_parent) == core_before
    assert _snapshot(d2b_parent) == d2b_before

    before_adjudicate = list(calls)
    assert (
        phase06_cli.main(
            [
                "adjudicate-d4b",
                "--core-parent-output",
                str(core_parent),
                "--d2b-parent-output",
                str(d2b_parent),
                "--output",
                str(child),
            ]
        )
        == 0
    )
    assert calls == before_adjudicate
    decision = child / "analysis" / "phase06_d4b_adjudication.json"
    assert decision.is_file()
    assert _snapshot(core_parent) == core_before
    assert _snapshot(d2b_parent) == d2b_before
    assert capsys.readouterr().out.strip() == "D4_CAPACITY_TIME"
