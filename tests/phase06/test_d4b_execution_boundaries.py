from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import afmc_fm.phase06.pipeline as pipeline_module
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase06.config import load_phase06_config
from afmc_fm.phase06.planning import plan_d2b_cells
from afmc_fm.phase06.protocol import (
    build_phase06_d2b_protocol_lock,
    build_phase06_d4b_protocol_lock,
)

_PHASE06_CONFIG = Path("configs/experiments/phase06.yaml")
_PHASE06_SPEC = Path("docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md")
_D2B_ADDENDUM = Path(
    "docs/superpowers/specs/2026-08-26-phase0-6-d2b-execution-addendum.md"
)
_D4B_ADDENDUM = Path(
    "docs/superpowers/specs/2026-08-27-phase0-6-d4b-execution-addendum.md"
)
_PHASE05_PROTOCOL = Path("docs/results/phase05/raw/official_output/protocol_lock.json")
_CORE_EXECUTION_SHA = "1718402df1d6ef344168677e6d26ea664708e1bc"
_CORE_PROTOCOL_SHA256 = (
    "c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51"
)
_CORE_D3_SHA256 = "6b9fffed7503fae6beeac2314238ae3d10ffdebfd27ea71ad952ab1f87916460"
_D2B_EXECUTION_SHA = "516c9e3c0e965582fa5cce976e9d8ebf32ea8404"
_D2B_PROTOCOL_CANONICAL_SHA256 = (
    "33f7cb1f6e71560f547a746cb7f5eb41130f794ab1e3a6fb1928c40e05b29f12"
)
_D2B_ADJUDICATION_CANONICAL_SHA256 = (
    "ea28fd4d5f6490a10fad20d5d3f3e76a1de08bf6b1be3b9805c9cf6c519e845f"
)


def _canonical_hash(payload: object) -> str:
    data = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _core_evidence() -> dict[str, object]:
    config = load_phase06_config(_PHASE06_CONFIG)
    return {
        "parent_execution_sha": _CORE_EXECUTION_SHA,
        "parent_protocol_lock_sha256": _CORE_PROTOCOL_SHA256,
        "parent_d3_sha256": _CORE_D3_SHA256,
        "parent_d3_next_required_stage": "D2B",
        "parent_phase06_config_sha256": canonical_config_hash(config),
        "parent_phase06_spec_sha256": hashlib.sha256(
            _PHASE06_SPEC.read_bytes()
        ).hexdigest(),
        "parent_forbidden_seed_sets": {
            "cohort": list(range(701, 711)),
            "subset": list(range(801, 811)),
            "model": list(range(901, 911)),
        },
    }


def _parent_evidence() -> dict[str, object]:
    config = load_phase06_config(_PHASE06_CONFIG)
    return {
        "core_parent_execution_sha": _CORE_EXECUTION_SHA,
        "core_parent_protocol_lock_sha256": _CORE_PROTOCOL_SHA256,
        "core_parent_d3_sha256": _CORE_D3_SHA256,
        "d2b_parent_execution_sha": _D2B_EXECUTION_SHA,
        "d2b_parent_protocol_canonical_sha256": _D2B_PROTOCOL_CANONICAL_SHA256,
        "d2b_parent_adjudication_canonical_sha256": _D2B_ADJUDICATION_CANONICAL_SHA256,
        "d2b_next_required_stage": "D4_OPTIMIZATION",
        "phase06_config_sha256": canonical_config_hash(config),
        "phase06_spec_sha256": hashlib.sha256(_PHASE06_SPEC.read_bytes()).hexdigest(),
        "forbidden_seed_sets": {
            "cohort": list(range(701, 711)),
            "subset": list(range(801, 811)),
            "model": list(range(901, 911)),
        },
    }


def _d2b_protocol() -> dict[str, object]:
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)
    payload = build_phase06_d2b_protocol_lock(
        config,
        phase05_config,
        execution_commit=_D2B_EXECUTION_SHA,
        phase06_spec_path=_PHASE06_SPEC,
        d2b_addendum_path=_D2B_ADDENDUM,
        phase05_protocol_path=_PHASE05_PROTOCOL,
        parent_evidence=_core_evidence(),
    )
    assert _canonical_hash(payload) == _D2B_PROTOCOL_CANONICAL_SHA256
    return payload


def _d2b_adjudication() -> dict[str, object]:
    payload: dict[str, object] = {
        "complementary_array_evidence": "sufficient",
        "dominant_factors": {
            "d2a": {"40": "model", "5": "none"},
            "d2b": {"40": "model", "5": "none"},
        },
        "input_artifact_hashes": {
            "d2a_effect_rows": "72cfda9c8daf0faeafe00203e11c9d15e06172812696d269b33c9e7e03aa101e",
            "d2a_variance_components": "71d997032c90dd060842751b7483ed5d44983876a75925d09a1e229fee3f2c67",
            "d2b_effect_rows": "15157ca4fe7d49b286ad4093e85b7f046455a0a1e2d70b821793f10fba7f7ca8",
            "d2b_variance_components": "cca16e93313571df6388c26c8eac1d8a6c5d656dcd0433f9851e0433dd6e18c3",
            "parent_d3": "52ba28e3a3eddd1663158d8f636bb7ef5c4d8da189139decff2b98f0e9545fdb",
        },
        "next_required_stage": "D4_OPTIMIZATION",
        "overlap_rerun_diagnostics": {
            "40": {
                "maximum_absolute_delta_mae_difference": 0.0,
                "mean_absolute_delta_mae_difference": 0.0,
                "overlap_pair_count": 5,
                "pearson_correlation": 1.0,
            },
            "5": {
                "maximum_absolute_delta_mae_difference": 0.0,
                "mean_absolute_delta_mae_difference": 0.0,
                "overlap_pair_count": 5,
                "pearson_correlation": 1.0,
            },
        },
        "rationale": [
            "D2-A/D2-B N40 dominant factors: model/model.",
            "D2-A/D2-B N5 dominant factors: none/none.",
            "Complementary-array evidence: sufficient.",
            "Next required stage: D4_OPTIMIZATION.",
        ],
        "remaining_parent_escalations": ["D4_OPTIMIZATION", "D4_CAPACITY_TIME"],
    }
    assert _canonical_hash(payload) == _D2B_ADJUDICATION_CANONICAL_SHA256
    return payload


def _write_d2b_parent(tmp_path: Path) -> tuple[Path, Path]:
    core = tmp_path / "core"
    d2b = tmp_path / "d2b"
    (core / "analysis").mkdir(parents=True)
    (d2b / "analysis").mkdir(parents=True)
    (core / "analysis" / "phase06_d3_adjudication.json").write_text(
        json.dumps({"next_required_stage": "D2B"}), encoding="utf-8"
    )
    (d2b / "protocol_lock.json").write_text(
        json.dumps(_d2b_protocol(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (d2b / "analysis" / "phase06_d2b_adjudication.json").write_text(
        json.dumps(_d2b_adjudication(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return core, d2b


class _FakeStore:
    def __init__(self, output: Path):
        self.output = output

    def load_stage_metrics(self, stage: str):
        return stage


def _patch_parent_recomputation(monkeypatch, core: Path, d2b: Path, recomputed: object):
    monkeypatch.setattr(
        pipeline_module,
        "load_phase06_d2b_parent_evidence",
        lambda *_args, **_kwargs: _core_evidence(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "open_bound_store",
        lambda output: _FakeStore(Path(output)),
    )
    completed: list[tuple[str, int]] = []

    def require_completed(_store, cells):
        planned = tuple(cells)
        completed.append((planned[0].stage, len(planned)))
        return frozenset(cell.cell_id for cell in planned)

    monkeypatch.setattr(pipeline_module, "require_completed_stage", require_completed)
    monkeypatch.setattr(pipeline_module, "analyze_d2a", lambda *_args: "d2a-analysis")
    monkeypatch.setattr(pipeline_module, "analyze_d2b", lambda *_args: "d2b-analysis")
    monkeypatch.setattr(
        pipeline_module,
        "adjudicate_d2b",
        lambda *_args: recomputed,
    )
    return completed


def test_d4b_protocol_binds_frozen_parent_chain_and_matrix() -> None:
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)

    lock = build_phase06_d4b_protocol_lock(
        config,
        phase05_config,
        execution_commit="d" * 40,
        phase06_spec_path=_PHASE06_SPEC,
        d4b_addendum_path=_D4B_ADDENDUM,
        phase05_protocol_path=_PHASE05_PROTOCOL,
        parent_evidence=_parent_evidence(),
    )

    assert lock["schema_version"] == 3
    assert lock["core_parent_execution_sha"] == _CORE_EXECUTION_SHA
    assert lock["core_parent_protocol_lock_sha256"] == _CORE_PROTOCOL_SHA256
    assert lock["core_parent_d3_sha256"] == _CORE_D3_SHA256
    assert lock["d2b_parent_execution_sha"] == _D2B_EXECUTION_SHA
    assert lock["d2b_parent_protocol_canonical_sha256"] == _D2B_PROTOCOL_CANONICAL_SHA256
    assert (
        lock["d2b_parent_adjudication_canonical_sha256"]
        == _D2B_ADJUDICATION_CANONICAL_SHA256
    )
    assert lock["d2b_next_required_stage"] == "D4_OPTIMIZATION"
    assert lock["d4b_contexts"] == [
        [401, 501],
        [402, 502],
        [403, 503],
        [404, 504],
        [405, 505],
    ]
    assert lock["d4b_model_seeds"] == list(range(1001, 1011))
    assert lock["d4b_n_train"] == 40
    assert lock["d4b_flow_modes"] == ["none", "time_scaled"]
    assert lock["d4b_bootstrap_resamples"] == 10_000
    assert lock["d4b_bootstrap_seed"] == 20260827


def test_d4b_protocol_rejects_adjudication_identity_drift() -> None:
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)
    evidence = _parent_evidence()
    evidence["d2b_parent_adjudication_canonical_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="D2-B adjudication"):
        build_phase06_d4b_protocol_lock(
            config,
            phase05_config,
            execution_commit="d" * 40,
            phase06_spec_path=_PHASE06_SPEC,
            d4b_addendum_path=_D4B_ADDENDUM,
            phase05_protocol_path=_PHASE05_PROTOCOL,
            parent_evidence=evidence,
        )


def test_d4b_protocol_rejects_non_optimization_route() -> None:
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)
    evidence = _parent_evidence()
    evidence["d2b_next_required_stage"] = "D4_CAPACITY_TIME"

    with pytest.raises(ValueError, match="D4_OPTIMIZATION"):
        build_phase06_d4b_protocol_lock(
            config,
            phase05_config,
            execution_commit="d" * 40,
            phase06_spec_path=_PHASE06_SPEC,
            d4b_addendum_path=_D4B_ADDENDUM,
            phase05_protocol_path=_PHASE05_PROTOCOL,
            parent_evidence=evidence,
        )


def test_d4b_parent_loader_revalidates_d2b_completion_and_recomputation(
    tmp_path, monkeypatch
):
    core, d2b = _write_d2b_parent(tmp_path)
    expected = _d2b_adjudication()
    completed = _patch_parent_recomputation(monkeypatch, core, d2b, expected)
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)

    evidence = pipeline_module.load_phase06_d4b_parent_evidence(
        core,
        d2b,
        config=config,
        phase05_config=phase05_config,
    )

    assert ("d2b", len(plan_d2b_cells(config))) in completed
    assert evidence == _parent_evidence()


def test_d4b_parent_loader_rejects_recomputed_adjudication_mismatch(
    tmp_path, monkeypatch
):
    core, d2b = _write_d2b_parent(tmp_path)
    recomputed = dict(_d2b_adjudication())
    recomputed["next_required_stage"] = "STOP"
    _patch_parent_recomputation(monkeypatch, core, d2b, recomputed)
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)

    with pytest.raises(ValueError, match="recomputed"):
        pipeline_module.load_phase06_d4b_parent_evidence(
            core,
            d2b,
            config=config,
            phase05_config=phase05_config,
        )
