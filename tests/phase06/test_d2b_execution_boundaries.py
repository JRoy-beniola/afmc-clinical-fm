from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import afmc_fm.phase06.cli as phase06_cli
import afmc_fm.phase06.pipeline as pipeline_module
import afmc_fm.phase06.protocol as protocol_module
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.phase06.config import load_phase06_config
from afmc_fm.phase06.planning import plan_d1_cells, plan_d2a_cells
from afmc_fm.phase06.protocol import build_phase06_protocol_lock

_PHASE06_CONFIG = Path("configs/experiments/phase06.yaml")
_PHASE06_SPEC = Path(
    "docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md"
)
_D2B_ADDENDUM = Path(
    "docs/superpowers/specs/2026-08-26-phase0-6-d2b-execution-addendum.md"
)
_PHASE05_PROTOCOL = Path(
    "docs/results/phase05/raw/official_output/protocol_lock.json"
)
_RUNNER = Path("tools/execution/phase06/run_stage.sh")
_PARENT_EXECUTION_SHA = "1718402df1d6ef344168677e6d26ea664708e1bc"
_PARENT_PROTOCOL_SHA256 = (
    "c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51"
)
_PARENT_D3_SHA256 = "6b9fffed7503fae6beeac2314238ae3d10ffdebfd27ea71ad952ab1f87916460"


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _write_marker_only_parent(
    tmp_path: Path,
    *,
    d3_payload: dict[str, object] | None = None,
) -> Path:
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)
    lock = build_phase06_protocol_lock(
        config,
        phase05_config,
        execution_commit=_PARENT_EXECUTION_SHA,
        phase06_spec_path=_PHASE06_SPEC,
        phase05_protocol_path=_PHASE05_PROTOCOL,
    )
    lock_bytes = _canonical_json_bytes(lock)
    assert hashlib.sha256(lock_bytes).hexdigest() == _PARENT_PROTOCOL_SHA256

    root = tmp_path / "parent"
    root.mkdir(parents=True)
    (root / "protocol_lock.json").write_bytes(lock_bytes)
    identity = {
        "protocol_lock_sha256": _PARENT_PROTOCOL_SHA256,
        "phase06_config_sha256": canonical_config_hash(config),
        "execution_commit": _PARENT_EXECUTION_SHA,
    }
    for stage, cells in {
        "d1": plan_d1_cells(config, phase05_config),
        "d2a": plan_d2a_cells(config),
    }.items():
        marker = root / "stages" / stage / "COMPLETE"
        marker.parent.mkdir(parents=True)
        marker.write_bytes(
            _canonical_json_bytes(
                {
                    "schema_version": 1,
                    "identity": identity,
                    "stage": stage,
                    "cell_ids": sorted(cell.cell_id for cell in cells),
                }
            )
        )

    if d3_payload is None:
        d3_payload = {
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
    analysis = root / "analysis"
    analysis.mkdir()
    (analysis / "phase06_d3_adjudication.json").write_bytes(
        _canonical_json_bytes(d3_payload)
    )
    return root


def _load_parent(root: Path):
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)
    return pipeline_module.load_phase06_d2b_parent_evidence(
        root,
        config=config,
        phase05_config=phase05_config,
    )


def _parent_evidence(d3_hash: str) -> dict[str, object]:
    config = load_phase06_config(_PHASE06_CONFIG)
    return {
        "parent_execution_sha": _PARENT_EXECUTION_SHA,
        "parent_protocol_lock_sha256": _PARENT_PROTOCOL_SHA256,
        "parent_d3_sha256": d3_hash,
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


def test_parent_d3_hash_is_frozen_to_executed_parent():
    assert getattr(protocol_module, "_PARENT_PHASE06_D3_SHA256", None) == _PARENT_D3_SHA256


def test_d2b_parent_loader_rejects_marker_only_parent_without_cell_bundles(tmp_path):
    root = _write_marker_only_parent(tmp_path)

    with pytest.raises(ValueError, match="d1 stage is not complete"):
        _load_parent(root)


def test_d2b_parent_loader_verifies_d3_input_hashes_against_parent_evidence(
    tmp_path, monkeypatch
):
    d3_payload = {
        "phenomenon_reproduction": "reproduced",
        "interaction_ambiguity": True,
        "next_required_stage": "D2B",
        "triggered_escalations": ["D2B", "D4_OPTIMIZATION"],
        "input_artifact_hashes": {"d1_table": "0" * 64},
    }
    root = _write_marker_only_parent(tmp_path, d3_payload=d3_payload)
    d3_path = root / "analysis" / "phase06_d3_adjudication.json"
    monkeypatch.setattr(
        pipeline_module,
        "_PARENT_PHASE06_D3_SHA256",
        hashlib.sha256(d3_path.read_bytes()).hexdigest(),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "open_bound_store",
        lambda _output: object(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "require_completed_stage",
        lambda _store, cells: frozenset(cell.cell_id for cell in cells),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_expected_parent_d3_input_hashes",
        lambda *_args, **_kwargs: {"d1_table": "1" * 64},
        raising=False,
    )

    with pytest.raises(ValueError, match="input_artifact_hashes"):
        _load_parent(root)


def test_d2b_protocol_rejects_non_frozen_parent_d3_hash():
    config = load_phase06_config(_PHASE06_CONFIG)
    phase05_config = load_phase05_config(config.phase05_config)

    with pytest.raises(ValueError, match="parent D3.*frozen"):
        protocol_module.build_phase06_d2b_protocol_lock(
            config,
            phase05_config,
            execution_commit="b" * 40,
            phase06_spec_path=_PHASE06_SPEC,
            d2b_addendum_path=_D2B_ADDENDUM,
            phase05_protocol_path=_PHASE05_PROTOCOL,
            parent_evidence=_parent_evidence("d" * 64),
        )


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_d2b_public_cli_rejects_non_cuda_device(tmp_path, device):
    parser = phase06_cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "d2b",
                "--config",
                str(_PHASE06_CONFIG),
                "--parent-output",
                str(tmp_path / "parent"),
                "--output",
                str(tmp_path / "child"),
                "--device",
                device,
            ]
        )


@pytest.mark.parametrize(
    ("parent_parts", "child_parts"),
    [
        (("parent",), ("parent", "child")),
        (("child", "parent"), ("child",)),
    ],
)
def test_d2b_python_boundary_rejects_nested_output_roots(
    tmp_path, parent_parts, child_parts
):
    parent = tmp_path.joinpath(*parent_parts)
    child = tmp_path.joinpath(*child_parts)

    with pytest.raises(ValueError, match="parent.*child"):
        phase06_cli._require_distinct_outputs(parent, child)


def test_d2b_shell_boundary_rejects_nested_output_roots():
    text = _RUNNER.read_text(encoding="utf-8")

    assert '[[ "$CHILD_CANONICAL" == "$PARENT_OUTPUT/"* ]]' in text
    assert '[[ "$PARENT_OUTPUT" == "$CHILD_CANONICAL/"* ]]' in text
