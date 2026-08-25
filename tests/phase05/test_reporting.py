import hashlib
import json
from pathlib import Path

import pandas as pd

from afmc_fm.phase05.reporting import write_phase05_report_artifacts


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _report_fixture(tmp_path: Path) -> Path:
    output = tmp_path / "phase05"
    output.mkdir()
    lock = {
        "schema_version": 1,
        "spec_commit": "1" * 40,
        "phase0_execution_sha": "2" * 40,
        "phase0_metrics_sha256": "3" * 64,
        "phase05_config_sha256": "4" * 64,
        "development_bundles": [[401 + i, 501 + i, 601 + i] for i in range(5)],
        "confirmatory_bundles": [[701 + i, 801 + i, 901 + i] for i in range(10)],
        "locked_min_relative_effect": 0.02,
        "locked_uncertainty_mae_tolerance": 0.02,
        "time_scale_days": 30.0,
    }
    (output / "protocol_lock.json").write_text(
        json.dumps(lock, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    protocol_hash = hashlib.sha256((output / "protocol_lock.json").read_bytes()).hexdigest()
    frozen = {
        "flow_mode": "time_scaled",
        "jump_mode": "residual",
        "uncertainty_mode": "deterministic",
        "strict_history": True,
        "state_dim": 24,
        "time_scale_days": 30.0,
        "trainable_parameters": 6900,
        "matched_gru_hidden_size": 39,
        "matched_gru_parameters": 6888,
        "matched_mlp_hidden_size": 216,
        "matched_mlp_parameters": 6915,
        "protocol_lock_sha256": protocol_hash,
        "development_artifact_hashes": {},
    }
    (output / "frozen_candidate.json").write_text(
        json.dumps(frozen, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    mechanism_rows = []
    variants = {
        "flow": ("none", "gated", "time_scaled"),
        "jump": ("none", "gru", "residual"),
        "uncertainty": ("joint", "decoupled", "deterministic"),
    }
    for stage, candidates in variants.items():
        for candidate_index, candidate in enumerate(candidates):
            for n_train in (5, 10, 20, 40):
                mechanism_rows.append(
                    {
                        "stage": stage,
                        "world": "smooth" if stage == "flow" else "jumps",
                        "n_train": n_train,
                        "variant": candidate,
                        "metric": "mae",
                        "value": 1.2 - 0.05 * candidate_index - 0.01 * n_train,
                    }
                )
    _write_csv(output / "development" / "mechanism_metrics.csv", mechanism_rows)
    _write_csv(
        output / "development" / "flow_gate.csv",
        [
            {"candidate": "gated", "passed": True, "selected": False},
            {"candidate": "time_scaled", "passed": True, "selected": True},
        ],
    )
    _write_csv(
        output / "development" / "jump_gate.csv",
        [
            {"candidate": "gru", "passed": True, "selected": False},
            {"candidate": "residual", "passed": True, "selected": True},
        ],
    )
    _write_csv(
        output / "development" / "uncertainty_gate.csv",
        [
            {"candidate": "joint", "selected": False},
            {"candidate": "decoupled", "selected": False},
            {"candidate": "deterministic", "selected": True},
        ],
    )
    _write_csv(
        output / "development" / "representation_timing_audit.csv",
        [{"world": "smooth", "strict_timing_required": True}],
    )

    worlds = ("smooth", "jumps", "informative_observation")
    models = (
        "phase05_candidate",
        "matched_gru",
        "matched_representation_mlp",
    )
    learning_rows = []
    for world_index, world in enumerate(worlds):
        for model_index, model in enumerate(models):
            for n_train in (5, 10, 20, 40, 80, 100):
                learning_rows.append(
                    {
                        "world": world,
                        "n_train": n_train,
                        "model": model,
                        "variant": model,
                        "mean_mae": 1.4
                        - 0.01 * n_train
                        + 0.05 * model_index
                        + 0.02 * world_index,
                        "sd_mae": 0.03 + 0.002 * model_index,
                        "n_bundles": 10,
                    }
                )
    _write_csv(output / "confirmation" / "learning_curves.csv", learning_rows)
    _write_csv(
        output / "confirmation" / "metrics.csv",
        [
            {
                "world": "smooth",
                "cohort_seed": 701,
                "subset_seed": 801,
                "model_seed": 901,
                "n_train": 5,
                "model": "phase05_candidate",
                "variant": "candidate",
                "split": "test",
                "site_or_shift": "all",
                "metric": "mae",
                "value": 1.0,
            }
        ],
    )
    effect_rows = []
    for world_index, world in enumerate(worlds):
        for comparator_index, comparator in enumerate(
            ("matched_gru", "matched_representation_mlp")
        ):
            for bundle_index in range(10):
                effect_rows.append(
                    {
                        "world": world,
                        "cohort_seed": 701 + bundle_index,
                        "subset_seed": 801 + bundle_index,
                        "model_seed": 901 + bundle_index,
                        "comparator": comparator,
                        "effect": 0.04
                        + 0.002 * bundle_index
                        - 0.003 * comparator_index
                        + 0.001 * world_index,
                    }
                )
    _write_csv(output / "confirmation" / "paired_naulc_effects.csv", effect_rows)
    _write_csv(
        output / "confirmation" / "primary_gate_summary.csv",
        [{"world": world, "headline_passed": True} for world in worlds],
    )
    _write_csv(
        output / "confirmation" / "bootstrap_intervals.csv",
        [
            {
                "world": "smooth",
                "comparator": "matched_gru",
                "lower": 0.01,
                "upper": 0.08,
            }
        ],
    )
    _write_csv(
        output / "confirmation" / "sign_tests.csv",
        [
            {
                "world": "smooth",
                "comparator": "matched_gru",
                "p_value": 0.05,
                "holm_adjusted_p": 0.30,
            }
        ],
    )
    _write_csv(
        output / "confirmation" / "capacity_audit.csv",
        [
            {"control": "matched_gru", "actual_parameters": 6888},
            {"control": "matched_representation_mlp", "actual_parameters": 6915},
        ],
    )
    confirmation = output / "confirmation"
    frozen_hash = hashlib.sha256((output / "frozen_candidate.json").read_bytes()).hexdigest()
    identity = {
        "schema_version": 1,
        "spec_sha256": "5" * 64,
        "config_sha256": "4" * 64,
        "protocol_lock_sha256": protocol_hash,
    }
    confirmation_started_at = "2026-08-25T12:34:56+00:00"
    (confirmation / "STARTED").write_text(
        json.dumps(
            {
                "identity": identity,
                "frozen_candidate_sha256": frozen_hash,
                "started_at": confirmation_started_at,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    runtime_metadata = {
        "python_version": "3.11.16",
        "platform": "Linux-fixture",
        "library_versions": {
            "numpy": "2.4.6",
            "pandas": "3.0.5",
            "torch": "2.13.0+cu128",
        },
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_devices": [
            {"index": 0, "name": "Fixture GPU", "capability": "9.0"}
        ],
    }
    execution_provenance = {
        "schema_version": 1,
        "identity": identity,
        "implementation_sha": "6" * 40,
        "simulator_config_sha256": "7" * 64,
        "invocations": [
            {
                "invocation_number": 1,
                "stage": "flow",
                "started_at": "2026-08-25T11:00:00+00:00",
                "ended_at": "2026-08-25T11:00:02.500000+00:00",
                "wall_time_seconds": 2.5,
                "requested_device": "cpu",
                "resolved_device": "cpu",
                "workers": 1,
                "resume": False,
                "fail_fast": False,
                "expected_cell_count": 5,
                "completed_before": 0,
                "completed_after": 5,
                "shard_count": 5,
                "failures": [],
                "runtime_metadata": runtime_metadata,
            },
            {
                "invocation_number": 2,
                "stage": "confirmation",
                "started_at": "2026-08-25T13:00:00+00:00",
                "ended_at": "2026-08-25T13:00:03+00:00",
                "wall_time_seconds": 3.0,
                "requested_device": "cuda",
                "resolved_device": "cuda",
                "workers": 2,
                "resume": False,
                "fail_fast": False,
                "expected_cell_count": 1260,
                "completed_before": 0,
                "completed_after": 600,
                "shard_count": 30,
                "failures": [
                    {"type": "RuntimeError", "message": "transient fit failure"}
                ],
                "runtime_metadata": runtime_metadata,
            },
            {
                "invocation_number": 3,
                "stage": "confirmation",
                "started_at": "2026-08-25T13:10:00+00:00",
                "ended_at": "2026-08-25T13:10:07+00:00",
                "wall_time_seconds": 7.0,
                "requested_device": "cuda",
                "resolved_device": "cuda",
                "workers": 2,
                "resume": True,
                "fail_fast": False,
                "expected_cell_count": 1260,
                "completed_before": 600,
                "completed_after": 1260,
                "shard_count": 30,
                "failures": [],
                "runtime_metadata": runtime_metadata,
            },
            {
                "invocation_number": 4,
                "stage": "robustness",
                "started_at": "2026-08-25T14:00:00+00:00",
                "ended_at": "2026-08-25T14:00:04+00:00",
                "wall_time_seconds": 4.0,
                "requested_device": "cuda",
                "resolved_device": "cuda",
                "workers": 2,
                "resume": False,
                "fail_fast": False,
                "expected_cell_count": 180,
                "completed_before": 0,
                "completed_after": 180,
                "shard_count": 20,
                "failures": [],
                "runtime_metadata": runtime_metadata,
            },
        ],
    }
    (output / "execution_provenance.json").write_text(
        json.dumps(execution_provenance, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    site_rows = []
    for model_index, model in enumerate(models):
        for bundle_index in range(10):
            site_rows.append(
                {
                    "world": "site_shift",
                    "cohort_seed": 701 + bundle_index,
                    "subset_seed": 801 + bundle_index,
                    "model_seed": 901 + bundle_index,
                    "n_train": 20,
                    "model": model,
                    "mae_site_0": 1.0,
                    "mae_site_1": 1.05 + 0.03 * model_index,
                    "mae_absolute_degradation": 0.05 + 0.03 * model_index,
                    "mae_relative_degradation": 0.05 + 0.03 * model_index,
                }
            )
    _write_csv(output / "robustness" / "site_shift_metrics.csv", site_rows)
    _write_csv(
        output / "robustness" / "misspecification_metrics.csv",
        [
            {
                "cohort_seed": 701 + index,
                "subset_seed": 801 + index,
                "model_seed": 901 + index,
                "candidate_naulc": 1.03,
                "matched_gru_naulc": 1.0,
                "matched_representation_mlp_naulc": 1.08,
                "best_control_model": "matched_gru",
                "best_control_naulc": 1.0,
                "relative_excess": 0.03,
            }
            for index in range(10)
        ],
    )

    for stage in ("confirmation", "robustness"):
        marker = output / "stages" / stage / "COMPLETE"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("fixture\n", encoding="utf-8")
    return output


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase05_report_writes_required_artifacts_deterministically_without_mutating_inputs(
    tmp_path: Path,
):
    output = _report_fixture(tmp_path)
    source_paths = [
        path
        for path in output.rglob("*")
        if path.is_file() and "figures" not in path.parts
    ]
    before = {path.relative_to(output): _sha(path) for path in source_paths}

    first = write_phase05_report_artifacts(output)

    required = {
        "protocol_manifest.json",
        "execution_provenance.json",
        "development/mechanism_metrics.csv",
        "development/flow_gate.csv",
        "development/jump_gate.csv",
        "development/uncertainty_gate.csv",
        "development/representation_timing_audit.csv",
        "frozen_candidate.json",
        "confirmation/metrics.csv",
        "confirmation/learning_curves.csv",
        "confirmation/paired_naulc_effects.csv",
        "confirmation/primary_gate_summary.csv",
        "confirmation/bootstrap_intervals.csv",
        "confirmation/sign_tests.csv",
        "confirmation/capacity_audit.csv",
        "robustness/site_shift_metrics.csv",
        "robustness/misspecification_metrics.csv",
        "figures/mechanism_controls.png",
        "figures/low_n_learning_curves.png",
        "figures/paired_naulc_effects.png",
        "figures/site_shift_degradation.png",
        "run_manifest.json",
    }
    assert required.issubset(
        {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    )
    assert set(first) >= {
        "protocol_manifest",
        "mechanism_controls",
        "low_n_learning_curves",
        "paired_naulc_effects",
        "site_shift_degradation",
        "run_manifest",
    }
    assert not (output / "figures" / "calibration.png").exists()

    after = {path.relative_to(output): _sha(path) for path in source_paths}
    assert after == before

    figure_paths = sorted((output / "figures").glob("*.png"))
    first_figures = {path.name: _sha(path) for path in figure_paths}
    first_manifest_hash = _sha(output / "run_manifest.json")
    write_phase05_report_artifacts(output)
    assert {path.name: _sha(path) for path in figure_paths} == first_figures
    assert _sha(output / "run_manifest.json") == first_manifest_hash

    protocol_manifest = json.loads((output / "protocol_manifest.json").read_text())
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    execution_provenance = json.loads(
        (output / "execution_provenance.json").read_text(encoding="utf-8")
    )
    assert protocol_manifest["spec_commit"] == "1" * 40
    assert run_manifest["spec_sha256"] == "5" * 64
    assert run_manifest["implementation_sha"] == "6" * 40
    assert run_manifest["simulator_config_sha256"] == "7" * 64
    assert run_manifest["phase0_execution_sha"] == "2" * 40
    assert run_manifest["phase05_config_sha256"] == "4" * 64
    assert run_manifest["confirmation_started_at"] == "2026-08-25T12:34:56+00:00"
    assert run_manifest["execution_provenance_sha256"] == _sha(
        output / "execution_provenance.json"
    )
    assert run_manifest["execution_provenance"] == execution_provenance
    assert run_manifest["execution_provenance"]["invocations"][1][
        "resolved_device"
    ] == "cuda"
    assert run_manifest["execution_provenance"]["invocations"][1][
        "runtime_metadata"
    ]["cuda_devices"][0]["name"] == "Fixture GPU"
    assert run_manifest["development_bundles"] == [
        [401 + i, 501 + i, 601 + i] for i in range(5)
    ]
    assert run_manifest["confirmatory_bundles"] == [
        [701 + i, 801 + i, 901 + i] for i in range(10)
    ]
    assert run_manifest["selected_mechanisms"] == {
        "flow_mode": "time_scaled",
        "jump_mode": "residual",
        "uncertainty_mode": "deterministic",
    }
    assert run_manifest["matched_control_parameters"] == {
        "matched_gru": 6888,
        "matched_representation_mlp": 6915,
    }
    assert run_manifest["stage_timings"] == {
        "confirmation": {"invocation_count": 2, "wall_time_seconds": 10.0},
        "flow": {"invocation_count": 1, "wall_time_seconds": 2.5},
        "robustness": {"invocation_count": 1, "wall_time_seconds": 4.0},
    }
    assert run_manifest["failures"] == [
        {
            "invocation_number": 2,
            "stage": "confirmation",
            "type": "RuntimeError",
            "message": "transient fit failure",
        }
    ]
    assert "report_runtime_metadata" not in run_manifest
    assert "provenance_limitations" not in run_manifest
    assert "execution_provenance.json" in run_manifest["source_artifact_sha256"]
    assert "grand_score" not in run_manifest
