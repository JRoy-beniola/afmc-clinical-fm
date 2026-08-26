from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_SOURCE_ARTIFACTS = (
    "protocol_lock.json",
    "frozen_candidate.json",
    "execution_provenance.json",
    "development/mechanism_metrics.csv",
    "development/flow_gate.csv",
    "development/jump_gate.csv",
    "development/uncertainty_gate.csv",
    "development/representation_timing_audit.csv",
    "confirmation/STARTED",
    "confirmation/metrics.csv",
    "confirmation/learning_curves.csv",
    "confirmation/paired_naulc_effects.csv",
    "confirmation/primary_gate_summary.csv",
    "confirmation/bootstrap_intervals.csv",
    "confirmation/sign_tests.csv",
    "confirmation/capacity_audit.csv",
    "robustness/site_shift_metrics.csv",
    "robustness/misspecification_metrics.csv",
    "stages/confirmation/COMPLETE",
    "stages/robustness/COMPLETE",
)
_WORLD_ORDER = ("smooth", "jumps", "informative_observation")
_PRIMARY_MODELS = (
    "phase05_candidate",
    "matched_gru",
    "matched_representation_mlp",
)
_FIGURE_METADATA = {"Software": "afmc-clinical-fm"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid reporting JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"reporting JSON artifact must contain an object: {path}")
    return payload


def _require_hex(value: object, length: int, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise RuntimeError(f"{name} is missing or invalid")
    return value.lower()


def _require_sources(output: Path) -> dict[str, Path]:
    paths = {name: output / name for name in _SOURCE_ARTIFACTS}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Phase-0.5 reporting requires complete persisted artifacts: "
            + ", ".join(missing)
        )
    return paths


def _atomic_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return path


def _save_figure(path: Path, figure) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(
        temporary,
        format="png",
        dpi=120,
        bbox_inches="tight",
        metadata=_FIGURE_METADATA,
    )
    plt.close(figure)
    temporary.replace(path)
    return path


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{name} is missing columns: {sorted(missing)}")


def _mechanism_controls_figure(output: Path) -> Path:
    frame = pd.read_csv(output / "development" / "mechanism_metrics.csv")
    _require_columns(
        frame,
        {"stage", "n_train", "variant", "metric", "value"},
        "mechanism_metrics.csv",
    )
    frame = frame.loc[frame["metric"] == "mae"].copy()
    if frame.empty:
        raise RuntimeError("mechanism_metrics.csv contains no MAE rows")

    stages = tuple(
        stage for stage in ("flow", "jump", "uncertainty") if stage in set(frame["stage"])
    )
    if not stages:
        raise RuntimeError("mechanism_metrics.csv contains no supported development stages")
    figure, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 4), squeeze=False)
    for axis, stage in zip(axes[0], stages, strict=True):
        selected = frame.loc[frame["stage"] == stage]
        summary = (
            selected.groupby(["variant", "n_train"], as_index=False)["value"]
            .mean()
            .sort_values(["variant", "n_train"], kind="mergesort")
        )
        for variant in sorted(summary["variant"].astype(str).unique()):
            rows = summary.loc[summary["variant"].astype(str) == variant]
            axis.plot(rows["n_train"], rows["value"], marker="o", label=variant)
        axis.set_title(stage.replace("_", " ").title())
        axis.set_xlabel("Labelled training patients")
        axis.set_ylabel("Mean MAE")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    figure.tight_layout()
    return _save_figure(output / "figures" / "mechanism_controls.png", figure)


def _learning_curves_figure(output: Path) -> Path:
    frame = pd.read_csv(output / "confirmation" / "learning_curves.csv")
    _require_columns(
        frame,
        {"world", "n_train", "model", "mean_mae", "sd_mae"},
        "learning_curves.csv",
    )
    worlds = tuple(world for world in _WORLD_ORDER if world in set(frame["world"]))
    if not worlds:
        raise RuntimeError("learning_curves.csv contains no locked target worlds")

    figure, axes = plt.subplots(1, len(worlds), figsize=(5 * len(worlds), 4), squeeze=False)
    for axis, world in zip(axes[0], worlds, strict=True):
        selected = frame.loc[frame["world"] == world].copy()
        for model in _PRIMARY_MODELS:
            rows = selected.loc[selected["model"] == model].sort_values(
                "n_train", kind="mergesort"
            )
            if rows.empty:
                continue
            x = rows["n_train"].to_numpy(dtype=float)
            mean = rows["mean_mae"].to_numpy(dtype=float)
            sd = rows["sd_mae"].fillna(0.0).to_numpy(dtype=float)
            axis.plot(x, mean, marker="o", label=model)
            axis.fill_between(x, mean - sd, mean + sd, alpha=0.15)
        axis.set_xscale("log", base=2)
        axis.set_title(world.replace("_", " ").title())
        axis.set_xlabel("Labelled training patients")
        axis.set_ylabel("MAE, mean ± seed SD")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="x-small")
    figure.tight_layout()
    return _save_figure(output / "figures" / "low_n_learning_curves.png", figure)


def _paired_effects_figure(output: Path) -> Path:
    frame = pd.read_csv(output / "confirmation" / "paired_naulc_effects.csv")
    _require_columns(
        frame,
        {"world", "cohort_seed", "subset_seed", "model_seed", "comparator", "effect"},
        "paired_naulc_effects.csv",
    )
    worlds = tuple(world for world in _WORLD_ORDER if world in set(frame["world"]))
    comparators = ("matched_gru", "matched_representation_mlp")
    figure, axes = plt.subplots(1, len(worlds), figsize=(5 * len(worlds), 4), squeeze=False)
    for axis, world in zip(axes[0], worlds, strict=True):
        selected = frame.loc[frame["world"] == world]
        for comparator_index, comparator in enumerate(comparators):
            rows = selected.loc[selected["comparator"] == comparator].sort_values(
                ["cohort_seed", "subset_seed", "model_seed"], kind="mergesort"
            )
            x = np.arange(len(rows), dtype=float) + comparator_index * 0.16
            axis.scatter(x, rows["effect"].to_numpy(dtype=float), label=comparator)
        axis.axhline(0.0, linewidth=1.0)
        axis.set_title(world.replace("_", " ").title())
        axis.set_xlabel("Paired confirmatory bundle")
        axis.set_ylabel("nAULC effect (control − candidate)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="x-small")
    figure.tight_layout()
    return _save_figure(output / "figures" / "paired_naulc_effects.png", figure)


def _site_shift_figure(output: Path) -> Path:
    frame = pd.read_csv(output / "robustness" / "site_shift_metrics.csv")
    _require_columns(
        frame,
        {"n_train", "model", "mae_absolute_degradation"},
        "site_shift_metrics.csv",
    )
    summary = (
        frame.groupby(["n_train", "model"], as_index=False)["mae_absolute_degradation"]
        .agg(mean_degradation="mean", sd_degradation="std")
        .sort_values(["model", "n_train"], kind="mergesort")
    )
    figure, axis = plt.subplots(figsize=(6, 4))
    for model in _PRIMARY_MODELS:
        rows = summary.loc[summary["model"] == model]
        if rows.empty:
            continue
        x = rows["n_train"].to_numpy(dtype=float)
        mean = rows["mean_degradation"].to_numpy(dtype=float)
        sd = rows["sd_degradation"].fillna(0.0).to_numpy(dtype=float)
        axis.plot(x, mean, marker="o", label=model)
        axis.fill_between(x, mean - sd, mean + sd, alpha=0.15)
    if summary["n_train"].nunique() > 1:
        axis.set_xscale("log", base=2)
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xlabel("Labelled training patients")
    axis.set_ylabel("Site 1 − site 0 MAE degradation")
    axis.set_title("Complete-truth site-shift degradation")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="x-small")
    figure.tight_layout()
    return _save_figure(output / "figures" / "site_shift_degradation.png", figure)


def _calibration_figure(output: Path) -> Path:
    metrics = pd.read_csv(output / "confirmation" / "metrics.csv")
    _require_columns(metrics, {"model", "metric", "value"}, "confirmation/metrics.csv")
    candidate = metrics.loc[metrics["model"] == "phase05_candidate"]
    supported = candidate.loc[candidate["metric"].isin(("coverage_90", "nll"))].copy()
    if supported.empty:
        raise RuntimeError(
            "probabilistic Phase-0.5 reporting requires candidate calibration metrics"
        )
    summary = supported.groupby("metric", as_index=False)["value"].mean()
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.bar(summary["metric"], summary["value"])
    axis.set_title("Confirmatory calibration diagnostics")
    axis.set_ylabel("Mean metric value")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _save_figure(output / "figures" / "calibration.png", figure)


def _protocol_manifest(
    output: Path,
    lock: dict[str, object],
    frozen: dict[str, object],
) -> dict[str, object]:
    return {
        **lock,
        "protocol_lock_sha256": _sha256(output / "protocol_lock.json"),
        "frozen_candidate_sha256": _sha256(output / "frozen_candidate.json"),
        "selected_mechanisms": {
            "flow_mode": frozen.get("flow_mode"),
            "jump_mode": frozen.get("jump_mode"),
            "uncertainty_mode": frozen.get("uncertainty_mode"),
        },
    }


def _stage_counts(output: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for stage in (
        "flow",
        "jump",
        "uncertainty",
        "timing_audit",
        "confirmation",
        "robustness",
    ):
        cell_dir = output / "stages" / stage / "cells"
        cell_paths = tuple(sorted(cell_dir.glob("*.json"))) if cell_dir.is_dir() else ()
        shard_ids = set()
        for path in cell_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            seed = payload.get("seed_bundle")
            world = payload.get("world")
            if isinstance(seed, dict) and world is not None:
                shard_ids.add(
                    (
                        str(world),
                        seed.get("cohort_seed"),
                        seed.get("subset_seed"),
                        seed.get("model_seed"),
                    )
                )
        counts[stage] = {"cell_count": len(cell_paths), "shard_count": len(shard_ids)}
    return counts


def _execution_provenance(
    output: Path,
    lock: dict[str, object],
) -> dict[str, object]:
    provenance = _json_object(output / "execution_provenance.json")
    if provenance.get("schema_version") != 1:
        raise RuntimeError("execution provenance schema is incompatible")
    identity = provenance.get("identity")
    if not isinstance(identity, dict):
        raise TypeError("execution provenance identity must be an object")
    spec_sha256 = _require_hex(identity.get("spec_sha256"), 64, "spec_sha256")
    config_sha256 = _require_hex(
        identity.get("config_sha256"), 64, "execution provenance config hash"
    )
    protocol_sha256 = _require_hex(
        identity.get("protocol_lock_sha256"),
        64,
        "execution provenance protocol-lock hash",
    )
    if config_sha256 != str(lock.get("phase05_config_sha256", "")).lower():
        raise RuntimeError("execution provenance Phase-0.5 config hash mismatch")
    if protocol_sha256 != _sha256(output / "protocol_lock.json"):
        raise RuntimeError("execution provenance protocol-lock hash mismatch")
    _require_hex(provenance.get("implementation_sha"), 40, "implementation_sha")
    _require_hex(
        provenance.get("simulator_config_sha256"),
        64,
        "simulator_config_sha256",
    )
    invocations = provenance.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise RuntimeError("execution provenance contains no execution invocations")
    for invocation in invocations:
        if not isinstance(invocation, dict):
            raise TypeError("execution provenance invocation must be an object")
    identity["spec_sha256"] = spec_sha256
    identity["config_sha256"] = config_sha256
    identity["protocol_lock_sha256"] = protocol_sha256
    return provenance


def _confirmation_start(
    output: Path,
    provenance: dict[str, object],
) -> dict[str, object]:
    marker = _json_object(output / "confirmation" / "STARTED")
    if marker.get("identity") != provenance.get("identity"):
        raise RuntimeError("confirmation start protocol identity mismatch")
    expected_frozen_hash = _sha256(output / "frozen_candidate.json")
    if marker.get("frozen_candidate_sha256") != expected_frozen_hash:
        raise RuntimeError("confirmation start frozen-candidate hash mismatch")
    started_at = marker.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise RuntimeError("confirmation start timestamp is missing")
    return marker


def _stage_timings(invocations: list[object]) -> dict[str, dict[str, int | float]]:
    summaries: dict[str, dict[str, int | float]] = {}
    for invocation in invocations:
        if not isinstance(invocation, dict):
            raise TypeError("execution provenance invocation must be an object")
        stage = invocation.get("stage")
        wall_time = invocation.get("wall_time_seconds")
        if not isinstance(stage, str) or not stage:
            raise RuntimeError("execution invocation stage is invalid")
        if type(wall_time) not in {int, float} or wall_time < 0:
            raise RuntimeError("execution invocation wall time is invalid")
        summary = summaries.setdefault(
            stage,
            {"invocation_count": 0, "wall_time_seconds": 0.0},
        )
        summary["invocation_count"] = int(summary["invocation_count"]) + 1
        summary["wall_time_seconds"] = float(summary["wall_time_seconds"]) + float(
            wall_time
        )
    return {stage: summaries[stage] for stage in sorted(summaries)}


def _execution_failures(invocations: list[object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for invocation in invocations:
        if not isinstance(invocation, dict):
            raise TypeError("execution provenance invocation must be an object")
        invocation_number = invocation.get("invocation_number")
        stage = invocation.get("stage")
        failures = invocation.get("failures")
        if type(invocation_number) is not int or invocation_number <= 0:
            raise RuntimeError("execution invocation number is invalid")
        if not isinstance(stage, str) or not stage:
            raise RuntimeError("execution invocation stage is invalid")
        if not isinstance(failures, list):
            raise TypeError("execution invocation failures must be a list")
        for failure in failures:
            if not isinstance(failure, dict):
                raise TypeError("execution failure record must be an object")
            failure_type = failure.get("type")
            message = failure.get("message")
            if not isinstance(failure_type, str) or not isinstance(message, str):
                raise TypeError("execution failure type and message must be strings")
            rows.append(
                {
                    "invocation_number": invocation_number,
                    "stage": stage,
                    "type": failure_type,
                    "message": message,
                }
            )
    return rows


def _run_manifest(
    output: Path,
    lock: dict[str, object],
    frozen: dict[str, object],
    source_paths: dict[str, Path],
) -> dict[str, object]:
    provenance = _execution_provenance(output, lock)
    marker = _confirmation_start(output, provenance)
    identity = provenance["identity"]
    if not isinstance(identity, dict):
        raise TypeError("execution provenance identity must be an object")
    invocations = provenance["invocations"]
    if not isinstance(invocations, list):
        raise TypeError("execution provenance invocations must be a list")
    return {
        "schema_version": 1,
        "synthetic": True,
        "spec_sha256": identity["spec_sha256"],
        "implementation_sha": provenance["implementation_sha"],
        "spec_commit": lock.get("spec_commit"),
        "phase0_execution_sha": lock.get("phase0_execution_sha"),
        "phase0_metrics_sha256": lock.get("phase0_metrics_sha256"),
        "simulator_config_sha256": provenance["simulator_config_sha256"],
        "phase05_config_sha256": lock.get("phase05_config_sha256"),
        "protocol_lock_sha256": _sha256(output / "protocol_lock.json"),
        "frozen_candidate_sha256": _sha256(output / "frozen_candidate.json"),
        "development_bundles": lock.get("development_bundles"),
        "confirmatory_bundles": lock.get("confirmatory_bundles"),
        "selected_mechanisms": {
            "flow_mode": frozen.get("flow_mode"),
            "jump_mode": frozen.get("jump_mode"),
            "uncertainty_mode": frozen.get("uncertainty_mode"),
        },
        "matched_control_parameters": {
            "matched_gru": frozen.get("matched_gru_parameters"),
            "matched_representation_mlp": frozen.get("matched_mlp_parameters"),
        },
        "confirmation_start_sha256": _sha256(output / "confirmation" / "STARTED"),
        "confirmation_started_at": marker["started_at"],
        "execution_provenance_sha256": _sha256(output / "execution_provenance.json"),
        "execution_provenance": provenance,
        "stage_timings": _stage_timings(invocations),
        "failures": _execution_failures(invocations),
        "source_artifact_sha256": {
            name: _sha256(path) for name, path in sorted(source_paths.items())
        },
        "persisted_stage_counts": _stage_counts(output),
    }


def write_phase05_report_artifacts(output: str | Path) -> dict[str, Path]:
    root = Path(output)
    sources = _require_sources(root)
    lock = _json_object(root / "protocol_lock.json")
    frozen = _json_object(root / "frozen_candidate.json")

    paths = {
        "protocol_manifest": _atomic_json(
            root / "protocol_manifest.json",
            _protocol_manifest(root, lock, frozen),
        ),
        "mechanism_controls": _mechanism_controls_figure(root),
        "low_n_learning_curves": _learning_curves_figure(root),
        "paired_naulc_effects": _paired_effects_figure(root),
        "site_shift_degradation": _site_shift_figure(root),
    }
    if frozen.get("uncertainty_mode") != "deterministic":
        paths["calibration"] = _calibration_figure(root)
    else:
        (root / "figures" / "calibration.png").unlink(missing_ok=True)

    paths["run_manifest"] = _atomic_json(
        root / "run_manifest.json",
        _run_manifest(root, lock, frozen, sources),
    )
    return paths


__all__ = ["write_phase05_report_artifacts"]
