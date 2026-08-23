"""Reproducibility metadata for persisted Phase-0 benchmark runs."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy
import pandas
import scipy
import sklearn
import torch
from matplotlib import pyplot as plt

from afmc_fm.execution.device import _is_wsl
from afmc_fm.execution.persistence import (
    CELL_SCHEMA_VERSION,
    PROTOCOL_ANCHOR,
    canonical_config_hash,
)
from afmc_fm.experiments.runner import ExperimentConfig
from afmc_fm.simulator.config import SimulatorConfig

plt.switch_backend("Agg")

OUTPUT_SCHEMA_VERSION = 1
SCIENTIFIC_SORT_KEY = (
    "benchmark",
    "world",
    "cohort_seed",
    "subset_seed",
    "model_seed",
    "n_train",
    "model",
    "ablation",
    "split",
    "site_or_shift",
    "metric",
)
METRIC_OUTPUT_COLUMNS = (
    "model",
    "ablation",
    "n_train",
    "n_fit",
    "n_validation",
    "seed",
    "cohort_seed",
    "subset_seed",
    "model_seed",
    "split",
    "site_or_shift",
    "metric",
    "value",
    "trainable_parameters",
    "backend",
    "world",
    "benchmark",
)
DERIVED_ARTIFACT_NAMES = (
    "metrics.csv",
    "ablation_metrics.csv",
    "gate_summary.csv",
    "learning_curves.png",
)
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def collect_runtime_metadata(resolved_device: str) -> dict[str, object]:
    """Collect portable software, host, and accelerator provenance."""
    cuda_available = torch.cuda.is_available()
    using_cuda = resolved_device == "cuda"
    capability = torch.cuda.get_device_capability(0) if using_cuda else None
    return {
        "python_version": platform.python_version(),
        "library_versions": {
            "matplotlib": matplotlib.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "os": platform.platform(),
        "wsl": _is_wsl(),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if using_cuda else None,
        "gpu_compute_capability": (
            f"{capability[0]}.{capability[1]}" if capability is not None else None
        ),
        "cpu_logical_count": os.cpu_count(),
    }


def execution_commit_sha() -> str:
    """Return the exact source checkout commit used for execution."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip().lower()
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("git rev-parse HEAD did not return a full commit SHA")
    return commit


def baseline_backend_ids(experiment: ExperimentConfig) -> dict[str, str]:
    """Describe the declared estimator backend for each configured model."""
    identifiers = {
        "engineered_linear": "torch_ridge",
        "gradient_boosting": "sklearn_hist_gradient_boosting",
        "gru_from_scratch": "torch",
        "representation_linear": "torch_ridge",
        "representation_mlp": "torch_lbfgs",
        "flow_jump": "torch",
        "flow_jump_observation": "torch",
    }
    return {model: identifiers[model] for model in experiment.models}


def _failure_payload(failure: object) -> dict[str, object]:
    if is_dataclass(failure) and not isinstance(failure, type):
        payload = asdict(failure)
    elif isinstance(failure, Mapping):
        payload = dict(failure)
    else:
        raise TypeError("failures must contain dataclass instances or mappings")
    return {str(key): value for key, value in payload.items()}


def build_run_manifest(
    *,
    simulator: SimulatorConfig,
    experiment: ExperimentConfig,
    resolved_device: str,
    worker_count: int,
    backend_ids: Mapping[str, str],
    started_at: datetime,
    ended_at: datetime,
    wall_time_seconds: float,
    expected_by_shard: Mapping[str, frozenset[str]],
    completed_by_shard: Mapping[str, frozenset[str]],
    failures: Sequence[object] = (),
    execution_commit_sha: str | None = None,
    runtime_metadata: Mapping[str, object] | None = None,
    template_seed: int = 0,
) -> dict[str, object]:
    """Build the complete JSON-compatible scientific and execution manifest."""
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    if resolved_device not in {"cpu", "cuda"}:
        raise ValueError("resolved_device must be cpu or cuda")
    if ended_at < started_at:
        raise ValueError("ended_at must not precede started_at")
    if wall_time_seconds < 0:
        raise ValueError("wall_time_seconds must be non-negative")

    failure_payloads = [_failure_payload(failure) for failure in failures]
    failed_shard_ids = {
        str(failure["shard_id"])
        for failure in failure_payloads
        if "shard_id" in failure
    }
    completed_cell_count = sum(
        len(expected.intersection(completed_by_shard.get(shard_id, frozenset())))
        for shard_id, expected in expected_by_shard.items()
    )
    completed_shard_count = sum(
        completed_by_shard.get(shard_id, frozenset()) == expected
        for shard_id, expected in expected_by_shard.items()
    )
    failed_cell_count = sum(
        len(expected - completed_by_shard.get(shard_id, frozenset()))
        for shard_id, expected in expected_by_shard.items()
        if shard_id in failed_shard_ids
    )
    runtime = dict(
        collect_runtime_metadata(resolved_device)
        if runtime_metadata is None
        else runtime_metadata
    )
    required_runtime_fields = {
        "python_version",
        "library_versions",
        "os",
        "wsl",
        "cuda_available",
        "cuda_runtime",
        "gpu_name",
        "gpu_compute_capability",
        "cpu_logical_count",
    }
    missing_runtime = required_runtime_fields - runtime.keys()
    if missing_runtime:
        raise ValueError(
            "runtime metadata is missing: " + ", ".join(sorted(missing_runtime))
        )
    commit = execution_commit_sha or globals()["execution_commit_sha"]()
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("execution_commit_sha must be a full lowercase Git SHA")

    return {
        "synthetic": True,
        "template_seed": template_seed,
        "protocol_anchor": PROTOCOL_ANCHOR,
        "execution_commit_sha": commit,
        "simulator_config_hash": canonical_config_hash(simulator),
        "experiment_config_hash": canonical_config_hash(experiment),
        "simulator_config": asdict(simulator),
        "experiment_config": asdict(experiment),
        "python_version": runtime["python_version"],
        "library_versions": runtime["library_versions"],
        "os": runtime["os"],
        "wsl": runtime["wsl"],
        "resolved_device": resolved_device,
        "cuda_available": runtime["cuda_available"],
        "cuda_runtime": runtime["cuda_runtime"],
        "gpu_name": runtime["gpu_name"],
        "gpu_compute_capability": runtime["gpu_compute_capability"],
        "cpu_logical_count": runtime["cpu_logical_count"],
        "worker_count": worker_count,
        "backend_ids": dict(sorted(backend_ids.items())),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "generated_at": ended_at.isoformat(),
        "wall_time_seconds": wall_time_seconds,
        "expected_shard_count": len(expected_by_shard),
        "completed_shard_count": completed_shard_count,
        "failed_shard_count": len(failed_shard_ids),
        "expected_cell_count": sum(map(len, expected_by_shard.values())),
        "completed_cell_count": completed_cell_count,
        "failed_cell_count": failed_cell_count,
        "failures": failure_payloads,
        "cell_schema_version": CELL_SCHEMA_VERSION,
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
    }


def _phase0_gate_summary(metrics: pandas.DataFrame) -> pandas.DataFrame:
    selected = metrics[
        (metrics["benchmark"] == "low_n")
        & (metrics["site_or_shift"] == "all")
        & (metrics["ablation"] == "none")
        & (metrics["metric"] == "mae")
    ]
    summary = (
        selected.groupby(["world", "n_train", "model"], as_index=False)["value"]
        .mean()
        .pivot(index=["world", "n_train"], columns="model", values="value")
        .reset_index()
    )
    summary.columns.name = None
    for model in ("flow_jump", "representation_linear", "gru_from_scratch"):
        if model not in summary:
            summary[model] = numpy.nan
    comparisons_available = summary[
        ["flow_jump", "representation_linear", "gru_from_scratch"]
    ].notna().all(axis=1)
    summary["flow_jump_beats_both_primary_comparators"] = pandas.array(
        [
            bool(flow < representation and flow < gru) if available else pandas.NA
            for flow, representation, gru, available in zip(
                summary["flow_jump"],
                summary["representation_linear"],
                summary["gru_from_scratch"],
                comparisons_available,
                strict=True,
            )
        ],
        dtype="boolean",
    )
    return summary


def _plot_learning_curves(metrics: pandas.DataFrame, output: Path) -> None:
    selected = metrics[metrics["metric"] == "mae"]
    figure, axis = plt.subplots(figsize=(7, 4))
    for model_name, group in selected.groupby("model"):
        summary = group.groupby("n_train")["value"].agg(["mean", "std"]).reset_index()
        axis.plot(summary["n_train"], summary["mean"], marker="o", label=model_name)
        deviation = summary["std"].fillna(0.0)
        axis.fill_between(
            summary["n_train"],
            summary["mean"] - deviation,
            summary["mean"] + deviation,
            alpha=0.15,
        )
    axis.set_xlabel("Training patients")
    axis.set_ylabel("MAE")
    axis.set_title("Synthetic low-N forecasting")
    if not selected.empty:
        axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output, dpi=150, format="png")
    plt.close(figure)


def aggregate_persisted_metrics(
    store: Any,
    expected_by_shard: Mapping[str, frozenset[str]],
) -> pandas.DataFrame:
    """Load exactly planned valid cells and apply the canonical scientific order."""
    store.validate_resume()
    completed_by_shard = store.load_completed_cell_ids_by_shard()
    expected_cell_ids = (
        frozenset().union(*expected_by_shard.values())
        if expected_by_shard
        else frozenset()
    )
    completed_cell_ids = (
        frozenset().union(*completed_by_shard.values())
        if completed_by_shard
        else frozenset()
    )
    unexpected = completed_cell_ids - expected_cell_ids
    if unexpected:
        raise ValueError(
            "unplanned persisted cells are not allowed: "
            + ", ".join(sorted(unexpected))
        )
    rows = list(store.iter_metric_rows(expected_cell_ids))
    metrics = pandas.DataFrame(rows, columns=METRIC_OUTPUT_COLUMNS)
    if not metrics.empty:
        if metrics.duplicated(list(SCIENTIFIC_SORT_KEY)).any():
            raise ValueError("duplicate scientific metric keys in persisted cells")
        metrics = metrics.sort_values(
            list(SCIENTIFIC_SORT_KEY),
            kind="mergesort",
            ignore_index=True,
        )
    return metrics


def _write_csv(frame: pandas.DataFrame, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)


def write_derived_artifacts(metrics: pandas.DataFrame, output: Path) -> None:
    """Write the unchanged public CSV and learning-curve definitions atomically."""
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(metrics, output / "metrics.csv")
    _write_csv(
        metrics[metrics["ablation"] != "none"],
        output / "ablation_metrics.csv",
    )
    _write_csv(_phase0_gate_summary(metrics), output / "gate_summary.csv")
    plot_output = output / "learning_curves.png"
    temporary_plot = plot_output.with_name(f".{plot_output.name}.tmp")
    _plot_learning_curves(metrics, temporary_plot)
    temporary_plot.replace(plot_output)


def write_run_manifest(manifest: Mapping[str, object], output: Path) -> None:
    """Atomically persist a validated JSON-compatible run manifest."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    final = output / "run_manifest.json"
    temporary = output / ".run_manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(final)


def reaggregate_benchmark_outputs(
    simulator: SimulatorConfig,
    experiment: ExperimentConfig,
    output: Path,
    *,
    template_seed: int = 0,
) -> pandas.DataFrame:
    """Regenerate public artifacts solely from persisted, validated cell JSON."""
    from afmc_fm.execution.jobs import plan_shards
    from afmc_fm.execution.persistence import RunStore
    from afmc_fm.execution.scheduler import _expected_cell_ids, _run_identity

    output = Path(output).resolve()
    shards = plan_shards(experiment, supplied_seed=template_seed)
    expected_by_shard = {
        shard.shard_id: _expected_cell_ids(shard, simulator, experiment)
        for shard in shards
    }
    store = RunStore(output, _run_identity(simulator, experiment))
    metrics = aggregate_persisted_metrics(store, expected_by_shard)
    write_derived_artifacts(metrics, output)
    return metrics


__all__ = [
    "DERIVED_ARTIFACT_NAMES",
    "METRIC_OUTPUT_COLUMNS",
    "OUTPUT_SCHEMA_VERSION",
    "SCIENTIFIC_SORT_KEY",
    "aggregate_persisted_metrics",
    "baseline_backend_ids",
    "build_run_manifest",
    "collect_runtime_metadata",
    "execution_commit_sha",
    "reaggregate_benchmark_outputs",
    "write_derived_artifacts",
    "write_run_manifest",
]
