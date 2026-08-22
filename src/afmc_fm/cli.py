import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from afmc_fm.config import load_yaml
from afmc_fm.experiments.runner import (
    ExperimentConfig,
    run_low_n_benchmark,
    run_observation_shift_benchmark,
)
from afmc_fm.schema.events import events_to_frame
from afmc_fm.simulator.cohort import (
    SimulatedCohort,
    simulate_cohort,
    simulate_world,
)
from afmc_fm.simulator.config import SimulatorConfig


def _simulator_from_yaml(path: str | Path) -> tuple[SimulatorConfig, int]:
    raw = load_yaml(path)
    seed = int(raw.pop("seed", 0))
    return SimulatorConfig(**raw), seed


def _experiment_from_yaml(path: str | Path) -> ExperimentConfig:
    raw = load_yaml(path)
    for key in (
        "train_sizes",
        "cohort_seeds",
        "subset_seeds",
        "model_seeds",
        "worlds",
        "models",
        "ablations",
        "test_sites",
    ):
        if key in raw:
            raw[key] = tuple(raw[key])
    return ExperimentConfig(**raw)


def _write_simulation(cohort: SimulatedCohort, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    events = [event for patient in cohort.patients for event in patient.timeline.events]
    events_to_frame(events).to_csv(output / "events.csv", index=False)
    latent_arrays = {}
    for patient in cohort.patients:
        latent_arrays[f"{patient.patient_id}_times"] = patient.latent.times
        latent_arrays[f"{patient.patient_id}_states"] = patient.latent.states
    np.savez_compressed(output / "latent_truth.npz", **latent_arrays)
    manifest = {
        "synthetic": True,
        "seed": cohort.seed,
        "simulator_config": asdict(cohort.config),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (output / "simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _plot_learning_curves(metrics, output: Path) -> None:
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
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)



def _phase0_gate_summary(metrics: pd.DataFrame) -> pd.DataFrame:
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
            summary[model] = np.nan
    comparisons_available = summary[
        ["flow_jump", "representation_linear", "gru_from_scratch"]
    ].notna().all(axis=1)
    summary["flow_jump_beats_both_primary_comparators"] = pd.array(
        [
            bool(flow < representation and flow < gru) if available else pd.NA
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


def _world_cohort(
    world: str,
    config: SimulatorConfig,
    seed: int,
) -> SimulatedCohort:
    if world == "custom":
        return simulate_cohort(config, seed)
    return simulate_world(world, config, seed)


def _simulate(args: argparse.Namespace) -> int:
    config, seed = _simulator_from_yaml(args.config)
    _write_simulation(simulate_cohort(config, seed), Path(args.output))
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    sim_config, seed = _simulator_from_yaml(args.sim_config)
    experiment = _experiment_from_yaml(args.exp_config)
    template_seed = experiment.cohort_seeds[0] if experiment.cohort_seeds else seed
    frames: list[pd.DataFrame] = []
    for world in experiment.worlds:
        cohort = _world_cohort(world, sim_config, template_seed)
        low_n = run_low_n_benchmark(cohort, experiment)
        low_n["world"] = world
        low_n["benchmark"] = "low_n"
        frames.append(low_n)
        if world == "site_shift" and any(
            model.startswith("flow_jump") for model in experiment.models
        ):
            shift = run_observation_shift_benchmark(cohort, experiment)
            shift["world"] = world
            shift["benchmark"] = "observation_shift"
            frames.append(shift)
    metrics = pd.concat(frames, ignore_index=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    metrics[metrics["ablation"] != "none"].to_csv(
        output / "ablation_metrics.csv", index=False
    )
    _phase0_gate_summary(metrics).to_csv(output / "gate_summary.csv", index=False)
    _plot_learning_curves(metrics, output / "learning_curves.png")
    manifest = {
        "synthetic": True,
        "template_seed": seed,
        "simulator_config": asdict(sim_config),
        "experiment_config": asdict(experiment),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afmc-phase0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--output", required=True)
    simulate.set_defaults(handler=_simulate)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--sim-config", required=True)
    benchmark.add_argument("--exp-config", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.set_defaults(handler=_benchmark)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
