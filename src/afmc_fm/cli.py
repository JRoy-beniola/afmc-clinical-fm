import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from afmc_fm.config import load_yaml
from afmc_fm.experiments.runner import ExperimentConfig, run_low_n_benchmark
from afmc_fm.schema.events import events_to_frame
from afmc_fm.simulator.cohort import SimulatedCohort, simulate_cohort
from afmc_fm.simulator.config import SimulatorConfig


def _simulator_from_yaml(path: str | Path) -> tuple[SimulatorConfig, int]:
    raw = load_yaml(path)
    seed = int(raw.pop("seed", 0))
    return SimulatorConfig(**raw), seed


def _experiment_from_yaml(path: str | Path) -> ExperimentConfig:
    raw = load_yaml(path)
    raw.pop("train_site", None)
    raw.pop("test_sites", None)
    if "train_sizes" in raw:
        raw["train_sizes"] = tuple(raw["train_sizes"])
    if "seeds" in raw:
        raw["seeds"] = tuple(raw["seeds"])
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


def _simulate(args: argparse.Namespace) -> int:
    config, seed = _simulator_from_yaml(args.config)
    _write_simulation(simulate_cohort(config, seed), Path(args.output))
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    sim_config, seed = _simulator_from_yaml(args.sim_config)
    cohort = simulate_cohort(sim_config, seed)
    experiment = _experiment_from_yaml(args.exp_config)
    max_train = int(sim_config.cohort_size * 0.7)
    experiment = replace(
        experiment,
        train_sizes=tuple(size for size in experiment.train_sizes if size <= max_train),
    )
    if not experiment.train_sizes:
        raise ValueError("cohort is too small for every configured training size")
    metrics = run_low_n_benchmark(cohort, experiment)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    _plot_learning_curves(metrics, output / "learning_curves.png")
    manifest = {
        "synthetic": True,
        "seed": seed,
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
