from __future__ import annotations

import hashlib
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pandas as pd

from afmc_fm.phase05.config import Phase05Config, load_phase05_config
from afmc_fm.phase05.execution import (
    Phase05ExecutionOptions,
    Phase05Job,
    Phase05ShardSpec,
)
from afmc_fm.phase05.protocol import FrozenCandidate, freeze_candidate
from afmc_fm.phase05.robustness import build_robustness_jobs, run_robustness_job
from afmc_fm.phase05.runner import PreparedPhase05Cohort, prepare_phase05_cohort
from afmc_fm.phase05.store import Phase05Store
from afmc_fm.simulator.cohort import simulate_world
from afmc_fm.simulator.config import SimulatorConfig


def _require_finalized_confirmation(output: Path) -> None:
    required = (
        output / "confirmation" / "STARTED",
        output / "stages" / "confirmation" / "COMPLETE",
        output / "confirmation" / "primary_gate_summary.csv",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(
            "finalized confirmation is required before Phase-0.5 robustness"
        )


def _prepare_robustness_shard(
    shard: Phase05ShardSpec,
    *,
    simulator_config: SimulatorConfig,
) -> PreparedPhase05Cohort:
    cohort = simulate_world(
        shard.world,
        simulator_config,
        seed=shard.seed_bundle.cohort_seed,
    )
    return prepare_phase05_cohort(cohort)


def _run_robustness_job(
    job: Phase05Job,
    prepared: PreparedPhase05Cohort,
    device,
    *,
    config: Phase05Config,
    frozen: FrozenCandidate,
) -> pd.DataFrame:
    return run_robustness_job(
        job,
        prepared,
        config=config,
        frozen=frozen,
        device=device,
    )


def _root_cli_dependencies():
    from afmc_fm import cli

    return cli.run_phase05_jobs, cli._simulator_from_yaml


def run_phase05_robustness_cli(
    args,
    *,
    locked_store: Callable[[Path, Phase05Config], tuple[dict[str, object], Phase05Store]],
    run_jobs: Callable[..., pd.DataFrame] | None = None,
    simulator_from_yaml: Callable[[str | Path], tuple[SimulatorConfig, int]] | None = None,
) -> int:
    if run_jobs is None or simulator_from_yaml is None:
        default_run_jobs, default_simulator_from_yaml = _root_cli_dependencies()
        if run_jobs is None:
            run_jobs = default_run_jobs
        if simulator_from_yaml is None:
            simulator_from_yaml = default_simulator_from_yaml

    config = load_phase05_config(args.exp_config)
    simulator_config, _ = simulator_from_yaml(args.sim_config)
    output = Path(args.output)
    _, store = locked_store(output, config)

    frozen_path = output / "frozen_candidate.json"
    if not frozen_path.is_file():
        raise RuntimeError("frozen_candidate.json is required before Phase-0.5 robustness")
    frozen = freeze_candidate(output, config)
    _require_finalized_confirmation(output)

    frozen_candidate_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    jobs = build_robustness_jobs(
        config,
        frozen,
        frozen_candidate_hash=frozen_candidate_hash,
    )
    options = Phase05ExecutionOptions(
        device=args.device,
        workers=args.workers,
        resume=args.resume,
        fail_fast=args.fail_fast,
    )
    run_jobs(
        jobs,
        store=store,
        config=config,
        options=options,
        prepare_shard=partial(
            _prepare_robustness_shard,
            simulator_config=simulator_config,
        ),
        run_job=partial(
            _run_robustness_job,
            config=config,
            frozen=frozen,
        ),
    )

    raise RuntimeError("Phase-0.5 robustness finalization is not yet wired")


__all__ = ["run_phase05_robustness_cli"]
