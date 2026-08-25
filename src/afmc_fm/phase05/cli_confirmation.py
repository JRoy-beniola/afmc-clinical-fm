from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import pandas as pd

from afmc_fm.phase05.config import Phase05Config, load_phase05_config
from afmc_fm.phase05.confirmation import build_confirmation_jobs, run_confirmation_job
from afmc_fm.phase05.execution import (
    Phase05ExecutionOptions,
    Phase05Job,
    Phase05ShardSpec,
)
from afmc_fm.phase05.protocol import FrozenCandidate, freeze_candidate
from afmc_fm.phase05.runner import PreparedPhase05Cohort, prepare_phase05_cohort
from afmc_fm.phase05.store import Phase05Store
from afmc_fm.simulator.cohort import simulate_world
from afmc_fm.simulator.config import SimulatorConfig


def _prepare_confirmation_shard(
    shard: Phase05ShardSpec,
    *,
    simulator_config: SimulatorConfig,
) -> PreparedPhase05Cohort:
    cohort = simulate_world(
        shard.world,
        simulator_config,
        seed=shard.seed_bundle.cohort_seed,
    )
    return prepare_phase05_cohort(cohort, include_historical=True)


def _run_confirmation_job(
    job: Phase05Job,
    prepared: PreparedPhase05Cohort,
    device,
    *,
    config: Phase05Config,
    frozen: FrozenCandidate,
) -> pd.DataFrame:
    return run_confirmation_job(
        job,
        prepared,
        config=config,
        frozen=frozen,
        device=device,
    )


def run_phase05_confirmation_cli(
    args,
    *,
    run_jobs: Callable[..., pd.DataFrame],
    locked_store: Callable[[Path, Phase05Config], tuple[dict[str, object], Phase05Store]],
    simulator_from_yaml: Callable[[str | Path], tuple[SimulatorConfig, int]],
) -> int:
    config = load_phase05_config(args.exp_config)
    simulator_config, _ = simulator_from_yaml(args.sim_config)
    output = Path(args.output)
    _, store = locked_store(output, config)

    frozen_path = output / "frozen_candidate.json"
    if not frozen_path.is_file():
        raise RuntimeError("frozen_candidate.json is required before Phase-0.5 confirmation")
    frozen = freeze_candidate(output, config)

    frozen_candidate_hash = store.mark_confirmation_started()
    jobs = build_confirmation_jobs(
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
            _prepare_confirmation_shard,
            simulator_config=simulator_config,
        ),
        run_job=partial(
            _run_confirmation_job,
            config=config,
            frozen=frozen,
        ),
    )
    return 0
