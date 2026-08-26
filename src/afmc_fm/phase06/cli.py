from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from afmc_fm.config import load_yaml
from afmc_fm.execution.manifest import execution_commit_sha
from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import Phase05Config, load_phase05_config
from afmc_fm.phase06.config import Phase06Config, load_phase06_config
from afmc_fm.phase06.execution import run_phase06_stage
from afmc_fm.phase06.planning import plan_d1_cells, plan_d2a_cells
from afmc_fm.phase06.protocol import build_phase06_protocol_lock
from afmc_fm.phase06.store import Phase06Store
from afmc_fm.simulator.config import SimulatorConfig

_PHASE05_PROTOCOL_PATH = Path(
    "docs/results/phase05/raw/official_output/protocol_lock.json"
)


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_simulator_config(path: str | Path) -> SimulatorConfig:
    raw = dict(load_yaml(path))
    raw.pop("seed", None)
    return SimulatorConfig(**raw)


def _bind_store(
    config: Phase06Config,
    phase05_config: Phase05Config,
    output: str | Path,
) -> Phase06Store:
    execution_commit = execution_commit_sha()
    lock = build_phase06_protocol_lock(
        config,
        phase05_config,
        execution_commit=execution_commit,
        phase06_spec_path=config.phase06_spec,
        phase05_protocol_path=_PHASE05_PROTOCOL_PATH,
    )
    protocol_hash = hashlib.sha256(_canonical_json_bytes(lock)).hexdigest()
    store = Phase06Store(
        Path(output),
        protocol_hash=protocol_hash,
        config_hash=canonical_config_hash(config),
        execution_commit=execution_commit,
    )
    store.write_protocol_lock(lock)
    return store


def _load_bound_inputs(args: argparse.Namespace) -> tuple[
    Phase06Config,
    Phase05Config,
    Phase06Store,
]:
    config = load_phase06_config(args.config)
    phase05_config = load_phase05_config(config.phase05_config)
    store = _bind_store(config, phase05_config, args.output)
    return config, phase05_config, store


def _run_d1(args: argparse.Namespace) -> int:
    config, phase05_config, store = _load_bound_inputs(args)
    cells = plan_d1_cells(config, phase05_config)
    simulator_config = _load_simulator_config(config.simulator_config)
    run_phase06_stage(
        cells,
        store,
        phase05_config,
        simulator_config,
        device=args.device,
        resume=args.resume,
    )
    return 0


def _run_d2a(args: argparse.Namespace) -> int:
    config, phase05_config, store = _load_bound_inputs(args)
    cells = plan_d2a_cells(config)
    simulator_config = _load_simulator_config(config.simulator_config)
    run_phase06_stage(
        cells,
        store,
        phase05_config,
        simulator_config,
        device=args.device,
        resume=args.resume,
    )
    return 0


def _adjudicate(_args: argparse.Namespace) -> int:
    raise RuntimeError(
        "Phase 0.6 adjudication is unavailable until the locked D1/D2 analysis path is implemented"
    )


def _add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afmc-phase06")
    subparsers = parser.add_subparsers(dest="command", required=True)

    d1 = subparsers.add_parser("d1")
    _add_stage_arguments(d1)
    d1.set_defaults(handler=_run_d1)

    d2a = subparsers.add_parser("d2a")
    _add_stage_arguments(d2a)
    d2a.set_defaults(handler=_run_d2a)

    adjudicate = subparsers.add_parser("adjudicate")
    adjudicate.add_argument("--output", required=True)
    adjudicate.set_defaults(handler=_adjudicate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
