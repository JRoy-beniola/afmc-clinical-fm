from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from afmc_fm.phase05.config import Phase05Config, load_phase05_config
from afmc_fm.phase05.protocol import freeze_candidate
from afmc_fm.phase05.store import Phase05Store


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


def run_phase05_robustness_cli(
    args,
    *,
    locked_store: Callable[[Path, Phase05Config], tuple[dict[str, object], Phase05Store]],
) -> int:
    config = load_phase05_config(args.exp_config)
    output = Path(args.output)
    locked_store(output, config)

    frozen_path = output / "frozen_candidate.json"
    if not frozen_path.is_file():
        raise RuntimeError("frozen_candidate.json is required before Phase-0.5 robustness")
    freeze_candidate(output, config)
    _require_finalized_confirmation(output)

    raise RuntimeError("Phase-0.5 robustness execution is not yet wired")


__all__ = ["run_phase05_robustness_cli"]
