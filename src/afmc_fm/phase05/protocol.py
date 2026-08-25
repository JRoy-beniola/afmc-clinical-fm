import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.sequences import JUMP_ELIGIBLE_EVENT_CODES

_PROTOCOL_SCHEMA_VERSION = 1
_NOISE_FLOOR_METHOD = "phase0-relative-se-mad-v1"
_TARGET_WORLDS = ("smooth", "jumps", "informative_observation")
_PRIMARY_TRAIN_SIZES = (5, 10, 20, 40)
_MODELS = ("flow_jump", "representation_linear", "gru_from_scratch")
_REQUIRED_COLUMNS = frozenset(
    {
        "benchmark",
        "site_or_shift",
        "ablation",
        "metric",
        "world",
        "n_train",
        "model",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "value",
    }
)
_SHA_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_NEAR_ZERO_MEDIAN = 1e-12


def _validate_columns(metrics: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(metrics.columns)
    if missing:
        raise ValueError(f"missing required Phase-0 metric columns: {sorted(missing)}")


def _required_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(metrics)
    frame = metrics.loc[
        (metrics["benchmark"] == "low_n")
        & (metrics["site_or_shift"] == "all")
        & (metrics["ablation"] == "none")
        & (metrics["metric"] == "mae")
        & metrics["world"].isin(_TARGET_WORLDS)
        & metrics["n_train"].isin(_PRIMARY_TRAIN_SIZES)
        & metrics["model"].isin(_MODELS)
    ].copy()
    return frame


def estimate_phase0_relative_noise_floor(metrics: pd.DataFrame) -> float:
    frame = _required_frame(metrics)
    expected_groups = {
        (world, n_train, model)
        for world in _TARGET_WORLDS
        for n_train in _PRIMARY_TRAIN_SIZES
        for model in _MODELS
    }
    observed_groups = set(
        frame[["world", "n_train", "model"]].itertuples(index=False, name=None)
    )
    missing_groups = expected_groups - observed_groups
    if missing_groups:
        raise ValueError("missing required Phase-0 groups")
    unexpected_groups = observed_groups - expected_groups
    if unexpected_groups:
        raise ValueError("unexpected required-frame Phase-0 groups")

    relative_standard_errors: list[float] = []
    grouped = frame.groupby(["world", "n_train", "model"], sort=False)
    for key in sorted(expected_groups):
        group = grouped.get_group(key)
        if len(group) != 5:
            raise ValueError("each required Phase-0 group must contain exactly five matched seed rows")
        bundles = group[["cohort_seed", "subset_seed", "model_seed"]].drop_duplicates()
        if len(bundles) != 5:
            raise ValueError("each required Phase-0 group must contain five distinct matched seed bundles")
        values = group["value"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Phase-0 noise-floor values must be non-finite free")
        median = float(np.median(values))
        if abs(median) <= _NEAR_ZERO_MEDIAN:
            raise ValueError("Phase-0 noise-floor group has near-zero median")
        mad = float(np.median(np.abs(values - median)))
        robust_sd = 1.4826 * mad
        relative_se = robust_sd / (abs(median) * math.sqrt(5.0))
        if not math.isfinite(relative_se):
            raise ValueError("Phase-0 relative standard error is non-finite")
        relative_standard_errors.append(relative_se)

    noise_floor = float(np.percentile(np.asarray(relative_standard_errors), 75.0))
    if not math.isfinite(noise_floor):
        raise ValueError("Phase-0 noise floor is non-finite")
    return noise_floor


def _validate_sha(value: str) -> None:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError("provenance values must be a 40- or 64-character hexadecimal SHA")


def build_protocol_lock(
    config: Phase05Config,
    phase0_metrics: pd.DataFrame,
    *,
    phase0_metrics_sha256: str,
    spec_commit: str,
    phase0_execution_sha: str,
) -> dict[str, object]:
    for value in (phase0_metrics_sha256, spec_commit, phase0_execution_sha):
        _validate_sha(value)

    noise_floor = estimate_phase0_relative_noise_floor(phase0_metrics)
    locked_threshold = max(config.provisional_relative_effect, noise_floor)
    return {
        "schema_version": _PROTOCOL_SCHEMA_VERSION,
        "phase0_noise_floor_method": _NOISE_FLOOR_METHOD,
        "phase0_noise_floor": noise_floor,
        "locked_min_relative_effect": locked_threshold,
        "locked_uncertainty_mae_tolerance": locked_threshold,
        "spec_commit": spec_commit,
        "phase0_execution_sha": phase0_execution_sha,
        "phase0_metrics_sha256": phase0_metrics_sha256,
        "phase05_config_sha256": canonical_config_hash(config),
        "development_bundles": [list(bundle.as_tuple()) for bundle in config.development_bundles],
        "confirmatory_bundles": [
            list(bundle.as_tuple()) for bundle in config.confirmatory_bundles
        ],
        "jump_eligible_event_codes": sorted(JUMP_ELIGIBLE_EVENT_CODES),
        "time_scale_days": config.time_scale_days,
    }


def _gate_passed(path: Path, gate_name: str) -> bool:
    if not path.is_file():
        raise RuntimeError(f"missing required {gate_name} gate artifact")
    frame = pd.read_csv(path)
    if frame.empty or "passed" not in frame.columns:
        raise RuntimeError(f"invalid {gate_name} gate artifact")
    normalized = {str(value).strip().lower() for value in frame["passed"].dropna()}
    if not normalized or not normalized.issubset({"true", "false", "1", "0"}):
        raise RuntimeError(f"invalid {gate_name} gate artifact")
    return bool(normalized & {"true", "1"})


def _write_development_failure(output: Path, failed_gates: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "development_failed",
        "failed_gates": failed_gates,
    }
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    (output / "development_failure.json").write_bytes(data)


def freeze_candidate(output: str | Path, config: Phase05Config):
    del config
    output_path = Path(output)
    development = output_path / "development"
    if not _gate_passed(development / "flow_gate.csv", "flow"):
        _write_development_failure(output_path, ["flow"])
        raise RuntimeError("flow gate failed; confirmation is not permitted")
    if not _gate_passed(development / "jump_gate.csv", "jump"):
        _write_development_failure(output_path, ["jump"])
        raise RuntimeError("jump gate failed; confirmation is not permitted")
    raise NotImplementedError("successful candidate freezing is not implemented yet")


__all__ = [
    "build_protocol_lock",
    "estimate_phase0_relative_noise_floor",
    "freeze_candidate",
]
