from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from afmc_fm.execution.persistence import canonical_config_hash
from afmc_fm.phase05.baselines import build_capacity_audit
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
_DEVELOPMENT_ARTIFACTS = (
    "flow_gate.csv",
    "jump_gate.csv",
    "uncertainty_gate.csv",
    "representation_timing_audit.csv",
)
_STATE_DIM = 24
_VALUE_DIM = 3
_EVENT_DIM = 3
_REPRESENTATION_INPUT_DIM = 19
_ASSIMILATION_SEMANTICS = "phase0_grucell_unchanged"


@dataclass(frozen=True, slots=True)
class FrozenCandidate:
    flow_mode: str
    jump_mode: str
    uncertainty_mode: str
    strict_history: bool
    state_dim: int
    time_scale_days: float
    jump_eligible_event_codes: tuple[str, ...]
    assimilation_semantics: str
    trainable_parameters: int
    matched_gru_hidden_size: int
    matched_gru_parameters: int
    matched_mlp_hidden_size: int
    matched_mlp_parameters: int
    protocol_lock_sha256: str
    development_artifact_hashes: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "flow_mode": self.flow_mode,
            "jump_mode": self.jump_mode,
            "uncertainty_mode": self.uncertainty_mode,
            "strict_history": self.strict_history,
            "state_dim": self.state_dim,
            "time_scale_days": self.time_scale_days,
            "jump_eligible_event_codes": list(self.jump_eligible_event_codes),
            "assimilation_semantics": self.assimilation_semantics,
            "trainable_parameters": self.trainable_parameters,
            "matched_gru_hidden_size": self.matched_gru_hidden_size,
            "matched_gru_parameters": self.matched_gru_parameters,
            "matched_mlp_hidden_size": self.matched_mlp_hidden_size,
            "matched_mlp_parameters": self.matched_mlp_parameters,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "development_artifact_hashes": dict(self.development_artifact_hashes),
        }


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


def _normalized_boolean_values(series: pd.Series) -> set[str]:
    return {str(value).strip().lower() for value in series.dropna()}


def _gate_passed(path: Path, gate_name: str) -> bool:
    if not path.is_file():
        raise RuntimeError(f"missing required {gate_name} gate artifact")
    frame = pd.read_csv(path)
    if frame.empty or "passed" not in frame.columns:
        raise RuntimeError(f"invalid {gate_name} gate artifact")
    normalized = _normalized_boolean_values(frame["passed"])
    if not normalized or not normalized.issubset({"true", "false", "1", "0"}):
        raise RuntimeError(f"invalid {gate_name} gate artifact")
    return bool(normalized & {"true", "1"})


def _selected_gate_row(path: Path, selection_column: str, gate_name: str) -> pd.Series:
    if not path.is_file():
        raise RuntimeError(f"missing required {gate_name} gate artifact")
    frame = pd.read_csv(path)
    required = {"candidate", selection_column, "trainable_parameters"}
    if frame.empty or not required.issubset(frame.columns):
        raise RuntimeError(f"invalid {gate_name} gate artifact")
    selected = frame.loc[
        frame[selection_column].astype(str).str.strip().str.lower().isin({"true", "1"})
    ]
    if len(selected) != 1:
        raise RuntimeError(f"{gate_name} gate must select exactly one candidate")
    return selected.iloc[0]


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing required freeze input: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _development_hashes(development: Path) -> dict[str, str]:
    return {name: _sha256(development / name) for name in _DEVELOPMENT_ARTIFACTS}


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


def _write_frozen_candidate(output: Path, candidate: FrozenCandidate) -> None:
    data = (
        json.dumps(candidate.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    (output / "frozen_candidate.json").write_bytes(data)


def _load_frozen_candidate(path: Path) -> FrozenCandidate:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return FrozenCandidate(
            flow_mode=str(payload["flow_mode"]),
            jump_mode=str(payload["jump_mode"]),
            uncertainty_mode=str(payload["uncertainty_mode"]),
            strict_history=bool(payload["strict_history"]),
            state_dim=int(payload["state_dim"]),
            time_scale_days=float(payload["time_scale_days"]),
            jump_eligible_event_codes=tuple(payload["jump_eligible_event_codes"]),
            assimilation_semantics=str(payload["assimilation_semantics"]),
            trainable_parameters=int(payload["trainable_parameters"]),
            matched_gru_hidden_size=int(payload["matched_gru_hidden_size"]),
            matched_gru_parameters=int(payload["matched_gru_parameters"]),
            matched_mlp_hidden_size=int(payload["matched_mlp_hidden_size"]),
            matched_mlp_parameters=int(payload["matched_mlp_parameters"]),
            protocol_lock_sha256=str(payload["protocol_lock_sha256"]),
            development_artifact_hashes=dict(payload["development_artifact_hashes"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("frozen candidate artifact is invalid") from error


def _reuse_or_reject_frozen_candidate(
    output: Path,
    development: Path,
    config: Phase05Config,
) -> FrozenCandidate | None:
    frozen_path = output / "frozen_candidate.json"
    if not frozen_path.is_file():
        return None

    existing = _load_frozen_candidate(frozen_path)
    protocol_hash = _sha256(output / "protocol_lock.json")
    artifact_hashes = _development_hashes(development)
    if (
        existing.protocol_lock_sha256 != protocol_hash
        or existing.development_artifact_hashes != artifact_hashes
        or existing.time_scale_days != config.time_scale_days
    ):
        raise RuntimeError("frozen candidate conflicts with current freeze inputs")
    return existing


def freeze_candidate(output: str | Path, config: Phase05Config) -> FrozenCandidate:
    output_path = Path(output)
    development = output_path / "development"
    existing = _reuse_or_reject_frozen_candidate(output_path, development, config)
    if existing is not None:
        return existing

    flow_path = development / "flow_gate.csv"
    jump_path = development / "jump_gate.csv"
    uncertainty_path = development / "uncertainty_gate.csv"
    timing_path = development / "representation_timing_audit.csv"

    if not _gate_passed(flow_path, "flow"):
        _write_development_failure(output_path, ["flow"])
        raise RuntimeError("flow gate failed; confirmation is not permitted")
    if not _gate_passed(jump_path, "jump"):
        _write_development_failure(output_path, ["jump"])
        raise RuntimeError("jump gate failed; confirmation is not permitted")

    flow = _selected_gate_row(flow_path, "passed", "flow")
    jump = _selected_gate_row(jump_path, "passed", "jump")
    uncertainty = _selected_gate_row(
        uncertainty_path,
        "selected",
        "uncertainty",
    )
    if not timing_path.is_file():
        raise RuntimeError("missing required representation timing audit artifact")

    trainable_parameters = int(uncertainty["trainable_parameters"])
    capacity = build_capacity_audit(
        target_parameters=trainable_parameters,
        value_dim=_VALUE_DIM,
        event_dim=_EVENT_DIM,
        representation_input_dim=_REPRESENTATION_INPUT_DIM,
    ).set_index("control")

    protocol_lock = output_path / "protocol_lock.json"
    development_artifact_hashes = _development_hashes(development)
    frozen = FrozenCandidate(
        flow_mode=str(flow["candidate"]),
        jump_mode=str(jump["candidate"]),
        uncertainty_mode=str(uncertainty["candidate"]),
        strict_history=True,
        state_dim=_STATE_DIM,
        time_scale_days=config.time_scale_days,
        jump_eligible_event_codes=tuple(sorted(JUMP_ELIGIBLE_EVENT_CODES)),
        assimilation_semantics=_ASSIMILATION_SEMANTICS,
        trainable_parameters=trainable_parameters,
        matched_gru_hidden_size=int(capacity.loc["matched_gru", "hidden_size"]),
        matched_gru_parameters=int(capacity.loc["matched_gru", "actual_parameters"]),
        matched_mlp_hidden_size=int(
            capacity.loc["matched_representation_mlp", "hidden_size"]
        ),
        matched_mlp_parameters=int(
            capacity.loc["matched_representation_mlp", "actual_parameters"]
        ),
        protocol_lock_sha256=_sha256(protocol_lock),
        development_artifact_hashes=development_artifact_hashes,
    )
    _write_frozen_candidate(output_path, frozen)
    return frozen


__all__ = [
    "FrozenCandidate",
    "build_protocol_lock",
    "estimate_phase0_relative_noise_floor",
    "freeze_candidate",
]
