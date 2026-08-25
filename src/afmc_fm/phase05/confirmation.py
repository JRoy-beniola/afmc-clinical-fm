from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from afmc_fm.phase05.analysis import paired_confirmatory_naulc_effects
from afmc_fm.phase05.config import Phase05Config
from afmc_fm.phase05.execution import Phase05Job, plan_phase05_shards
from afmc_fm.phase05.protocol import FrozenCandidate

CONFIRMATORY_MODELS = (
    "phase05_candidate",
    "matched_gru",
    "matched_representation_mlp",
    "representation_linear",
    "representation_mlp_original",
    "gru_original",
    "phase0_flow_jump_reference",
)
PRIMARY_COMPARATORS = ("matched_gru", "matched_representation_mlp")
_TARGET_WORLD_ORDER = ("smooth", "jumps", "informative_observation")
_EARLY_TRAIN_SIZES = (5, 10, 20)
_BUNDLE_COLUMNS = ["cohort_seed", "subset_seed", "model_seed"]


def paired_naulc_effects(
    metrics: pd.DataFrame,
    candidate: str,
    comparator: str,
) -> pd.DataFrame:
    return paired_confirmatory_naulc_effects(metrics, candidate, comparator)


def _ten_effects(effects) -> np.ndarray:
    values = np.asarray(effects, dtype=float)
    if values.ndim != 1 or values.size != 10:
        raise ValueError("confirmatory inference requires exactly ten paired seed effects")
    if not np.isfinite(values).all():
        raise ValueError("confirmatory seed effects must be finite")
    return values


def paired_bootstrap_mean_ci(
    effects,
    *,
    resamples: int = 10_000,
    seed: int = 20260824,
) -> tuple[float, float]:
    values = _ten_effects(effects)
    if type(resamples) is not int or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def exact_sign_test(effects) -> dict[str, object]:
    values = _ten_effects(effects)
    wins = int(np.sum(values > 0))
    losses = int(np.sum(values < 0))
    ties = int(np.sum(values == 0))
    n = wins + losses
    p_value = 1.0 if n == 0 else float(
        binomtest(wins, n, 0.5, alternative="greater").pvalue
    )
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n": n,
        "p_value": p_value,
    }


def holm_adjust(p_values) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Holm correction requires a non-empty one-dimensional array")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite and between zero and one")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    scaled = (values.size - np.arange(values.size)) * sorted_values
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def _model_variants(frozen: FrozenCandidate) -> dict[str, str]:
    return {
        "phase05_candidate": (
            f"{frozen.flow_mode}__{frozen.jump_mode}__{frozen.uncertainty_mode}"
        ),
        "matched_gru": f"hidden{frozen.matched_gru_hidden_size}",
        "matched_representation_mlp": f"hidden{frozen.matched_mlp_hidden_size}",
        "representation_linear": "original",
        "representation_mlp_original": "hidden32",
        "gru_original": "hidden32",
        "phase0_flow_jump_reference": "historical_inclusive_history",
    }


def build_confirmation_jobs(
    config: Phase05Config,
    frozen: FrozenCandidate,
    *,
    frozen_candidate_hash: str,
) -> tuple[Phase05Job, ...]:
    variants = _model_variants(frozen)
    return tuple(
        Phase05Job(
            shard=shard,
            n_train=n_train,
            model=model,
            variant=variants[model],
            frozen_candidate_hash=frozen_candidate_hash,
        )
        for shard in plan_phase05_shards(config, "confirmation")
        for n_train in config.train_sizes
        for model in CONFIRMATORY_MODELS
    )


def _paired_mae_at_n(
    metrics: pd.DataFrame,
    *,
    world: str,
    n_train: int,
    candidate: str,
    comparator: str,
) -> np.ndarray:
    required = {
        "world",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "n_train",
        "model",
        "metric",
        "value",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"missing confirmatory metric columns: {sorted(missing)}")
    frame = metrics.loc[
        (metrics["world"] == world)
        & (metrics["n_train"] == n_train)
        & (metrics["metric"] == "mae")
        & metrics["model"].isin((candidate, comparator))
    ].copy()
    if "split" in frame.columns:
        frame = frame.loc[frame["split"] == "test"]
    if "site_or_shift" in frame.columns:
        frame = frame.loc[frame["site_or_shift"] == "all"]
    candidate_rows = frame.loc[frame["model"] == candidate, _BUNDLE_COLUMNS + ["value"]]
    comparator_rows = frame.loc[
        frame["model"] == comparator,
        _BUNDLE_COLUMNS + ["value"],
    ]
    merged = candidate_rows.merge(
        comparator_rows,
        on=_BUNDLE_COLUMNS,
        suffixes=("_candidate", "_comparator"),
        validate="one_to_one",
    )
    if len(merged) != 10:
        raise ValueError("early-N comparison requires exactly ten matched confirmatory bundles")
    effects = merged["value_comparator"].to_numpy(dtype=float) - merged[
        "value_candidate"
    ].to_numpy(dtype=float)
    if not np.isfinite(effects).all():
        raise ValueError("early-N paired MAE effects must be finite")
    return effects


def evaluate_primary_gate(
    metrics: pd.DataFrame,
    *,
    candidate: str = "phase05_candidate",
    comparators: tuple[str, str] = PRIMARY_COMPARATORS,
    win_requirement: int = 8,
) -> dict[str, object]:
    if len(comparators) != 2 or len(set(comparators)) != 2:
        raise ValueError("primary gate requires exactly two distinct comparators")
    if not 1 <= win_requirement <= 10:
        raise ValueError("win_requirement must be between one and ten")

    world_rows: list[dict[str, object]] = []
    effects_by_comparator: dict[str, pd.DataFrame] = {}
    for comparator in comparators:
        effects = paired_naulc_effects(metrics, candidate, comparator)
        effects_by_comparator[comparator] = effects
        for world, group in effects.groupby("world", sort=True):
            values = _ten_effects(group["effect"].to_numpy(dtype=float))
            mean_effect = float(np.mean(values))
            wins = int(np.sum(values > 0))
            world_rows.append(
                {
                    "world": world,
                    "comparator": comparator,
                    "mean_effect": mean_effect,
                    "wins": wins,
                    "win_fraction": wins / 10.0,
                    "passed": bool(mean_effect > 0 and wins >= win_requirement),
                }
            )
    world_summary = pd.DataFrame(world_rows).sort_values(
        ["world", "comparator"]
    ).reset_index(drop=True)
    observed_worlds = set(world_summary["world"].unique())
    unknown_worlds = observed_worlds - set(_TARGET_WORLD_ORDER)
    if unknown_worlds:
        raise ValueError(f"unexpected primary target worlds: {sorted(unknown_worlds)}")
    worlds = tuple(world for world in _TARGET_WORLD_ORDER if world in observed_worlds)
    passing_sets = []
    for comparator in comparators:
        passing_sets.append(
            set(
                world_summary.loc[
                    (world_summary["comparator"] == comparator)
                    & world_summary["passed"],
                    "world",
                ]
            )
        )
    shared = set.intersection(*passing_sets)
    shared_passing_worlds = tuple(world for world in worlds if world in shared)

    early_rows: list[dict[str, object]] = []
    for world in worlds:
        for n_train in _EARLY_TRAIN_SIZES:
            row: dict[str, object] = {"world": world, "n_train": n_train}
            all_pass = True
            for comparator in comparators:
                effects = _paired_mae_at_n(
                    metrics,
                    world=world,
                    n_train=n_train,
                    candidate=candidate,
                    comparator=comparator,
                )
                mean_effect = float(np.mean(effects))
                wins = int(np.sum(effects > 0))
                passed = bool(mean_effect > 0 and wins >= win_requirement)
                row[f"{comparator}_mean_effect"] = mean_effect
                row[f"{comparator}_wins"] = wins
                row[f"{comparator}_passed"] = passed
                all_pass = all_pass and passed
            row["passed"] = all_pass
            early_rows.append(row)
    early_summary = pd.DataFrame(early_rows).sort_values(
        ["world", "n_train"]
    ).reset_index(drop=True)
    early_n_passed = bool(
        early_summary.loc[early_summary["world"].isin(shared_passing_worlds), "passed"].any()
    )
    headline_passed = bool(len(shared_passing_worlds) >= 2 and early_n_passed)
    return {
        "headline_passed": headline_passed,
        "shared_passing_worlds": shared_passing_worlds,
        "early_n_passed": early_n_passed,
        "world_summary": world_summary,
        "early_n_summary": early_summary,
        "effects_by_comparator": effects_by_comparator,
    }


def _atomic_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def persist_confirmation_analysis(
    output: str | Path,
    metrics: pd.DataFrame,
    *,
    config: Phase05Config,
    candidate: str = "phase05_candidate",
) -> dict[str, Path]:
    if tuple(config.target_worlds) != _TARGET_WORLD_ORDER:
        raise ValueError("primary confirmatory analysis requires the three locked target worlds")
    gate = evaluate_primary_gate(
        metrics,
        candidate=candidate,
        comparators=PRIMARY_COMPARATORS,
        win_requirement=config.confirmatory_win_requirement,
    )

    effect_frames = []
    bootstrap_rows = []
    sign_rows = []
    for comparator in PRIMARY_COMPARATORS:
        effects = gate["effects_by_comparator"][comparator].copy()
        effect_frames.append(effects)
        for world in config.target_worlds:
            group = effects.loc[effects["world"] == world]
            values = _ten_effects(group["effect"].to_numpy(dtype=float))
            lower, upper = paired_bootstrap_mean_ci(
                values,
                resamples=config.bootstrap_resamples,
                seed=config.bootstrap_seed,
            )
            bootstrap_rows.append(
                {
                    "world": world,
                    "comparator": comparator,
                    "mean_effect": float(np.mean(values)),
                    "ci_lower_95": lower,
                    "ci_upper_95": upper,
                    "resamples": config.bootstrap_resamples,
                    "seed": config.bootstrap_seed,
                }
            )
            sign = exact_sign_test(values)
            sign_rows.append(
                {
                    "world": world,
                    "comparator": comparator,
                    **sign,
                }
            )

    effects_frame = pd.concat(effect_frames, ignore_index=True).sort_values(
        ["world", "comparator", *_BUNDLE_COLUMNS]
    ).reset_index(drop=True)
    bootstrap_frame = pd.DataFrame(bootstrap_rows).sort_values(
        ["world", "comparator"]
    ).reset_index(drop=True)
    sign_frame = pd.DataFrame(sign_rows).sort_values(
        ["world", "comparator"]
    ).reset_index(drop=True)
    sign_frame["holm_p_value"] = holm_adjust(sign_frame["p_value"].to_numpy(dtype=float))

    primary_summary = gate["world_summary"].copy()
    primary_summary["shared_passing_worlds"] = ",".join(gate["shared_passing_worlds"])
    primary_summary["early_n_passed"] = gate["early_n_passed"]
    primary_summary["headline_passed"] = gate["headline_passed"]

    confirmation = Path(output) / "confirmation"
    return {
        "effects": _atomic_csv(confirmation / "paired_naulc_effects.csv", effects_frame),
        "bootstrap": _atomic_csv(
            confirmation / "bootstrap_intervals.csv",
            bootstrap_frame,
        ),
        "sign_tests": _atomic_csv(confirmation / "sign_tests.csv", sign_frame),
        "primary_gate": _atomic_csv(
            confirmation / "primary_gate_summary.csv",
            primary_summary,
        ),
    }


__all__ = [
    "CONFIRMATORY_MODELS",
    "PRIMARY_COMPARATORS",
    "build_confirmation_jobs",
    "evaluate_primary_gate",
    "exact_sign_test",
    "holm_adjust",
    "paired_bootstrap_mean_ci",
    "paired_naulc_effects",
    "persist_confirmation_analysis",
]
