import numpy as np
import pandas as pd

from afmc_fm.phase05.analysis import normalized_log_n_aulc, paired_development_gate

_DEVELOPMENT_WORLDS = ("smooth", "jumps", "informative_observation")
_BUNDLE_COLUMNS = ["cohort_seed", "subset_seed", "model_seed"]


def _variant_name(flow: str, jump: str, uncertainty: str) -> str:
    return f"{flow}__{jump}__{uncertainty}"


def _aulc_by_bundle(frame: pd.DataFrame, *, variant: str, world: str) -> pd.DataFrame:
    subset = frame.loc[
        (frame["variant"] == variant)
        & (frame["world"] == world)
        & (frame["metric"] == "mae")
    ].copy()
    if subset.empty:
        raise ValueError(f"missing MAE rows for variant {variant} in world {world}")
    rows: list[dict[str, object]] = []
    for bundle, group in subset.groupby(_BUNDLE_COLUMNS, sort=True):
        rows.append(
            {
                "cohort_seed": bundle[0],
                "subset_seed": bundle[1],
                "model_seed": bundle[2],
                "aulc": normalized_log_n_aulc(group),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 5:
        raise ValueError("development selection requires exactly five matched seed bundles")
    return result.sort_values(_BUNDLE_COLUMNS).reset_index(drop=True)


def _paired_aulcs(
    frame: pd.DataFrame,
    *,
    candidate_variant: str,
    control_variant: str,
    world: str,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = _aulc_by_bundle(frame, variant=candidate_variant, world=world)
    control = _aulc_by_bundle(frame, variant=control_variant, world=world)
    merged = candidate.merge(control, on=_BUNDLE_COLUMNS, suffixes=("_candidate", "_control"))
    if len(merged) != 5:
        raise ValueError("development candidates must share five matched seed bundles")
    return (
        merged["aulc_candidate"].to_numpy(dtype=float),
        merged["aulc_control"].to_numpy(dtype=float),
    )


def _variant_parameters(frame: pd.DataFrame, variant: str) -> int:
    subset = frame.loc[frame["variant"] == variant, "trainable_parameters"]
    if subset.empty:
        raise ValueError(f"missing trainable parameter count for variant {variant}")
    values = pd.unique(subset.dropna())
    if len(values) != 1:
        raise ValueError(f"variant {variant} has inconsistent trainable parameter counts")
    return int(values[0])


def _choose_passing_candidate(
    gate_table: pd.DataFrame,
    *,
    tie_tolerance: float = 0.005,
) -> str | None:
    passing = gate_table.loc[gate_table["passed"]].copy()
    if passing.empty:
        return None
    if len(passing) == 1:
        return str(passing.iloc[0]["candidate"])
    passing = passing.sort_values("candidate_mean_aulc", ascending=True).reset_index(drop=True)
    best = passing.iloc[0]
    second = passing.iloc[1]
    if abs(
        float(best["mean_relative_improvement"])
        - float(second["mean_relative_improvement"])
    ) < tie_tolerance:
        passing = passing.sort_values(
            ["trainable_parameters", "candidate_mean_aulc", "candidate"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        return str(passing.iloc[0]["candidate"])
    return str(best["candidate"])


def _mark_selected(gate_table: pd.DataFrame, selected: str | None) -> pd.DataFrame:
    result = gate_table.copy()
    result["selected"] = result["candidate"].eq(selected) if selected is not None else False
    return result


def select_flow(
    frame: pd.DataFrame,
    *,
    locked_min_relative_effect: float,
) -> dict[str, object]:
    world = "smooth"
    control = _variant_name("none", "none", "deterministic")
    rows = []
    for candidate in ("gated", "time_scaled"):
        variant = _variant_name(candidate, "none", "deterministic")
        candidate_aulc, control_aulc = _paired_aulcs(
            frame,
            candidate_variant=variant,
            control_variant=control,
            world=world,
        )
        gate = paired_development_gate(
            candidate_aulc,
            control_aulc,
            min_relative_effect=locked_min_relative_effect,
        )
        rows.append(
            {
                "candidate": candidate,
                "control": "none",
                "trainable_parameters": _variant_parameters(frame, variant),
                **{key: value for key, value in gate.items() if key != "paired_improvements"},
            }
        )
    gate_table = pd.DataFrame(rows)
    selected = _choose_passing_candidate(gate_table)
    gate_table = _mark_selected(gate_table, selected)
    return {
        "passed": selected is not None,
        "selected": selected,
        "gate_table": gate_table,
    }


def select_jump(
    frame: pd.DataFrame,
    *,
    selected_flow: str | None,
    flow_passed: bool,
    locked_min_relative_effect: float,
) -> dict[str, object]:
    if not flow_passed or selected_flow is None:
        raise RuntimeError("flow selection must pass before jump selection")
    world = "jumps"
    control = _variant_name(selected_flow, "none", "deterministic")
    rows = []
    for candidate in ("gru", "residual"):
        variant = _variant_name(selected_flow, candidate, "deterministic")
        candidate_aulc, control_aulc = _paired_aulcs(
            frame,
            candidate_variant=variant,
            control_variant=control,
            world=world,
        )
        gate = paired_development_gate(
            candidate_aulc,
            control_aulc,
            min_relative_effect=locked_min_relative_effect,
        )
        rows.append(
            {
                "candidate": candidate,
                "control": "none",
                "trainable_parameters": _variant_parameters(frame, variant),
                **{key: value for key, value in gate.items() if key != "paired_improvements"},
            }
        )
    gate_table = pd.DataFrame(rows)
    selected = _choose_passing_candidate(gate_table)
    gate_table = _mark_selected(gate_table, selected)
    return {
        "passed": selected is not None,
        "selected": selected,
        "gate_table": gate_table,
    }


def _mean_world_aulc(frame: pd.DataFrame, *, variant: str, world: str) -> float:
    return float(_aulc_by_bundle(frame, variant=variant, world=world)["aulc"].mean())


def _paired_metric_world_wins(
    frame: pd.DataFrame,
    *,
    winner_variant: str,
    loser_variant: str,
    world: str,
    metric: str,
) -> tuple[bool, int, float, float]:
    winner = frame.loc[
        (frame["variant"] == winner_variant)
        & (frame["world"] == world)
        & (frame["metric"] == metric),
        _BUNDLE_COLUMNS + ["value"],
    ].copy()
    loser = frame.loc[
        (frame["variant"] == loser_variant)
        & (frame["world"] == world)
        & (frame["metric"] == metric),
        _BUNDLE_COLUMNS + ["value"],
    ].copy()
    if winner.empty or loser.empty:
        return False, 0, float("nan"), float("nan")
    winner = winner.groupby(_BUNDLE_COLUMNS, as_index=False)["value"].mean()
    loser = loser.groupby(_BUNDLE_COLUMNS, as_index=False)["value"].mean()
    merged = winner.merge(loser, on=_BUNDLE_COLUMNS, suffixes=("_winner", "_loser"))
    if len(merged) != 5:
        raise ValueError("probabilistic comparison requires five matched development bundles")
    winner_values = merged["value_winner"].to_numpy(dtype=float)
    loser_values = merged["value_loser"].to_numpy(dtype=float)
    if metric == "coverage_90":
        winner_values = np.abs(winner_values - 0.90)
        loser_values = np.abs(loser_values - 0.90)
    if not np.isfinite(winner_values).all() or not np.isfinite(loser_values).all():
        raise ValueError("probabilistic comparison values must be finite")
    wins = int(np.sum(winner_values < loser_values))
    return (
        bool(float(np.mean(winner_values)) < float(np.mean(loser_values)) and wins >= 4),
        wins,
        float(np.mean(winner_values)),
        float(np.mean(loser_values)),
    )


def select_uncertainty(
    frame: pd.DataFrame,
    *,
    selected_flow: str,
    selected_jump: str,
    locked_uncertainty_mae_tolerance: float,
) -> dict[str, object]:
    variants = {
        name: _variant_name(selected_flow, selected_jump, name)
        for name in ("joint", "decoupled", "deterministic")
    }
    point_admissible: dict[str, bool] = {}
    for probabilistic in ("joint", "decoupled"):
        admissible = True
        for world in _DEVELOPMENT_WORLDS:
            prob = _mean_world_aulc(frame, variant=variants[probabilistic], world=world)
            deterministic = _mean_world_aulc(
                frame,
                variant=variants["deterministic"],
                world=world,
            )
            if prob > (1.0 + locked_uncertainty_mae_tolerance) * deterministic:
                admissible = False
        point_admissible[probabilistic] = admissible

    probabilistic_world_wins = {"joint": 0, "decoupled": 0}
    for candidate, other in (("joint", "decoupled"), ("decoupled", "joint")):
        if not point_admissible[candidate]:
            continue
        for world in _DEVELOPMENT_WORLDS:
            nll_ok, _, _, _ = _paired_metric_world_wins(
                frame,
                winner_variant=variants[candidate],
                loser_variant=variants[other],
                world=world,
                metric="nll",
            )
            ce90_ok, _, _, _ = _paired_metric_world_wins(
                frame,
                winner_variant=variants[candidate],
                loser_variant=variants[other],
                world=world,
                metric="coverage_90",
            )
            if nll_ok and ce90_ok:
                probabilistic_world_wins[candidate] += 1

    selected = "deterministic"
    eligible = [
        name
        for name in ("joint", "decoupled")
        if point_admissible[name] and probabilistic_world_wins[name] >= 2
    ]
    if len(eligible) == 1:
        selected = eligible[0]
    elif len(eligible) == 2:
        selected = max(
            eligible,
            key=lambda name: (
                probabilistic_world_wins[name],
                -_variant_parameters(frame, variants[name])
                if "trainable_parameters" in frame.columns
                else 0,
            ),
        )

    return {
        "selected": selected,
        "point_admissible": point_admissible,
        "probabilistic_world_wins": probabilistic_world_wins,
    }


def representation_timing_audit(strict_aulc, inclusive_aulc) -> dict[str, object]:
    strict = np.asarray(strict_aulc, dtype=float)
    inclusive = np.asarray(inclusive_aulc, dtype=float)
    if strict.shape != inclusive.shape or strict.ndim != 1 or strict.size != 5:
        raise ValueError("timing audit requires five matched paired values")
    if not np.isfinite(strict).all() or not np.isfinite(inclusive).all():
        raise ValueError("timing audit values must be finite")
    paired = inclusive - strict
    return {
        "strict_timing_required": True,
        "paired_delta_inclusive_minus_strict": paired,
        "mean_delta_inclusive_minus_strict": float(np.mean(paired)),
        "median_delta_inclusive_minus_strict": float(np.median(paired)),
    }


__all__ = [
    "representation_timing_audit",
    "select_flow",
    "select_jump",
    "select_uncertainty",
]
