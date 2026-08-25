from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from afmc_fm.experiments.runner import build_complete_truth_targets
from afmc_fm.phase05.runner import padded_phase05_batch
from afmc_fm.phase05.sequences import Phase05Sequence
from afmc_fm.simulator.cohort import SimulatedPatient

_BUNDLE_COLUMNS = ["cohort_seed", "subset_seed", "model_seed"]
_DEFAULT_CANDIDATE = "phase05_candidate"
_DEFAULT_COMPARATORS = ("matched_gru", "matched_representation_mlp")


def complete_truth_phase05_batch(
    sequences: list[Phase05Sequence],
    patients: list[SimulatedPatient],
) -> dict[str, torch.Tensor]:
    if len(sequences) != len(patients):
        raise ValueError("patients and sequences must have equal lengths")
    batch = padded_phase05_batch(sequences, patients=patients)
    max_steps = batch["target_values"].shape[1]
    for row, patient in enumerate(patients):
        targets, masks = build_complete_truth_targets(patient)
        length = len(targets)
        if length > max_steps:
            raise ValueError("complete-truth targets exceed Phase-0.5 sequence length")
        batch["target_values"][row, :length] = torch.from_numpy(targets)
        batch["target_masks"][row, :length] = torch.from_numpy(masks)
    return batch


def _site_metric_value(
    group: pd.DataFrame,
    metric: str,
    site: int,
    *,
    required: bool,
) -> float | None:
    rows = group.loc[
        (group["metric"] == metric) & (group["site_or_shift"] == f"site_{site}"),
        "value",
    ]
    if rows.empty and not required:
        return None
    if len(rows) != 1:
        expectation = "exactly one" if required else "at most one"
        raise ValueError(
            f"site-shift evaluation requires {expectation} {metric} row for site {site}"
        )
    value = float(rows.iloc[0])
    if not np.isfinite(value):
        raise ValueError("site-shift metric values must be finite")
    return value


def evaluate_site_shift(
    metrics: pd.DataFrame,
    *,
    candidate: str = _DEFAULT_CANDIDATE,
    comparators: tuple[str, str] = _DEFAULT_COMPARATORS,
    win_requirement: int = 8,
) -> dict[str, pd.DataFrame]:
    if len(comparators) != 2 or len(set(comparators)) != 2:
        raise ValueError("site-shift gate requires exactly two distinct comparators")
    if candidate in comparators:
        raise ValueError("candidate must be distinct from site-shift comparators")
    if not 1 <= win_requirement <= 10:
        raise ValueError("win_requirement must be between one and ten")

    required_columns = {
        "world",
        "cohort_seed",
        "subset_seed",
        "model_seed",
        "n_train",
        "model",
        "site_or_shift",
        "metric",
        "value",
    }
    missing = required_columns - set(metrics.columns)
    if missing:
        raise ValueError(f"missing site-shift metric columns: {sorted(missing)}")

    selected_models = (candidate, *comparators)
    frame = metrics.loc[
        (metrics["world"] == "site_shift")
        & metrics["model"].isin(selected_models)
        & metrics["site_or_shift"].isin(("site_0", "site_1"))
    ].copy()
    if "split" in frame.columns:
        frame = frame.loc[frame["split"] == "test"]
    if frame.empty:
        raise ValueError("site-shift metrics contain no supported rows")

    rows: list[dict[str, object]] = []
    group_columns = ["n_train", "model", *_BUNDLE_COLUMNS]
    for key, group in frame.groupby(group_columns, sort=True):
        n_train, model, cohort_seed, subset_seed, model_seed = key
        site0_mae = _site_metric_value(group, "mae", 0, required=True)
        site1_mae = _site_metric_value(group, "mae", 1, required=True)
        assert site0_mae is not None and site1_mae is not None
        if site0_mae <= 0:
            raise ValueError("site-0 MAE must be positive for relative degradation")
        row: dict[str, object] = {
            "world": "site_shift",
            "cohort_seed": int(cohort_seed),
            "subset_seed": int(subset_seed),
            "model_seed": int(model_seed),
            "n_train": int(n_train),
            "model": str(model),
            "mae_site_0": site0_mae,
            "mae_site_1": site1_mae,
            "mae_absolute_degradation": site1_mae - site0_mae,
            "mae_relative_degradation": (site1_mae - site0_mae) / site0_mae,
        }

        site0_nll = _site_metric_value(group, "nll", 0, required=False)
        site1_nll = _site_metric_value(group, "nll", 1, required=False)
        if (site0_nll is None) != (site1_nll is None):
            raise ValueError("site-shift NLL must be present for both sites or neither")
        if site0_nll is not None and site1_nll is not None:
            row["nll_site_0"] = site0_nll
            row["nll_site_1"] = site1_nll
            row["nll_absolute_degradation"] = site1_nll - site0_nll

        site0_coverage = _site_metric_value(group, "coverage_90", 0, required=False)
        site1_coverage = _site_metric_value(group, "coverage_90", 1, required=False)
        if (site0_coverage is None) != (site1_coverage is None):
            raise ValueError(
                "site-shift 90% coverage must be present for both sites or neither"
            )
        if site0_coverage is not None and site1_coverage is not None:
            ce90_site0 = abs(site0_coverage - 0.90)
            ce90_site1 = abs(site1_coverage - 0.90)
            row["coverage_90_site_0"] = site0_coverage
            row["coverage_90_site_1"] = site1_coverage
            row["ce90_site_0"] = ce90_site0
            row["ce90_site_1"] = ce90_site1
            row["ce90_absolute_degradation"] = ce90_site1 - ce90_site0
        rows.append(row)

    per_seed = pd.DataFrame(rows).sort_values(
        ["n_train", "model", *_BUNDLE_COLUMNS], kind="mergesort"
    ).reset_index(drop=True)

    gate_rows: list[dict[str, object]] = []
    for n_train in sorted(per_seed["n_train"].unique()):
        candidate_rows = per_seed.loc[
            (per_seed["n_train"] == n_train) & (per_seed["model"] == candidate),
            [*_BUNDLE_COLUMNS, "mae_absolute_degradation"],
        ]
        candidate_bundles = set(
            candidate_rows[_BUNDLE_COLUMNS].itertuples(index=False, name=None)
        )
        if len(candidate_bundles) != 10:
            raise ValueError(
                "site-shift gate requires exactly ten candidate confirmatory bundles per N"
            )
        for comparator in comparators:
            comparator_rows = per_seed.loc[
                (per_seed["n_train"] == n_train)
                & (per_seed["model"] == comparator),
                [*_BUNDLE_COLUMNS, "mae_absolute_degradation"],
            ]
            comparator_bundles = set(
                comparator_rows[_BUNDLE_COLUMNS].itertuples(index=False, name=None)
            )
            if comparator_bundles != candidate_bundles:
                raise ValueError(
                    "site-shift gate requires ten exactly matched confirmatory bundles"
                )
            paired = candidate_rows.merge(
                comparator_rows,
                on=_BUNDLE_COLUMNS,
                suffixes=("_candidate", "_comparator"),
                validate="one_to_one",
            )
            advantages = (
                paired["mae_absolute_degradation_comparator"].to_numpy(dtype=float)
                - paired["mae_absolute_degradation_candidate"].to_numpy(dtype=float)
            )
            wins = int(np.sum(advantages > 0))
            candidate_mean = float(
                paired["mae_absolute_degradation_candidate"].mean()
            )
            comparator_mean = float(
                paired["mae_absolute_degradation_comparator"].mean()
            )
            gate_rows.append(
                {
                    "n_train": int(n_train),
                    "comparator": comparator,
                    "candidate_mean_mae_degradation": candidate_mean,
                    "comparator_mean_mae_degradation": comparator_mean,
                    "mean_degradation_advantage": float(np.mean(advantages)),
                    "wins": wins,
                    "win_fraction": wins / 10.0,
                    "passed": bool(
                        candidate_mean < comparator_mean
                        and wins >= win_requirement
                    ),
                }
            )

    gate_summary = pd.DataFrame(gate_rows).reset_index(drop=True)
    return {"per_seed": per_seed, "gate_summary": gate_summary}


__all__ = ["complete_truth_phase05_batch", "evaluate_site_shift"]
