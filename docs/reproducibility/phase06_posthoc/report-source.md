# Phase 0.6 Post-Hoc Optimization-Conditioned Analysis — Source-First Documentary Record

## Documentary status

This file is the canonical version-controlled documentary source for the archived post-Phase-0.6 optimization-conditioned analysis. There is **no official historical report** or official DOCX registered for this exploratory analysis, and this source does not retroactively create one. The authoritative scientific evidence remains the immutable archive under `docs/results/phase06_posthoc_optimization/`.

This source is derived only from the archived decision record, execution provenance, and checksummed analysis artifacts. It is a documentary reproducibility aid, not a new experiment, a new adjudication, or confirmatory evidence.

## Frozen source bindings

- Decision record: `docs/results/phase06_posthoc_optimization/decision_record.md`
- Execution provenance: `docs/results/phase06_posthoc_optimization/execution_provenance.json`
- Evidence checksum manifest: `docs/results/phase06_posthoc_optimization/MANIFEST.sha256`
- Paired mechanism table: `docs/results/phase06_posthoc_optimization/analysis/phase06_posthoc_pair_mechanisms.csv`
- Primary/secondary association table: `docs/results/phase06_posthoc_optimization/analysis/phase06_posthoc_associations.csv`
- Leave-one-out robustness table: `docs/results/phase06_posthoc_optimization/analysis/phase06_posthoc_leave_one_out.csv`
- Screening record: `docs/results/phase06_posthoc_optimization/analysis/phase06_posthoc_screening.json`

The archived analysis implementation commit is `8c9de2aaeee1e26f65e28a1dd682f8ac3effad72`. The analysis used the frozen Phase 0.6 D4-B archive as its source matrix. It evaluated 50 paired effects across five contexts and ten model seeds, with 10,000 permutation resamples and permutation seed `20260827`.

## Exploratory screening result

The frozen exploratory classification is:

`structured optimization-conditioned heterogeneity worth prospective testing`

The screening record identifies four passing primary diagnostics:

- `delta_selected_epoch`
- `delta_stop_epoch`
- `delta_shadow_mae_epoch`
- `delta_patience_exhausted`

`delta_selection_shadow_gap` did not pass the primary screening criteria. The archived screening record marks the analysis `exploratory_not_confirmatory: true`.

## Primary association summary

| Diagnostic | Two-way Pearson | Two-way Spearman | Permutation p | Holm p |
| --- | ---: | ---: | ---: | ---: |
| `delta_selected_epoch` | 0.824903 | 0.785930 | 0.000100 | 0.000500 |
| `delta_stop_epoch` | 0.835082 | 0.793584 | 0.000100 | 0.000500 |
| `delta_shadow_mae_epoch` | 0.658699 | 0.635117 | 0.000100 | 0.000500 |
| `delta_selection_shadow_gap` | -0.197819 | -0.181373 | 0.228777 | 0.228777 |
| `delta_patience_exhausted` | -0.713579 | -0.691780 | 0.000100 | 0.000500 |

The effect orientation is `Delta_MAE = MAE_control - MAE_time_scaled`; mechanism deltas are oriented `candidate - control`.

## Interpretation boundary

The archived decision record treats the coherent signal as one optimization-horizon / premature-termination phenomenon expressed through correlated diagnostics, rather than four independent mechanisms. Later selected, stopping, and shadow-MAE epochs are associated with more favorable `time_scaled` effects, while candidate-side patience exhaustion is associated with worse `time_scaled` effects.

The null `delta_selection_shadow_gap` result argues against checkpoint-objective mismatch as the primary explanation. The secondary gradient, parameter-norm, and early flow-displacement diagnostics do not provide comparable evidence.

This remains retrospective analysis on the same frozen D4-B 50-pair matrix. It is hypothesis-generating and does not establish causal mediation.

## Frozen Phase 0.6 boundary

The terminal historical Phase 0.6 decision remains exactly:

`D4-B AMBIGUOUS -> STOP`

The exploratory post-hoc result **does not authorize** D4 capacity/time execution, D4-D or full-factorial execution, confirmatory execution, Phase 0.7 execution, reuse of protected confirmatory seeds, or reinterpretation of the frozen Phase 0.6 adjudication.

The protected confirmatory seed firewall remains untouched:

- cohort: `701..710`
- subset: `801..810`
- model: `901..910`

## Prospective implication

The archived next-step recommendation is to freeze a prospective experiment before any new cells are run. Any future experiment would need to manipulate optimization termination prospectively rather than re-correlate stopping diagnostics with performance. This documentary source records that recommendation only; it does not authorize or execute it.
