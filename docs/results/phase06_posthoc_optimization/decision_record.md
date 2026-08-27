# Phase 0.6 Post-Hoc Optimization-Conditioned Analysis — Decision Record

Status: **EXPLORATORY RESULT RECORDED; PHASE 0.6 TERMINAL VERDICT UNCHANGED**

Analysis implementation commit: `8c9de2aaeee1e26f65e28a1dd682f8ac3effad72`

Frozen source archive: `docs/results/phase06/evidence/d4b/`

Execution settings:
- paired effects: 50 (`5 contexts × 10 model seeds`)
- source cells: 100 (`none` and `time_scaled`)
- permutation resamples: 10,000
- permutation seed: `20260827`
- effect orientation: `Delta_MAE = MAE_control - MAE_time_scaled`
- mechanism orientation: `candidate - control`

## Screening result

`structured optimization-conditioned heterogeneity worth prospective testing`

Passing primary mechanisms:
- `delta_selected_epoch`
- `delta_stop_epoch`
- `delta_shadow_mae_epoch`
- `delta_patience_exhausted`

Non-passing primary mechanism:
- `delta_selection_shadow_gap`

## Primary association results

| Mechanism | Raw Pearson | Raw Spearman | Context-demeaned Pearson | Context-demeaned Spearman | Two-way Pearson | Two-way Spearman | Permutation p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `delta_selected_epoch` | 0.872755 | 0.857916 | 0.749952 | 0.671418 | 0.824903 | 0.785930 | 0.000100 | 0.000500 |
| `delta_stop_epoch` | 0.882253 | 0.842109 | 0.747827 | 0.665876 | 0.835082 | 0.793584 | 0.000100 | 0.000500 |
| `delta_shadow_mae_epoch` | 0.766847 | 0.799561 | 0.630313 | 0.524295 | 0.658699 | 0.635117 | 0.000100 | 0.000500 |
| `delta_selection_shadow_gap` | -0.352329 | -0.108986 | -0.226932 | -0.158660 | -0.197819 | -0.181373 | 0.228777 | 0.228777 |
| `delta_patience_exhausted` | -0.735026 | -0.718105 | -0.678387 | -0.623110 | -0.713579 | -0.691780 | 0.000100 | 0.000500 |

## Secondary association results

| Mechanism | Two-way Pearson | Two-way Spearman | Permutation p |
| --- | ---: | ---: | ---: |
| `delta_prefix10_gradient_l2_mean` | -0.201172 | -0.155534 | 0.232777 |
| `delta_prefix10_parameter_l2_mean` | -0.088479 | -0.085426 | 0.604440 |
| `candidate_prefix10_flow_displacement_mean` | 0.161179 | 0.135078 | 0.345365 |

## Interpretation boundary

The coherent signal is best treated as one optimization-horizon / premature-termination phenomenon observed through several correlated diagnostics, not as four independent mechanisms. Later selected, stopping, and shadow-MAE epochs are associated with more favorable `time_scaled` effects; candidate-side patience exhaustion is associated with worse `time_scaled` effects.

The null `delta_selection_shadow_gap` result argues against checkpoint-objective mismatch as the primary explanation. The secondary gradient, parameter-norm, and early flow-displacement diagnostics do not provide comparable evidence.

This is retrospective mechanism analysis on the same frozen D4-B 50-pair matrix. It is hypothesis-generating only and does not establish causal mediation.

## Frozen Phase 0.6 boundary

The terminal Phase 0.6 classification remains exactly:

`D4-B AMBIGUOUS -> STOP`

This post-hoc result does not authorize `D4_CAPACITY_TIME`, D4-D/full factorial execution, confirmatory execution, reuse of protected confirmatory seeds, or any reinterpretation of the Phase 0.6 terminal adjudication.

Protected confirmatory seed firewall remains untouched:
- cohort: `701..710`
- subset: `801..810`
- model: `901..910`

## Next scientific action

Freeze a prospective experiment before any new cells are run. The prospective intervention must manipulate optimization termination rather than merely re-correlate stopping diagnostics with performance. The intended question is whether preventing candidate-specific premature termination causally shifts the `time_scaled` treatment effect upward and reduces initialization-dependent heterogeneity on unseen development contexts and model seeds.

## Raw artifact archival status

The exact execution artifacts are archived byte-for-byte under `docs/results/phase06_posthoc_optimization/analysis/`:
- `phase06_posthoc_pair_mechanisms.csv`
- `phase06_posthoc_associations.csv`
- `phase06_posthoc_leave_one_out.csv`
- `phase06_posthoc_screening.json`

Their archive is checksummed by `docs/results/phase06_posthoc_optimization/MANIFEST.sha256`, with execution provenance recorded in `docs/results/phase06_posthoc_optimization/execution_provenance.json`. These archived artifacts must not be regenerated or hand-edited in place.
