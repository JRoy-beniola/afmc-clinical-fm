# AFMC Phase 0.6 Final Decision Record

Date: 2026-08-27

## Final status

**PHASE 0.6 TERMINATED — D4-B AMBIGUOUS → STOP**

Phase 0.6 was a development-only diagnostic program created to explain the structured N-dependent behavior observed after the failed Phase 0.5 flow gate. It did not reopen or reinterpret the Phase 0.5 result.

The complete Phase 0.6 diagnostic chain is:

| Component | Status | Frozen consequence |
|---|---|---|
| D0 instrumentation | COMPLETE | Diagnostic trajectories available |
| D1 exact reproduction | COMPLETE | Proceed to variance diagnosis |
| D2-A orthogonal variance screen | COMPLETE | Model initialization dominant at N=40; D2-B required |
| D3 adjudication | COMPLETE | `next_required_stage = D2B` |
| D2-B complementary replication | COMPLETE | 100/100 cells |
| Cross-array adjudication | COMPLETE | Evidence sufficient; `D4_OPTIMIZATION` |
| D4-B optimization/init stability | COMPLETE | 100/100 cells |
| D4-B adjudication | COMPLETE | `ambiguous` |
| Final route | TERMINAL | `STOP` |
| D4_CAPACITY_TIME | NOT AUTHORIZED | Stability gate not passed |
| Confirmatory seeds | PROTECTED | Never executed |

## D2-A / D2-B diagnostic conclusion

The two independently aliased orthogonal arrays agreed on the named-factor state:

- N=5: no named dominant factor.
- N=40: model initialization is the named dominant factor.

D2-B reproduced the N-dependent predictive shift and independently supported the model-initialization diagnosis.

The D2-A/D2-B overlap reruns were exact:

- mean absolute paired-effect difference: 0.0
- maximum absolute paired-effect difference: 0.0
- Pearson correlation: 1.0

This satisfied the frozen complementary-array sufficiency rule and authorized D4-B without a full 5x5x5 development factorial.

## D4-B frozen question

D4-B asked:

> Does the N=40 time-scaled predictive advantage remain consistently positive across a broader diagnostic-only bank of model initializations when cohort/subset context and all training semantics are held fixed?

Frozen matrix:

- fixed contexts: `(401,501)` through `(405,505)`
- model seeds: `1001..1010`
- N: `40`
- flow modes: `none`, `time_scaled`
- jump: `none`
- uncertainty: `deterministic`
- total cells: `100`
- paired effects: `50`

Primary effect orientation:

`Delta_MAE = MAE_none - MAE_time_scaled`

Positive values favor `time_scaled`.

## D4-B final result

Observed:

- `Delta_overall = +0.00894437611103058`
- positive model-seed means: `6/10`
- positive fixed-context means: `3/5`
- model-seed-cluster bootstrap 95% CI:
  `[-0.0016367578506469728, +0.019871510744094847]`
- individual paired effects:
  `26 positive / 24 negative`

Frozen classification rules required all of the following for `stable`:

- `Delta_overall > 0`
- at least `8/10` positive model-seed means
- at least `4/5` positive context means
- bootstrap lower bound `> 0`

The observed result does not satisfy those conditions.

The frozen `fragile` rule was also not triggered because:

- overall effect is positive,
- positive model-seed count is greater than 5,
- positive context count is greater than 2.

Therefore the result falls exactly into the preregistered middle state:

**classification = `ambiguous`**

and the frozen route is:

**next_required_stage = `STOP`**

## Scientific conclusion

Phase 0.6 establishes that the late N=40 predictive phenomenon is reproducible and that model initialization is a reproducible dominant source of its heterogeneity.

However, expanded initialization stress testing does not establish a robust positive time-scaled treatment effect. The average effect remains slightly positive, but its sign and magnitude vary materially across model seeds and fixed contexts, and the model-seed-cluster confidence interval includes zero.

The correct scientific conclusion is therefore:

> The N=40 time-scaled predictive advantage is not established as robust to model-initialization variation under the frozen Phase 0.6 training formulation.

This is a failure of the robustness claim, not proof that time scaling is universally ineffective or harmful.

## Protected boundary

Phase 0.6 used development-only evidence.

The reserved confirmatory namespaces remained untouched:

- cohort seeds `701..710`
- subset seeds `801..810`
- model seeds `901..910`

No D4 capacity/time experiment, full-factorial expansion, confirmatory execution, Phase 0.5 continuation, or post-hoc rescue experiment is authorized within Phase 0.6.

Any subsequent analysis of optimization-conditioned effects must be explicitly labeled exploratory and must not alter this frozen decision record.
