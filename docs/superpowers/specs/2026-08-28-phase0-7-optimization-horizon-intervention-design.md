# Phase 0.7 Prospective Optimization-Horizon Intervention Design

Status: **FROZEN — IMPLEMENTATION AUTHORIZED — NO OFFICIAL EXECUTION AUTHORIZED**

Date: 2026-08-28

## 1. Scientific role in the research program

Phase 0.7 is a narrow prospective causal checkpoint, not the endpoint of the research program and not an attempt to rescue `time_scaled` indefinitely.

The completed evidence trail is:

- Phase 0: original extreme-low-N architecture hypothesis failed.
- Phase 0.5: mechanistic low-N redesign failed its flow gate; an N=40 diagnostic signal remained.
- Phase 0.6: variance diagnosis found model initialization dominant; D4-B ended `AMBIGUOUS -> STOP`.
- Post-Phase-0.6: retrospective analysis of the same D4-B paired effects found structured optimization-horizon associations.
- Reproducibility closure: the historical line through the post-Phase-0.6 analysis is now integrated and preserved without reopening any frozen scientific verdict.

Phase 0.7 asks whether one specific retrospective explanation survives a prospective intervention. Regardless of its verdict, the longer-horizon program then moves outward toward the general methodological question:

> When does optimization confound low-data inductive-bias evaluation?

The intended generalization ladder after Phase 0.7 is multi-architecture replication, multi-world synthetic evidence, real longitudinal datasets, and prospective/frozen confirmation. Phase 0.7 alone cannot establish that field-level claim.

## 2. Frozen historical boundaries

The following remain immutable:

- Phase 0.6 D4-B terminal classification: `D4-B AMBIGUOUS -> STOP`.
- The post-Phase-0.6 optimization-conditioned analysis is exploratory only.
- No retrospective association is promoted to causal evidence.
- No causal mediation claim is permitted.
- Protected confirmatory seeds remain untouched: cohort `701..710`, subset `801..810`, model `901..910`.
- No Phase 0.7 result may rewrite Phase 0, Phase 0.5, or Phase 0.6 conclusions.
- No threshold search or post-hoc rescue rule is permitted after Phase 0.7 execution begins.

## 3. Scientific question

When `time_scaled` changes predictive performance at N=40, does preventing premature optimization termination causally change its relative performance versus the locked control architecture under the same synthetic formulation?

The narrower mechanistic question is whether the architecture comparison is sensitive to the stopping policy because the two architectures interact differently with trajectory truncation.

This is a prospective intervention on optimization termination. It is not formal causal mediation analysis and does not identify a natural direct or indirect effect.

## 4. Experimental scope

World: `smooth` only.

Training N: `40` only.

Flow variants:

- control: `none__none__deterministic`
- candidate: `time_scaled__none__deterministic`

Optimization policies:

1. `standard_early_stop`: exact locked D4-B early-stopping behavior.
2. `forced_horizon`: train through epoch 100 even if the ordinary patience rule would have terminated training earlier.

No other N, world, flow, jump, capacity, checkpoint objective, or architecture comparison is authorized in Phase 0.7.

All model, optimizer, loss, data, capacity, checkpoint-scoring, and evaluation semantics remain locked to the validated D4-B implementation unless this specification explicitly changes them.

## 5. Development-only design and seed firewall

Use five unseen development contexts:

- `(406,506)`
- `(407,507)`
- `(408,508)`
- `(409,509)`
- `(410,510)`

Use ten unseen model-initialization seeds:

- `1101..1110`

These are disjoint from the Phase 0.6 D4-B contexts/model seeds and from the protected confirmatory firewall.

Protected confirmatory seeds remain forbidden:

- cohort `701..710`
- subset `801..810`
- model `901..910`

Total planned cells:

`5 contexts × 10 model seeds × 2 flow variants × 2 optimization policies = 200 cells`.

The exact 200-cell plan must be materialized, validated, and hash-bound before the first official Phase 0.7 training cell runs.

## 6. Forced-horizon intervention semantics

`forced_horizon` changes exactly one termination behavior: patience exhaustion no longer causes the optimizer loop to break before epoch 100.

It must not change:

- architecture or parameter count;
- initialization;
- optimizer construction or optimizer state evolution before the standard-policy stopping point;
- learning rate, scheduler, losses, or regularization;
- data realization, subset, batch order, or stochastic inputs;
- input representation;
- train/validation/test split;
- production checkpoint scoring objective;
- shadow validation-MAE checkpoint computation;
- maximum epoch count of 100;
- test evaluation semantics.

The ordinary patience condition must still be evaluated under `forced_horizon`. The first epoch at which it would have terminated the standard run must be recorded as `would_patience_exhaust_epoch`. If the condition never fires by epoch 100, the value is null.

Under `forced_horizon`, production and shadow checkpoints are selected retrospectively over the complete 100-epoch trajectory using the same scoring rules used by the standard policy. This isolates trajectory truncation from checkpoint-objective choice.

## 7. Strong paired-trajectory invariant

For every matched `(context, model_seed, flow_variant)` triple, `standard_early_stop` and `forced_horizon` must be trajectory-identical through the epoch at which the standard run terminates.

Before that point, the matched policies must have identical:

- simulator realization and subset;
- model initialization;
- batch sequence and RNG-derived stochastic inputs;
- optimizer updates and optimizer state;
- training and validation losses;
- production-checkpoint state and selected-epoch history;
- shadow-MAE checkpoint state and selected-epoch history;
- diagnostic values.

The only permitted policy divergence is that `standard_early_stop` exits when patience is exhausted while `forced_horizon` records that event and continues training.

For every `(context, model_seed, optimization_policy)` pair, control and `time_scaled` use the same cohort/subset/model seeds.

Any cell using a protected confirmatory seed is invalid.

## 8. Cell-level quantities

For each of the 50 `(context, model_seed)` pairs, define:

`Delta_standard = MAE_control,standard - MAE_time_scaled,standard`

`Delta_forced = MAE_control,forced - MAE_time_scaled,forced`

Positive `Delta` favors `time_scaled`.

Define the architecture-by-policy interaction:

`G = Delta_forced - Delta_standard`.

Positive `G` means forced horizon shifts the relative comparison in favor of `time_scaled`.

For interpretability, also define architecture-specific policy effects:

`H_time_scaled = MAE_time_scaled,standard - MAE_time_scaled,forced`

`H_control = MAE_control,standard - MAE_control,forced`

so that:

`G = H_time_scaled - H_control`.

Positive `H` means that architecture obtains lower test MAE under forced horizon. These `H` quantities are secondary and cannot change the primary causal classification.

## 9. Primary causal estimand

Primary estimand A is the crossed-design mean architecture-by-policy interaction:

`mean(G)`.

The primary scientific claim is whether forced horizon causally increases the relative N=40 `time_scaled` contrast under this locked setting.

## 10. Prespecified heterogeneity estimand

For policy `p`, let the 5×10 matrix of paired architecture contrasts be `Delta_p(i,j)` for context `i` and model seed `j`.

Define the two-way residual exactly as:

`r_p(i,j) = Delta_p(i,j) - rowmean_p(i) - colmean_p(j) + grandmean_p`.

Use sample standard deviation with `ddof=1` over the 50 residuals.

Define:

`R_SD = SD(r_forced) / SD(r_standard)`.

Values `R_SD < 1` indicate reduced crossed-design residual heterogeneity under forced horizon.

If the standard-policy residual SD is zero or non-finite, `R_SD` is undefined and the heterogeneity-support gate fails. No epsilon, alternate denominator, or rescue computation is permitted.

## 11. Crossed bootstrap inference

Use exactly 10,000 crossed bootstrap resamples with seed `20260827`.

For each resample:

1. sample the five context indices with replacement;
2. sample the ten model-seed indices with replacement;
3. form the Cartesian 5×10 resampled matrix;
4. recompute `mean(G)` from that resampled matrix;
5. separately recompute row means, column means, grand means, and two-way residuals for `Delta_standard` and `Delta_forced` inside that resample;
6. recompute `R_SD` from those resampled residuals.

Report 95% percentile intervals for `mean(G)` and `R_SD`.

Also report:

- positive pair-level `G` values out of 50;
- positive context-mean `G` values out of 5;
- positive model-seed-mean `G` values out of 10.

## 12. Frozen adjudication structure

Phase 0.7 separates the causal architecture-by-policy effect from the heterogeneity-reduction claim so that one scientifically interpretable result is not erased by failure of the other.

### 12.1 Primary causal effect gate

The causal effect gate passes only if all four conditions hold:

1. observed `mean(G) > 0`;
2. lower 95% crossed-bootstrap bound for `mean(G) > 0`;
3. at least 4/5 context-mean `G` values are positive;
4. at least 8/10 model-seed-mean `G` values are positive.

### 12.2 Prespecified heterogeneity gate

The heterogeneity gate passes only if both conditions hold:

1. observed `R_SD < 1`;
2. upper 95% crossed-bootstrap bound for `R_SD < 1`.

### 12.3 Overall classification

Use exactly one of three classifications:

- `P07_OPTIMIZATION_HORIZON_EFFECT_AND_HETEROGENEITY_SUPPORTED` if both gates pass.
- `P07_OPTIMIZATION_HORIZON_EFFECT_ONLY` if the primary causal effect gate passes and the heterogeneity gate does not.
- `P07_OPTIMIZATION_HORIZON_NOT_ESTABLISHED` if the primary causal effect gate fails, regardless of the heterogeneity gate.

If the causal gate fails while the heterogeneity gate passes, the allowed statement is that forced horizon reduced residual heterogeneity under the locked setting but did not establish the prespecified positive architecture-by-policy effect. It is not classified as causal support for the architecture contrast.

There is no post-hoc rescue rule, alternate threshold, endpoint substitution, or threshold search.

## 13. Secondary mechanistic diagnostics

These are supportive only and cannot alter the frozen classification:

- fraction of standard control and `time_scaled` cells that stop before epoch 100;
- fraction of forced-horizon cells with non-null `would_patience_exhaust_epoch` before epoch 100;
- architecture-stratified `would_patience_exhaust_epoch` distributions;
- best validation MAE obtained after `would_patience_exhaust_epoch` relative to the best value available at that point;
- selected production-checkpoint epoch shift under forced versus standard policy;
- shadow-MAE checkpoint epoch shift under forced versus standard policy;
- `H_time_scaled` and `H_control` distributions and crossed summaries;
- association between candidate-side patience diagnostics and `Delta_standard`;
- corresponding association with `Delta_forced`.

The Phase 0.6 post-hoc gradient, parameter-norm, checkpoint-mismatch, and flow-displacement variables are not promoted to Phase 0.7 primary endpoints.

## 14. Interpretation rules

### Both gates pass

Allowed claim:

> Under unseen development contexts and model initializations in the locked N=40 smooth synthetic setting, preventing premature optimization termination causally increased the relative `time_scaled` effect and reduced crossed-design residual heterogeneity.

This supports proceeding to an optimization-equated architecture evaluation. It does not establish generalization across architectures, worlds, datasets, or the field.

### Causal effect gate passes; heterogeneity gate fails

Allowed claim:

> Under unseen development contexts and model initializations in the locked N=40 smooth synthetic setting, preventing premature optimization termination causally increased the relative `time_scaled` effect, but reduced crossed-design residual heterogeneity was not established.

This supports a causal architecture-by-policy interaction but does not support the stronger claim that premature termination explains the observed initialization heterogeneity.

### Causal effect gate fails

Allowed claim:

> The Phase 0.6 optimization-horizon association was diagnostically structured but did not prospectively establish the prespecified positive architecture-by-policy effect as a sufficient causal explanation under the locked Phase 0.7 intervention.

The specific stopping-horizon explanation must then be retired or reformulated rather than rescued post hoc.

## 15. Long-horizon branch after adjudication

Phase 0.7 is intentionally terminal for repeated diagnosis of this same 200-cell problem.

If causal support is established, the next architecture-specific step is an optimization-equated architecture evaluation. If it is not established, the specific `time_scaled` stopping-horizon explanation is retired or reformulated.

Both branches then feed the broader methodological program rather than another rescue cycle on the same setting:

1. formulate the general question of when optimization confounds low-data inductive-bias evaluation;
2. replicate prospectively across multiple architectures;
3. test across multiple synthetic data-generating worlds and low-data regimes;
4. characterize ranking stability and architecture-by-optimization interactions rather than only mean benchmark performance;
5. evaluate whether the phenomenon changes practical conclusions on real longitudinal datasets;
6. conduct a final prospective/frozen confirmation before any broad methodological claim.

The eventual target is a general evaluation methodology, not a claim that `time_scaled` itself generalizes across the field.

## 16. Firewall and non-claims

Phase 0.7 is development-only.

It must not:

- alter the frozen Phase 0.6 `AMBIGUOUS -> STOP` verdict;
- authorize D4-D or a broader factorial search;
- touch confirmatory seeds `701..710`, `801..810`, or `901..910`;
- tune success thresholds after results are observed;
- add mechanisms after official execution begins;
- reinterpret the post-Phase-0.6 retrospective analysis as confirmatory evidence;
- claim formal mediation;
- claim generalization beyond the locked smooth synthetic N=40 setting;
- claim a clinically relevant architecture advantage.

## 17. Pre-execution implementation requirements

Before any official Phase 0.7 cell may run, implementation must provide:

- a protocol lock cryptographically binding the exact frozen specification bytes by SHA-256;
- exact 200-cell planning validation;
- hard rejection of protected confirmatory seeds;
- tests proving matched standard/forced cells are trajectory-identical through the standard-policy stopping epoch;
- tests proving the only policy divergence is termination versus continuation;
- tests proving forced-horizon runs continue through epoch 100 while preserving `would_patience_exhaust_epoch`;
- deterministic tests for `Delta_standard`, `Delta_forced`, `G`, `H_time_scaled`, and `H_control`;
- deterministic two-way-residual and `R_SD` tests including zero/non-finite denominator behavior;
- deterministic crossed-bootstrap tests;
- deterministic tests for the three-way frozen adjudication;
- CPU smoke validation and CUDA execution support consistent with the existing harness;
- full CI green on the exact candidate execution SHA;
- an execution manifest/hash freeze before official launch.

## 18. Authorization boundaries

This specification is frozen and authorizes implementation work only.

Implementation must be performed on the Phase 0.7 branch through a pull request and must pass CI on the exact candidate head before merge. The frozen design itself does not authorize any official Phase 0.7 training.

After implementation, tests, CI, and execution-SHA freeze are complete, official execution of the 200 Phase 0.7 cells requires a separate explicit human authorization.

No official Phase 0.7 training is authorized by approval of the design, implementation plan, pull request, or merge.