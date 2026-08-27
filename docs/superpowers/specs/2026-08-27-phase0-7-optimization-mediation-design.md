# Phase 0.7 Prospective Optimization-Mediation Falsification Design

Status: **DRAFT — NOT AUTHORIZED FOR EXECUTION UNTIL REVIEWED AND FROZEN**

Date: 2026-08-27

## Scientific question

When `time_scaled` changes predictive performance at N=40, how much of the apparent architecture effect is caused by candidate-specific premature optimization termination rather than by the representation itself?

The Phase 0.6 D4-B terminal result remains `AMBIGUOUS -> STOP`. A retrospective analysis of the same 50 D4-B paired effects found strong crossed-design associations between `Delta_MAE` and selected epoch, stop epoch, shadow-MAE epoch, and patience exhaustion. That result is exploratory and cannot establish mediation. Phase 0.7 therefore performs a prospective intervention on optimization termination.

## Primary hypothesis

Let

`Delta_policy = MAE_control,policy - MAE_time_scaled,policy`

so positive values favor `time_scaled`.

For each unseen `(context, model_seed)` pair define the optimization-policy interaction:

`G = Delta_forced_horizon - Delta_standard_early_stop`.

Primary hypothesis:

> Preventing candidate-specific premature termination while preserving architecture, data, initialization, optimizer, maximum epoch budget, loss, and checkpoint-selection semantics will increase the relative `time_scaled` effect and reduce initialization-dependent heterogeneity.

This is a causal intervention on the training procedure, not another correlation analysis.

## Experimental scope

World: `smooth` only.

Training N: `40` only.

Flow variants:
- control: `none__none__deterministic`
- candidate: `time_scaled__none__deterministic`

Optimization policies:
1. `standard_early_stop`: exact D4-B stopping behavior.
2. `forced_horizon`: train through epoch 100 regardless of patience exhaustion; still record the epoch at which the ordinary patience rule would have fired, but do not terminate the run.

All other model, optimizer, loss, data, capacity, and evaluation semantics remain locked to the validated D4-B implementation unless this spec explicitly changes them.

## New development-only seeds

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

No other N, world, flow, jump, capacity, or selection-objective condition is authorized in Phase 0.7.

## Forced-horizon intervention semantics

`forced_horizon` changes one behavior only: patience exhaustion no longer stops optimization before epoch 100.

It must not change:
- model architecture or parameter count;
- initialization for a matched cell;
- optimizer, learning rate, scheduler, losses, or regularization;
- input representation;
- train/validation/test split;
- production checkpoint scoring objective;
- shadow validation-MAE checkpoint computation;
- maximum epoch count of 100;
- test evaluation semantics.

The production checkpoint and shadow-MAE checkpoint are both selected retrospectively from the full 100-epoch trajectory under `forced_horizon`. This isolates trajectory truncation from checkpoint-objective choice.

The ordinary patience rule must still be evaluated and logged as a counterfactual diagnostic (`would_patience_exhaust_epoch`) so that later analysis can verify that the intervention actually affected the runs expected to be truncated under the standard policy.

## Pairing and execution invariants

For every `(context, model_seed, flow_variant)` triple, the two optimization-policy cells must share the same simulator realization, subset, model initialization, and all stochastic inputs except the stopping-policy intervention.

For every `(context, model_seed, optimization_policy)` pair, control and `time_scaled` must share the same cohort/subset/model seeds.

Cells are invalid if any protected confirmatory seed appears.

The complete 200-cell plan must be materialized and hash-bound before the first official Phase 0.7 training cell runs.

## Primary estimands

For each of the 50 `(context, model_seed)` pairs:

1. `Delta_standard = MAE_control,standard - MAE_time_scaled,standard`
2. `Delta_forced = MAE_control,forced - MAE_time_scaled,forced`
3. `G = Delta_forced - Delta_standard`

Primary estimand A — mean policy interaction:

`mean(G)`.

Primary estimand B — residual heterogeneity ratio:

1. two-way residualize `Delta_standard` by context and model-seed main effects;
2. two-way residualize `Delta_forced` identically;
3. compute `R_SD = SD(residual_forced) / SD(residual_standard)`.

Values `G > 0` support an upward shift in the relative `time_scaled` effect under forced horizon. Values `R_SD < 1` support reduced crossed-design heterogeneity.

## Primary inference

Use a fixed crossed context/model bootstrap with 10,000 resamples and seed `20260827`.

For each resample:
- sample the five context indices with replacement;
- sample the ten model-seed indices with replacement;
- take their Cartesian product from the 5×10 paired matrix;
- recompute `mean(G)`;
- recompute the two-way-residualized `R_SD`.

Report 95% percentile intervals for both primary estimands.

Also report:
- number of positive pair-level `G` values out of 50;
- number of positive context-mean `G` values out of 5;
- number of positive model-seed-mean `G` values out of 10.

## Frozen success rule

Classify `P07_OPTIMIZATION_MEDIATION_SUPPORTED` only if all conditions hold:

1. observed `mean(G) > 0`;
2. lower 95% crossed-bootstrap bound for `mean(G) > 0`;
3. observed `R_SD < 1`;
4. upper 95% crossed-bootstrap bound for `R_SD < 1`;
5. at least 4/5 context-mean `G` values are positive;
6. at least 8/10 model-seed-mean `G` values are positive.

If any condition fails, classify `P07_OPTIMIZATION_MEDIATION_NOT_ESTABLISHED`.

There is no post-hoc rescue rule and no alternate threshold search within Phase 0.7.

## Secondary mechanistic diagnostics

These are supportive only and cannot change the primary classification:

- fraction of standard `time_scaled` cells that stop before epoch 100;
- fraction of forced-horizon `time_scaled` cells whose logged `would_patience_exhaust_epoch` is before epoch 100;
- recovery after the counterfactual patience point: best validation MAE after `would_patience_exhaust_epoch` minus best validation MAE available at that point;
- shift in selected checkpoint epoch under forced horizon versus standard;
- shift in shadow-MAE checkpoint epoch under forced horizon versus standard;
- association between candidate-side patience diagnostics and `Delta_standard`;
- corresponding association with `Delta_forced`, expected to weaken if truncation is causal.

The Phase 0.6 post-hoc secondary gradient/parameter/flow-displacement variables are not promoted to Phase 0.7 primary endpoints.

## Interpretation rules

If supported, the allowed claim is:

> Under unseen development contexts and model initializations, preventing premature optimization termination causally increased the relative N=40 `time_scaled` effect and reduced seed-dependent heterogeneity under the locked synthetic training formulation.

This still does not establish a clinically relevant advantage, generalization beyond the smooth synthetic world, or a confirmatory architecture win.

If not established, the allowed claim is:

> The Phase 0.6 optimization-horizon association was diagnostically structured but did not prospectively survive direct intervention as a sufficient causal explanation of the `time_scaled` effect heterogeneity.

## Firewall and non-claims

Phase 0.7 is development-only.

It must not:
- alter the frozen Phase 0.6 `AMBIGUOUS -> STOP` verdict;
- authorize D4-D/full factorial work;
- touch confirmatory seeds `701..710`, `801..810`, or `901..910`;
- tune success thresholds after observing Phase 0.7 results;
- add new mechanisms after execution begins;
- interpret a post-hoc association as causal before the prospective intervention completes.

## Authorization boundary

Before execution, the implementation must provide:
- a hash-bound Phase 0.7 protocol lock containing this spec hash;
- exact 200-cell planning validation;
- tests proving only the stopping-policy behavior differs between matched standard and forced-horizon cells;
- tests proving forced-horizon runs continue to epoch 100 while preserving the ordinary patience diagnostic;
- deterministic analysis/adjudication tests for the six-part success rule;
- a CPU smoke path and CUDA execution path consistent with the existing harness.

**No official Phase 0.7 training is authorized until this written design has been reviewed and explicitly frozen.**
