# Phase 0.6 D4-B Optimization/Initialization Stability Execution Addendum

**Date:** 2026-08-27  
**Status:** Frozen before D4-B implementation/execution  
**Branch:** `phase0-6-d4b-work`  
**Base execution head:** `516c9e3c0e965582fa5cce976e9d8ebf32ea8404`  
**Validated D2-B implementation SHA:** `cb61c3eb3f39a00980033e6b61108a0834e5bfca`  
**Parent D2-B decision:** `next_required_stage = D4_OPTIMIZATION`

## 1. Purpose

D4-B is the smallest targeted falsification experiment justified by the completed Phase 0.6 D2-A/D2-B evidence.

The question is:

> Is the N=40 predictive advantage of `time_scaled` robust to model initialization/optimization variation when cohort and subset conditions are held fixed?

D2-A and D2-B independently identified model initialization as the unique dominant named factor at N=40 and no named dominant factor at N=5. Their cross-array adjudication was sufficient and selected `D4_OPTIMIZATION` as the next required stage.

D4-B does not test capacity or temporal semantics. It does not change the optimizer, learning rate, weight decay, patience, architecture, checkpoint objective, data-generation semantics, or Phase 0.5 training objective. Those are deliberately held fixed so model-seed stability can be falsified in isolation.

A negative, fragile, or ambiguous D4-B result is a valid stopping result.

## 2. Immutable parent evidence

D4-B is a child diagnostic stage. It must never rewrite the completed D1/D2-A/D3 parent store or the completed D2-B child store.

The D4-B implementation must bind to the completed D2-B child execution whose protocol lock reports:

```text
execution_commit = 516c9e3c0e965582fa5cce976e9d8ebf32ea8404
parent_execution_sha = 1718402df1d6ef344168677e6d26ea664708e1bc
parent_protocol_lock_sha256 = c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51
parent_d3_sha256 = 6b9fffed7503fae6beeac2314238ae3d10ffdebfd27ea71ad952ab1f87916460
parent_d3_next_required_stage = D2B
d2b_mapping = model_index=(cohort_index+2*subset_index)%5
```

The completed D2-B cross-array adjudication must semantically match all of the following:

```text
complementary_array_evidence = sufficient
dominant_factors.d2a.40 = model
dominant_factors.d2b.40 = model
dominant_factors.d2a.5 = none
dominant_factors.d2b.5 = none
next_required_stage = D4_OPTIMIZATION
remaining_parent_escalations = [D4_OPTIMIZATION, D4_CAPACITY_TIME]
```

Using canonical JSON serialization (`sort_keys=True`, separators `(',', ':')`, `allow_nan=False`), the completed D2-B adjudication has SHA-256:

```text
ea28fd4d5f6490a10fad20d5d3f3e76a1de08bf6b1be3b9805c9cf6c519e845f
```

Before any D4-B child store is created, the implementation must:

1. open the D2-B parent as a hash-bound `Phase06Store`;
2. validate its protocol identity and frozen D1/D2-A/D3 linkage;
3. require exact completion of all 100 D2-B cells and validate their persisted bundles;
4. recompute/load the D2-B analysis required for cross-array adjudication;
5. verify the persisted D2-B adjudication against the frozen canonical hash and semantic decision above;
6. reject any parent evidence whose bytes, hashes, stage completeness, or decision no longer match the frozen chain.

## 3. Scientific scope

### 3.1 Fixed cohort/subset contexts

D4-B reuses five already exposed development contexts:

```text
(401, 501)
(402, 502)
(403, 503)
(404, 504)
(405, 505)
```

The cohort and subset seeds are fixed across the new model-seed bank. Their original model seeds are not reused as D4-B model seeds.

### 3.2 New diagnostic model-seed bank

D4-B uses the dedicated model-seed namespace:

```text
1001..1010
```

These ten seeds are diagnostic-only. They are distinct from:

```text
D1/D2 model seeds: 601..605
reserved Phase 0.5 confirmatory model seeds: 901..910
```

The existing confirmatory firewall remains absolute. No Phase 0.6 code may plan or execute model seeds `901..910`, cohort seeds `701..710`, or subset seeds `801..810`.

### 3.3 Run matrix

The D4-B matrix is fixed as:

```text
stage: d4b
world: smooth
N: 40
flow: none, time_scaled
jump: none
uncertainty: deterministic
cohort/subset contexts: 5
model seeds: 10
```

Total execution count:

```text
5 contexts x 10 model seeds x 2 flow modes = 100 cells
```

No N=5 cells are run in D4-B. The purpose is specifically to falsify the observed N=40 model-initialization instability.

## 4. Training and execution invariants

Each D4-B cell must use the existing Phase 0.5/Phase 0.6 training path without changing:

```text
optimizer
learning rate
weight decay
patience
epoch limit
training objective
checkpoint objective
architecture
representation/state dimensions
time_scale_days
data generation
train/validation/test splitting
low-N budget selection semantics
```

Only `model_seed` varies within a fixed cohort/subset context. `none` and `time_scaled` for the same context/model seed form a paired comparison.

Official D4-B execution is CUDA-only, one worker, and must record the same environment/provenance metadata required by prior Phase 0.6 CUDA stages.

## 5. Primary estimand

For each fixed context `c` and model seed `m`:

```text
Delta_MAE(c,m) = MAE_none(c,m) - MAE_time_scaled(c,m)
```

Positive values favor `time_scaled`.

There are exactly 50 paired effects.

Define the model-seed mean effect:

```text
Delta_model(m) = mean_c Delta_MAE(c,m)
```

and context mean effect:

```text
Delta_context(c) = mean_m Delta_MAE(c,m)
```

The overall effect is:

```text
Delta_overall = mean_(c,m) Delta_MAE(c,m)
```

## 6. Locked bootstrap

D4-B uses a model-seed-cluster bootstrap because model initialization is the factor being stress-tested.

Settings:

```text
bootstrap_resamples = 10000
bootstrap_seed = 20260827
confidence_interval = percentile 95%
```

For each bootstrap resample:

1. sample 10 model seeds with replacement from the frozen seed bank;
2. include all five fixed cohort/subset contexts for each sampled model seed;
3. compute the grand mean paired `Delta_MAE` over the resulting 50 context/seed effects;
4. record the bootstrap mean.

The 2.5th and 97.5th percentiles form the locked 95% interval.

No post-hoc alternative bootstrap unit, confidence level, or seed may replace this primary analysis.

## 7. D4-B stability classification

The classification is based only on the frozen paired MAE effects.

Let:

```text
positive_model_seed_count = number of m with Delta_model(m) > 0
positive_context_count = number of c with Delta_context(c) > 0
bootstrap_ci_lower = lower endpoint of the locked 95% cluster-bootstrap interval
```

### `stable`

D4-B is classified `stable` only if all four conditions hold:

```text
Delta_overall > 0
positive_model_seed_count >= 8/10
positive_context_count >= 4/5
bootstrap_ci_lower > 0
```

### `fragile`

D4-B is classified `fragile` if any of the following holds:

```text
Delta_overall <= 0
positive_model_seed_count <= 5/10
positive_context_count <= 2/5
```

### `ambiguous`

Every other outcome is `ambiguous`.

The thresholds are frozen before D4-B implementation and may not be tuned after seeing D4-B outputs.

## 8. Descriptive optimization diagnostics

The primary scientific gate is Section 7. The following diagnostics are descriptive and may explain instability but may not alter the classification threshold.

For each D4-B cell, persist the existing Phase 0.6 training trace and summary, including:

```text
selected production-checkpoint epoch
shadow validation-MAE checkpoint epoch
stop epoch
early-stop reason
per-epoch gradient L2 norm
per-epoch parameter L2 norm
per-epoch mean/median/p95 flow displacement
validation core-loss trajectory
validation MAE trajectory
```

Produce a D4-B optimization-dispersion summary grouped by fixed context and flow mode. Across the ten model seeds report, at minimum:

```text
selected_checkpoint_epoch: mean, std, IQR
stop_epoch: mean, std, IQR
per-cell mean gradient_l2_norm: mean, std, IQR
per-cell mean flow_displacement: mean, std, IQR
early_stop_reason counts
```

These quantities are explanatory only. No post-hoc numerical tolerance on them becomes a scientific gate.

## 9. Required D4-B artifacts

Execution must persist at least:

```text
phase06_d4b_effects.csv
phase06_d4b_model_seed_summary.csv
phase06_d4b_context_summary.csv
phase06_d4b_bootstrap_diagnostics.json
phase06_d4b_optimization_dispersion.csv
```

The effect table contains exactly 50 paired rows and includes:

```text
cohort_seed
subset_seed
model_seed
n_train
control_mae
time_scaled_mae
Delta_MAE
control_selected_epoch
time_scaled_selected_epoch
control_stop_epoch
time_scaled_stop_epoch
```

The model-seed summary contains exactly 10 rows. The context summary contains exactly 5 rows.

Execution itself must not write the final D4-B adjudication artifact.

## 10. Explicit adjudication and next state

Cross-stage continuation is a separate explicit command after D4-B execution artifacts are audited.

The canonical adjudication artifact is:

```text
phase06_d4b_adjudication.json
```

It records at least:

```text
classification
Delta_overall
positive_model_seed_count
positive_context_count
bootstrap_ci_lower
bootstrap_ci_upper
next_required_stage
rationale
input_artifact_hashes
```

The next state is frozen as:

```text
stable    -> D4_CAPACITY_TIME
fragile   -> STOP
ambiguous -> STOP
```

`STOP` means Phase 0.6 does not execute another intervention from this chain. An ambiguous result is not permission to enlarge the model-seed bank post hoc.

`D4_CAPACITY_TIME` is only a route token. This addendum does not implement or authorize the D4-D capacity/time experiment.

## 11. Child provenance boundary

D4-B must execute in a new child output root distinct from and non-nested with every parent evidence root.

The D4-B child protocol lock must bind at least:

```text
child execution SHA
Phase 0.6 config/spec identities
this D4-B addendum SHA-256
Phase 0.5 config/protocol identity
D2-B parent execution SHA
D2-B parent protocol identity
canonical D2-B adjudication SHA-256
D2-B next_required_stage = D4_OPTIMIZATION
fixed cohort/subset contexts
D4-B model-seed bank 1001..1010
N = 40
flow modes = none,time_scaled
bootstrap_resamples = 10000
bootstrap_seed = 20260827
forbidden seed sets
```

A resume must revalidate the entire parent chain and reuse only a child store whose protocol identity exactly matches the current D4-B execution SHA and frozen addendum.

## 12. CLI and tooling boundary

The D4-B implementation may expose only the following new Phase 0.6 actions:

```text
d4b
adjudicate-d4b
```

`d4b` is CUDA-only in the public CLI.

The execution launcher must support fresh/resume D4-B execution with:

```text
separate child output
immutable parent binding
clean tracked worktree check
validated implementation SHA check
CUDA preflight
tmux runner/monitor support
100-cell completion accounting
hard stop after analysis
```

The launcher must not run `adjudicate-d4b` automatically.

## 13. Hard boundaries

This addendum does not authorize or implement:

```text
D4 checkpoint-policy intervention
D4 cohort/subset expansion
D4 capacity-matched control
D4 time-shuffled control
D4 mechanism redesign
full 5x5x5 seed factorial
Phase 0.5 continuation
reserved confirmatory seeds
```

No D4-B result may retroactively change the archived Phase 0.5 negative gate result.

A `stable` D4-B result means only that the N=40 predictive effect survives this frozen initialization/optimization stress test. It is not evidence by itself that temporal semantics cause the effect or that latent-mechanism recovery improved.

## 14. Implementation acceptance criteria

Before D4-B CUDA execution may be authorized, implementation validation must demonstrate:

1. the planner produces exactly 100 unique D4-B cells and no other stage/matrix;
2. only the five frozen cohort/subset contexts and model seeds `1001..1010` are reachable;
3. all reserved confirmatory seeds remain rejected;
4. parent D2-B completion and adjudication are cryptographically and structurally validated before child creation;
5. parent and child roots reject equality and nesting in either direction;
6. the public D4-B CLI is CUDA-only;
7. fresh/resume semantics preserve hash-bound child-store integrity;
8. the primary analysis reproduces the frozen 50 paired effects, summaries, bootstrap, and stability thresholds exactly;
9. execution and adjudication remain separate;
10. no D4-D, full-factorial, Phase 0.5 continuation, or confirmatory execution path is exposed;
11. Ruff and the full pytest suite pass;
12. an independent execution-readiness review reports no remaining Critical or Important blocker before the implementation SHA is frozen.

Only after those gates pass may a separate D4-B readiness record authorize local CUDA execution.