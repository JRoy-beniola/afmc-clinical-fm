# Phase 0.6 Diagnostic Program Design

**Date:** 2026-08-26  
**Status:** Approved architecture; implementation not started  
**Branch:** `phase0-6-diagnostics`  
**Base:** `main` at `63e478d3bd2b26348d9d32ef5921c7f8db1b8c6c`  
**Predecessor:** Phase 0.5 official Stage I-A negative result

## 1. Purpose

Phase 0.6 is a diagnostic program for explaining the Phase 0.5 Stage I-A result. It is not a continuation of the failed Phase 0.5 development sequence, not a replacement confirmatory study, and not an architecture search.

The Phase 0.5 official decision is immutable:

> **PHASE-0.5 TERMINATED AT STAGE I-A — FLOW MECHANISM GATE NOT ESTABLISHED**

Phase 0.5 completed the 60 preregistered smooth-world flow-isolation cells and stopped at the locked flow gate because neither `gated` nor `time_scaled` established the required development effect. Later Stage I mechanisms and all confirmatory stages were not entered.

The postmortem nevertheless exposed a diagnostic pattern worth explaining: `time_scaled` was negative at very low N, approximately neutral around N=20, and showed a positive predictive MAE effect at N=40 while latent-state recovery did not improve. Phase 0.6 exists to determine whether that pattern is reproducible and, if so, whether it is explained by optimization, checkpoint selection, seed composition, sample complexity, model capacity, or a predictive/mechanistic disconnect.

## 2. Immutable scientific boundaries

Phase 0.6 MUST preserve all of the following boundaries.

1. The Phase 0.5 official gate result remains failed regardless of any Phase 0.6 result.
2. Phase 0.6 evidence is diagnostic/exploratory and MUST NOT be relabeled as Phase 0.5 confirmatory evidence.
3. The Phase 0.5 official output, reports, manifests, and archived provenance MUST NOT be rewritten.
4. Phase 0.6 MUST NOT resume the Phase 0.5 `jump`, `uncertainty`, `timing_audit`, `confirmation`, or `robustness` sequence.
5. Phase 0.6 MUST NOT inspect, execute, recombine, or otherwise use the reserved confirmatory seed bundles `(701..710, 801..810, 901..910)`.
6. No D4 intervention may retroactively convert Phase 0.5 into a positive result.
7. A negative Phase 0.6 conclusion is a valid terminal result.

## 3. Goals and non-goals

### 3.1 Goals

Phase 0.6 will:

- add non-invasive training instrumentation to the exact Phase 0.5 training path;
- verify that instrumentation does not alter training semantics;
- reproduce the original 40-cell `none` versus `time_scaled` smooth-world comparison with the original development bundles;
- separate cohort, subset, and model-initialization variance using an orthogonal seed design;
- apply predeclared adjudication rules to competing explanations;
- permit only the smallest targeted falsification experiment justified by the diagnostic evidence;
- preserve machine-readable traces, summaries, provenance, and integrity hashes sufficient for later audit.

### 3.2 Non-goals

Phase 0.6 will not:

- optimize a new architecture before the diagnostic chain justifies one;
- change the Phase 0.5 training objective, optimizer, learning rate, weight decay, patience, or production checkpoint rule during D0-D3;
- use test performance to select checkpoints;
- tune D1/D2 decision rules after seeing their outputs;
- run the Phase 0.5 confirmatory seed bank;
- claim latent-mechanism recovery from predictive MAE alone;
- expand to a full 5x5x5 seed factorial unless the lower-cost orthogonal design remains ambiguous.

## 4. Diagnostic hypotheses

Phase 0.6 carries seven explicit hypotheses.

| ID | Hypothesis | Primary diagnostic evidence |
| --- | --- | --- |
| H1 | Model initialization drives the observed instability | model-seed main effect; trajectory dispersion |
| H2 | Low-N subset composition drives the observed instability | subset-seed main effect |
| H3 | Synthetic cohort realization drives the observed instability | cohort-seed main effect |
| H4 | `time_scaled` exhibits a genuine N-dependent sample-complexity transition | reproducible sign change and positive paired N=40 minus N=5 shift |
| H5 | The composite validation checkpoint objective masks predictive behavior | production-versus-shadow checkpoint divergence |
| H6 | Extra trainable capacity, rather than temporal semantics, explains the crossover | capacity-matched and time-destroyed controls in D4 |
| H7 | Predictive improvement is disconnected from latent-mechanism recovery | positive predictive effect with non-positive latent-R2 effect |

No single Phase 0.6 stage is expected to resolve every hypothesis.

## 5. Program structure

Phase 0.6 proceeds in order:

```text
D0  non-invasive instrumentation
    |
D1  exact 40-cell phenomenon reproduction
    |
D2  orthogonal seed variance decomposition
    |
D3  predeclared hypothesis adjudication
    |
D4  targeted falsification only
    |
    +-- stop with diagnosis
    `-- justify a separate redesigned mechanism study
```

D0-D3 are diagnostic. D4 is conditional and may not run if the earlier evidence provides a sufficient stopping conclusion.

---

# 6. D0 — non-invasive instrumentation

## 6.1 Architectural choice

D0 uses **Option A: opt-in instrumentation in the existing Phase 0.5 trainer**.

The exact training path that produced Phase 0.5 remains authoritative. The existing `fit_phase05_model()` / `_fit_core()` optimization and checkpoint-selection semantics are retained. Diagnostics are an optional observer. The default call path remains diagnostics-free.

Conceptually:

```python
fit_phase05_model(..., diagnostics=None)
```

and diagnostic execution uses:

```python
fit_phase05_model(..., diagnostics=recorder)
```

The observer MUST NOT own optimization, mutate the model, mutate tensors used by training, call `backward()`, call `optimizer.step()`, change early stopping, or choose the returned checkpoint.

## 6.2 Non-interference invariant

For a fixed deterministic CPU case:

```text
diagnostics OFF == diagnostics ON
```

with respect to:

- parameter initialization;
- optimizer parameter set;
- forward/loss computation used for training;
- gradient update ordering;
- optimizer hyperparameters;
- validation core-loss sequence;
- stale-epoch and patience behavior;
- selected production checkpoint epoch;
- final returned `state_dict`;
- final evaluation metrics.

This invariant is a hard implementation gate. If the equivalence test fails, Phase 0.6 execution MUST NOT begin.

CUDA reproduction is not required to be bitwise identical to a historical run because GPU execution may contain nondeterministic kernels. CUDA execution MUST instead be fully provenance-bound and interpreted against rerun variability.

## 6.3 Observer boundary

The trainer may know that observations exist, but it MUST NOT know about Phase 0.6 output directories, CSV files, manifests, or analysis logic.

The interface is intentionally narrow:

```text
TrainingDiagnosticObserver
    on_epoch(record)
    on_training_end(summary)
```

The trainer emits immutable/plain records. Phase 0.6 code owns persistence.

A later implementation may use a `Protocol`, callback interface, or equivalent typed abstraction, but the dependency direction MUST remain:

```text
phase05 training -> generic observation contract
phase06 diagnostics -> concrete recorder/persistence
```

The Phase 0.5 trainer MUST NOT import Phase 0.6 execution or persistence code.

## 6.4 Epoch timing and recorded quantities

Epoch numbering is 1-based in persisted artifacts.

The existing update order remains:

1. training forward;
2. training core loss;
3. `backward()`;
4. observe gradient norm;
5. `optimizer.step()`;
6. observe post-step parameter norm;
7. existing validation core-loss computation;
8. optional no-grad diagnostic validation pass;
9. existing checkpoint/early-stopping decision;
10. emit the epoch record.

The normal validation core-loss call MUST remain the value used by production selection. The optional diagnostic validation pass MUST NOT replace it.

Each epoch record contains:

```text
epoch
train_core_loss
validation_core_loss
validation_mae
validation_rmse
gradient_l2_norm
parameter_l2_norm
mean_flow_displacement
median_flow_displacement
p95_flow_displacement
best_validation_core_loss_so_far
best_core_epoch
shadow_best_validation_mae_so_far
shadow_best_mae_epoch
stale_epochs
```

`train_core_loss` and `validation_core_loss` refer to the existing Phase 0.5 `_core_loss`, including the configured event-loss contribution.

The gradient norm is the global L2 norm over the exact core parameter set optimized by `_fit_core()`, measured after `backward()` and before `optimizer.step()`.

The parameter norm is the global L2 norm over that same core parameter set after `optimizer.step()`.

## 6.5 Flow displacement

Flow displacement measures the actual magnitude of the pre-assimilation state transformation already present in the model output.

For patient/time position `(i,t)`:

```text
previous_state(i,0) = 0
previous_state(i,t) = post_event_state(i,t-1), t > 0
flow_displacement(i,t) = ||pre_event_state(i,t) - previous_state(i,t)||_2
```

Only valid sequence positions contribute to the aggregate. The observer records mean, median, and 95th percentile displacement.

For `flow_mode="none"`, displacement is expected to be exactly zero up to floating-point identity. A non-zero result for `none` is treated as a diagnostic invariant failure.

No per-patient state trajectory is persisted during D0.

## 6.6 Production and shadow checkpoints

The production checkpoint remains:

```text
argmin epoch validation_core_loss
```

The shadow checkpoint is observational:

```text
argmin epoch validation_mae
```

The shadow checkpoint MUST NOT influence stale-epoch counting, early stopping, returned model state, D1 final metrics, or any Phase 0.5 decision.

To make a later D4 checkpoint falsification possible without reconstructing a potentially different CUDA trajectory, diagnostic runs may retain exactly two compact checkpoint snapshots per cell:

- the production-best state;
- the shadow-MAE-best state.

This is a narrow exception to the no-tensor-logging rule. Phase 0.6 MUST NOT persist per-epoch parameter tensors, per-parameter gradients, activations, optimizer states, or full patient-state trajectories. The two snapshots are deferred diagnostic evidence and MUST NOT be evaluated comparatively during D1 unless D3 later selects D4-A.

## 6.7 Training summary

At training end the recorder emits:

```text
epochs_run
stop_epoch
selected_checkpoint_epoch
selected_validation_core_loss
shadow_mae_checkpoint_epoch
shadow_validation_mae
early_stop_reason
```

`early_stop_reason` is one of:

```text
patience_exhausted
max_epochs_reached
```

Any non-finite training/validation loss or norm is an execution failure, not a scientific result.

---

# 7. D1 — exact phenomenon reproduction

## 7.1 Scope

D1 reproduces only the smooth-world comparison relevant to the Phase 0.5 postmortem.

Locked matrix:

```text
world: smooth
flow modes: none, time_scaled
jump mode: none
uncertainty mode: deterministic
N: 5, 10, 20, 40
bundles:
  (401, 501, 601)
  (402, 502, 602)
  (403, 503, 603)
  (404, 504, 604)
  (405, 505, 605)
```

Total:

```text
5 bundles x 4 N x 2 flow modes = 40 cells
```

`gated` is excluded because D1 is specifically testing the `time_scaled` crossover diagnostic.

D1 MUST reuse the Phase 0.5 training hyperparameters and data-generation semantics. Phase 0.6 MUST not silently duplicate and drift those values; the runtime must bind to the Phase 0.5 config/protocol identity and record the exact hashes used.

## 7.2 D1-A — execution fidelity

Before interpreting the 40-cell result:

- diagnostics-on/off CPU equivalence MUST pass;
- all 40 planned cells MUST complete successfully;
- each cell MUST have final metrics, trace, summary, and provenance binding;
- no confirmatory seed MUST appear anywhere in the plan or output;
- no NaN/Inf may appear in required diagnostics or final metrics.

The intended diagnostic reproduction environment is CUDA with one worker, matching the official Stage I-A execution topology as closely as practical. Runtime metadata MUST record device, CUDA/PyTorch/Python versions, OS/WSL context, execution SHA, config hashes, and wall time.

## 7.3 Effect orientation

For MAE, define the paired effect:

```text
Delta_MAE(b,N) = MAE_none(b,N) - MAE_time_scaled(b,N)
```

Positive favors `time_scaled`.

For aligned latent R2, define:

```text
Delta_R2(b,N) = R2_time_scaled(b,N) - R2_none(b,N)
```

Positive favors `time_scaled`.

## 7.4 D1-B — phenomenon reproduction criterion

The historical qualitative shape is called **reproduced** only if all of the following hold:

```text
mean Delta_MAE(N=5)  <= 0
mean Delta_MAE(N=10) <= 0
mean Delta_MAE(N=20) <= 0
mean Delta_MAE(N=40) > 0
N=40 time_scaled wins >= 4/5 bundles
```

This is a diagnostic shape criterion, not a new efficacy gate and not a replacement for the Phase 0.5 nAULC gate.

Classification:

- `reproduced`: all five conditions above hold;
- `not_reproduced`: mean N=40 effect is non-positive OR N=40 wins are <= 2/5;
- `ambiguous`: every other outcome.

The D1 rerun may report its own nAULC descriptively, but no D1 nAULC result may alter the archived Phase 0.5 decision.

## 7.5 D1 artifacts

Each cell persists:

```text
final cell metric record
training trace
training summary
production checkpoint snapshot
shadow-MAE checkpoint snapshot
hash bindings for all of the above
```

A consolidated table `phase06_d1_reproduction.csv` contains at minimum:

```text
bundle
n_train
control_mae
time_scaled_mae
Delta_MAE
winner
control_selected_epoch
time_scaled_selected_epoch
control_shadow_mae_epoch
time_scaled_shadow_mae_epoch
control_stop_epoch
time_scaled_stop_epoch
```

Additional diagnostic columns may be added, but the locked fields may not be removed or redefined.

D1 MUST NOT evaluate the shadow checkpoint on the test split for comparative inference. Shadow checkpoint evaluation is reserved for conditional D4-A.

---

# 8. D2 — orthogonal seed variance decomposition

## 8.1 Motivation

The five Phase 0.5 development triples couple cohort, subset, and model seeds. A bundle-level win/loss cannot reveal which source of randomness drove the result.

D2 recombines only the already exposed development seed levels:

```text
cohort: 401..405
subset: 501..505
model: 601..605
```

Reserved confirmatory seed levels remain forbidden.

## 8.2 D2-A orthogonal array

Let `i,j in {0,1,2,3,4}` index cohort and subset levels. Assign model level:

```text
k = (i + j) mod 5
```

The 25 combinations are:

```text
                 subset
             501  502  503  504  505
cohort 401    601  602  603  604  605
       402    602  603  604  605  601
       403    603  604  605  601  602
       404    604  605  601  602  603
       405    605  601  602  603  604
```

This is an `OA(25,3,5,2)` design: every factor level occurs five times and every pair of factor levels occurs exactly once.

D2-A runs only the two endpoints of the apparent regime change:

```text
N: 5, 40
flow: none, time_scaled
jump: none
uncertainty: deterministic
world: smooth
```

Total:

```text
25 seed combinations x 2 N x 2 flow modes = 100 cells
```

## 8.3 Additive decomposition

For each N:

```text
Delta_ijk = MAE_none_ijk - MAE_time_scaled_ijk
```

Fit the diagnostic additive model:

```text
Delta_ijk = mu + C_i + S_j + M_k + epsilon_ijk
```

Report:

- factor-level mean effects;
- main-effect sums of squares;
- fraction of total variation assigned to cohort, subset, model, and residual;
- residual distribution;
- fixed-seed bootstrap uncertainty.

Use 10,000 bootstrap resamples with a Phase 0.6-fixed bootstrap seed. Each bootstrap resample refits the additive decomposition.

A factor is called **diagnostically dominant** only if:

1. it has the largest main-effect variance share;
2. its share is at least 2x the next-largest main-effect share; and
3. it is the largest main-effect component in at least 80% of bootstrap resamples.

A factor is called **diagnostically weak** only if it is the smallest main-effect component and is the largest component in no more than 20% of bootstrap resamples. Otherwise it remains unresolved.

## 8.4 Interaction limitation and D2-B escalation

D2-A estimates main effects efficiently but does not separately identify arbitrary two-way or three-way interactions; those effects are aliased.

If the residual dominates, factor rankings are unstable, or D3 cannot adjudicate because of interaction ambiguity, run the complementary 25-combination array:

```text
k = (i + 2j) mod 5
```

This is D2-B.

Only if D2-A plus D2-B remain insufficient may Phase 0.6 escalate to the full 5x5x5 development-seed factorial. Such an escalation requires a separately frozen execution addendum before compute begins.

---

# 9. D3 — predeclared hypothesis adjudication

D3 maps diagnostic signatures to hypothesis states. It MUST NOT select a preferred narrative by visual inspection alone.

The canonical output is:

```text
phase06_d3_adjudication.json
```

## 9.1 Gate 1 — phenomenon reproduction

Use the D1 classification from Section 7.4:

```text
reproduced | not_reproduced | ambiguous
```

If D1 is `not_reproduced`, Phase 0.6 MUST NOT claim a genuine sample-complexity transition.

## 9.2 Gate 2 — checkpoint-objective mismatch

For each cell define:

```text
e_core = production-selected epoch
e_MAE  = shadow best-validation-MAE epoch
G_ckpt = validation_MAE(e_core) - validation_MAE(e_MAE)
```

Positive `G_ckpt` is the predictive validation cost of the production checkpoint rule.

For each original D1 bundle at N=40 define:

```text
D_ckpt(bundle) = G_ckpt_time_scaled - G_ckpt_none
```

H5 is:

- `strengthened` if mean `D_ckpt > 0`, at least 4/5 bundles have `D_ckpt > 0`, and `time_scaled` itself has `G_ckpt > 0` in at least 4/5 bundles;
- `weakened` if mean `D_ckpt <= 0` and no more than 2/5 bundles have `D_ckpt > 0`;
- `unresolved` otherwise.

Epoch distance is reported descriptively but is not itself a mismatch criterion.

## 9.3 Gate 3 — seed-source dominance

For H1/H2/H3:

- a factor satisfying the D2 diagnostic-dominance rule is `strengthened`;
- a factor satisfying the D2 diagnostic-weak rule is `weakened`;
- otherwise it is `unresolved`.

Large unexplained residual variance is not assigned to a named factor. It triggers interaction ambiguity and may require D2-B.

## 9.4 Gate 4 — N-dependent regime change

For each of the 25 D2-A combinations define:

```text
T_ij = Delta_MAE_ij,N40 - Delta_MAE_ij,N5
```

H4 is `strong` only if:

1. D1 is `reproduced`;
2. mean D2 `Delta_MAE(N=5) <= 0`;
3. mean D2 `Delta_MAE(N=40) > 0`;
4. mean `T > 0`;
5. at least 20/25 `T_ij` values are positive; and
6. the fixed-seed bootstrap 95% interval for mean `T` lies wholly above zero.

H4 is `partial` if D1 is reproduced and mean `T > 0` but one or more of conditions 5-6 fail.

H4 is `absent` if D1 is not reproduced, or if mean `T <= 0` after D2-A.

H4 is `unresolved` if D1 is ambiguous or D2 evidence is incomplete/interaction-ambiguous.

## 9.5 Predictive/mechanistic disconnect

At N=40 use the paired orientations from Section 7.3.

H7 is `strengthened` if:

```text
mean Delta_MAE > 0
MAE time_scaled wins >= 4/5 D1 bundles
mean Delta_R2 <= 0
latent-R2 time_scaled wins <= 2/5 D1 bundles
```

H7 is `weakened` if both predictive MAE and latent R2 favor `time_scaled` on average and each wins at least 4/5 D1 bundles.

Otherwise H7 is `unresolved`.

A positive H4 result combined with strengthened H7 MUST be described as an N-dependent predictive effect, not mechanistic recovery.

## 9.6 Capacity hypothesis

H6 is explicitly `not_yet_tested` after D3. Parameter count differences alone do not adjudicate H6.

## 9.7 D3 output schema

The canonical JSON records at least:

```text
phenomenon_reproduction
H1_initialization
H2_subset_composition
H3_cohort_heterogeneity
H4_sample_complexity
H5_checkpoint_objective
H6_capacity
H7_predictive_mechanistic_disconnect
triggered_escalations
next_required_stage
rationale
input_artifact_hashes
```

`next_required_stage` is one of:

```text
D2B
D4_CHECKPOINT
D4_OPTIMIZATION
D4_DATA_REGIME
D4_CAPACITY_TIME
STOP
```

---

# 10. D4 — targeted intervention and falsification hierarchy

D4 is conditional. No D4 run matrix may execute merely because code exists for it.

The rule is:

> Perform only the smallest intervention needed to falsify the explanation selected by D3.

Before any D4 compute begins, the exact conditional run matrix, diagnostic-only seed allocation, and artifact contract MUST be frozen in a short D4 execution addendum committed to the Phase 0.6 branch. This requirement is intentional rather than an unresolved design placeholder: D3 determines which D4 path is scientifically justified, so unused D4 experiments must not be preregistered as though they will all run.

## 10.1 D4-A — checkpoint-policy falsification

Run only if H5 is strengthened.

For each selected diagnostic trajectory compare the two already-retained checkpoints:

```text
A: production validation-core-loss checkpoint
B: shadow validation-MAE checkpoint
```

No retraining difference is introduced. The estimand is:

```text
Delta_selection = test_MAE(core_checkpoint) - test_MAE(shadow_MAE_checkpoint)
```

This is the first stage allowed to evaluate the shadow checkpoint comparatively on the test split.

The production checkpoint policy remains unchanged regardless of the D4-A result unless a later, separately designed study chooses to change it.

## 10.2 D4-B — optimization/initialization stability

Run if H1 is strengthened or D1 trajectories show substantial optimization instability.

Hold cohort/subset conditions fixed and expand only diagnostic model seeds from a dedicated Phase 0.6 namespace that cannot collide with the confirmatory range. The D4 addendum fixes the exact seed bank before execution.

Measure:

- treatment-effect variance over model seeds;
- selected-epoch dispersion;
- gradient-norm trajectory dispersion;
- flow-displacement trajectory dispersion;
- early-stopping behavior;
- sign consistency of the N=40 effect.

The first D4-B experiment MUST NOT simultaneously change optimizer, learning rate, weight decay, patience, architecture, or checkpoint objective.

## 10.3 D4-C — cohort/subset robustness

Run if H2 or H3 is strengthened.

If subset composition dominates, expand diagnostic subset resampling while cohort/model factors are controlled.

If cohort realization dominates, expand diagnostic synthetic cohorts while subset/model factors are controlled.

The D4 addendum fixes the exact diagnostic seed namespaces and sample counts before execution. Confirmatory seeds remain forbidden.

Report the effect distribution, not merely its mean:

```text
mean
median
standard deviation
IQR
sign consistency
5th percentile
95th percentile
```

## 10.4 D4-D — capacity and temporal-semantics falsification

Run if H4 remains plausible while H6 is unresolved, or if H7 is strengthened.

Two controls are required.

### Control 1: capacity-matched non-temporal residual

Use the same learned state transformation parameter count as `time_scaled` but remove per-example elapsed-time information:

```text
h' = h + c_ref * tanh(W h + b)
```

`W,b` match the `time_scaled` flow transform exactly in dimensionality and trainable parameter count. `c_ref` is a fixed, non-trainable scalar computed once from development-only D1 elapsed-time scales as the median of:

```text
delta_t / (time_scale_days + delta_t)
```

across valid D1 development positions. `c_ref` is frozen before D4-D execution and cannot use test outcomes.

This control asks whether an equally sized learned residual transform is sufficient without patient-specific timing semantics.

### Control 2: time-shuffled flow

Keep the `time_scaled` architecture and parameter count but destroy the association between elapsed time and the patient/time position.

Within each split and diagnostic seed bundle:

1. derive non-negative inter-event `delta_t` values;
2. permute valid `delta_t` values with a fixed Phase 0.6 permutation seed;
3. reassign them while preserving sequence lengths;
4. reconstruct monotone cumulative times;
5. train/evaluate using those reconstructed times.

This preserves the elapsed-time marginal distribution while breaking its semantic alignment.

Evidence for temporal inductive bias is substantially stronger only if:

```text
time_scaled > none
time_scaled > capacity-matched non-temporal residual
time_scaled > time-shuffled flow
```

A predictive advantage without improved latent recovery MUST still not be called mechanistic recovery.

## 10.5 D4-E — mechanism redesign boundary

A new architecture study is justified only after simpler explanations have survived appropriate falsification.

A defensible escalation chain is:

```text
D1 crossover reproduces
D2 not explained by one seed source
D3 strengthens H4
D4-A not a checkpoint artifact
D4-B/C not optimization/data instability
D4-D not explained by capacity and requires meaningful time
```

If that chain holds while latent recovery remains poor, the justified conclusion is that temporal dynamics appear predictively useful but the current mechanism formulation is inadequate. A redesigned mechanism must then be a new separately specified study (for example, Phase 0.7), not an unannounced Phase 0.6 architecture mutation.

---

# 11. Confirmatory-seed firewall

Phase 0.6 must enforce the confirmatory boundary in code, not only documentation.

The reserved Phase 0.5 confirmatory levels are:

```text
cohort seeds: 701..710
subset seeds: 801..810
model seeds: 901..910
```

Before any Phase 0.6 job is planned or launched, validation MUST reject a job if any seed belongs to its corresponding reserved set.

Phase 0.6 execution interfaces MUST NOT expose `confirmation` or `robustness` as valid Phase 0.6 stages.

A Phase 0.6 protocol lock MUST persist:

- the forbidden seed sets;
- the exact D1 development bundles;
- the D2 orthogonal mapping;
- the base Phase 0.5 config/protocol hashes;
- the Phase 0.6 execution commit;
- the diagnostic classification thresholds.

Any mismatch between runtime configuration and the persisted protocol identity is a hard failure.

---

# 12. Output and provenance architecture

Phase 0.6 runtime output MUST be separate from all Phase 0/0.5 official directories.

A recommended structure is:

```text
outputs/phase06_<execution-sha>/
├── protocol_lock.json
├── execution_provenance.json
├── stages/
│   ├── d1/
│   │   ├── cells/
│   │   ├── traces/
│   │   ├── summaries/
│   │   └── checkpoints/
│   └── d2/
│       ├── cells/
│       ├── traces/
│       ├── summaries/
│       └── checkpoints/
└── analysis/
    ├── phase06_d1_reproduction.csv
    ├── phase06_d2_effects.csv
    ├── phase06_d2_variance_components.csv
    └── phase06_d3_adjudication.json
```

Runtime output remains ignored by Git until a stage is scientifically complete and intentionally archived under `docs/results/phase06/` using the established repository-result archival pattern.

Every cell summary MUST bind by SHA-256 to:

- final metric record;
- training trace;
- production checkpoint;
- shadow checkpoint;
- config/protocol identity;
- execution commit.

The stage-level provenance MUST record:

- planned cell count;
- completed cell count;
- failures;
- start/end timestamps and wall time;
- device and worker settings;
- Python/PyTorch/CUDA/runtime metadata;
- repository commit;
- configuration hashes;
- protocol-lock hash;
- forbidden-seed validation result.

No stage may be labeled complete if its planned cell set is incomplete or contains duplicate cell IDs.

---

# 13. Failure and error semantics

Engineering failure and scientific failure remain distinct.

Engineering failures include:

- invalid protocol/config hash;
- forbidden confirmatory seed detected;
- missing/duplicate cell plan;
- non-finite required values;
- malformed trace/summary artifacts;
- diagnostics non-interference failure;
- checksum mismatch;
- callback/observer mutation of training state;
- unexpected exception during a planned cell.

Scientific negative outcomes include:

- D1 pattern does not reproduce;
- no seed factor dominates;
- H4 is absent;
- checkpoint mismatch is weakened;
- capacity/time controls falsify the temporal interpretation;
- predictive improvement remains disconnected from latent recovery.

Scientific negative outcomes MUST produce normal decision artifacts and normal process exit semantics for a completed diagnostic stage. They are not engineering crashes.

---

# 14. Testing strategy

Implementation follows TDD.

## 14.1 D0 unit tests

At minimum:

1. diagnostics-on/off fixed-CPU training produces byte-identical final core parameters for deterministic mode;
2. selected production checkpoint epoch is identical with diagnostics on/off;
3. final evaluation metrics are identical within exact/deterministic tolerance;
4. observer receives one epoch record per completed epoch;
5. `flow_mode="none"` reports zero flow displacement;
6. gradient/parameter norms are finite and non-negative;
7. shadow checkpoint never changes the returned production checkpoint;
8. observer exceptions fail clearly rather than silently corrupt training;
9. no Phase 0.6 persistence dependency is imported by the Phase 0.5 trainer.

## 14.2 Protocol/firewall tests

At minimum:

1. all five D1 development bundles are accepted;
2. every reserved confirmatory cohort/subset/model seed is rejected in its corresponding namespace;
3. D2-A contains exactly 25 unique combinations;
4. every D2 factor level appears exactly five times;
5. every factor pair appears exactly once;
6. no D2-A combination uses a confirmatory value;
7. duplicate cell plans are rejected;
8. config/protocol hash mismatch is rejected.

## 14.3 Analysis tests

Synthetic fixtures MUST test each D3 classification branch, including:

- D1 reproduced / not reproduced / ambiguous;
- H1/H2/H3 dominance and weak classifications;
- H4 strong / partial / absent / unresolved;
- H5 strengthened / weakened / unresolved;
- H7 strengthened / weakened / unresolved;
- D2-B escalation on interaction ambiguity;
- correct effect orientation for MAE and latent R2.

## 14.4 Full regression gate

Before Phase 0.6 experimental execution:

- Ruff passes;
- the full existing test suite passes;
- all new Phase 0.6 tests pass;
- no existing Phase 0.5 test expectation is weakened merely to accommodate diagnostics.

---

# 15. Implementation boundaries

The expected code responsibilities are:

```text
src/afmc_fm/phase05/training.py
    exact historical trainer with optional observation points only

src/afmc_fm/phase06/
    diagnostic records/recorder
    Phase 0.6 config/protocol validation
    D1/D2 planning and execution
    persistence and integrity
    D1/D2/D3 analysis

configs/experiments/phase06.yaml
    Phase 0.6 diagnostic-only settings and fixed analysis seeds
    references/binds the Phase 0.5 training configuration identity

tests/phase06/
    instrumentation invariants
    seed firewall
    orthogonal design
    persistence/provenance
    adjudication
```

The implementation plan may refine file names to follow existing repository conventions, but it MUST preserve these responsibility boundaries.

No production model interface change is justified merely for diagnostics unless required to expose already-computed observational state. Prefer deriving diagnostics from existing `Phase05FlowJumpOutput` fields and no-grad observer computations.

---

# 16. Stage-completion criteria

## D0 complete

D0 is complete only when diagnostics non-interference, schema, flow-displacement, shadow-selection, and full regression tests pass.

## D1 complete

D1 is complete only when all 40 cells complete, provenance/integrity validation passes, the confirmatory-seed firewall is clean, and the reproduction classification is emitted.

## D2 complete

D2-A is complete only when all 100 cells complete and the orthogonal main-effect decomposition plus bootstrap diagnostics are emitted. D2-B is run only if triggered by the declared ambiguity rules.

## D3 complete

D3 is complete only when the canonical adjudication JSON is produced from hash-bound D1/D2 artifacts and identifies either a required next stage or `STOP`.

## D4 complete

A D4 path is complete only against its separately frozen conditional execution addendum. Unselected D4 paths are not considered incomplete work.

---

# 17. Stopping rules

Phase 0.6 MUST be allowed to stop without proposing a new model.

Examples of valid stopping conclusions:

```text
D1 crossover does not reproduce
-> no architectural escalation

model initialization explains most effect variance
-> diagnose optimization instability; no mechanism redesign yet

subset/cohort composition dominates
-> diagnose data-regime sensitivity; no mechanism claim

capacity-matched residual matches time_scaled
-> temporal-flow interpretation weakened/falsified

time-shuffled flow matches time_scaled
-> meaningful elapsed-time semantics not established

prediction improves but latent recovery remains absent
-> predictive effect only; no mechanistic-validity claim
```

A diagnostic program that cannot conclude `STOP` would be architecture search rather than falsification.

---

# 18. Interpretation language

Phase 0.6 reports MUST preserve evidence class explicitly.

Allowed examples:

- "The N=40 predictive crossover reproduced across the development bundles."
- "Model-seed variance dominated the diagnostic treatment-effect variance."
- "Checkpoint-objective mismatch was strengthened by the diagnostic trace."
- "Time-scaled flow improved predictive MAE without demonstrating improved latent-state recovery."

Disallowed examples without later evidence:

- "Phase 0.5 succeeded at N=40."
- "The flow mechanism was recovered."
- "The confirmatory result is positive."
- "Temporal dynamics are validated" based only on `time_scaled > none`.

---

# 19. Final design decision

Phase 0.6 is a staged diagnostic and falsification program, not a rescue attempt for Phase 0.5.

The central architectural decision is to instrument the exact existing Phase 0.5 trainer through an opt-in observer while preserving production semantics. D1 then attempts an exact development-seed reproduction of the apparent sample-size crossover. D2 breaks the coupled seed bundles with a balanced orthogonal design. D3 converts those observations into predefined hypothesis states. D4 runs only the smallest targeted intervention justified by D3.

The confirmatory seed bank remains untouched throughout. Any future redesigned mechanism is a new study with a new specification and evidence boundary.
