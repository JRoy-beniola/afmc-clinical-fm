# Phase 0.6 D2-B Execution Addendum

**Date:** 2026-08-26  
**Status:** Frozen before D2-B implementation/execution  
**Parent execution SHA:** `1718402df1d6ef344168677e6d26ea664708e1bc`  
**Parent D3 decision:** `next_required_stage = D2B`

## 1. Purpose

This addendum authorizes only the conditional D2-B diagnostic stage already predeclared in the Phase 0.6 core design. It does not authorize D4, a full 5x5x5 development-seed factorial, any confirmatory seed, or any continuation of Phase 0.5.

The immutable parent evidence consists of the completed D1, D2-A, and D3 artifacts under the parent Phase 0.6 output root. D2-B must execute in a new child output root because the parent store is hash-bound to the parent execution SHA and must not be rewritten.

## 2. Scientific scope

D2-B uses only the already exposed development seed levels:

```text
cohort: 401..405
subset: 501..505
model: 601..605
```

Reserved confirmatory seeds remain forbidden.

The complementary orthogonal array is fixed as:

```text
k = (i + 2j) mod 5
```

for `i,j in {0,1,2,3,4}` indexing cohort and subset levels. The stage runs:

```text
world: smooth
N: 5, 40
flow: none, time_scaled
jump: none
uncertainty: deterministic
```

Total execution count:

```text
25 seed triples x 2 N x 2 flow modes = 100 cells
```

All 100 cells are rerun under the child execution SHA. The five seed triples that overlap D2-A are intentionally rerun rather than borrowed from the parent store; their repeated effects are reported as rerun-fidelity evidence.

## 3. Child provenance boundary

The D2-B child protocol lock must record and validate all of the following:

```text
child execution SHA
Phase 0.6 core config SHA-256
Phase 0.6 core spec SHA-256
this D2-B addendum SHA-256
Phase 0.5 config/protocol identity
parent execution SHA
parent protocol-lock SHA-256
parent D3 artifact SHA-256
parent D3 next_required_stage = D2B
forbidden seed sets
D2-B mapping k=(i+2j) mod 5
bootstrap_resamples = 10000
bootstrap_seed = 20260826
```

The parent output is read-only evidence. D2-B must never write into the parent root.

## 4. D2-B analysis

D2-B uses the same effect orientation as D2-A:

```text
Delta_MAE = MAE_none - MAE_time_scaled
```

For each N, fit the same additive main-effect model:

```text
Delta_ijk = mu + C_i + S_j + M_k + epsilon_ijk
```

Use the same locked bootstrap settings and the same factor classifications:

- `dominant`: largest main-effect share, at least 2x the next-largest named share, and largest in at least 80% of 10,000 bootstrap resamples;
- `weak`: smallest named main-effect share and largest in no more than 20% of bootstrap resamples;
- otherwise `unresolved`.

Persist D2-B effect rows, factor-level effects, variance components, bootstrap diagnostics, N-shift rows/summary, and overlap-rerun diagnostics.

## 5. Cross-array adjudication

Cross-array adjudication is explicit and separate from execution. It compares the recomputed parent D2-A result with the child D2-B result and consumes the frozen parent D3 artifact.

For each N, define the named dominant factor of an array as the unique factor classified `dominant`, or `none` if no factor is dominant.

The complementary-array evidence is considered **sufficient** only when all of the following hold:

1. At N=40, D2-A and D2-B identify the same non-`none` dominant factor.
2. At N=5, the two arrays are structurally consistent: either both have no dominant factor, or both identify the same dominant factor.
3. Neither array identifies a different named dominant factor at the same N.

If these conditions fail, D2-A plus D2-B remain insufficient and the next state is:

```text
FULL_FACTORIAL_ADDENDUM
```

This is a stop-and-design state only. No full factorial command is implemented by this addendum.

If the conditions pass, D2-B removes `D2B` from the parent D3 escalation queue and preserves the remaining predeclared escalation order. With the current parent D3 evidence this means the next route is expected to be `D4_OPTIMIZATION` if model-seed dominance is reproduced at N=40; `D4_CAPACITY_TIME` remains queued behind it. D4 is still not implemented by this addendum.

## 6. Overlap rerun diagnostics

The five D2-A/D2-B overlapping seed triples are descriptive reproducibility evidence. For each N, report:

```text
overlap_pair_count
mean absolute difference in Delta_MAE
maximum absolute difference in Delta_MAE
Pearson correlation when defined
```

No post-hoc numerical tolerance is used as a scientific gate.

## 7. Hard boundaries

D2-B implementation and tooling must not expose commands for:

```text
confirmatory seeds
Phase 0.5 continuation
full 5x5x5 factorial
D4 checkpoint intervention
D4 optimization intervention
D4 data-regime intervention
D4 capacity/time intervention
```

A negative, diffuse, contradictory, or unresolved D2-B result is valid evidence.
