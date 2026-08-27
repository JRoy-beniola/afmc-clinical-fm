# Phase 0.6 D4-B Execution Readiness Validation

Date finalized: 2026-08-27

## Status and scope

- Implementation status: **READY FOR D4-B EXECUTION**
- Branch: `phase0-6-d4b-work`
- Validated implementation SHA: `18f391fa89d80687f38f6c50a60a062e4524edd1`
- Core Phase 0.6 parent execution SHA: `1718402df1d6ef344168677e6d26ea664708e1bc`
- D2-B parent execution SHA: `516c9e3c0e965582fa5cce976e9d8ebf32ea8404`
- Frozen parent decision entering D4-B: `next_required_stage = D4_OPTIMIZATION`
- Scope: execution readiness for the frozen Phase 0.6 D4-B optimization/initialization stability diagnostic only.
- This validation did not execute D4-B CUDA science, D4-B adjudication, D4 capacity/time, D4-D, a full-factorial expansion, Phase 0.5 continuation, or confirmatory seeds.

The validated implementation SHA above is the execution-critical freeze point. A later validation-document-only commit may move branch HEAD, but D4-B source, configuration, launcher/monitor tooling, and the frozen D4-B execution addendum must remain byte-identical to this validated SHA. The launcher enforces this boundary before execution.

## Immutable two-parent evidence chain

D4-B consumes two immutable roots and must never rewrite either one.

### Core D1/D2-A/D3 parent

| Evidence | Frozen identity |
|---|---|
| Core parent execution SHA | `1718402df1d6ef344168677e6d26ea664708e1bc` |
| Core parent protocol-lock SHA-256 | `c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51` |
| Core parent D3 artifact SHA-256 | `6b9fffed7503fae6beeac2314238ae3d10ffdebfd27ea71ad952ab1f87916460` |
| Core D3 route | `next_required_stage = D2B` |

### Completed D2-B parent

| Evidence | Frozen identity |
|---|---|
| D2-B execution SHA | `516c9e3c0e965582fa5cce976e9d8ebf32ea8404` |
| D2-B canonical protocol SHA-256 | `33f7cb1f6e71560f547a746cb7f5eb41130f794ab1e3a6fb1928c40e05b29f12` |
| D2-B canonical adjudication SHA-256 | `ea28fd4d5f6490a10fad20d5d3f3e76a1de08bf6b1be3b9805c9cf6c519e845f` |
| D2-B adjudicated route | `next_required_stage = D4_OPTIMIZATION` |
| Remaining predeclared escalation order | `D4_OPTIMIZATION`, then `D4_CAPACITY_TIME` |

Before any D4-B child store is created, the validated implementation:

- deeply validates the core D1/D2-A/D3 parent through the existing bound-store validation path;
- verifies the frozen core protocol, config/spec, forbidden-seed and D3 identities;
- opens the D2-B parent as a hash-bound `Phase06Store`;
- validates exact completion of all 100 persisted D2-B bundles rather than trusting a marker alone;
- verifies the D2-B child protocol linkage to the exact core parent;
- verifies the canonical D2-B protocol and adjudication identities above;
- recomputes D2-A and D2-B analyses from the two immutable roots;
- recomputes the cross-array D2-B adjudication;
- requires the recomputed adjudication to equal the persisted frozen adjudication;
- requires the resulting route to remain `D4_OPTIMIZATION` before D4-B planning or child creation proceeds.

Any provenance or semantic mismatch is a pre-execution failure.

## Frozen D4-B scientific question

D4-B tests one narrow falsification question:

> Is the N=40 `time_scaled` predictive advantage robust to model initialization/optimization variation when cohort and subset contexts are held fixed?

D4-B does not test capacity or temporal semantics. It therefore does not change optimizer, learning rate, weight decay, patience, epoch limit, objective, checkpoint policy, architecture, representation/state dimensions, time scale, data generation, split semantics, or low-N budget selection.

## Frozen D4-B execution matrix

Fixed cohort/subset contexts:

```text
(401, 501)
(402, 502)
(403, 503)
(404, 504)
(405, 505)
```

Dedicated diagnostic model-seed bank:

```text
1001..1010
```

Cell settings:

```text
stage: d4b
world: smooth
N: 40
flow: none, time_scaled
jump: none
uncertainty: deterministic
```

Cardinality:

```text
5 fixed contexts x 10 model seeds x 2 flow modes = 100 cells
```

The two flow variants for the same context/model seed form one paired comparison, giving exactly 50 paired D4-B effects.

## Seed firewall

Reserved confirmatory namespaces remain forbidden:

```text
cohort: 701..710
subset: 801..810
model: 901..910
```

The D4-B model seeds `1001..1010` are diagnostic-only and do not overlap the earlier D1/D2 development model seeds `601..605` or the reserved confirmatory model seeds `901..910`.

No D4-B code path in this validation authorizes confirmatory execution, Phase 0.5 continuation, full `5x5x5` seed expansion, or any later D4 intervention.

## Frozen primary estimand and bootstrap

For each fixed context `c` and model seed `m`:

```text
Delta_MAE(c,m) = MAE_none(c,m) - MAE_time_scaled(c,m)
```

Positive values favor `time_scaled`.

The locked bootstrap is model-seed-cluster resampling:

```text
bootstrap_resamples = 10000
bootstrap_seed = 20260827
resampling unit = model seed
contexts retained per sampled seed = all 5
confidence interval = percentile 95%
```

The implementation derives exactly ten model-seed means, five fixed-context means, the grand paired mean, and the model-seed-cluster bootstrap interval from the frozen 50-pair effect surface.

## Frozen D4-B classification

D4-B is `stable` only if all four conditions hold:

```text
Delta_overall > 0
positive model-seed means >= 8/10
positive fixed-context means >= 4/5
bootstrap 95% CI lower bound > 0
```

D4-B is `fragile` if any of the following holds:

```text
Delta_overall <= 0
positive model-seed means <= 5/10
positive fixed-context means <= 2/5
```

Every other result is `ambiguous`.

The next route is frozen as:

```text
stable    -> D4_CAPACITY_TIME
fragile   -> STOP
ambiguous -> STOP
```

`D4_CAPACITY_TIME` is only a route token. This implementation and readiness record do not implement or authorize capacity/time execution.

## Required execution artifacts

A successful D4-B execution must persist the exact hash-bound D4-B cell set and at least:

```text
phase06_d4b_effects.csv
phase06_d4b_model_seed_summary.csv
phase06_d4b_context_summary.csv
phase06_d4b_bootstrap_diagnostics.json
phase06_d4b_optimization_dispersion.csv
```

Execution itself must not write `phase06_d4b_adjudication.json`.

The separate explicit `adjudicate-d4b` action revalidates both immutable parents and the child protocol/completion boundary before recomputing analysis and writing only the final D4-B adjudication artifact.

## TDD and implementation hardening evidence

D4-B implementation was developed through explicit RED/GREEN boundaries rather than by adding execution code first.

Key RED checkpoints included:

- CI #317: planner surface failed because D4-B planning did not yet exist;
- CI #319: provenance/protocol RED failed because the D4-B protocol boundary did not yet exist;
- CI #321: deep two-parent RED left exactly two intended failures at the missing D4-B recomputation boundary while 571 tests passed and five CUDA-only tests skipped;
- CI #323: tooling RED produced six intended failures corresponding to the frozen D4-B runner/starter/monitor/README contract while 586 tests passed and five CUDA-only tests skipped;
- CI #324: persistence RED produced exactly one failure because `Phase06Store` still rejected stage `d4b`, with 592 tests passing and five CUDA-only tests skipped.

The persistence defect exposed by CI #324 was fixed minimally by adding `d4b` to the Phase 0.6 store stage allowlist and preserving the existing hash-bound cell/store semantics.

The final implementation also includes an execution-level fake D4-B smoke covering a complete 100-cell child, required analysis persistence, immutable-parent snapshots, and explicit execution/adjudication separation.

## Independent Critical/Important review

The exact validated candidate SHA

```text
18f391fa89d80687f38f6c50a60a062e4524edd1
```

received an independent execution-readiness review scoped to the frozen D4-B matrix, seed firewall, two-parent provenance, child protocol, root isolation, CUDA-only execution, resume integrity, persistence, paired analysis, cluster bootstrap, classification logic, execution/adjudication separation, and absence of later-stage execution exposure.

The review reported:

```text
Critical
None.

Important
None.
```

and concluded:

```text
VERDICT: 18f391fa89d80687f38f6c50a60a062e4524edd1 is suitable to freeze as the validated Phase 0.6 D4-B implementation SHA and proceed to the readiness-record/final-CI gate before CUDA execution.
```

No Critical or Important blocker remained at the freeze point.

## Final implementation CI evidence

Exact candidate CI #326 ran against the D4-B branch stacked on the frozen D2-B base and completed successfully.

```text
ruff check src tests: All checks passed!
pytest -q: 594 passed, 5 skipped in 170.64s (0:02:50)
```

The five skips were exclusively CUDA-unavailable tests on the GitHub CPU runner:

```text
tests/execution/test_cuda_smoke.py
tests/models/test_baselines.py (2)
tests/phase05/test_model.py
tests/phase05/test_validation_smoke.py
```

No CPU-runner skip is treated as proof that official D4-B CUDA science has already been executed. Local CUDA preflight remains a separate gate.

## Guarded launcher and child-store boundary

The validated D4-B launcher requires:

- this D4-B readiness record exposing the validated implementation SHA;
- `PHASE06_CORE_PARENT_OUTPUT` bound to the immutable D1/D2-A/D3 root;
- `PHASE06_D2B_PARENT_OUTPUT` bound to the immutable completed D2-B root;
- all three roots pairwise distinct and non-nested in either direction;
- a separate child output of the form `phase06_d4b_<HEAD>` by default;
- a clean tracked/staged working tree;
- D4-B execution-critical paths byte-identical to the validated implementation SHA;
- `nvidia-smi` availability;
- `torch.cuda.is_available() == True`;
- public D4-B invocation with `--device cuda` only;
- fresh/resume child-store protocol integrity;
- exact D4-B `COMPLETE` and analysis artifacts before launcher completion.

The launch manifest records the child execution identity, readiness identity, both parent roots and their relevant artifact hashes, the D4-B addendum hash, Python/PyTorch/CUDA metadata, and GPU identity.

The launcher hard-stops after D4-B execution and does not invoke `adjudicate-d4b` automatically.

## Final readiness boundary

This record authorizes only the next execution gate:

1. update the local WSL checkout to the final docs-only D4-B branch head;
2. verify D4-B execution-critical paths remain byte-identical to `18f391fa89d80687f38f6c50a60a062e4524edd1`;
3. bind `PHASE06_CORE_PARENT_OUTPUT` to the immutable D1/D2-A/D3 parent;
4. bind `PHASE06_D2B_PARENT_OUTPUT` to the completed immutable D2-B parent;
5. pass the local CUDA preflight;
6. launch the frozen 100-cell D4-B diagnostic in a separate child store;
7. hard-stop after execution for artifact audit before any explicit adjudication.

This record does not authorize automatic D4-B adjudication, D4 capacity/time execution, D4-D, full-factorial expansion, confirmatory seeds, Phase 0.5 continuation, or reinterpretation of the archived Phase 0.5 negative result.

**READY FOR D4-B EXECUTION**
