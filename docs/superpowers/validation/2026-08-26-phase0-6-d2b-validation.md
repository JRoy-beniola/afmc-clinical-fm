# Phase 0.6 D2-B Execution Readiness Validation

Date finalized: 2026-08-27

## Status and scope

- Implementation status: **READY FOR D2-B EXECUTION**
- Branch: `phase0-6-d2b-work`
- Validated implementation SHA: `cb61c3eb3f39a00980033e6b61108a0834e5bfca`
- Parent Phase 0.6 execution SHA: `1718402df1d6ef344168677e6d26ea664708e1bc`
- Parent D3 decision: `next_required_stage = D2B`
- Scope: execution readiness for the frozen Phase 0.6 D2-B complementary diagnostic only.
- This validation did not execute D2-B CUDA science, cross-array adjudication, D4, a full-factorial addendum, Phase 0.5 continuation, or confirmatory seeds.

The implementation SHA above is the execution-critical freeze point. A later validation-document-only commit is permitted, but D2-B source, configuration, launcher/monitor tooling, and the frozen execution addendum must remain byte-identical to this validated SHA. The launcher enforces this boundary before execution.

## Frozen parent identities

| Evidence | Frozen identity |
|---|---|
| Parent execution SHA | `1718402df1d6ef344168677e6d26ea664708e1bc` |
| Parent protocol-lock SHA-256 | `c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51` |
| Parent D3 artifact SHA-256 | `6b9fffed7503fae6beeac2314238ae3d10ffdebfd27ea71ad952ab1f87916460` |
| Phase 0.6 core config SHA-256 | `b07a87ff2958b6b5e91a2a93cde81fc7d78d93910d5e1769978437f0bdc951f2` |
| Phase 0.6 core spec SHA-256 | `1e86f8f3f1848f1c1aa83a5f611fd3967a4d501a6213a8fbc270ae30442ef81c` |
| Phase 0.5 config SHA-256 | `508d6703f6d7b3ae2b92068510b58f16cffa985a233a340e9228d96f476d4693` |

D2-B parent loading at the validated implementation SHA additionally verifies:

- the parent protocol bytes and execution identity;
- the parent Phase 0.6/Phase 0.5 configuration identities and Phase 0.6 spec identity;
- the forbidden-seed identity;
- exact D1 and D2-A `COMPLETE` markers;
- the actual persisted D1 and D2-A cell bundles through bound-store completion validation;
- the frozen parent D3 SHA-256;
- `next_required_stage = D2B`;
- the D3 `input_artifact_hashes` against recomputed/loaded parent evidence.

## Frozen D2-B scientific matrix

D2-B uses only the already exposed development seed levels:

```text
cohort: 401..405
subset: 501..505
model: 601..605
```

The complementary orthogonal mapping is frozen as:

```text
k = (i + 2j) mod 5
```

for `i,j in {0,1,2,3,4}` indexing cohort and subset levels.

The execution matrix is:

```text
world: smooth
N: 5, 40
flow: none, time_scaled
jump: none
uncertainty: deterministic
```

Cardinality:

```text
25 seed triples x 2 N x 2 flow modes = 100 cells
```

All 100 cells are rerun in the child output under the child execution SHA. The five triples overlapping D2-A are intentionally rerun and are later used only as descriptive rerun-fidelity evidence.

## Seed firewall

Reserved confirmatory seeds remain forbidden:

```text
cohort: 701..710
subset: 801..810
model: 901..910
```

The D2-B addendum and implementation do not authorize or expose confirmatory execution. They also do not authorize Phase 0.5 continuation, a full `5x5x5` development-seed factorial, or any D4 intervention.

## Execution-boundary hardening review

The final blocker-fix cycle addressed all four Important findings from the independent execution-readiness review:

1. Parent D1/D2-A completion is validated against persisted cell bundles, not marker contents alone.
2. The real frozen parent D3 SHA-256 is enforced, and its `input_artifact_hashes` are checked against parent evidence.
3. The public D2-B CLI is CUDA-only; `cpu` and `auto` are not accepted for D2-B execution.
4. Parent and child output roots reject equality and nesting in either direction in both Python and the shell launcher.

The follow-up independent re-review of `cb61c3eb3f39a00980033e6b61108a0834e5bfca` reported:

```text
Critical: None
Important: None
```

and concluded that the SHA is suitable to freeze for the final readiness/CI gate before CUDA execution.

## TDD and CI evidence

The blocker fixes were exercised through an explicit RED/GREEN cycle.

RED checkpoint, CI #306:

```text
Ruff: PASS
pytest: 9 failed, 556 passed, 5 skipped
```

All nine failures mapped to the four execution-boundary defects under repair.

Intermediate GREEN candidate, CI #313:

```text
Ruff: PASS
pytest: 2 failed, 563 passed, 5 skipped
```

The two remaining failures were test-contract fallout only: one over-specific error-message assertion and one legacy fake end-to-end test still invoking D2-B with `--device cpu`.

Final implementation GREEN, CI #315 against the validated implementation SHA:

```text
ruff check src tests: All checks passed!
pytest -q: 565 passed, 5 skipped in 194.28s (0:03:14)
```

The five skips were exclusively CUDA-unavailable tests on the GitHub CPU runner:

```text
tests/execution/test_cuda_smoke.py
tests/models/test_baselines.py (2)
tests/phase05/test_model.py
tests/phase05/test_validation_smoke.py
```

No CPU-runner skip is treated as evidence that CUDA execution itself has already been validated. Actual GPU preflight remains a separate local gate before official D2-B execution.

## Launcher and child-store boundaries

The validated D2-B launcher requires:

- a dedicated D2-B readiness record exposing this validated implementation SHA;
- `PHASE06_PARENT_OUTPUT` pointing to the frozen parent output;
- parent/child roots that are distinct and non-nested;
- a separate child output of the form `phase06_d2b_<HEAD>` by default;
- a clean tracked working tree;
- execution-critical paths byte-identical to this validated implementation SHA;
- `nvidia-smi` availability;
- `torch.cuda.is_available() == True`;
- D2-B invocation with `--device cuda`;
- an explicit D2-B `COMPLETE` marker and persisted D2-B analysis outputs before reporting stage completion.

The launcher hard-stops after D2-B execution. Cross-array adjudication is not run automatically. D4 is not run automatically or authorized by this record.

## Final readiness boundary

This record authorizes only the next execution gate:

1. update the local WSL checkout to the final docs-only PR head;
2. verify execution-critical paths are byte-identical to `cb61c3eb3f39a00980033e6b61108a0834e5bfca`;
3. bind `PHASE06_PARENT_OUTPUT` to the immutable parent output;
4. pass the local CUDA preflight;
5. launch the frozen 100-cell D2-B complementary diagnostic in a separate child store.

It does not authorize automatic cross-array adjudication, D4, full-factorial execution, confirmatory seeds, or reinterpretation of the archived Phase 0.5 result.

**READY FOR D2-B EXECUTION**
