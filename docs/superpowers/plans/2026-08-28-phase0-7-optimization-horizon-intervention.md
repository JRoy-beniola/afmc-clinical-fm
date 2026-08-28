# Phase 0.7 Optimization-Horizon Intervention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, validate, and CI-gate the frozen Phase 0.7 prospective stopping-policy intervention without executing any official Phase 0.7 training cells.

**Architecture:** Add Phase 0.7 as an isolated protocol/planning/runner/analysis layer that reuses the validated Phase 0.6 cohort and evaluation machinery. Introduce one narrow shared-training control for whether patience exhaustion terminates optimization; historical callers retain the current default behavior, while `forced_horizon` records the ordinary patience event and continues to epoch 100. All scientific inference, adjudication, manifesting, and execution boundaries live under `phase07` so Phase 0.6 evidence and semantics remain immutable.

**Tech Stack:** Python 3.11+, NumPy, pandas, PyTorch, PyYAML, argparse, pytest, Ruff, GitHub Actions, existing AFMC Phase 0.5/0.6 training and execution infrastructure.

**Spec:** `docs/superpowers/specs/2026-08-28-phase0-7-optimization-horizon-intervention-design.md`

## Global Constraints

- Phase 0.6 terminal result remains exactly `D4-B AMBIGUOUS -> STOP`.
- Phase 0.7 is development-only: `smooth`, N=40, flow modes `{none,time_scaled}`, policies `{standard_early_stop,forced_horizon}`.
- Contexts are exactly `(406,506)` through `(410,510)` and model seeds exactly `1101..1110`.
- Protected confirmatory seeds remain forbidden: cohort `701..710`, subset `801..810`, model `901..910`.
- The official matrix is exactly `5 × 10 × 2 × 2 = 200` unique cells.
- `forced_horizon` changes only the termination action on patience exhaustion; all matched-prefix trajectory semantics must remain identical.
- Maximum epoch count remains exactly 100 and ordinary patience remains exactly the locked Phase 0.5/0.6 value.
- Bootstrap inference uses exactly 10,000 crossed context/model resamples with seed `20260827`.
- The three frozen classifications are exactly `P07_OPTIMIZATION_HORIZON_EFFECT_AND_HETEROGENEITY_SUPPORTED`, `P07_OPTIMIZATION_HORIZON_EFFECT_ONLY`, and `P07_OPTIMIZATION_HORIZON_NOT_ESTABLISHED`.
- No official Phase 0.7 cell may be executed by tests, smoke validation, CI, PR review, merge, or implementation approval.
- All work stays on `phase0-7-optimization-horizon-intervention`, is reviewed through a PR, and must pass CI on the exact PR head before merge.
- Use TDD for every production behavior change; historical Phase 0.5/0.6 tests are regression gates whenever shared training code changes.

---

### Task 1: Frozen Phase 0.7 config and protocol identity

**Files:**
- Create: `configs/experiments/phase07.yaml`
- Create: `src/afmc_fm/phase07/__init__.py`
- Create: `src/afmc_fm/phase07/config.py`
- Create: `src/afmc_fm/phase07/protocol.py`
- Create: `tests/phase07/test_config.py`
- Create: `tests/phase07/test_protocol.py`

**Interfaces:**
- Produces: `Phase07Config` with exact contexts, model seeds, flow modes, policies, N=40, max epoch, bootstrap constants, and forbidden seed sets.
- Produces: `load_phase07_config(path) -> Phase07Config`.
- Produces: `validate_phase07_seed_triplet(cohort_seed, subset_seed, model_seed) -> None`.
- Produces: `build_phase07_protocol_lock(config, *, execution_commit, phase07_spec_path) -> dict[str, object]` binding the exact frozen spec bytes by SHA-256 and recording all frozen design constants.

- [ ] **Step 1: Write failing config tests**

Require exact constants and rejection of any deviation in world, N, contexts, model seeds, flow modes, policies, epoch budget, bootstrap constants, or forbidden seeds.

- [ ] **Step 2: Run the focused config tests and verify RED**

Run: `pytest -q tests/phase07/test_config.py`

Expected: import/module failure because `afmc_fm.phase07.config` does not yet exist.

- [ ] **Step 3: Implement the minimal config layer and YAML**

The YAML and dataclass must represent only the frozen experiment; no generic hyperparameter-search surface is added.

- [ ] **Step 4: Run config tests and Ruff; verify GREEN**

Run: `pytest -q tests/phase07/test_config.py && ruff check src/afmc_fm/phase07 tests/phase07`

- [ ] **Step 5: Write failing protocol tests**

Tests must verify that changing one byte in a temporary copy of the frozen spec changes `phase07_spec_sha256`; malformed execution SHAs fail; protected seed sets and all frozen constants appear in the lock; and the lock cannot describe any matrix other than the frozen 200-cell design.

- [ ] **Step 6: Run protocol tests and verify RED**

Run: `pytest -q tests/phase07/test_protocol.py`

- [ ] **Step 7: Implement the minimal protocol lock and verify GREEN**

Run: `pytest -q tests/phase07/test_config.py tests/phase07/test_protocol.py && ruff check src/afmc_fm/phase07 tests/phase07`

- [ ] **Step 8: Commit**

Commit message: `feat: freeze Phase 0.7 protocol identity`

### Task 2: Exact 200-cell planner and plan hash

**Files:**
- Create: `src/afmc_fm/phase07/planning.py`
- Create: `tests/phase07/test_planning.py`

**Interfaces:**
- Produces immutable `Phase07CellSpec` containing `world`, `cohort_seed`, `subset_seed`, `model_seed`, `n_train`, `flow_mode`, and `optimization_policy`.
- Produces: `plan_phase07_cells(config) -> tuple[Phase07CellSpec, ...]`.
- Produces: `phase07_plan_sha256(cells) -> str` over deterministic canonical JSON serialization.

- [ ] **Step 1: Write failing planner tests**

Require exactly 200 unique cells, exactly 50 `(context, model_seed)` pairs, exactly four cells per pair, exact two flow modes and two policies, N=40/smooth only, no protected seeds, deterministic ordering, and deterministic plan SHA-256.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/phase07/test_planning.py`

- [ ] **Step 3: Implement the minimal planner and canonical plan serializer**

Planner construction is Cartesian product only; no adaptive pruning, sampling, or threshold logic.

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest -q tests/phase07/test_planning.py tests/phase07/test_protocol.py && ruff check src/afmc_fm/phase07 tests/phase07`

Commit message: `feat: add exact Phase 0.7 planner`

### Task 3: Narrow stopping-policy injection with historical regression protection

**Files:**
- Modify: `src/afmc_fm/phase05/training.py`
- Create: `tests/phase07/test_training_policy.py`
- Regression-test: existing `tests/phase05/**`, `tests/phase06/**`

**Interfaces:**
- Extend `fit_phase05_model(..., *, stop_on_patience: bool = True)` and the internal `_fit_core` with the same keyword-only control.
- Preserve the default `True` behavior byte-for-behavior for all historical callers.
- With `stop_on_patience=False`, continue to `config.max_epochs` while still computing the same patience condition every epoch.
- Do not create a parallel Phase 0.7 trainer.

- [ ] **Step 1: Write a failing historical-default regression test**

Construct a deterministic tiny fixture where patience exhausts early; assert default `fit_phase05_model` stops at the same epoch and emits `patience_exhausted` exactly as before.

- [ ] **Step 2: Write a failing forced-horizon test**

Using the same initialization/data/config, run `stop_on_patience=False`; assert the diagnostic trace reaches `max_epochs`, while the prefix through the standard stop epoch is element-wise identical for all existing diagnostic fields.

- [ ] **Step 3: Verify RED for the new keyword**

Run: `pytest -q tests/phase07/test_training_policy.py`

Expected: failure because `stop_on_patience` is not accepted.

- [ ] **Step 4: Implement the minimal conditional break**

Keep `should_stop = stale_epochs >= config.patience` unchanged. Only execute the existing break when `should_stop and stop_on_patience`; do not alter optimizer creation, checkpoint updates, diagnostics, validation, RNG, or loss code.

- [ ] **Step 5: Verify focused GREEN and historical regressions**

Run: `pytest -q tests/phase07/test_training_policy.py tests/phase05 tests/phase06`

Run: `ruff check src tests`

- [ ] **Step 6: Commit**

Commit message: `feat: add nonterminating patience policy`

### Task 4: Phase 0.7 runner and paired-trajectory evidence

**Files:**
- Create: `src/afmc_fm/phase07/runner.py`
- Create: `src/afmc_fm/phase07/diagnostics.py`
- Create: `tests/phase07/test_runner.py`

**Interfaces:**
- Reuse `prepare_phase06_cohort` semantics or equivalent shared primitives without changing Phase 0.6 behavior.
- Produces `Phase07CellRun` with metrics, epoch trace, summary, production/shadow checkpoints, `optimization_policy`, and `would_patience_exhaust_epoch: int | None`.
- `standard_early_stop` calls shared training with `stop_on_patience=True`.
- `forced_horizon` calls shared training with `stop_on_patience=False` and derives the first ordinary patience-exhaustion epoch from the unchanged stale-epoch trace.

- [ ] **Step 1: Write failing runner tests**

Require identical simulator/subset/model seeds, identical initial state, identical standard/forced diagnostic prefix through standard termination, forced `epochs_run == 100`, and correct first `would_patience_exhaust_epoch` or null if patience never fires.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/phase07/test_runner.py`

- [ ] **Step 3: Implement the isolated runner/diagnostic adapter**

No Phase 0.6 result files or protocol code may be mutated.

- [ ] **Step 4: Verify GREEN plus Phase 0.6 runner regression**

Run: `pytest -q tests/phase07/test_runner.py tests/phase06/test_runner.py tests/phase07/test_training_policy.py`

- [ ] **Step 5: Commit**

Commit message: `feat: add Phase 0.7 paired-policy runner`

### Task 5: Deterministic estimands, crossed bootstrap, and frozen adjudication

**Files:**
- Create: `src/afmc_fm/phase07/analysis.py`
- Create: `tests/phase07/test_analysis.py`

**Interfaces:**
- Produces pair table with `Delta_standard`, `Delta_forced`, `G`, `H_time_scaled`, `H_control`.
- Produces exact two-way residuals `r_p(i,j) = Delta_p(i,j) - rowmean_p(i) - colmean_p(j) + grandmean_p`.
- Uses sample SD with `ddof=1` and returns undefined `R_SD` when standard residual SD is zero/non-finite.
- Produces exactly 10,000 crossed bootstrap replicates using seed `20260827`, resampling contexts and model seeds independently and recomputing residualization inside each replicate.
- Produces the exact three-way frozen classification.

- [ ] **Step 1: Write failing algebra tests**

Use a hand-computable 5×10 synthetic matrix to verify all five pair-level quantities and the two-way residual formula.

- [ ] **Step 2: Write failing `R_SD` edge-case tests**

Require no epsilon rescue and heterogeneity-gate failure for zero/non-finite standard residual SD.

- [ ] **Step 3: Write failing deterministic-bootstrap tests**

Same input and seed must return identical percentile intervals; changing input must change the result; resampled residualization must be recomputed within each bootstrap sample.

- [ ] **Step 4: Write failing adjudication tests**

Cover all three classifications plus the case where heterogeneity passes but the causal gate fails.

- [ ] **Step 5: Verify RED, implement minimally, and verify GREEN**

Run: `pytest -q tests/phase07/test_analysis.py`

Run: `ruff check src/afmc_fm/phase07 tests/phase07`

- [ ] **Step 6: Commit**

Commit message: `feat: add frozen Phase 0.7 analysis`

### Task 6: Execution boundaries, CLI, smoke validation, and manifest freeze

**Files:**
- Create: `src/afmc_fm/phase07/execution.py`
- Create: `src/afmc_fm/phase07/cli.py`
- Create: `tests/phase07/test_execution.py`
- Create: `tests/phase07/test_cli.py`
- Modify: `pyproject.toml`
- Create: `tools/execution/phase07/README.txt`

**Interfaces:**
- Add console script `afmc-phase07 = "afmc_fm.phase07.cli:main"`.
- Expose read-only/provisional commands for protocol validation, plan materialization/verification, and a small synthetic CPU smoke.
- Any command capable of official cell execution must require an explicit official-execution artifact/authorization boundary that is deliberately absent during this PR.
- Produce an execution manifest containing execution commit, frozen spec SHA-256, config SHA-256, protocol canonical hash, exact plan SHA-256, expected cell count 200, and forbidden seed sets.

- [ ] **Step 1: Write failing CLI/boundary tests**

Require plan/verify/smoke commands; require official execution to fail closed when the authorization artifact is absent; verify no smoke path expands into the 200-cell official matrix.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/phase07/test_execution.py tests/phase07/test_cli.py`

- [ ] **Step 3: Implement minimal execution/CLI boundary**

CPU smoke may use tiny fixture data and a reduced epoch config only when clearly labeled non-official and structurally unable to write official evidence.

- [ ] **Step 4: Verify focused GREEN**

Run: `pytest -q tests/phase07/test_execution.py tests/phase07/test_cli.py`

- [ ] **Step 5: Run the non-official CPU smoke only**

Run the CLI smoke command; confirm it does not produce or consume the official 200-cell plan.

- [ ] **Step 6: Commit**

Commit message: `feat: gate Phase 0.7 execution tooling`

### Task 7: Full PR verification and execution-SHA freeze

**Files:**
- Modify only if needed for CI coverage: `.github/workflows/ci.yml`
- Create: `docs/superpowers/validation/2026-08-28-phase0-7-implementation-validation.md`

**Interfaces:**
- The validation record names the exact PR head SHA and records the exact lint/test/CPU-smoke evidence.
- No official training command is run.

- [ ] **Step 1: Run complete local-equivalent validation through CI**

Required commands on the exact PR head:

`ruff check src tests tools/reproducibility`

`pytest -q`

plus the Phase 0.7 non-official CPU smoke.

- [ ] **Step 2: Inspect every CI failure, review thread, and changed file**

No merge while any required check is failing, any important review issue is unresolved, or the branch has drifted from `main` without reconciliation.

- [ ] **Step 3: Verify immutable/historical boundaries**

Diff audit must show no mutation of official Phase 0/0.5/0.6 evidence and no use of protected confirmatory seeds.

- [ ] **Step 4: Record the candidate execution SHA**

The validation record must identify the exact implementation commit that passed CI. If any code changes after validation, the execution SHA is invalid and CI/validation must be repeated.

- [ ] **Step 5: Merge the PR only after exact-head CI is green**

Use an expected-head guard. Preserve the frozen spec and complete implementation history.

- [ ] **Step 6: STOP**

After merge, verify `main` contains the tested implementation. Do not execute any of the 200 official Phase 0.7 cells. Obtain a separate explicit authorization before official training.
