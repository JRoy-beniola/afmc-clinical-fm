# Phase 0.7 Execution Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Phase 0.7 safe for an official 200-cell local-GPU run by adding fail-closed bootstrap handling, exact dependency/config/worktree identity binding, crash-resilient cell persistence/resume, completion-gated official analysis, and frozen 100/12 policy invariants.

**Architecture:** Keep the scientific runner and pure statistics core narrow. Add a dedicated `Phase07Store` modeled on Phase 0.6 for authoritative evidence persistence, and let the execution layer own identity, clean-checkout, resume, provenance, aggregation, and official-analysis trust boundaries. Do not modify historical Phase 0.6 stores or evidence.

**Tech Stack:** Python 3.11, pytest, pandas, NumPy, PyTorch, pathlib/tempfile/os, SHA-256 canonical config hashing, Git subprocess checks, GitHub Actions CI.

**Spec:** `docs/superpowers/specs/2026-08-28-phase0-7-crash-resilient-execution-design.md`

## Global Constraints

- Scientific design remains smooth/N=40 only, 5 contexts, 10 model seeds, 2 architectures, 2 policies, exactly 200 official cells.
- Protected seeds remain forbidden: cohort 701..710, subset 801..810, model 901..910.
- No production CLI path may accept a reduced/test plan.
- Official execution requires a clean Git checkout and exact loaded Phase 0.5/simulator config hashes.
- Any invalid `R_SD` bootstrap replicate invalidates heterogeneity inference; no dropping/retry/epsilon stabilization.
- Resume granularity is one cell; no mid-cell continuation.
- Official analysis must require valid completion + exact validated 200-cell bundle set.
- No merge, authorization creation, or official 200-cell execution during implementation.

---

### Task 1: Fail-closed heterogeneity bootstrap

**Files:**
- Modify: `tests/phase07/test_analysis.py`
- Modify: `src/afmc_fm/phase07/analysis.py`

**Interfaces:**
- Consumes: `crossed_phase07_bootstrap(pairs)`, `Phase07Statistics`, `adjudicate_phase07(statistics)`.
- Produces: `R_SD_ci_lower/R_SD_ci_upper = NaN` whenever `R_SD_valid_replicates != bootstrap_resamples`; adjudication requires full valid-replicate count.

- [ ] **Step 1: Write failing tests**

Change the helper used by the existing three-level adjudication test to use `R_SD_valid_replicates=10_000` for the valid case, and add:

```python
def test_incomplete_r_sd_bootstrap_invalidates_heterogeneity_gate():
    module = _analysis_api()
    statistics = _statistics(module, causal=True, heterogeneity=True)
    statistics = module.Phase07Statistics(
        **{**statistics.__dict__, "R_SD_valid_replicates": 9_999}
    )
    assert module.adjudicate_phase07(statistics) == "P07_OPTIMIZATION_HORIZON_EFFECT_ONLY"
```

If slots/frozen dataclass has no `__dict__`, construct the object explicitly with the same values and `R_SD_valid_replicates=9_999`.

Also strengthen the deterministic bootstrap fixture:

```python
assert first["R_SD_valid_replicates"] < first["bootstrap_resamples"]
assert np.isnan(first["R_SD_ci_lower"])
assert np.isnan(first["R_SD_ci_upper"])
```

- [ ] **Step 2: Verify RED**

Run targeted CI/test command:

```bash
pytest tests/phase07/test_analysis.py -q
```

Expected: existing implementation fails because it computes a finite CI from filtered finite ratios and allows incomplete replicate count to pass heterogeneity.

- [ ] **Step 3: Implement minimal fail-closed behavior**

In `crossed_phase07_bootstrap`:

```python
finite_ratios = ratios[np.isfinite(ratios)]
if finite_ratios.size == _BOOTSTRAP_RESAMPLES:
    ratio_lower, ratio_upper = np.quantile(ratios, [0.025, 0.975])
else:
    ratio_lower = ratio_upper = float("nan")
```

In `adjudicate_phase07`, add:

```python
and statistics.R_SD_valid_replicates == statistics.bootstrap_resamples
```

to the heterogeneity predicate.

- [ ] **Step 4: Verify GREEN**

```bash
pytest tests/phase07/test_analysis.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/phase07/test_analysis.py src/afmc_fm/phase07/analysis.py
git commit -m "fix: fail closed on invalid Phase 0.7 bootstrap replicates"
```

---

### Task 2: Bind execution to loaded configs and a clean checkout

**Files:**
- Modify: `tests/phase07/test_execution.py`
- Modify: `src/afmc_fm/phase07/execution.py`
- Modify: `src/afmc_fm/phase07/cli.py`

**Interfaces:**
- Produces: `require_clean_phase07_checkout() -> None`.
- `build_phase07_execution_manifest(...)` adds `phase05_config_sha256` and `simulator_config_sha256` from canonicalized loaded dataclasses.
- `_expected_authorization(manifest)` includes both dependency hashes.

- [ ] **Step 1: Write failing identity tests**

Add assertions to manifest test:

```python
from afmc_fm.phase05.config import load_phase05_config
from afmc_fm.simulator.config import SimulatorConfig
from afmc_fm.config import load_yaml

phase05 = load_phase05_config(config.phase05_config)
raw = dict(load_yaml(config.simulator_config)); raw.pop("seed", None)
simulator = SimulatorConfig(**raw)
assert manifest["phase05_config_sha256"] == canonical_config_hash(phase05)
assert manifest["simulator_config_sha256"] == canonical_config_hash(simulator)
```

Update the exact authorization payload to include both hashes, and add a mutation test proving either hash mismatch rejects authorization.

Add a clean-checkout unit test by monkeypatching the internal subprocess helper so dirty porcelain output causes `ValueError` before a forbidden callback can run.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/phase07/test_execution.py -q
```

Expected: FAIL for missing dependency hashes/clean-worktree API.

- [ ] **Step 3: Implement identity binding**

In `execution.py`, load the exact Phase 0.5 and simulator dataclasses and hash with `canonical_config_hash`. Add both fields to manifest and expected authorization. Add a clean checkout guard based on:

```bash
git status --porcelain --untracked-files=all
```

using the repository root as cwd; non-empty stdout raises `ValueError("Phase 0.7 official execution requires a clean worktree")`.

Ensure official CLI checks cleanliness before authorization/execution and before device resolution. The non-official smoke must not require a clean checkout.

- [ ] **Step 4: Verify GREEN**

```bash
pytest tests/phase07/test_execution.py tests/phase07/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/phase07/test_execution.py tests/phase07/test_cli.py src/afmc_fm/phase07/execution.py src/afmc_fm/phase07/cli.py
git commit -m "feat: bind Phase 0.7 execution to exact checkout and configs"
```

---

### Task 3: Add crash-resilient `Phase07Store`

**Files:**
- Create: `src/afmc_fm/phase07/store.py`
- Create: `tests/phase07/test_store.py`

**Interfaces:**
- `Phase07Store(output: str | Path, *, identity: Mapping[str, object])`.
- `initialize(manifest, protocol_lock, plan_payload, *, resume: bool) -> None`.
- `write_cell_bundle(cell, *, metrics, trace, summary, production_state_dict, shadow_state_dict) -> str`.
- `validate_resume(expected_cells) -> frozenset[str]`.
- `load_metrics(expected_cells) -> pd.DataFrame`.
- `mark_complete(expected_cells) -> None`.
- `require_complete(expected_cells) -> None`.

- [ ] **Step 1: Write failing store tests**

Cover real persistence behavior using a synthetic valid `Phase07CellSpec` and tiny DataFrames/state dicts:

```python
def test_cell_bundle_is_committed_atomically_and_validates(...): ...
def test_abandoned_staging_directory_is_not_authoritative(...): ...
def test_corrupt_authoritative_bundle_fails_closed(...): ...
def test_unexpected_authoritative_cell_fails_closed(...): ...
def test_complete_requires_exact_expected_set(...): ...
```

The test helper may use a reduced expected set only through `Phase07Store`'s Python test-facing validation methods; production planner/CLI remain exact-200.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/phase07/test_store.py -q
```

Expected: FAIL with missing `afmc_fm.phase07.store`.

- [ ] **Step 3: Implement minimal store**

Model validation on `Phase06Store`, but use per-cell directories. For each cell:

```text
stages/phase07/.tmp/<cell>.<nonce>/
  cell.json
  metrics.csv
  training_trace.csv
  summary.json
  production_checkpoint.pt
  shadow_mae_checkpoint.pt
  artifact_manifest.json
  COMPLETE
```

Write/hashes/validate staged evidence, write staged COMPLETE last, then same-filesystem `os.replace(staging, final_dir)`. Existing final bundle must either fully validate and be semantically identical or fail closed; never overwrite conflicting authoritative evidence.

Root identity files are atomically written and exact-matched on resume. `validate_resume` removes abandoned `.tmp` directories only after root identity is established, rejects unexpected final cell IDs, and validates every observed bundle.

- [ ] **Step 4: Verify GREEN**

```bash
pytest tests/phase07/test_store.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/afmc_fm/phase07/store.py tests/phase07/test_store.py
git commit -m "feat: add crash-resilient Phase 0.7 store"
```

---

### Task 4: Store-backed official orchestration, resume, completion-gated analysis

**Files:**
- Modify: `tests/phase07/test_execution.py`
- Modify: `tests/phase07/test_cli.py`
- Modify: `src/afmc_fm/phase07/execution.py`
- Modify: `src/afmc_fm/phase07/cli.py`

**Interfaces:**
- `run_phase07_official_cells(..., store: Phase07Store, resume: bool, execute_cell: Callable) -> tuple[str, ...]` keeps production exact-plan validation.
- `load_completed_phase07_metrics(store, expected_cells) -> pd.DataFrame` requires valid COMPLETE + exact cell set.
- CLI `official` gains `--resume`.
- CLI may expose `analyze --output <root>` to produce adjudication only through completion-gated store validation.

- [ ] **Step 1: Write failing orchestration tests**

Use an internal injected reduced cell sequence only after bypassing the production exact-plan wrapper in a private helper. Required behaviors:

```python
# first invocation commits A/B then callback raises on C
# resume validates A/B and invokes callback only for C/D
assert resumed_calls == [C.cell_id, D.cell_id]
```

Also test:

- existing persisted cells without `resume=True` fail;
- missing/corrupt COMPLETE blocks official analysis;
- valid completion permits metrics load;
- aggregate CSV is rebuilt from cell bundles, not trusted as resume state;
- CLI parses `--resume` and forwards it.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/phase07/test_execution.py tests/phase07/test_cli.py -q
```

Expected: FAIL because current official flow requires empty output and writes directly/non-atomically.

- [ ] **Step 3: Implement store-backed orchestration**

Remove `_persist_official_cell` direct writes from CLI. Initialize `Phase07Store`, validate identity/authorization/clean checkout, validate resume set, prepare cohorts lazily, run only remaining cells, and call `store.write_cell_bundle` after each successful `Phase07CellRun`.

On callback exception, persist invocation provenance atomically and re-raise. On success, rescan exact 200 cells, rebuild aggregate metrics, validate exact analysis matrix, then `mark_complete` last.

Official analysis loads only via `store.require_complete(expected_cells)` and `store.load_metrics(expected_cells)` before calling `build_phase07_pair_table`, `statistics_from_phase07_pairs`, and `adjudicate_phase07`.

- [ ] **Step 4: Verify GREEN**

```bash
pytest tests/phase07/test_store.py tests/phase07/test_execution.py tests/phase07/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/afmc_fm/phase07/execution.py src/afmc_fm/phase07/cli.py tests/phase07/test_execution.py tests/phase07/test_cli.py
git commit -m "feat: make Phase 0.7 official execution resumable"
```

---

### Task 5: Frozen 100/12 policy invariants and regression gate

**Files:**
- Modify: `tests/phase07/test_training_policy.py`

**Interfaces:**
- Tests shared `fit_phase05_model(..., stop_on_patience=...)` under both `flow_mode="none"` and `flow_mode="time_scaled"` with `max_epochs=100`, `patience=12`.

- [ ] **Step 1: Write invariant test**

Parameterize `_model(flow_mode)` over both frozen flow modes and use:

```python
config = Phase05Config(
    max_epochs=100,
    patience=12,
    learning_rate=1e-30,
    weight_decay=0.0,
)
```

For each flow mode, clone one initialization into standard/forced models. Assert:

```python
standard_summary["early_stop_reason"] == "patience_exhausted"
forced_summary["epochs_run"] == 100
forced_summary["early_stop_reason"] == "max_epochs_reached"
pd.testing.assert_frame_equal(
    standard.trace_frame().reset_index(drop=True),
    forced.trace_frame().iloc[: len(standard.trace_frame())].reset_index(drop=True),
    check_exact=True,
)
```

- [ ] **Step 2: Run targeted policy suite**

```bash
pytest tests/phase07/test_training_policy.py -q
```

Expected: PASS if shared stopping implementation satisfies frozen constants. If it fails, treat as a real blocker and fix only the minimum shared behavior under a new RED test.

- [ ] **Step 3: Run Phase 0.7 regression**

```bash
pytest tests/phase07 -q
```

Expected: PASS.

- [ ] **Step 4: Run neighboring historical regression**

```bash
pytest tests/phase05 tests/phase06 tests/phase07 -q
```

Expected: PASS except environment-dependent CUDA skips already present historically.

- [ ] **Step 5: Final exact-head verification**

```bash
ruff check src tests tools/reproducibility
pytest -q
afmc-phase07 smoke --device cpu
```

Expected: Ruff clean, full pytest green with only known CUDA-unavailable skips, CPU smoke reports `official_cells_executed=0` and `prefix_identical=true`.

- [ ] **Step 6: Commit**

```bash
git add tests/phase07/test_training_policy.py
git commit -m "test: freeze Phase 0.7 policy invariants at 100 epochs"
```

---

## Final review checklist

- No scientific constants changed.
- No protected seeds used.
- No official outputs committed.
- Authorization includes loaded dependency-config hashes.
- Dirty checkout blocks official run/resume.
- Any invalid R_SD replicate prevents heterogeneity support.
- Store validates exact identity and authoritative bundles.
- Resume skips only fully validated completed cells.
- Official completion requires exact 200/200.
- Official analysis cannot accept arbitrary CSV evidence.
- Both architectures are tested at max_epochs=100/patience=12.
- Full suite and CPU smoke green before CUDA handoff.
- Independent Codex audit repeats on final exact head before user CUDA smoke.