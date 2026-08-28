# Phase 0.7 Optimization Mediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive the exact Phase 0.6 post-hoc outputs with provenance, then implement—but do not execute—the frozen Phase 0.7 prospective stopping-policy intervention and adjudication pipeline.

**Architecture:** Keep the frozen Phase 0.6 archive immutable. Add repo-native post-hoc run/archive commands, a separate `docs/results/phase06_posthoc_optimization/` evidence namespace, then add Phase 0.7 as a new isolated protocol/planning/execution/analysis layer reusing the existing Phase 0.6 harness primitives. Phase 0.7 changes only stopping-policy behavior in its `forced_horizon` arm.

**Tech Stack:** Python 3.11+, pandas, NumPy, PyTorch, argparse, pytest, Ruff, existing AFMC Phase 0.6 store/protocol/execution infrastructure.

**Spec:** `docs/superpowers/specs/2026-08-27-phase0-7-optimization-mediation-design.md`

## Global Constraints

- Phase 0.6 terminal result remains `D4-B AMBIGUOUS -> STOP`.
- Never modify `docs/results/phase06/`.
- Protected confirmatory seeds remain cohort `701..710`, subset `801..810`, model `901..910`.
- Phase 0.7 is `smooth`, N=40, flows `{none,time_scaled}`, policies `{standard_early_stop,forced_horizon}` only.
- New Phase 0.7 seeds: contexts `(406,506)..(410,510)`, model seeds `1101..1110`.
- Planned official matrix: exactly 200 cells.
- No official Phase 0.7 training occurs as part of implementation or CI.
- Use TDD for every production behavior change.

---

### Task 1: Repo-native post-hoc execution command

**Files:**
- Modify: `src/afmc_fm/phase06/cli.py`
- Test: `tests/phase06/test_posthoc_tooling.py`

**Interfaces:**
- Consumes: `run_posthoc_archive_analysis(archive_root, output_dir, permutation_resamples=..., permutation_seed=...)`.
- Produces: CLI command `afmc-phase06 posthoc-optimization`.

- [ ] **Step 1: Write the failing parser/default test**

```python
def test_posthoc_cli_exposes_locked_defaults():
    parser = phase06_cli.build_parser()
    args = parser.parse_args(["posthoc-optimization"])
    assert args.archive == "docs/results/phase06/evidence/d4b"
    assert args.output == "outputs/phase06_posthoc_optimization"
    assert args.permutation_resamples == 10_000
    assert args.permutation_seed == 20260827
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
TMPDIR=/tmp python -m pytest -q tests/phase06/test_posthoc_tooling.py::test_posthoc_cli_exposes_locked_defaults
```

Expected: parser rejects `posthoc-optimization`.

- [ ] **Step 3: Implement the minimal CLI handler and parser**

Add a handler that calls the existing archive runner, prints classification, passing primary mechanisms, and the association table. Defaults must match the spec exactly.

- [ ] **Step 4: Verify GREEN**

Run the focused test plus `ruff check src tests`.

- [ ] **Step 5: Commit**

```bash
git add src/afmc_fm/phase06/cli.py tests/phase06/test_posthoc_tooling.py
git commit -m "feat: expose Phase 0.6 posthoc analysis CLI"
```

### Task 2: Exact post-hoc artifact archiver

**Files:**
- Create: `src/afmc_fm/phase06/posthoc_archive.py`
- Modify: `src/afmc_fm/phase06/cli.py`
- Test: `tests/phase06/test_posthoc_tooling.py`

**Interfaces:**
- Produces: `archive_posthoc_outputs(source_dir, destination_dir, analysis_commit) -> dict[str, object]`.
- Produces: CLI command `afmc-phase06 archive-posthoc-optimization`.

- [ ] **Step 1: Write failing tests**

Tests must require that the archiver:
- accepts only the four expected result files;
- copies them byte-for-byte into `docs/results/phase06_posthoc_optimization/analysis/`;
- rejects any destination inside `docs/results/phase06/`;
- verifies screening JSON says `exploratory_not_confirmatory=true` and `phase06_terminal_decision="AMBIGUOUS -> STOP"`;
- writes `execution_provenance.json` containing the supplied analysis commit and the screening classification;
- writes deterministic SHA-256 hashes for all four raw artifacts.

- [ ] **Step 2: Verify RED**

Run the two archiver tests and confirm failure because the module/command does not exist.

- [ ] **Step 3: Implement minimal archive module and CLI**

No statistical recomputation is permitted. The archiver copies and hashes the existing local artifacts only.

- [ ] **Step 4: Verify GREEN**

Run focused tests and Ruff.

- [ ] **Step 5: Commit**

```bash
git add src/afmc_fm/phase06/posthoc_archive.py src/afmc_fm/phase06/cli.py tests/phase06/test_posthoc_tooling.py
git commit -m "feat: archive exact Phase 0.6 posthoc artifacts"
```

### Task 3: Import the executed post-hoc evidence

**Files:**
- Existing: `docs/results/phase06_posthoc_optimization/decision_record.md`
- Create locally via archive command: `docs/results/phase06_posthoc_optimization/analysis/*`
- Create locally via archive command: `docs/results/phase06_posthoc_optimization/execution_provenance.json`
- Create locally via archive command: `docs/results/phase06_posthoc_optimization/MANIFEST.sha256`

- [ ] **Step 1: Pull the tooling commit into the WSL checkout**

- [ ] **Step 2: Archive the exact already-executed outputs**

```bash
afmc-phase06 archive-posthoc-optimization \
  --source outputs/phase06_posthoc_optimization \
  --destination docs/results/phase06_posthoc_optimization \
  --analysis-commit 8c9de2aaeee1e26f65e28a1dd682f8ac3effad72
```

- [ ] **Step 3: Verify hashes and git diff**

Run:

```bash
sha256sum -c docs/results/phase06_posthoc_optimization/MANIFEST.sha256
git diff --check
git status --short
```

Expected: all four artifact checksums pass; no files under `docs/results/phase06/` change.

- [ ] **Step 4: Commit the exact evidence**

```bash
git add docs/results/phase06_posthoc_optimization
git commit -m "results: archive Phase 0.6 posthoc optimization evidence"
```

### Task 4: Freeze Phase 0.7 protocol after human review

**Files:**
- Modify after approval: `docs/superpowers/specs/2026-08-27-phase0-7-optimization-mediation-design.md`
- Create: `configs/experiments/phase07.yaml`
- Create: `src/afmc_fm/phase07/config.py`
- Create: `src/afmc_fm/phase07/protocol.py`
- Test: `tests/phase07/test_config.py`, `tests/phase07/test_protocol.py`

**Interfaces:**
- Produces a hash-bound protocol containing exact seeds, 200-cell scope, stopping-policy definitions, bootstrap settings, success rule, and confirmatory firewall.

- [ ] **Step 1: Human reviews the design and explicitly authorizes freeze**

No production implementation proceeds past this point without that approval.

- [ ] **Step 2: Write failing config/protocol tests**

Tests must assert all frozen constants from the design, exact forbidden seeds, exact 10,000/bootstrap seed, and that a protocol lock binds the design SHA-256.

- [ ] **Step 3: Verify RED**

- [ ] **Step 4: Implement minimal config/protocol layer**

- [ ] **Step 5: Verify GREEN and commit**

### Task 5: Exact 200-cell Phase 0.7 planner

**Files:**
- Create: `src/afmc_fm/phase07/planning.py`
- Test: `tests/phase07/test_planning.py`

**Interfaces:**
- Produces: `plan_phase07_cells(config) -> tuple[Phase07Cell, ...]` containing exactly 200 unique cells.

- [ ] Write failing tests for exact counts, seed sets, flow/policy levels, and forbidden-seed rejection.
- [ ] Verify RED.
- [ ] Implement the minimal planner.
- [ ] Verify GREEN.
- [ ] Commit.

### Task 6: Forced-horizon execution semantics

**Files:**
- Create: `src/afmc_fm/phase07/execution.py`
- Modify only shared training code if a narrow injection point is required.
- Test: `tests/phase07/test_execution.py`

**Interfaces:**
- `standard_early_stop` reproduces D4-B stopping semantics.
- `forced_horizon` ignores patience as a termination action, trains to epoch 100, and records `would_patience_exhaust_epoch`.

- [ ] Write failing paired-policy tests proving all non-stopping semantics are identical.
- [ ] Write a failing test proving forced horizon reaches epoch 100 even when patience fires earlier.
- [ ] Verify RED.
- [ ] Implement only the stopping-policy injection.
- [ ] Verify GREEN plus relevant Phase 0.6 regression tests.
- [ ] Commit.

### Task 7: Phase 0.7 analysis and adjudication

**Files:**
- Create: `src/afmc_fm/phase07/analysis.py`
- Test: `tests/phase07/test_analysis.py`

**Interfaces:**
- Computes `Delta_standard`, `Delta_forced`, `G`, crossed-bootstrap CI for `mean(G)`, residualized `R_SD`, context/model positivity, and the exact six-part classification.

- [ ] Write synthetic fixtures with known SUPPORT and NOT_ESTABLISHED cases.
- [ ] Verify RED.
- [ ] Implement the deterministic 10,000-resample analysis and six-part gate.
- [ ] Verify GREEN.
- [ ] Commit.

### Task 8: Phase 0.7 CLI and non-execution verification

**Files:**
- Create: `src/afmc_fm/phase07/cli.py` or add an isolated `phase07` command surface following repository convention.
- Modify: `pyproject.toml` if a new console script is used.
- Test: `tests/phase07/test_cli.py`

**Interfaces:**
- Must expose planning/validation/smoke paths separately from official execution.

- [ ] Write failing parser/boundary tests.
- [ ] Verify RED.
- [ ] Implement CLI.
- [ ] Run CPU smoke only; do not execute official 200 cells.
- [ ] Run `ruff check src tests` and full `pytest -q`.
- [ ] Commit.

### Task 9: Final pre-execution audit

- [ ] Verify the Phase 0.7 design is marked frozen and hash-bound.
- [ ] Verify exact 200-cell plan before training.
- [ ] Verify no protected seed appears.
- [ ] Verify no code path can silently substitute standard stopping for forced horizon.
- [ ] Verify full CI is green at the intended execution commit.
- [ ] Record the validated implementation SHA.
- [ ] Stop and obtain explicit authorization before official Phase 0.7 execution.
