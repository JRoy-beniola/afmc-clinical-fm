# Reproducibility R4: Isolated Historical Rerun and Scientific Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe historical rerun orchestration, explicit environment reconstruction labeling, layered structural/numerical/scientific comparison, and source-first closure for post-Phase-0.6/future reporting without authorizing Phase 0.7 execution.

**Architecture:** R4 separates planning, execution, and comparison. Historical reruns are always launched in an isolated detached Git worktree at the phase's recorded implementation SHA, with output paths forced beneath `outputs/reproduction/<phase>/rerun/<run-id>/`. Each phase adapter declares the historically authentic command, required evidence/parents, environment evidence, protected-seed policy, and comparison policy. The runner refuses unsupported or under-specified reruns rather than falling back to current HEAD. Layered comparison returns `EXACT`, `NUMERICALLY_REPRODUCED`, `SCIENTIFICALLY_REPRODUCED`, or `FAILED_REPRODUCTION` while preserving lower-layer failures.

**Tech Stack:** Python 3.11 standard library (`subprocess`, `tempfile`, `json`, `hashlib`, `platform`, `importlib.metadata`), NumPy/Pandas, PyYAML, pytest, Ruff, Git worktrees.

**Spec:** `docs/superpowers/specs/2026-08-27-reproducibility-closure-design.md`

## Global Constraints

- Historical scientific decisions are immutable; Phase 0.6 remains `D4-B AMBIGUOUS -> STOP`.
- A historical rerun must never silently use current HEAD.
- Fresh outputs must never write under `docs/results/`.
- Environment reconstruction is labeled `reconstructed`, never `exact`, unless an exact historical lock/container is independently present and verified.
- Reserved confirmatory seed namespaces remain forbidden except for an explicitly authorized historical rerun of a phase that originally used those exact seeds; no new confirmatory experiment is authorized by R4.
- No Phase 0.7 execution is implemented or triggered by this plan.
- Expensive historical reruns are not required in ordinary CI; CI tests orchestration with fixture Git repositories and small smoke commands.

---

### Task 1: Historical environment evidence and capture

**Files:**
- Create: `src/afmc_fm/reproducibility/environment.py`
- Create: `tests/reproducibility/test_environment.py`

**Interfaces:**
- Produces: `EnvironmentRecord(status, python_version, platform, packages, torch, cuda, gpu, source_paths, limitations)`
- Produces: `classify_historical_environment(root: Path, phase: PhaseDefinition) -> EnvironmentRecord`
- Produces: `capture_current_environment() -> EnvironmentRecord`

- [ ] **Step 1: Write failing classification tests**

Fixture tests must distinguish:

- exact: an explicitly registered immutable environment lock/container hash exists and validates;
- reconstructed: dependency/project metadata exists but no exact execution snapshot;
- unknown: insufficient evidence exists.

A requirements file created after execution cannot produce `exact` merely because its packages look plausible.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_environment.py -q`

- [ ] **Step 3: Implement evidence-based classification**

Capture package versions with `importlib.metadata`, Python/platform with stdlib, and Torch/CUDA/GPU only when available. Persist no file during classification. Current environment capture is explicitly a *new reproduction environment record*, not evidence of the historical environment.

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_environment.py -q && ruff check src tests`

```bash
git add src/afmc_fm/reproducibility/environment.py tests/reproducibility/test_environment.py
git commit -m "feat: classify and capture reproduction environments"
```

---

### Task 2: Isolated historical Git worktree runner

**Files:**
- Create: `src/afmc_fm/reproducibility/worktree.py`
- Create: `tests/reproducibility/test_worktree.py`

**Interfaces:**
- Produces: `historical_worktree(repository: Path, sha: str) -> ContextManager[Path]`
- Produces: `run_in_worktree(worktree: Path, command: tuple[str, ...], env: Mapping[str, str], log_path: Path) -> CompletedProcess[str]`

- [ ] **Step 1: Write a fixture Git-repository test**

Create two commits where current HEAD contains `CURRENT` and the historical commit contains `HISTORICAL`. Enter `historical_worktree(repo, historical_sha)` and assert `git rev-parse HEAD` equals the historical SHA and the file content is `HISTORICAL`.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_worktree.py -q`

- [ ] **Step 3: Implement worktree isolation**

Use:

```text
git worktree add --detach <temporary-path> <recorded-sha>
```

Then verify the resolved HEAD equals the requested 40-character SHA before yielding. Cleanup uses `git worktree remove --force` and never deletes an existing user worktree. Reject dirty repository metadata operations that would require force-moving refs.

- [ ] **Step 4: Add current-HEAD fallback regression**

Monkeypatch the requested SHA to a nonexistent commit and require a hard failure. The code must not retry with current HEAD or branch name.

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_worktree.py -q && ruff check src tests`

```bash
git add src/afmc_fm/reproducibility/worktree.py tests/reproducibility/test_worktree.py
git commit -m "feat: isolate historical reruns in detached worktrees"
```

---

### Task 3: Phase rerun specifications and safety policy

**Files:**
- Create: `src/afmc_fm/reproducibility/rerun_models.py`
- Create: `docs/reproducibility/phase0/rerun.yaml`
- Create: `docs/reproducibility/phase05/rerun.yaml`
- Create: `docs/reproducibility/phase06/rerun.yaml`
- Create: `tests/reproducibility/test_rerun_models.py`

**Interfaces:**
- Produces: `RerunSpec(phase_id, implementation_sha, command, environment, required_paths, required_parent_bindings, seed_policy, comparison_policy)`
- Produces: `load_rerun_spec(path: Path) -> RerunSpec`
- Produces: `validate_rerun_spec(root: Path, phase: PhaseDefinition, spec: RerunSpec) -> tuple[CheckResult, ...]`

- [ ] **Step 1: Write failing safety tests**

Reject specs that:

- use `HEAD`, a branch name, or a SHA different from the registry's historical implementation SHA;
- write output beneath `docs/results/`;
- declare `seed_policy: unrestricted`;
- reference a protected-confirmatory namespace without `historical_exact_only: true` and a recorded historical seed set;
- omit required immutable parent bindings for an execution mode that historically required them.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_rerun_models.py -q`

- [ ] **Step 3: Encode authentic historical commands conservatively**

Use repository evidence only. Where a complete current-repository rerun command cannot be justified, set `supported: false` with a concrete `blocked_reason` instead of inventing one. Phase 0.6 must preserve the documented execution-kit boundaries; R4 does not authorize D4-D, capacity/time, full-factorial expansion, Phase 0.5 continuation, or confirmatory-seed execution.

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_rerun_models.py -q && ruff check src tests`

```bash
git add src/afmc_fm/reproducibility/rerun_models.py docs/reproducibility/phase0/rerun.yaml docs/reproducibility/phase05/rerun.yaml docs/reproducibility/phase06/rerun.yaml tests/reproducibility/test_rerun_models.py
git commit -m "feat: bind historical rerun specifications"
```

---

### Task 4: Rerun planner/executor and output isolation

**Files:**
- Create: `src/afmc_fm/reproducibility/rerun.py`
- Create: `tests/reproducibility/test_rerun.py`

**Interfaces:**
- Produces: `plan_rerun(root: Path, phase_id: str, *, run_id: str | None = None) -> RerunPlan`
- Produces: `execute_rerun(root: Path, plan: RerunPlan) -> RerunExecution`
- Default output: `outputs/reproduction/<phase>/rerun/<run-id>/`

- [ ] **Step 1: Write failing fixture smoke**

Create a temporary Git repo with a historical commit containing a tiny Python script. `plan_rerun` must bind that commit and isolated output. `execute_rerun` must run the historical script in a detached worktree, emit execution/environment provenance, and leave the fixture official archive unchanged.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_rerun.py -q`

- [ ] **Step 3: Implement planning**

Planning performs no execution and returns explicit readiness:

```text
READY
BLOCKED_MISSING_HISTORY
BLOCKED_MISSING_ENVIRONMENT
BLOCKED_MISSING_PARENT
BLOCKED_POLICY
UNSUPPORTED
```

A blocked plan contains reasons and required remediation. It may never silently relax a missing prerequisite.

- [ ] **Step 4: Implement execution**

Execution requires `plan.status == READY`. It:

1. snapshots all historical evidence roots;
2. creates the isolated output directory;
3. creates a detached historical worktree at the bound SHA;
4. captures the new reproduction environment;
5. injects only declared output/parent environment variables;
6. executes the declared command;
7. writes `execution.json`, stdout/stderr logs, and an output inventory;
8. verifies historical archive snapshots unchanged.

- [ ] **Step 5: Add failure-path tests**

A nonzero child process returns a failed execution record without moving/copying partial output into official evidence. A missing SHA, parent, or policy requirement fails before process launch.

- [ ] **Step 6: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_rerun.py tests/reproducibility/test_worktree.py -q && ruff check src tests`

```bash
git add src/afmc_fm/reproducibility/rerun.py tests/reproducibility/test_rerun.py
git commit -m "feat: orchestrate isolated historical reruns"
```

---

### Task 5: Layered reproduction comparison

**Files:**
- Create: `src/afmc_fm/reproducibility/comparison.py`
- Create: `tests/reproducibility/test_comparison.py`

**Interfaces:**
- Produces: `ComparisonPolicy` with structural keys, exact files, numeric tables/columns, per-field `atol`/`rtol`, and scientific decision resolver.
- Produces: `compare_reproduction(official_root: Path, rerun_root: Path, policy: ComparisonPolicy) -> ReproductionComparison`
- Verdicts exactly: `EXACT`, `NUMERICALLY_REPRODUCED`, `SCIENTIFICALLY_REPRODUCED`, `FAILED_REPRODUCTION`.

- [ ] **Step 1: Write failing verdict-lattice tests**

Construct fixture official/rerun outputs for:

1. exact identity -> `EXACT`;
2. deterministic metadata differs but numerical values are within tolerance -> `NUMERICALLY_REPRODUCED`;
3. numerical tolerance fails but the frozen scientific decision resolves identically -> `SCIENTIFICALLY_REPRODUCED`;
4. scientific decision differs -> `FAILED_REPRODUCTION`.

Assert lower-layer failures remain visible in all non-EXACT reports.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_comparison.py -q`

- [ ] **Step 3: Implement layers**

Layer 1: structural row/key/config/stopping semantics.

Layer 2: exact SHA/equality for deterministic files declared exact.

Layer 3: numeric tables with declared per-column `atol`/`rtol` using NumPy; NaN handling must be explicit in policy.

Layer 4: scientific decision resolver returns the phase's frozen classification from rerun evidence and must equal the registered historical classification after the same normalization used by R1 verification.

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_comparison.py -q && ruff check src tests`

```bash
git add src/afmc_fm/reproducibility/comparison.py tests/reproducibility/test_comparison.py
git commit -m "feat: add layered scientific reproduction comparison"
```

---

### Task 6: Public `rerun` CLI with explicit execution acknowledgement

**Files:**
- Modify: `src/afmc_fm/reproducibility/cli.py`
- Modify: `tests/reproducibility/test_cli.py`

**Interfaces:**
- Adds: `afmc-reproduce rerun <phase>` -> plan/readiness only by default.
- Adds: `afmc-reproduce rerun <phase> --execute` -> execute only a `READY` historical plan.
- Adds: `--run-id <id>`.

- [ ] **Step 1: Write failing CLI tests**

Default invocation must not start a process. `--execute` on `UNSUPPORTED` or blocked plans exits nonzero and prints reasons. A fixture READY plan executes and prints isolated worktree SHA, environment label, output directory, and final comparison verdict if comparison inputs exist.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_cli.py -q`

- [ ] **Step 3: Implement CLI**

The command must never offer `phase07`. `rerun all` is intentionally not implemented: expensive historical reruns require explicit per-phase acknowledgement. No command auto-promotes output into official evidence.

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_cli.py tests/reproducibility/test_rerun.py -q && ruff check src tests`

```bash
git add src/afmc_fm/reproducibility/cli.py tests/reproducibility/test_cli.py
git commit -m "feat: expose guarded historical rerun CLI"
```

---

### Task 7: Source-first post-Phase-0.6/future reporting policy

**Files:**
- Create: `docs/reproducibility/phase06_posthoc/report-source.md`
- Create: `docs/reproducibility/phase06_posthoc/report.yaml`
- Create: `docs/reproducibility/source-first-policy.md`
- Modify: `src/afmc_fm/reproducibility/registry.py`
- Modify: `docs/reproducibility/artifact-map.yaml`
- Test: `tests/reproducibility/test_source_first.py`

**Interfaces:**
- Post-Phase-0.6 documentary source is canonical source-first, not reverse-extracted from an official DOCX because no official DOCX is registered.
- Future phase manifests require source, environment capture declaration, editable diagram source where diagrams exist, and an explicit archive/freeze operation outside `rebuild`/`rerun`.

- [ ] **Step 1: Write failing policy/source tests**

Require post-hoc `report_source` to be non-null, classification to remain exploratory, and source-first manifest to bind only archived post-hoc evidence and the existing decision record. Reject any source-first manifest that points official evidence into `outputs/reproduction/`.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_source_first.py -q`

- [ ] **Step 3: Add source-first post-hoc documentary source**

Use the existing archived decision record and post-hoc tables/JSON as source material. Do not create an official historical report that did not exist. The source explicitly states its exploratory classification and that Phase 0.6 remains unchanged.

- [ ] **Step 4: Document prospective capture requirements**

`source-first-policy.md` requires future official executions to capture Python/package/platform/CUDA/GPU/Git/config hashes at execution time and future reports to keep narrative/table/figure/diagram source in Git before archival.

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_source_first.py tests/reproducibility/test_documentation.py -q && ruff check src tests`

```bash
git add docs/reproducibility/phase06_posthoc docs/reproducibility/source-first-policy.md src/afmc_fm/reproducibility/registry.py docs/reproducibility/artifact-map.yaml tests/reproducibility/test_source_first.py
git commit -m "docs: make post-Phase-0.6 reporting source-first"
```

---

### Task 8: Final reproducibility closure gate

**Files:**
- Modify: `docs/reproducibility/README.md`
- Modify: `tests/reproducibility/test_immutability.py`

- [ ] **Step 1: Full historical immutability test**

Exercise `verify`, supported `rebuild`, rerun planning, environment classification, and a fixture rerun. Assert all four real historical evidence roots remain byte-identical and no test writes official evidence.

- [ ] **Step 2: Verify public status semantics**

`afmc-reproduce status` must report per phase:

- archive/manifest status;
- documentary source status;
- environment status;
- rebuild support;
- rerun support/readiness;
- historical decision and result kind.

Support/readiness must remain honest when missing historical checkpoints, parents, environment snapshots, or commands prevent a real rerun.

- [ ] **Step 3: Run reproducibility suite**

Run: `pytest tests/reproducibility -q`

- [ ] **Step 4: Run Ruff**

Run: `ruff check src tests tools/reproducibility`

- [ ] **Step 5: Run full suite**

Run: `pytest -q`

Expected: all tests green; only already-established CUDA-unavailable skips remain.

- [ ] **Step 6: Audit diff boundary**

Run:

```bash
git diff --name-only <R4-base-sha>...HEAD -- docs/results/
```

Expected: no output.

Also verify no Phase 0.7 execution output, no protected-confirmatory output, and no official archive promotion was produced.

- [ ] **Step 7: Update README and commit closure**

Document the distinction between software support and historical readiness. It is acceptable and required for a phase to report `rerun_supported=false` or a blocked readiness status when the archive genuinely lacks necessary execution artifacts; R4 must not fabricate completeness.

```bash
git add docs/reproducibility/README.md tests/reproducibility/test_immutability.py
git commit -m "test: close reproducibility through post-Phase-0.6"
```

## R4 Completion Gate

R4 is complete when historical environment evidence is honestly classified, rerun planning/execution cannot use current HEAD or official output paths, fixture reruns prove worktree/output isolation, phase rerun specs are provenance-bound and policy-safe, layered comparison returns deterministic verdicts, post-Phase-0.6 reporting is source-first, all historical archives remain byte-identical, Ruff is clean, the complete pytest suite is green, and no Phase 0.7 or new confirmatory execution has occurred. Real historical rerun readiness may remain blocked where the historical archive lacks necessary checkpoints, parents, or exact environment evidence; that limitation must be surfaced rather than hidden.
