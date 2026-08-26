# Phase 0 CUDA-First Execution Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the serial Phase-0 benchmark execution path with a portable WSL/Linux-first, CUDA-aware, resumable, process-parallel execution engine without silently changing the scientific protocol.

**Architecture:** Keep simulator/model semantics in the existing research modules, add a small `execution` package that resolves runtime/device state, represents deterministic world/seed shards, persists completed cells atomically, schedules shards with `spawn`-based multiprocessing, and aggregates persisted cells back into the current Phase-0 artifacts. Torch-native neural models move to the selected device; Torch ridge and Torch MLP become declared pre-results baseline backends while sklearn HistGradientBoosting remains unchanged.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, pandas, SciPy, scikit-learn, `concurrent.futures.ProcessPoolExecutor`, `multiprocessing`, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-23-phase0-cuda-execution-design.md`

## Global Constraints

- Scientific protocol anchor is `be5a66b2e45362f60c90844e4e25673fb7bb3e21`.
- Canonical research runtime is native WSL/Linux with a Linux virtual environment and Linux Python executable.
- Windows-native and CPU-only Linux must remain supported.
- `--device auto` resolves to CUDA when available, otherwise CPU; `--device cuda` must fail if CUDA is unavailable.
- Multiprocessing must use the `spawn` context.
- No silent CPU fallback after a declared CUDA run starts.
- No changes to simulator worlds, train sizes, seed roles, patient splitting, flow-jump semantics, ablations, targets, losses, metrics, learning rate, weight decay, maximum epochs, or patience.
- The only declared baseline implementation revisions are Torch ridge and Torch MLP; sklearn HistGradientBoosting remains the gradient-boosting baseline.
- Existing CPU tests and flow-jump behavioral invariants must remain green.
- Completed work must be persisted atomically and resumable.
- Do not launch the official full benchmark until validation Gates A-F in the spec pass.

---

### Task 1: Runtime and device diagnostics

**Files:**
- Create: `src/afmc_fm/execution/__init__.py`
- Create: `src/afmc_fm/execution/device.py`
- Create: `tests/execution/test_device.py`
- Modify: `src/afmc_fm/cli.py`

**Interfaces:**
- Consumes: `torch.cuda.is_available()`, Python runtime metadata, OS environment.
- Produces: `resolve_device(requested: str) -> torch.device`, `runtime_diagnostics(requested_device: str, workers: int) -> dict[str, object]`, `move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]`.

- [ ] **Step 1: Write failing device-resolution tests**

Create tests that monkeypatch CUDA availability and assert:

```python
assert resolve_device("cpu").type == "cpu"
assert resolve_device("auto").type in {"cpu", "cuda"}
```

and that explicit `cuda` raises a clear `RuntimeError` when CUDA is unavailable.

- [ ] **Step 2: Run the focused test and confirm failure**

Run:

```bash
pytest tests/execution/test_device.py -v
```

Expected: import/function failures because `afmc_fm.execution.device` does not yet exist.

- [ ] **Step 3: Implement `device.py` minimally**

Implement:

```python
def resolve_device(requested: str) -> torch.device: ...
def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]: ...
def runtime_diagnostics(requested_device: str, workers: int) -> dict[str, object]: ...
```

Diagnostics must include Python executable, platform, WSL detection, PyTorch version, CUDA availability/runtime, GPU name when present, resolved device, CPU count, and worker count. Detect a Windows interpreter from WSL and expose a warning string rather than hard-failing.

- [ ] **Step 4: Add a CLI `diagnostics` subcommand**

Extend `src/afmc_fm/cli.py` so:

```bash
afmc-phase0 diagnostics --device auto --workers 4
```

prints stable JSON diagnostics.

- [ ] **Step 5: Run tests and Ruff**

Run:

```bash
pytest tests/execution/test_device.py tests/test_cli.py -v
ruff check src tests
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/afmc_fm/execution src/afmc_fm/cli.py tests/execution/test_device.py
git commit -m "feat: add runtime and device diagnostics"
```

---

### Task 2: Make neural training/evaluation device-aware

**Files:**
- Modify: `src/afmc_fm/experiments/runner.py`
- Modify: `tests/experiments/test_runner.py`
- Create: `tests/execution/test_cuda_smoke.py`

**Interfaces:**
- Consumes: `move_batch(...)`, `torch.device`.
- Produces: `_fit_neural(..., device: torch.device) -> nn.Module`, `_evaluate_neural(..., device: torch.device) -> dict[str, float]` with CPU NumPy metrics at the boundary.

- [ ] **Step 1: Write failing CPU device-plumbing tests**

Add a tiny neural benchmark test that explicitly passes `torch.device("cpu")` and verifies all current metrics still return finite scalars.

- [ ] **Step 2: Write CUDA smoke test guarded by availability**

Use:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
```

and run one one-epoch GRU/flow-jump smoke fit on CUDA.

- [ ] **Step 3: Run focused tests and confirm expected failure**

```bash
pytest tests/experiments/test_runner.py tests/execution/test_cuda_smoke.py -v
```

Expected: device argument/path does not yet exist.

- [ ] **Step 4: Update `_fit_neural`**

Move the model and both batches once:

```python
model = model.to(device)
train = move_batch(train, device)
validation = move_batch(validation, device)
```

Preserve AdamW, learning rate, weight decay, maximum epochs, patience, and best-state restoration exactly.

- [ ] **Step 5: Update `_evaluate_neural`**

Move the evaluation batch to the selected device and replace every direct `.numpy()` call with `.detach().cpu().numpy()`.

- [ ] **Step 6: Thread device through low-N and observation-shift execution**

Add a device parameter with CPU default at public runner boundaries so existing callers remain valid while the new scheduler can select CUDA explicitly.

- [ ] **Step 7: Run regression tests**

```bash
pytest tests/models tests/experiments -v
ruff check src tests
```

Expected: all existing CPU invariants pass; CUDA smoke passes only on GPU machines.

- [ ] **Step 8: Commit**

```bash
git add src/afmc_fm/experiments/runner.py tests/experiments/test_runner.py tests/execution/test_cuda_smoke.py
git commit -m "feat: make neural benchmark device aware"
```

---

### Task 3: Add validated Torch ridge backend

**Files:**
- Modify: `src/afmc_fm/models/baselines.py`
- Modify: `tests/models/test_baselines.py`

**Interfaces:**
- Produces: `TorchRidgeRegressor(alpha: float = 1.0, device: torch.device | str = "cpu")` with `fit`, `predict`, and `trainable_parameter_count` if needed by the runner.
- Reference: existing `ProbeRegressor` remains available for parity tests.

- [ ] **Step 1: Write failing parity tests**

Generate fixed well-conditioned matrices and compare predictions from sklearn `ProbeRegressor` and Torch ridge on CPU with a strict tolerance, including intercept semantics.

- [ ] **Step 2: Run the test and confirm failure**

```bash
pytest tests/models/test_baselines.py -k ridge -v
```

- [ ] **Step 3: Implement Torch ridge with stable linear algebra**

Prefer centering plus regularized Cholesky/solve rather than explicit matrix inversion. Preserve sklearn Ridge's unpenalized intercept semantics.

- [ ] **Step 4: Add CUDA parity smoke**

When CUDA is available, fit the same problem on CUDA and compare CPU/CUDA predictions within documented float32/float64 tolerance.

- [ ] **Step 5: Update the runner backend selection**

Use Torch ridge for `engineered_linear` and `representation_linear` in the optimized backend while retaining `ProbeRegressor` as the explicit reference backend.

- [ ] **Step 6: Run tests**

```bash
pytest tests/models/test_baselines.py tests/experiments/test_runner.py -v
ruff check src tests
```

- [ ] **Step 7: Commit**

```bash
git add src/afmc_fm/models/baselines.py src/afmc_fm/experiments/runner.py tests/models/test_baselines.py tests/experiments/test_runner.py
git commit -m "feat: add torch ridge baseline backend"
```

---

### Task 4: Add declared Torch representation MLP backend

**Files:**
- Modify: `src/afmc_fm/models/baselines.py`
- Modify: `src/afmc_fm/experiments/runner.py`
- Modify: `tests/models/test_baselines.py`
- Modify: `tests/experiments/test_runner.py`

**Interfaces:**
- Produces: `TorchMLPRegressorBaseline(input_dim: int, seed: int, device: torch.device | str)` implementing one hidden layer of width 32 with tanh activation and L2 regularization.
- Reference: existing sklearn `MLPRegressorBaseline` remains available for audit runs.

- [ ] **Step 1: Write failing architecture/parameter-count tests**

For a 19-dimensional representation probe, assert the Torch network is:

```text
Linear(19, 32) -> Tanh -> Linear(32, 1)
```

and parameter count is exactly `(19*32+32) + (32*1+1) = 673`.

- [ ] **Step 2: Write failing nonlinear positive-control test**

Reuse the existing quadratic-signal test and require finite predictions plus MSE below the current positive-control threshold.

- [ ] **Step 3: Implement the Torch MLP and LBFGS-family training path**

Use `torch.optim.LBFGS` with deterministic seed initialization, one-hidden-layer tanh architecture, and the declared L2 objective. Do not silently substitute AdamW.

- [ ] **Step 4: Add repeatability tests**

Two CPU fits with the same seed must agree within strict tolerance. CUDA fits with the same seed must be repeatable within documented CUDA tolerance.

- [ ] **Step 5: Switch optimized runner backend**

Make `representation_mlp` use the Torch backend in optimized execution while retaining a selectable sklearn reference backend for audit/parity commands.

- [ ] **Step 6: Run tests**

```bash
pytest tests/models/test_baselines.py tests/experiments/test_runner.py -v
ruff check src tests
```

- [ ] **Step 7: Commit**

```bash
git add src/afmc_fm/models/baselines.py src/afmc_fm/experiments/runner.py tests/models/test_baselines.py tests/experiments/test_runner.py
git commit -m "feat: add torch representation mlp backend"
```

---

### Task 5: Extract deterministic shard specifications and shard execution

**Files:**
- Create: `src/afmc_fm/execution/jobs.py`
- Modify: `src/afmc_fm/experiments/runner.py`
- Create: `tests/execution/test_jobs.py`

**Interfaces:**
- Produces: immutable `ShardSpec(world: str, cohort_seed: int, subset_seed: int, model_seed: int)` and deterministic `shard_id`.
- Produces: `plan_shards(experiment: ExperimentConfig, supplied_seed: int) -> tuple[ShardSpec, ...]`.
- Produces: `run_shard(spec: ShardSpec, sim_config: SimulatorConfig, experiment: ExperimentConfig, device: torch.device, cell_callback: Callable[[CellResult], None] | None = None) -> pd.DataFrame`.

- [ ] **Step 1: Write failing 25-shard planning test**

Load the full experiment config and assert exactly 25 unique shard IDs from five worlds times five seed bundles.

- [ ] **Step 2: Write seed-order invariance tests**

Assert each `ShardSpec` preserves the matched `(cohort_seed, subset_seed, model_seed)` tuple and never creates Cartesian products between seed roles.

- [ ] **Step 3: Extract a one-shard runner from current benchmark logic**

Refactor without changing scientific semantics so one shard simulates its cohort once, constructs sequences once, and executes all configured train sizes/models/ablations for that seed bundle.

- [ ] **Step 4: Preserve existing public benchmark wrappers**

Keep `run_low_n_benchmark` and `run_observation_shift_benchmark` working for old tests by implementing them in terms of the new one-shard primitives where practical.

- [ ] **Step 5: Run runner regression tests**

```bash
pytest tests/experiments/test_runner.py tests/execution/test_jobs.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/afmc_fm/execution/jobs.py src/afmc_fm/experiments/runner.py tests/execution/test_jobs.py tests/experiments/test_runner.py
git commit -m "refactor: expose deterministic benchmark shards"
```

---

### Task 6: Add atomic per-cell persistence and resume validation

**Files:**
- Create: `src/afmc_fm/execution/persistence.py`
- Create: `tests/execution/test_persistence.py`

**Interfaces:**
- Produces: `RunStore(output: Path, identity: RunIdentity)`.
- Produces methods: `write_cell`, `load_completed_cell_ids`, `mark_shard_complete`, `validate_resume`, `iter_metric_rows`.
- Uses deterministic cell IDs built from benchmark type, shard, train size, model, and ablation.

- [ ] **Step 1: Write failing atomic-write test**

Write a cell through the store and assert only the final `.json` exists after success and contains complete metadata plus metric rows.

- [ ] **Step 2: Write incompatible-resume test**

Create a store with one config hash, reopen with a different hash, and require a hard failure instead of mixing results.

- [ ] **Step 3: Implement atomic write via sibling temp file and `Path.replace`**

Persist JSON with stable sorting and a schema version. Write `COMPLETE` only after all expected cell IDs for the shard exist and validate.

- [ ] **Step 4: Implement resume inspection**

Return the valid completed cell IDs so shard execution can skip them after rebuilding its in-memory cohort/sequences.

- [ ] **Step 5: Run tests**

```bash
pytest tests/execution/test_persistence.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/afmc_fm/execution/persistence.py tests/execution/test_persistence.py
git commit -m "feat: persist and resume benchmark cells"
```

---

### Task 7: Connect shard execution to persistence callbacks

**Files:**
- Modify: `src/afmc_fm/execution/jobs.py`
- Modify: `src/afmc_fm/experiments/runner.py`
- Modify: `src/afmc_fm/execution/persistence.py`
- Modify: `tests/execution/test_jobs.py`
- Modify: `tests/execution/test_persistence.py`

**Interfaces:**
- `run_shard(..., completed_cell_ids: frozenset[str], on_cell_complete: Callable[[CellResult], None])` skips completed cells and emits each newly completed cell immediately.

- [ ] **Step 1: Write interrupted-run reproduction test**

Configure a tiny shard, persist the first cell, simulate interruption, rerun with that cell ID preloaded, and assert the first cell is not recomputed.

- [ ] **Step 2: Add cell-level callback to low-N loops**

After each model/ablation/train-size result is complete, build one `CellResult` containing all metric rows and provenance fields and call the persistence callback before continuing.

- [ ] **Step 3: Add observation-shift cell identity**

Ensure site-shift cells are distinguishable from ordinary low-N cells and that their raw site and delta metric rows are stored together under one fitted model cell.

- [ ] **Step 4: Run resume tests**

```bash
pytest tests/execution/test_jobs.py tests/execution/test_persistence.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/afmc_fm/execution src/afmc_fm/experiments/runner.py tests/execution
git commit -m "feat: checkpoint benchmark cells during shard execution"
```

---

### Task 8: Add spawn-based process scheduler and progress events

**Files:**
- Create: `src/afmc_fm/execution/scheduler.py`
- Create: `tests/execution/test_scheduler.py`

**Interfaces:**
- Produces: `ExecutionOptions(device: str, workers: int, resume: bool, fail_fast: bool)`.
- Produces: `run_scheduled_benchmark(sim_config, experiment, output: Path, options: ExecutionOptions) -> pd.DataFrame`.

- [ ] **Step 1: Write failing serial-vs-parallel CPU equivalence test**

Use a tiny two-shard experiment. Run once with one worker and once with two workers, sort by scientific key, and require identical cell IDs and machine-tolerance-equivalent metrics.

- [ ] **Step 2: Implement `spawn` process context explicitly**

Use:

```python
ctx = multiprocessing.get_context("spawn")
ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
```

Do not rely on platform default start methods.

- [ ] **Step 3: Limit per-worker CPU threading**

Inside each worker set PyTorch CPU threads to 1 and set numerical-library thread environment variables before heavy fitting where appropriate.

- [ ] **Step 4: Implement progress accounting**

Coordinator prints/logs completed shards, completed cells, expected cells, failures, elapsed wall time, active workers, and resolved device mode. Completion percentage is based on cells, not claimed as an exact ETA.

- [ ] **Step 5: Implement failure records**

A worker exception must produce structured failure metadata. With `fail_fast=False`, independent shards continue; with `fail_fast=True`, cancel pending futures and propagate the failure.

- [ ] **Step 6: Run scheduler tests**

```bash
pytest tests/execution/test_scheduler.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/afmc_fm/execution/scheduler.py tests/execution/test_scheduler.py
git commit -m "feat: schedule benchmark shards in parallel"
```

---

### Task 9: Deterministic aggregation and full manifest

**Files:**
- Create: `src/afmc_fm/execution/manifest.py`
- Modify: `src/afmc_fm/execution/persistence.py`
- Modify: `src/afmc_fm/execution/scheduler.py`
- Modify: `src/afmc_fm/cli.py`
- Create: `tests/execution/test_manifest.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `build_run_manifest(...) -> dict[str, object]`.
- Produces: deterministic aggregation from persisted cells to `metrics.csv`, `ablation_metrics.csv`, `gate_summary.csv`, and `learning_curves.png`.

- [ ] **Step 1: Write failing manifest-content test**

Assert the manifest records protocol anchor, execution commit, config hashes, Python/library versions, OS/WSL, CUDA/GPU details, worker count, backend IDs, start/end times, and expected/completed/failed counts.

- [ ] **Step 2: Implement stable config hashing**

Hash canonical JSON serializations of simulator and experiment configs. Never use Python object hash values.

- [ ] **Step 3: Implement deterministic aggregation**

Read all valid cell files, concatenate metric rows, sort by a fixed scientific key, and then generate the same four public artifacts the current CLI generates.

- [ ] **Step 4: Add `benchmark` CLI options**

Extend the current command with:

```text
--device auto|cpu|cuda
--workers N
--resume
--fail-fast
```

and route to `run_scheduled_benchmark`.

- [ ] **Step 5: Add a no-training re-aggregation path**

Provide an internal function or CLI option that rebuilds derived CSV/plot artifacts from persisted cells without fitting models again.

- [ ] **Step 6: Run tests**

```bash
pytest tests/execution/test_manifest.py tests/test_cli.py -v
ruff check src tests
```

- [ ] **Step 7: Commit**

```bash
git add src/afmc_fm/execution src/afmc_fm/cli.py tests/execution/test_manifest.py tests/test_cli.py
git commit -m "feat: aggregate resumable benchmark outputs"
```

---

### Task 10: Prove interruption/resume end-to-end

**Files:**
- Create: `tests/execution/test_resume_end_to_end.py`

**Interfaces:**
- Exercises the public scheduled benchmark path only.

- [ ] **Step 1: Build a tiny fixed benchmark config in the test**

Use one world, one seed bundle, `N=5`, one linear and one neural model, and one epoch for the neural model.

- [ ] **Step 2: Run an uninterrupted reference**

Persist to one temporary directory and capture final sorted metrics.

- [ ] **Step 3: Simulate interruption after the first persisted cell**

Use a test hook that raises after the first successful `write_cell`.

- [ ] **Step 4: Resume the interrupted directory**

Rerun with `resume=True` and assert the completed cell's timestamp/content is unchanged and only unfinished cells execute.

- [ ] **Step 5: Compare final aggregates**

Require the resumed final metrics to match the uninterrupted CPU reference to machine tolerance.

- [ ] **Step 6: Commit**

```bash
git add tests/execution/test_resume_end_to_end.py
git commit -m "test: verify end to end benchmark resume"
```

---

### Task 11: Native WSL runtime documentation and guardrails

**Files:**
- Modify: `README.md`
- Modify: `src/afmc_fm/execution/device.py`
- Modify: `tests/test_readme.py`

**Interfaces:**
- Documents canonical environment setup without making Windows-native unsupported.

- [ ] **Step 1: Add README assertions first**

Extend README tests to require documented native WSL setup and the diagnostics command.

- [ ] **Step 2: Document canonical setup**

Include:

```bash
cd ~/afmc-clinical-fm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
which python
python -m afmc_fm.cli diagnostics --device auto --workers 1
```

Explicitly warn against invoking `.venv/Scripts/python.exe` or `.venv/Scripts/afmc-phase0.exe` from WSL.

- [ ] **Step 3: Document CUDA verification**

Show `torch.cuda.is_available()`, `torch.version.cuda`, and GPU-name checks.

- [ ] **Step 4: Run docs tests**

```bash
pytest tests/test_readme.py tests/test_cli.py -v
```

- [ ] **Step 5: Commit**

```bash
git add README.md src/afmc_fm/execution/device.py tests/test_readme.py
git commit -m "docs: standardize native wsl benchmark runtime"
```

---

### Task 12: Full validation gates and throughput calibration

**Files:**
- Modify only if a genuine defect is found by a failing test; otherwise no source changes.
- Record validation output in: `docs/superpowers/validation/2026-08-23-phase0-cuda-execution-validation.md`

**Interfaces:**
- Validates Gates A-F from the design spec before any official full run.

- [ ] **Step 1: Gate A - run the complete regression suite**

```bash
ruff check src tests
pytest -q
```

Record exact pass count and versions.

- [ ] **Step 2: Gate B - native WSL/CUDA diagnostics**

```bash
afmc-phase0 diagnostics --device auto --workers 1
```

Require a Linux Python executable under `.venv/bin/python` and CUDA-visible RTX 4060 on the target machine.

- [ ] **Step 3: Gate C - baseline validation**

Run the focused Torch ridge and Torch MLP parity/positive-control tests on CPU and CUDA.

- [ ] **Step 4: Gate D - scheduler equivalence**

Run the fixed compact CPU matrix once serially and once with multiple workers; compare sorted metrics and cell identities.

- [ ] **Step 5: Gate E - resume correctness**

Run the end-to-end interruption/resume test and record that persisted cells were not recomputed.

- [ ] **Step 6: Gate F - RTX 4060 throughput calibration**

Run the same fixed smoke matrix with candidate worker counts `1`, `2`, and `4`. Record wall time, cells/minute, peak GPU memory, and failures. Choose the worker count with the best stable throughput; do not infer from GPU utilization alone.

- [ ] **Step 7: Write validation record**

Document commands, exact execution commit, environment, selected worker count, and whether every gate passed. If any gate fails, the official full benchmark remains blocked.

- [ ] **Step 8: Commit**

```bash
git add docs/superpowers/validation/2026-08-23-phase0-cuda-execution-validation.md
git commit -m "docs: validate cuda execution engine"
```

---

### Task 13: Launch the first official optimized Phase-0 benchmark

**Files:**
- No source changes permitted during launch.
- Output: a new directory such as `outputs/phase0_full_cuda_<execution-sha>/`.

**Interfaces:**
- Uses the validated CLI and frozen full configs.

- [ ] **Step 1: Verify clean repository and exact execution commit**

```bash
git status --short
git rev-parse HEAD
```

Require a clean tree and record the SHA.

- [ ] **Step 2: Run preflight diagnostics**

```bash
afmc-phase0 diagnostics --device auto --workers <validated-workers>
```

- [ ] **Step 3: Launch the full matrix**

```bash
afmc-phase0 benchmark \
  --sim-config configs/simulator/full.yaml \
  --exp-config configs/experiments/low_n.yaml \
  --output outputs/phase0_full_cuda_<execution-sha> \
  --device auto \
  --workers <validated-workers> \
  --resume
```

- [ ] **Step 4: Monitor progress without changing protocol**

Use persisted cell/shard counts and structured failures. Do not tune hyperparameters, worker-independent scientific settings, or selectively rerun favorable cells.

- [ ] **Step 5: Perform completeness audit before interpretation**

Confirm expected shard/cell counts, no incompatible manifest state, no duplicate metric keys, finite metrics where required, explicit failure count, backend provenance, and complete aggregate artifacts.

- [ ] **Step 6: Only then begin scientific interpretation**

Evaluate low-N learning curves, primary comparator wins, calibration, latent recovery, misspecified-world competitiveness, observation-head robustness, and warning/failure patterns against the predeclared go/no-go criteria.
