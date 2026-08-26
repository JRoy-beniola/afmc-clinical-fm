# Phase 0 CUDA execution engine validation

Date: 2026-08-23
Validation completed: 2026-08-23T13:49:39Z
Branch: `phase0-synthetic-harness`
Starting/execution SHA: `9186b7d3027c0fa26e0e86c28b5be2cdb19415b3`
Scientific protocol anchor: `be5a66b2e45362f60c90844e4e25673fb7bb3e21`

## Verdict

All validation Gates A-F passed. The selected CUDA worker count is **2**, based
on the highest stable completed-cell throughput on the fixed smoke matrix.

**OFFICIAL FULL BENCHMARK: UNBLOCKED**

This record does not launch the official full benchmark.

## Environment

- Python: 3.14.4; package entry point shebang
  `#!/home/royja/afmc-clinical-fm/.venv/bin/python`.
- Canonical executable reported by diagnostics:
  `/home/royja/afmc-clinical-fm/.venv/bin/python`. The venv interpreter is a
  native Linux symlink whose resolved target is `/usr/bin/python3.14`; no
  Windows `.exe` interpreter or entry point was used.
- OS: `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.43`;
  kernel `6.6.114.1-microsoft-standard-WSL2`; WSL detection `true`.
- `/dev/dxg`: present as character device `10,125`.
- CPU logical count: 12.
- PyTorch: 2.13.0+cu130.
- CUDA available: `true`; CUDA runtime: 13.0.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU; compute capability 8.9; NVIDIA
  driver 610.74; total memory 8188 MiB.
- NumPy 2.5.2; pandas 3.0.5; SciPy 1.18.1; scikit-learn 1.9.0;
  matplotlib 3.11.1; PyYAML 6.0.3; threadpoolctl 3.6.0.
- pytest 9.1.1; Ruff 0.16.4.
- All pytest commands used `TMPDIR=/tmp`; all commands used binaries under
  `./.venv/bin`.

## Gate A: complete regression suite — PASS

Commands:

```bash
./.venv/bin/ruff check src tests
TMPDIR=/tmp ./.venv/bin/pytest -q
```

Results:

- Ruff: `All checks passed!`
- pytest: `231 passed in 417.51s (0:06:57)`; no failures or skips were
  reported.
- The full suite includes the existing CPU and flow-jump behavioral invariant
  coverage. No source defect was found and no source file was changed.

## Gate B: native WSL/CUDA device correctness — PASS

Preflight commands:

```bash
readlink -f ./.venv/bin/python
readlink -f ./.venv/bin/afmc-phase0
head -n 1 ./.venv/bin/afmc-phase0
ls -l /dev/dxg
./.venv/bin/afmc-phase0 diagnostics --device auto --workers 1
./.venv/bin/afmc-phase0 diagnostics --device cpu --workers 1
./.venv/bin/afmc-phase0 diagnostics --device cuda --workers 1
```

Device results:

| Request | CUDA available | Selected device | GPU name | Result |
|---|---:|---|---|---|
| `auto` | true | `cuda` | NVIDIA GeForce RTX 4060 Laptop GPU | PASS |
| `cpu` | true | `cpu` | null | PASS; explicit CPU stayed on CPU |
| `cuda` | true | `cuda` | NVIDIA GeForce RTX 4060 Laptop GPU | PASS; no fallback |

Focused CPU/CUDA neural smoke command:

```bash
TMPDIR=/tmp ./.venv/bin/pytest -q -vv \
  tests/experiments/test_runner.py::test_explicit_cpu_matches_default_neural_results \
  tests/execution/test_cuda_smoke.py::test_neural_models_complete_one_epoch_on_cuda
```

Result: `2 passed in 6.67s`. Explicit CPU produced the same deterministic GRU
result as the CPU default. Real CUDA completed one-epoch GRU and flow-jump fits
and returned finite CPU-side metric values.

## Gate C: baseline validation — PASS

Command:

```bash
TMPDIR=/tmp ./.venv/bin/pytest -q -vv \
  tests/models/test_baselines.py::test_torch_ridge_cpu_predictions_match_sklearn_reference \
  tests/models/test_baselines.py::test_torch_ridge_does_not_penalize_intercept \
  tests/models/test_baselines.py::test_torch_ridge_cuda_predictions_match_cpu \
  tests/models/test_baselines.py::test_torch_mlp_has_declared_representation_architecture \
  tests/models/test_baselines.py::test_torch_mlp_representation_head_fits_nonlinear_signal \
  tests/models/test_baselines.py::test_torch_mlp_cpu_fits_repeat_with_same_seed \
  tests/models/test_baselines.py::test_torch_mlp_cuda_fits_repeat_with_same_seed
```

Result: `7 passed in 223.25s (0:03:43)` with no CUDA skips.

Validated tolerances and controls:

- Torch ridge versus sklearn reference: `atol=1e-10`, `rtol=1e-10`.
- Torch ridge unpenalized intercept: constant 7.25 prediction,
  `atol=1e-12`, `rtol=0`.
- Torch ridge CUDA versus CPU: `atol=1e-10`, `rtol=1e-10`, with coefficients
  resident on CUDA.
- Torch MLP architecture: `Linear(19,32) -> Tanh -> Linear(32,1)`, exactly
  673 trainable parameters.
- Torch MLP nonlinear positive control: finite predictions and MSE `< 0.03`.
- Torch MLP CPU same-seed repeatability: `atol=1e-12`, `rtol=0`.
- Torch MLP real-CUDA same-seed repeatability: `atol=1e-10`, `rtol=0`, with
  parameters resident on CUDA.
- Gradient boosting remains the existing sklearn
  `HistGradientBoostingRegressor`; no baseline source or hyperparameter was
  changed during validation.

## Gate D: scheduler equivalence — PASS

Command:

```bash
TMPDIR=/tmp ./.venv/bin/pytest -q -s \
  tests/execution/test_scheduler.py::test_one_and_two_spawn_workers_are_scientifically_equivalent
```

Result: `1 passed in 11.41s`; printed
`serial_parallel_max_delta=0`.

The fixed compact CPU matrix used `cohort_size=30`, `followup_days=45.0`, one
`smooth` world, two matched seed bundles `(17,23,31)` and `(19,29,37)`,
`N=5`, model `engineered_linear`, ablation `none`, `max_epochs=1`, and
`patience=1`. It ran once with one spawned worker and once with two spawned
workers. Sorted persisted cell identities and all non-value scientific fields
were identical, there were no duplicate or missing identities, and metric
values passed `rtol=1e-12`, `atol=1e-12` with exact maximum delta 0.

## Gate E: public interruption/resume correctness — PASS

Command:

```bash
TMPDIR=/tmp ./.venv/bin/pytest -q -s \
  tests/execution/test_resume_end_to_end.py::test_public_scheduler_resume_preserves_completed_cell_and_skips_its_fits
```

Result: `1 passed in 6.17s`.

The public `run_scheduled_benchmark` test used the Task 10 fixed CPU matrix:
one `smooth` shard `(cohort=17, subset=23, model=31)`, `N=5`,
`cohort_size=30`, `followup_days=45.0`, models `engineered_linear` and
`flow_jump`, and one epoch/patience. It interrupted after the first successful
durable write. The completed file was exactly
`low_n__smooth__cohort17__subset23__model31__n5__engineered_linear__none.json`.
Its bytes and `st_mtime_ns` were identical before and after resume. The
interrupted call fit only `engineered_linear`; the resumed call fit only
`flow_jump`, proving no refit of the durable cell. The final resumed aggregate
matched an uninterrupted run at `rtol=1e-12`, `atol=1e-12`, including NaNs.

## Gate F: fixed CUDA smoke throughput calibration — PASS

Every candidate used the same command shape with only `<workers>` and a fresh
`/tmp` output directory varying:

```bash
./.venv/bin/afmc-phase0 benchmark \
  --sim-config configs/simulator/smoke.yaml \
  --exp-config configs/experiments/smoke.yaml \
  --output <fresh-/tmp-output> \
  --device cuda \
  --workers <workers> \
  --fail-fast
```

The fixed matrix has simulator seed 7, cohort size 24, two worlds (`smooth`,
`site_shift`), one matched seed bundle `(101,201,301)`, `N=5`, models
`engineered_linear`, `representation_linear`, `gru_from_scratch`, and
`flow_jump`, ablations `none` and `no_flow`, `max_epochs=1`, and `patience=1`.
The scientific configuration and hyperparameters were not changed between
candidates. The simulator config hash was
`e04d4558716f87da8c6a7af95b566a66dc5a1d168c9e01e4d952dbb97f60bbb6`; the
experiment config hash was
`11ece0d674e695d550d414a5f0bbd5c71723807637e02ec4f2778136808c136e`.

GPU memory was sampled from `nvidia-smi --query-gpu=memory.used` every 0.2 s.
Because WSL did not expose per-process compute rows, the table reports peak
device-total memory and the increase over the immediately preceding device
baseline. This is an operational device-memory sample, not a PyTorch allocator
counter. The 4-worker run began with a higher WSL/driver cache baseline; total
peak remains the directly comparable capacity check.

| Requested workers | External wall (s) | Manifest wall (s) | Cells | Shards | Cells/min (external) | GPU baseline (MiB) | Peak total (MiB) | Peak increase (MiB) | Failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.606287871 | 8.852449465 | 12/12 | 2/2 | 49.293838815 | 958 | 1149 | 191 | 0 |
| 2 | 11.259559689 | 6.829981203 | 12/12 | 2/2 | 63.945662165 | 957 | 1471 | 514 | 0 |
| 4 | 14.406094480 | 9.330136241 | 12/12 | 2/2 | 49.978847563 | 1316 | 1594 | 278 | 0 |

All commands returned 0 with empty stdout/stderr, every manifest recorded
`resolved_device: cuda`, all three runs persisted the same 12 unique cell
identities, and no failed, incomplete, or fallback cell was recorded. The
fixed matrix has only two shards, so requesting four workers cannot expose more
than two concurrently runnable top-level shards.

**Selected worker count: 2.** It achieved the highest measured stable external
throughput, 63.945662165 cells/minute, with 12/12 cells, no failures, and a peak
device-total memory sample of 1471 MiB out of 8188 MiB. GPU utilization alone
was not used for selection.

## Failures and warnings

- Validation failures: none.
- Test skips: none in the complete suite or the focused Gate B/C commands.
- CUDA OOM or fallback: none.
- Source changes made in response to validation: none.
- Telemetry caveat: WSL exposed device-total memory but no per-process compute
  memory rows; peak memory is therefore documented as sampled device-total
  memory with an immediate baseline, as described above.
- The legacy run/process and `/mnt/d/afmc-clinical-fm` were not accessed or
  modified.
- Task 13 and the official full benchmark were not launched.
