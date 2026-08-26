# Phase 0 CUDA-First Execution Engine Design

Date: 2026-08-23
Status: Approved design, pending implementation plan
Scientific protocol anchor: `be5a66b2e45362f60c90844e4e25673fb7bb3e21`

## 1. Purpose

The first full Phase-0 benchmark attempt was stopped after approximately one hour because the reference runner executed the experiment matrix serially, used a Windows Python environment through WSL, wrote final artifacts only after the monolithic run completed, and left the available NVIDIA GPU unused. No completed benchmark result from that run will be used as scientific evidence.

This design replaces the execution layer before the first official Phase-0 result set. The goal is to reduce wall-clock time, make the benchmark resumable and observable, and use CUDA wherever that is technically and scientifically appropriate while retaining a portable CPU fallback.

The scientific protocol remains anchored to commit `be5a66b2`: simulator worlds, patient-level splitting, train-size budgets, seed roles, targets, losses, model architecture semantics, ablations, metrics, and interpretation rules remain fixed unless this document explicitly declares a baseline implementation change before the first official run.

## 2. Canonical runtime

WSL/Linux is the canonical research runtime. Windows-native and CPU-only Linux remain supported fallbacks.

The canonical development and benchmark environment must use a native Linux virtual environment under WSL rather than invoking `.venv/Scripts/*.exe` from Linux. CUDA support is a runtime capability detected through PyTorch; the package must not require a GPU to import, test, or run.

The benchmark command remains exposed through the Python package entry point. Device and parallelism options are added to the CLI rather than through platform-specific shell scripts.

## 3. Design principles

1. **Scientific identity is separate from execution placement.** A benchmark cell is defined by its scientific inputs, not by whether it executes on CPU or CUDA.
2. **No work exists only in process memory for hours.** Completed cells are persisted atomically and can be resumed.
3. **Parallelism is process-based.** Separate processes isolate NumPy, PyTorch, and scikit-learn random state and work on both Linux and Windows through the `spawn` multiprocessing context.
4. **CUDA is the preferred backend for Torch-compatible fitting.** CPU is a fallback or retained backend where the exact estimator is not CUDA-capable.
5. **Data preparation is reused within a shard.** The expensive synthetic cohort and sequence construction for one world/seed bundle must not be repeated for every model.
6. **The first official Phase-0 run begins only after execution-equivalence and resume tests pass.**

## 4. Unit of parallel execution

The natural top-level shard is one matched world/seed bundle:

```text
(world, cohort_seed, subset_seed, model_seed)
```

The frozen experiment contains five worlds and five matched seed bundles, giving 25 independent shards. A shard builds its synthetic cohort and encoded patient sequences once, then executes all configured train sizes, model families, and applicable ablations for that bundle.

This granularity is intentionally coarser than one model fit. It preserves reuse of the 5,000-patient synthetic cohort and encoded sequences while still exposing enough independent work for multi-core and multi-process execution.

Each shard has a deterministic identifier, for example:

```text
jumps__cohort101__subset201__model301
```

Within a shard, each fitted configuration has a deterministic cell identifier containing benchmark type, train size, model, and ablation. Result rows retain the existing world, seed, split, site/shift, metric, and trainable-parameter fields.

## 5. Scheduler

A lightweight native-Python scheduler is used rather than Ray, Dask, or another distributed framework.

The coordinator performs the following steps:

1. load and validate simulator and experiment configuration;
2. enumerate the 25 deterministic shard specifications;
3. inspect the output directory for valid completed shard/cell artifacts when `--resume` is enabled;
4. submit pending shards to a `ProcessPoolExecutor` created with multiprocessing `spawn` context;
5. stream progress and failures to the console and run log;
6. aggregate completed shard outputs deterministically into the existing final Phase-0 artifacts.

The scheduler exposes at least:

```text
--device auto|cpu|cuda
--workers N
--resume
--fail-fast
```

`--device auto` resolves to CUDA when `torch.cuda.is_available()` is true and otherwise CPU. `--device cuda` fails clearly when CUDA is unavailable. `--device cpu` never initializes CUDA.

Worker count is explicit and recorded in the run manifest. The initial safe default is conservative; throughput tuning on the RTX 4060 may justify a larger value after the validation benchmark. The design does not assume that one CUDA job per GPU is optimal because the neural models are very small.

## 6. Device-aware Torch execution

All PyTorch models and batches gain explicit device placement.

The execution boundary is:

```text
CPU preparation -> tensor batch -> selected torch.device -> model fit/evaluation -> CPU metric arrays
```

A shared helper moves every tensor in a batch dictionary to the selected device. Neural model construction is followed by `.to(device)`. Evaluation converts tensors through `.detach().cpu().numpy()` before NumPy/scikit-learn metrics are called.

No model architecture should contain hard-coded CUDA assumptions. `FlowJumpAdapter`, `GRUBaseline`, losses, and metrics remain device-agnostic.

The current `torch.set_num_threads(1)` policy remains appropriate inside each worker to prevent CPU oversubscription. CPU-backed numerical libraries must also be constrained to one thread per worker where necessary.

## 7. Baseline backend policy

CUDA capability is separated from estimator semantics.

### 7.1 Torch-native neural models

The following models run on the selected Torch device without changing their scientific architecture:

- `gru_from_scratch`
- `flow_jump`
- `flow_jump_observation`
- all flow-jump ablations

These models use CUDA by default when `--device auto` resolves to CUDA.

### 7.2 Linear probes

The mathematical linear baseline is not intrinsically CPU-bound. The official optimized runner introduces a Torch ridge implementation for:

- `engineered_linear`
- `representation_linear`

It must match the existing ridge objective, including intercept handling and L2 regularization semantics. A scikit-learn `Ridge` implementation remains available as a reference backend for parity tests.

The Torch implementation may use a stable least-squares/Cholesky solve or a deterministic optimization formulation. It must pass a CPU parity test against the current scikit-learn implementation on fixed synthetic matrices before becoming the default optimized backend.

### 7.3 Representation MLP

The current implementation is a 19-input, one-hidden-layer, 32-unit tanh MLP with L2 regularization and an L-BFGS-family optimizer. The optimized runner introduces a Torch implementation with the same input contract, hidden width, activation, and regularization objective so it can run on CUDA.

Because `torch.optim.LBFGS` is not the same optimizer implementation as scikit-learn/SciPy L-BFGS, exact numerical identity is not required and must not be claimed. This is a declared pre-results baseline implementation revision. The existing scikit-learn MLP remains available as a reference backend for audit runs.

Before the first official benchmark, the Torch MLP must demonstrate:

- correct 19-dimensional input contract;
- deterministic seeding on CPU and repeatable seeding on CUDA within documented tolerance;
- finite training and predictions across low-N smoke cases;
- no systematic failure relative to the reference MLP on positive-control tasks;
- documented trainable parameter count.

The official run manifest records which MLP backend was used.

### 7.4 Gradient boosting

`gradient_boosting` remains the scikit-learn histogram gradient-boosting baseline in this execution revision. Replacing it with XGBoost, CatBoost, cuML, or another GPU tree implementation would change estimator semantics more substantially and is not required to remove the primary runtime bottleneck.

A future GPU-tree backend may be added as a separately named comparator or a declared protocol revision, not silently substituted.

## 8. Shard lifecycle and data reuse

A worker receiving one shard performs:

1. simulate the configured world with the shard cohort seed;
2. create the leakage-safe patient split;
3. construct the fixed 16-dimensional SummaryHistoryEncoder outputs and patient sequences once;
4. iterate the six configured train sizes;
5. derive the fit/validation budget for that `N` using the shard subset seed;
6. fit each model/ablation using the shard model seed;
7. persist each completed fit's result bundle atomically;
8. execute the site-shift evaluation path when the shard world is `site_shift`;
9. mark the shard complete only after all expected cell identifiers are present and valid.

A restarted shard may repeat cohort/sequence preparation, but it must skip already completed model cells when resuming. This provides fine enough recovery without serializing the entire 5,000-patient in-memory sequence object graph.

## 9. Persistence and atomicity

The output structure is versioned and resumable:

```text
outputs/<run-id>/
  run_manifest.json
  run.log
  shards/
    <shard-id>/
      shard_manifest.json
      cells/
        <cell-id>.json
      COMPLETE
  metrics.csv
  ablation_metrics.csv
  gate_summary.csv
  learning_curves.png
```

A cell file is written to a temporary sibling path and renamed atomically after validation. A shard `COMPLETE` marker is written only when its expected cell set is complete.

`--resume` validates scientific config hashes, protocol anchor, execution schema version, and expected shard/cell identifiers. It must refuse to mix incompatible runs in one output directory.

Final CSVs and plots are derived artifacts. They can be deleted and regenerated from the persisted cell files without retraining models.

## 10. Progress reporting

The coordinator reports at least:

- shards complete / total;
- cells complete / expected;
- current elapsed wall time;
- active worker count;
- device mode;
- failed cell/shard count;
- estimated completion percentage based on completed cells, clearly labelled as a throughput estimate rather than a guaranteed ETA.

Worker logs record model, ablation, train size, seed bundle, start/end time, duration, backend, device, and warnings. Scikit-learn convergence warnings are captured as metadata instead of being lost in an unstructured console stream.

## 11. Reproducibility manifest

The final manifest records:

- scientific protocol anchor SHA (`be5a66b2...`);
- execution commit SHA;
- simulator and experiment config content hashes;
- resolved device mode;
- CUDA availability;
- GPU name and compute capability when applicable;
- PyTorch and CUDA runtime versions;
- Python, NumPy, pandas, SciPy, scikit-learn, and matplotlib versions;
- OS and WSL detection;
- CPU logical count;
- worker count;
- baseline backend identifiers;
- start/end timestamps and wall time;
- expected/completed/failed shard and cell counts;
- output schema version.

The stopped serial attempt is recorded only in research notes/log provenance as an aborted execution test. It contributes no metric rows to the official run.

## 12. Scientific invariants

This execution redesign must not silently change:

- five world definitions;
- cohort size and simulator parameters;
- train sizes `{5, 10, 20, 40, 80, 100}`;
- cohort/subset/model seed values and pairing;
- patient-level splitting rules;
- 16-dimensional fixed history representation contract;
- flow, assimilation, event-conditioned update, observation-head, and output-head semantics;
- flow-jump ablation definitions;
- value and event targets;
- observation-process loss semantics;
- loss weights;
- neural learning rate, weight decay, maximum epochs, and early-stopping patience;
- metrics and aggregation definitions;
- the interpretation that frozen v1's event module is broader than an intervention-only jump.

The Torch ridge and Torch MLP backend changes are the only declared baseline implementation revisions in this design. They occur before any official Phase-0 result exists and are versioned explicitly.

## 13. Validation gates before the official full benchmark

The optimized engine is not considered ready until all gates below pass.

### Gate A: existing regression suite

- Ruff passes.
- Existing unit tests pass on CPU.
- Existing behavioural invariants for flow-jump remain green.

### Gate B: device correctness

- CPU mode completes a small neural fit.
- CUDA mode completes the same smoke fit when CUDA is available.
- `auto` chooses the expected device.
- every evaluation path returns CPU NumPy arrays without device errors.
- CI skips CUDA-specific tests when no GPU exists rather than failing.

### Gate C: baseline validation

- Torch ridge agrees with scikit-learn Ridge within a strict numerical tolerance on fixed matrices and positive controls.
- Torch MLP passes deterministic/finite-output checks and the existing nonlinear positive control.
- parameter counts are reported correctly.
- gradient boosting remains unchanged.

### Gate D: scheduler equivalence

On a compact CPU-only benchmark matrix, serial execution and multi-process execution produce:

- identical expected cell identifiers;
- no duplicate or missing rows;
- the same patient budgets and seed assignments;
- numerically identical or machine-tolerance-equivalent metrics for deterministic CPU backends.

### Gate E: resume correctness

A validation run is deliberately interrupted after at least one completed cell. Restarting with `--resume` must:

- preserve completed cell files;
- rerun only unfinished cells;
- produce the same final aggregate as an uninterrupted run.

### Gate F: CUDA throughput smoke benchmark

Run a fixed small matrix with one and several workers on the RTX 4060. Choose the official worker count based on completed cells per minute while staying within GPU memory and maintaining stable execution. The chosen worker count is operational metadata, not part of the scientific protocol.

Only after Gates A-F pass may the first official full Phase-0 benchmark be launched.

## 14. Failure handling

A failed model cell writes structured failure metadata containing its identity, exception type, message, device/backend, and duration.

Default full-run behaviour is to continue other independent cells and report failures at the end. `--fail-fast` is available for development validation.

A failed cell is never represented by an invented metric or silently omitted from the manifest. The official benchmark is complete only when every expected cell either has a valid result or is explicitly classified as a failed run requiring protocol-level review.

CUDA out-of-memory errors must not trigger silent CPU fallback inside an already declared CUDA run. The worker fails clearly so the operator can restart with a lower worker count. Silent fallback would make resource provenance ambiguous.

## 15. CLI diagnostics

Add a lightweight diagnostic command or equivalent preflight output that reports:

```text
Python executable
OS / WSL status
Torch version
CUDA available
CUDA runtime
GPU name
selected device
CPU count
requested worker count
```

When running under WSL with a Windows `.exe` Python interpreter, emit a strong warning recommending a native Linux virtual environment. The runner remains portable and does not hard-fail solely because of that configuration.

## 16. Expected performance direction

The previous run effectively consumed approximately one CPU core continuously. The redesigned engine attacks three independent sources of wasted wall time:

1. 25 world/seed shards execute concurrently instead of serially;
2. Torch-compatible model fitting uses the RTX 4060 when available;
3. interrupted work is preserved and resumed instead of discarded.

No fixed speedup is promised before measurement. Success for the execution redesign is defined by correctness first and a material reduction in wall time second. Throughput is measured as completed benchmark cells per minute on the fixed smoke matrix and on the first official full run.

## 17. Scope boundaries

This design does not introduce:

- Ray, Dask, Kubernetes, Slurm integration, or multi-node distribution;
- mixed precision as a default;
- changes to simulator truth mechanisms;
- new scientific baselines such as GRU-D;
- a new selective `jump_mask` architecture;
- hyperparameter retuning based on benchmark results;
- GPU tree boosting substitution;
- interpretation of the aborted serial attempt.

Those are separate future decisions.

## 18. Acceptance criteria

Implementation is complete when:

1. the package runs from a native WSL/Linux environment with CPU and optional CUDA support;
2. all current tests plus new execution/device/resume tests pass;
3. the 25-shard benchmark can run with configurable process parallelism;
4. Torch neural models use CUDA in `auto` mode on the RTX 4060;
5. declared Torch ridge and Torch MLP backends are validated and versioned;
6. gradient boosting remains the frozen scikit-learn estimator;
7. completed cells persist atomically and are resumable;
8. aggregate artifacts are reproducible from persisted cells;
9. the run manifest fully records scientific and execution provenance;
10. the fixed smoke matrix passes serial/parallel/device validation before the official full run.
