# Phase-0.5 Core Task-16 Validation

Date: 2026-08-26

## Scope and identity

- Validated commit: `3fb62ff71e3d9f9750b6dbc1ebf5525db3f71e71`
- Branch: `phase0-5-implementation`
- Pulled remote checkpoint: `e22138b96c67d48b81eb12686c27eb3150373947`
- Phase-0 compatibility base: `18b96fec332c521af3474bbf7afcec21ebb9dc5b`
- Scope: Task 16 only. No official five-bundle Stage I, Stage III confirmation, or Stage IV robustness execution was launched.

## Runtime

- Python: `3.14.4` (`GCC 15.2.0`)
- Platform: `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.43`
- PyTorch: `2.13.0+cu130`
- PyTorch CUDA runtime: `13.0`
- cuDNN: `92000`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`, 8,188 MiB
- NVIDIA driver: `610.74`
- `nvidia-smi` CUDA UMD: `13.3`
- Ruff: `0.16.4`
- pytest: `9.1.1`

## Full validation

Commands:

```bash
export TMPDIR=/tmp
./.venv/bin/ruff check src tests
./.venv/bin/pytest -v
git diff --check
```

Results:

- Ruff: PASS, `All checks passed!`
- pytest: PASS, `388 passed in 218.28s (0:03:38)`
- Skips: none. CUDA was available, so all CUDA-marked tests executed.
- `git diff --check`: PASS, exit 0 with no output.

## Phase-0.5 CPU smoke

`tests/phase05/test_validation_smoke.py::test_phase05_cpu_validation_smoke_one_bundle_n5_n10_locked_mechanisms` passed in both the focused and full runs. It performed eight real one-epoch CPU fits using development bundle `(401, 501, 601)` and `N=5/10`, covering:

- gated flow;
- time-scaled flow;
- residual jump; and
- decoupled uncertainty.

Every defined metric was finite. The decoupled runs produced finite NLL and 90% coverage. `event_roc_auc` was excluded only when mathematically undefined for a single-class tiny test sample, as encoded by the existing metric contract.

## Staged lifecycle smoke

`tests/phase05/test_cli_develop.py::test_phase05_lifecycle_calibrates_develops_sequentially_and_freezes` passed. It exercises the real staged CLI and persistence boundaries in this order:

```text
calibrate
-> develop flow
-> develop jump
-> develop uncertainty
-> representation-timing audit
-> freeze
```

The executed smoke plan is constrained to development bundle `(401, 501, 601)`, `N=5/10`, one epoch, and early-stopping patience of one epoch. It executes 6 flow cells, 6 jump cells, 18 uncertainty cells, and 12 representation-timing cells. The unchanged scientific selectors separately receive explicit synthetic fixture rows with the required five-bundle and N=5/10/20/40 gate shape, as allowed by the approved Task-16 plan; fixture rows are not persisted as executed smoke cells. Calibration, sequential selection, stage completion, artifact persistence, and freeze are real. The frozen result selected `time_scaled` flow, `residual` jump, `decoupled` uncertainty, and strict pre-event history. No scientific threshold, seed, gate, comparator, or inference rule was changed to make a tiny stochastic run pass.

## RTX 4060 CUDA smoke

The project interpreter reported CUDA available and identified the RTX 4060. Four genuine one-epoch Phase-0.5 fits ran with explicit `torch.device("cuda")`, one each for:

1. `gated__none__deterministic`;
2. `time_scaled__none__deterministic`;
3. `time_scaled__residual__deterministic`; and
4. `time_scaled__residual__decoupled`.

The reproducible focused command is:

```bash
TMPDIR=/tmp ./.venv/bin/pytest \
  tests/phase05/test_validation_smoke.py::test_phase05_cuda_validation_smoke_four_real_fits_use_cuda \
  -v
```

The committed test wraps the real training boundary and requires every fit to receive `cuda`, every fitted model parameter to remain resident on CUDA, and peak allocated CUDA memory to be nonzero. It requires all defined outputs to be finite and requires finite NLL and 90% coverage for the decoupled fit. The focused command passed, and the same test passed again in the full suite. An instrumented run allocated 17,371,136 peak CUDA bytes and produced finite decoupled NLL (`1.1522973473616729`) and 90% coverage (`0.9411764705882353`). No tensor/device-placement error occurred.

The full suite additionally passed the historical CUDA one-epoch smoke, Torch ridge CPU/CUDA parity, repeatable CUDA MLP fitting, and the Phase-0.5 CUDA forward smoke.

## Historical Phase-0 compatibility

The compact historical scheduler regression passed:

- `tests/test_cli.py::test_benchmark_cli_runs_configured_worlds_and_writes_gate_outputs`
- `tests/execution/test_jobs.py::test_benchmark_cell_id_preserves_the_task_6_persistence_schema`

The compact run retained the expected historical outputs, two persisted cells, CPU execution metadata, and zero failures. The canonical legacy cell ID remained:

```text
low_n__jumps__cohort101__subset201__model301__n5__engineered_linear__none
```

The branch diff against the current Phase-0 base leaves the historical execution, experiment, data, simulator, and flow-jump model modules unchanged. The only shared baseline change is an optional representation-MLP hidden-size argument whose default remains the historical size of 32; full regressions verified unchanged default capacity and behavior.

## Freeze and confirmation guards

Focused guard tests passed:

- `test_confirmation_start_creates_hard_mutation_boundary` proved that, after confirmation starts, protocol-lock, development-artifact, and frozen-candidate mutation attempts fail.
- `test_validate_resume_rejects_candidate_or_protocol_drift` rejected both frozen-candidate hash drift and protocol-identity drift.
- `test_confirmatory_jobs_require_started_marker_and_exact_frozen_candidate` rejected confirmatory execution before the start marker and rejected a wrong frozen-candidate hash afterward.
- `test_phase05_confirm_binds_frozen_candidate_before_any_fit` proved that the complete confirmatory job plan binds the exact persisted candidate hash and only the ten disjoint confirmatory bundles. The test intentionally stopped before any confirmatory fit.

## Interruption and resume

Focused resume tests passed:

- `test_resume_preserves_completed_cell_bytes_and_runs_only_missing_jobs` injected an interruption, preserved the completed cell bytes and modification time, executed only the missing `N=10` fit on resume, and matched an uninterrupted reference.
- The three `test_cli_develop_resume_boundaries.py` tests preserved completed stage artifacts and cells, skipped finalized flow/jump/uncertainty work, and made a fully finalized development resume a no-op.

## Warnings and blockers

- No Task-16 blocker remains.
- The pre-existing untracked `monitor_phase0.sh` helper was preserved and excluded from commits and validation identity.
- CUDA smoke values are validation-only outputs from a tiny synthetic cohort and are not scientific results.

READY FOR OFFICIAL STAGE-I: YES
