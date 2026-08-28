# Phase 0.7 Crash-Resilient Official Execution Design

> **STATUS: REVIEW CANDIDATE — OPERATIONAL ONLY — DOES NOT AUTHORIZE OFFICIAL EXECUTION**

## 1. Purpose

Phase 0.7 has a frozen scientific design and an implemented 200-cell official execution path. This document adds only the operational durability required for a long local-GPU run.

It does **not** change:

- the Phase 0.7 scientific question;
- the N=40-only scope;
- the five frozen contexts;
- the ten Phase 0.7 model seeds;
- the two architectures;
- the two stopping policies;
- the 200-cell matrix;
- the protected confirmatory-seed firewall;
- the estimands, bootstrap, or adjudication rules.

The objective is:

> A process interruption must never require rerunning already valid completed cells, must never allow a partial/corrupt cell to masquerade as complete, and must never allow resumption under a different scientific or execution identity.

## 2. Provenance of the design

Phase 0.6 already established the repository's preferred execution semantics:

- hash-bound protocol/execution identity;
- atomic persistence of per-cell artifacts;
- resume validation before execution;
- skipping already valid completed cells;
- fail-closed handling of conflicting/corrupt persisted evidence;
- a stage-level `COMPLETE` marker written only after the exact expected cell set exists.

Phase 0.7 must carry those semantics forward rather than generalize or modify the Phase 0.6 implementation itself.

The preferred implementation is therefore a dedicated `Phase07Store` and Phase 0.7 execution orchestrator modeled on `Phase06Store` / `run_phase06_stage`.

## 3. Scope and non-goals

### In scope

- durable per-cell persistence;
- deterministic cell-boundary resume;
- execution-identity validation;
- artifact integrity validation;
- failure provenance;
- deterministic rebuild of aggregate Phase 0.7 metrics from persisted cells;
- exact 200/200 completion validation;
- CLI `--resume` semantics.

### Explicitly out of scope

- mid-epoch or mid-cell checkpoint resume;
- distributed training;
- multi-GPU sharding;
- changing optimizer, batching, RNG, checkpoint selection, or stopping-policy semantics;
- changing any Phase 0.7 scientific constant;
- modifying historical Phase 0.6 result/evidence files;
- modifying the battle-tested `Phase06Store` contract merely to share code.

## 4. Run identity

An official Phase 0.7 output root is bound immutably to one exact run identity:

```text
execution_commit
phase07_spec_sha256
phase07_config_sha256
protocol_lock_sha256
phase07_plan_sha256
expected_cell_count = 200
```

The output root persists at minimum:

```text
<output>/
├── execution_manifest.json
├── protocol_lock.json
├── plan.json
└── stages/
    └── phase07/
```

On first execution, the files are created atomically and validated against the current checkout/config/spec/plan.

On every resume attempt, the current process recomputes the expected identity and requires exact equality with the persisted identity before device resolution, cohort preparation, or model execution.

Any mismatch in execution SHA, spec hash, config hash, protocol hash, plan hash, or expected cell count fails closed.

An existing non-empty output root without `--resume` fails closed.

## 5. Cell identity

Phase 0.7 already defines deterministic cell IDs containing all frozen scientific dimensions:

```text
p07__smooth__cohort<C>__subset<S>__model<M>__n40__<flow_mode>__<optimization_policy>
```

`optimization_policy` is part of cell identity and therefore part of persisted evidence identity.

The exact expected cell set is the frozen 200-cell plan. No observed persisted cell may exist outside that set.

## 6. Cell bundle

Each successfully completed official cell persists a self-validating bundle containing at least:

```text
cell.json
metrics.csv
training_trace.csv
summary.json
production_checkpoint.pt
shadow_mae_checkpoint.pt
artifact_manifest.json
COMPLETE
```

The semantic evidence includes the Phase 0.7-specific fields already produced by the runner, including:

- `optimization_policy`;
- `would_patience_exhaust_epoch`;
- production checkpoint selection diagnostics;
- shadow MAE checkpoint diagnostics;
- complete training trace;
- test metrics.

`artifact_manifest.json` contains:

- run identity;
- cell metadata / `cell_id`;
- SHA-256 for every persisted evidence artifact;
- cell-bundle schema version.

The cell-level `COMPLETE` marker is written only after the bundle has been fully written and validated.

## 7. Atomic cell commit

A cell is never written directly into its final authoritative directory.

The store creates a same-filesystem temporary staging directory under the Phase 0.7 stage, for example:

```text
stages/phase07/.tmp/<cell_id>.<nonce>/
```

The complete bundle is written there. The implementation must:

1. write all artifacts;
2. flush and close them;
3. compute artifact hashes;
4. write the artifact manifest;
5. re-read and validate the staged bundle;
6. write staged `COMPLETE` last;
7. atomically rename/replace the staging directory into the final cell location.

Final authoritative location:

```text
stages/phase07/cells/<cell_id>/
```

The temporary and final locations must reside on the same filesystem. The implementation uses the platform's atomic same-filesystem rename/replace primitive and must not fall back to copy-then-delete semantics. If the atomic rename fails, the cell commit fails and the staged directory remains non-authoritative.

## 8. Interruption classes

### 8.1 Interruption before final cell commit

Examples:

- Python exception;
- Ctrl-C / `KeyboardInterrupt`;
- CUDA error;
- process kill;
- power loss;
- OS crash.

A staged but uncommitted cell is not authoritative and is not counted as complete.

On resume, abandoned Phase 0.7 staging directories are removed only after the run identity is validated. The corresponding cell is rerun from epoch 1.

### 8.2 Corruption of an authoritative completed cell

A final cell directory whose hashes, metadata, schema, or run identity do not validate is **not** treated as an interrupted cell.

Resume fails closed and reports the offending cell. It must not automatically delete or rerun apparently completed but corrupt scientific evidence.

This separates recoverable interruption from evidence corruption/tampering.

## 9. Resume algorithm

`afmc-phase07 official --resume` performs the following order:

1. load frozen Phase 0.7 config;
2. recompute current execution SHA, protocol lock, plan and hashes;
3. validate authorization against the current frozen identity;
4. validate persisted output-root identity;
5. validate the exact expected 200-cell set;
6. clean only abandoned same-run staging directories;
7. scan final cell directories;
8. reject unexpected cell IDs;
9. fully validate every observed completed bundle;
10. form `completed_before` from only valid authoritative cells;
11. resolve the requested device;
12. iterate cells in deterministic frozen plan order;
13. skip every `cell_id` in `completed_before`;
14. execute and atomically commit each remaining cell;
15. after execution, rescan and revalidate the authoritative cell set;
16. require exact 200/200 equality before any final completion marker.

Without `--resume`, any authoritative persisted cell causes a hard failure.

## 10. Resume granularity

Resume granularity is exactly one cell.

Phase 0.7 does not resume a model from an intermediate epoch. A failed cell is rerun from its original deterministic initialization.

This is deliberate. Mid-cell continuation would require freezing and validating model state, optimizer state, RNG state, stale-patience state, diagnostic recorder state and any other training state, then proving equivalence with uninterrupted training.

The added complexity is not justified because individual Phase 0.7 cells are short relative to the full 200-cell run.

The design therefore favors scientific simplicity:

```text
completed cells survive;
interrupted current cell reruns from epoch 1.
```

## 11. Failure provenance

The execution layer maintains atomically written run provenance including at least:

- planned cell count;
- completed-before count;
- completed-after count;
- current/last attempted cell ID when available;
- failures with exception type/message when catchable;
- device;
- runtime metadata;
- execution commit;
- protocol hash;
- plan hash;
- start/end timestamps and wall time.

This record is informative. It is not the authority for resume state. The authoritative completed set is reconstructed from fully validated final cell bundles.

Uncatchable failures such as power loss need not produce a failure record; the atomic cell protocol is the protection mechanism in those cases.

## 12. Aggregate metrics are derived evidence

`phase07_metrics.csv` is not an execution checkpoint.

It is rebuilt deterministically from the 200 validated authoritative cell bundles after all cells are present. A stale, missing, or partially written aggregate file cannot cause a cell to be skipped.

Before writing the root/stage completion marker, the rebuilt aggregate must satisfy the frozen analysis contract, including exactly 200 Phase 0.7 test-MAE rows matching the frozen design.

## 13. Completion boundary

The authoritative Phase 0.7 `COMPLETE` marker is written atomically and last.

It contains at minimum:

- schema version;
- run identity;
- exact sorted list of 200 cell IDs;
- completed cell count = 200.

It may be written only when:

```text
observed_valid_cell_ids == expected_frozen_cell_ids
```

A pre-existing `COMPLETE` marker is valid only if its payload exactly matches the current run identity and exact expected cell set and all 200 bundles independently validate.

A marker at 199/200, a marker containing an unexpected cell, or a marker whose identity differs from the current frozen identity fails closed.

Official Phase 0.7 aggregation/adjudication must refuse an output root without a valid completion marker and independently valid 200-cell set.

## 14. Determinism and scientific invariants

Crash resistance must not alter training behavior.

The persistence layer must sit outside `run_phase07_cell()` and must not change:

- model initialization;
- NumPy/PyTorch seeding;
- minibatches;
- optimizer updates;
- standard early stopping;
- forced-horizon behavior;
- checkpoint selection;
- diagnostic recording;
- evaluation.

A resumed cell begins from the same frozen cell specification and deterministic seeds as an uninterrupted first attempt.

The required scientific acceptance property is:

> Given the same execution identity, interruption + cell-boundary resume must produce the same scientific evidence as uninterrupted execution: metrics, traces, semantic summaries and checkpoint tensor values must be equal under the repository's deterministic test fixture. Runtime-only provenance may differ. Serialized checkpoint bytes and their per-run artifact hashes are not required to match across two independently produced output roots if the serialization format introduces non-semantic byte differences; each persisted bundle must still validate against its own recorded hashes.

## 15. CLI behavior

Official local execution remains single-worker and sequential for this version.

First start:

```bash
afmc-phase07 official \
  --authorization <authorization.json> \
  --output <output-root> \
  --device cuda
```

Resume:

```bash
afmc-phase07 official \
  --authorization <authorization.json> \
  --output <same-output-root> \
  --device cuda \
  --resume
```

`--resume` is explicit; the tool never silently resumes an existing output directory.

No cloud/multi-GPU/shard arguments are added by this design.

## 16. Required TDD coverage

Implementation is not accepted unless tests establish the following RED → GREEN behaviors:

1. a failure after N committed cells followed by `--resume` executes only the remaining cells;
2. a failure during cell staging creates no authoritative final cell;
3. an abandoned same-run staging directory is discarded and its cell reruns;
4. a corrupt authoritative completed bundle fails closed and is not silently rerun;
5. mismatched execution SHA fails before any cell callback;
6. mismatched spec/config/protocol/plan identity fails before any cell callback;
7. an unexpected persisted cell ID fails closed;
8. existing valid cells without `--resume` fail closed;
9. valid completed cells are never passed to `run_phase07_cell()` again during resume;
10. a completion marker cannot be produced with fewer or more than the exact 200 expected cells;
11. aggregate metrics are rebuilt from authoritative cell bundles rather than trusted as resume state;
12. interruption + resume and uninterrupted execution produce semantically identical scientific evidence for a controlled deterministic test fixture as defined in Section 14;
13. existing Phase 0.5/0.6 tests remain unchanged and green;
14. the non-official CPU smoke remains green;
15. the final candidate requires a user-run CUDA smoke after all crash-resilience changes are complete.

Tests may exercise persistence/orchestration with a small injected test-only cell set or fake cell callback, but that injection point must not be exposed through the production CLI and must not weaken `plan_phase07_cells()` / `phase07_plan_sha256()` or the production requirement that official execution contains exactly the frozen 200 cells. No test or CI path may execute the official 200-cell matrix.

## 17. Files / component boundaries

Expected implementation boundaries:

- **New:** `src/afmc_fm/phase07/store.py`
  - dedicated Phase 0.7 bundle persistence and resume validation;
  - modeled on Phase 0.6 semantics;
  - no modification of historical Phase 0.6 stores/evidence.

- **Modify:** `src/afmc_fm/phase07/execution.py`
  - orchestrate deterministic sequential execution through `Phase07Store`;
  - skip valid completed cells only under explicit resume;
  - persist run provenance;
  - establish exact completion boundary.

- **Modify:** `src/afmc_fm/phase07/cli.py`
  - add `--resume` to `official`;
  - remove the current empty-directory-only persistence flow in favor of the store.

- **Tests:** dedicated Phase 0.7 store/execution/CLI crash-resume tests.

The existing `Phase07CellRun`, frozen planner, protocol lock, authorization gate, analysis, and scientific runner remain conceptually unchanged.

## 18. Security / integrity posture

The system is intentionally fail-closed:

- identity mismatch → stop;
- unexpected authoritative cell → stop;
- corrupt authoritative cell → stop;
- conflicting completion marker → stop;
- failed atomic final rename → stop;
- partial staging work → recoverable only by discard + rerun after identity validation.

There is no automatic evidence repair.

## 19. Authorization boundary

This document approves no official training.

After this operational spec is approved:

1. write the TDD implementation plan;
2. implement crash resistance on PR #9;
3. run exact-head PR CI;
4. freeze a new candidate execution SHA;
5. run the non-official CUDA smoke locally on that exact SHA;
6. review the resulting PR state;
7. merge only after the user explicitly authorizes merge;
8. create/bind the separate official execution authorization;
9. require a separate explicit GO before the official 200-cell Phase 0.7 run.

The current pre-crash-resilience candidate SHA must not be treated as the final execution SHA.
