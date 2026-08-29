# Phase 0.7 Crash-Resilient Official Execution Design

> **STATUS: APPROVED OPERATIONAL DESIGN — DOES NOT AUTHORIZE OFFICIAL EXECUTION**

## 1. Purpose

Phase 0.7 has a frozen scientific design and an implemented 200-cell official execution path. This document adds the operational durability, provenance binding, fail-closed statistical handling, and completion gating required before a long local-GPU run.

It does **not** change:

- the Phase 0.7 scientific question;
- the N=40-only scope;
- the five frozen contexts;
- the ten Phase 0.7 model seeds;
- the two architectures;
- the two stopping policies;
- the 200-cell matrix;
- the protected confirmatory-seed firewall;
- the estimand definitions or adjudication thresholds.

The objectives are:

> A process interruption must never require rerunning already valid completed cells, must never allow a partial/corrupt cell to masquerade as complete, and must never allow resumption under a different scientific or execution identity.

> Official heterogeneity inference must fail closed if even one of the 10,000 crossed-bootstrap `R_SD` replicates is undefined/non-finite.

> Official execution and official analysis must consume only evidence bound to the exact clean checkout and exact loaded Phase 0.5 / simulator configurations.

## 2. Provenance of the design

Phase 0.6 established the repository's preferred execution semantics:

- hash-bound protocol/execution identity;
- atomic persistence of per-cell artifacts;
- resume validation before execution;
- skipping already valid completed cells;
- fail-closed handling of conflicting/corrupt persisted evidence;
- a stage-level `COMPLETE` marker written only after the exact expected cell set exists.

Phase 0.7 carries those semantics forward through a dedicated `Phase07Store` and Phase 0.7 execution orchestrator modeled on `Phase06Store` / `run_phase06_stage`. Historical Phase 0.6 implementation and evidence remain unchanged.

An independent pre-execution audit at commit `c8c32fb42518634ca153f915d8916af4952bf360` additionally identified three execution blockers that this design now incorporates:

1. missing crash-safe/resumable official persistence;
2. invalid `R_SD` bootstrap replicates being dropped before percentile computation;
3. insufficient binding of authorization/execution to loaded dependency configs and a clean worktree.

It also identified two hardening requirements: completion-gated official analysis and invariant coverage for both architectures at the frozen 100-epoch/patience-12 constants.

## 3. Scope and non-goals

### In scope

- durable per-cell persistence;
- deterministic cell-boundary resume;
- exact execution-identity validation;
- loaded Phase 0.5 config content hashing;
- loaded simulator config content hashing;
- clean-worktree requirement for official execution and resume;
- artifact integrity validation;
- failure provenance;
- deterministic rebuild of aggregate Phase 0.7 metrics from persisted cells;
- exact 200/200 completion validation;
- completion-gated official analysis/adjudication;
- CLI `--resume` semantics;
- fail-closed `R_SD` bootstrap behavior;
- frozen 100-epoch/patience-12 stopping-policy invariant tests for both architectures.

### Explicitly out of scope

- mid-epoch or mid-cell checkpoint resume;
- distributed training;
- multi-GPU sharding;
- changing optimizer, batching, RNG, checkpoint selection, or stopping-policy semantics;
- changing any Phase 0.7 scientific constant or gate threshold;
- modifying historical Phase 0.6 result/evidence files;
- modifying `Phase06Store` merely to share code;
- allowing arbitrary test plans through the production CLI.

## 4. Run identity

An official Phase 0.7 output root is bound immutably to one exact run identity:

```text
execution_commit
phase07_spec_sha256
phase07_config_sha256
phase05_config_sha256
simulator_config_sha256
protocol_lock_sha256
phase07_plan_sha256
expected_cell_count = 200
```

`phase05_config_sha256` and `simulator_config_sha256` are hashes of the canonicalized **loaded configuration values**, using the repository's canonical configuration serializer, not hashes of path strings alone. This ensures defaults and parsed values that actually govern execution are committed into the identity.

The output root persists at minimum:

```text
<output>/
├── execution_manifest.json
├── protocol_lock.json
├── plan.json
└── stages/
    └── phase07/
```

On first execution, these files are created atomically and validated against the current checkout/config/spec/plan.

On every resume attempt, the current process recomputes the expected identity and requires exact equality with persisted identity before device resolution, cohort preparation, or model execution.

Any mismatch in execution SHA, Phase 0.7 spec/config hash, loaded Phase 0.5 config hash, loaded simulator config hash, protocol hash, plan hash, or expected cell count fails closed.

The official authorization payload must include and exactly match both dependency-config hashes in addition to the execution commit, protocol hash, and plan hash.

## 5. Clean-checkout invariant

Official Phase 0.7 execution may run only from a clean Git worktree.

Before the first official execution **and every resume**, the execution layer must verify that tracked changes, staged changes, and untracked non-ignored files are absent. A dirty checkout fails before CUDA/device resolution or any cell callback.

The execution SHA alone is insufficient because `git rev-parse HEAD` does not describe uncommitted changes.

Generated official outputs should reside under an ignored path or outside the repository so that legitimate persisted output does not make a subsequent resume dirty. Ignored files do not invalidate the clean-checkout requirement.

No dirty-checkout bypass flag exists in the production CLI.

## 6. Cell identity

Phase 0.7 defines deterministic cell IDs containing all frozen scientific dimensions:

```text
p07__smooth__cohort<C>__subset<S>__model<M>__n40__<flow_mode>__<optimization_policy>
```

`optimization_policy` is part of cell identity and persisted evidence identity.

The exact expected cell set is the frozen 200-cell plan. No observed persisted cell may exist outside that set.

## 7. Cell bundle

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

The semantic evidence includes:

- `optimization_policy`;
- `would_patience_exhaust_epoch`;
- production checkpoint diagnostics;
- shadow MAE checkpoint diagnostics;
- complete training trace;
- test metrics.

`artifact_manifest.json` contains:

- complete run identity;
- cell metadata / `cell_id`;
- SHA-256 for every persisted evidence artifact;
- cell-bundle schema version.

The cell-level `COMPLETE` marker is written only after the staged bundle has been fully written and validated.

## 8. Atomic cell commit

A cell is never written directly into its final authoritative directory.

The store creates a same-filesystem temporary staging directory under the Phase 0.7 stage:

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
7. atomically rename the staging directory into the final cell location.

Final authoritative location:

```text
stages/phase07/cells/<cell_id>/
```

Temporary and final locations must reside on the same filesystem. The implementation uses same-filesystem atomic rename semantics and does not fall back to copy-then-delete. If final rename fails, the cell commit fails and staged work remains non-authoritative.

## 9. Interruption classes

### 9.1 Interruption before final cell commit

Examples include Python exception, Ctrl-C, CUDA error, process kill, power loss, or OS crash.

A staged but uncommitted cell is not authoritative and is not counted complete.

On resume, abandoned Phase 0.7 staging directories are removed only after run identity and clean-checkout validation. The corresponding cell reruns from epoch 1.

### 9.2 Corruption of an authoritative completed cell

A final cell directory whose hashes, metadata, schema, completion marker, or run identity do not validate is **not** treated as an interrupted cell.

Resume fails closed and reports the offending cell. It must not automatically delete or rerun apparently completed but corrupt scientific evidence.

## 10. Resume algorithm

`afmc-phase07 official --resume` performs this order:

1. load frozen Phase 0.7 config;
2. verify clean Git worktree;
3. recompute current execution SHA;
4. load Phase 0.5 and simulator configs and compute canonical content hashes;
5. recompute protocol lock, exact 200-cell plan and hashes;
6. validate authorization against the complete current identity;
7. validate persisted output-root identity;
8. validate the exact expected 200-cell set;
9. clean only abandoned same-run staging directories;
10. scan final cell directories;
11. reject unexpected cell IDs;
12. fully validate every observed completed bundle;
13. form `completed_before` from valid authoritative cells only;
14. resolve the requested device;
15. iterate cells in deterministic frozen plan order;
16. skip every `cell_id` in `completed_before`;
17. execute and atomically commit each remaining cell;
18. after execution, rescan and revalidate the authoritative cell set;
19. require exact 200/200 equality;
20. rebuild aggregate metrics from validated bundles;
21. validate aggregate contract;
22. write authoritative completion marker last.

Without `--resume`, a non-empty authoritative output root fails closed.

## 11. Resume granularity

Resume granularity is exactly one cell.

Phase 0.7 does not resume a model from an intermediate epoch. A failed cell reruns from its original deterministic initialization.

This avoids having to freeze and prove restoration equivalence for model, optimizer, RNG, stale-patience, diagnostic, and checkpoint-selector state.

```text
completed cells survive;
interrupted current cell reruns from epoch 1.
```

## 12. Failure provenance

The execution layer maintains atomically written run provenance including at least:

- planned cell count;
- completed-before count;
- completed-after count;
- current/last attempted cell ID when available;
- catchable failures with exception type/message;
- device;
- runtime metadata;
- execution commit;
- all run-identity hashes;
- start/end timestamps and wall time.

This record is informative only. Resume state is reconstructed from validated authoritative cell bundles.

Uncatchable failures such as power loss need not produce a failure record; atomic cell persistence protects those cases.

## 13. Aggregate metrics and official analysis

`phase07_metrics.csv` is derived evidence, not an execution checkpoint. It is rebuilt deterministically from the exact validated 200-cell set after all authoritative bundles exist.

A stale, missing, or partially written aggregate file can never cause a cell to be skipped.

The pure statistical functions in `phase07.analysis` may continue accepting an in-memory DataFrame for unit testing and reusable analysis logic. However, an **official analysis/adjudication entry point** must load evidence through `Phase07Store` and must refuse to proceed unless:

```text
valid root/stage COMPLETE
AND observed_valid_cell_ids == expected_frozen_cell_ids
AND all 200 cell bundles validate against the same run identity
```

Only after those checks may it rebuild/load the aggregate and invoke the pure pair-table/statistics/adjudication functions.

This keeps filesystem trust boundaries out of the mathematical core while preventing arbitrary 200-row CSVs from masquerading as official evidence.

## 14. Completion boundary

The authoritative Phase 0.7 `COMPLETE` marker is written atomically and last.

It contains at minimum:

- schema version;
- complete run identity;
- exact sorted list of 200 cell IDs;
- completed cell count = 200.

It may be written only when:

```text
observed_valid_cell_ids == expected_frozen_cell_ids
```

A pre-existing marker is valid only if its payload matches the current run identity and exact expected cell set and all 200 bundles independently validate.

A marker at 199/200, a marker containing an unexpected cell, or a marker with mismatched identity fails closed.

## 15. Fail-closed crossed bootstrap

The frozen crossed-bootstrap contract uses exactly 10,000 resamples with seed `20260827`.

For every replicate, context and model axes are resampled independently with replacement, the Cartesian matrix is formed, row/column/grand means and two-way residuals are recomputed, and `R_SD` is computed with sample SD (`ddof=1`).

If the standard-policy residual SD is zero or non-finite in **any** replicate, that replicate's `R_SD` is undefined. The implementation must then invalidate heterogeneity inference for the entire bootstrap run:

```text
R_SD_valid_replicates < bootstrap_resamples
    => R_SD_ci_lower = NaN
    => R_SD_ci_upper = NaN
    => heterogeneity gate = false
```

No undefined replicate may be dropped before percentile computation. There is no retry, replacement resample, epsilon stabilization, or threshold search.

The primary causal-effect bootstrap for `mean(G)` remains valid if its own 10,000 values are finite; invalid `R_SD` replicates do not by themselves invalidate the causal-effect gate. Consequently a supported primary effect with invalid heterogeneity inference may classify only as:

```text
P07_OPTIMIZATION_HORIZON_EFFECT_ONLY
```

The adjudicator must additionally require:

```text
R_SD_valid_replicates == bootstrap_resamples
```

before the heterogeneity gate can pass.

## 16. Determinism and scientific invariants

Persistence must sit outside `run_phase07_cell()` and must not alter:

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

Given the same execution identity, interruption + cell-boundary resume must produce the same scientific evidence as uninterrupted execution: metrics, traces, semantic summaries and checkpoint tensor values are equal under deterministic tests. Runtime-only provenance may differ. Serialized checkpoint bytes need not match across independently produced roots if serialization introduces non-semantic byte differences; every bundle must validate against its own hashes.

In addition to reduced smoke fixtures, tests must exercise both `none` and `time_scaled` architectures using the actual frozen `max_epochs=100` and `patience=12` constants. The test may use tiny synthetic data and a learning rate chosen to force deterministic patience behavior; it must not use official Phase 0.7 seeds or execute the official 200-cell plan.

The invariant test must establish that:

- standard policy can terminate through unchanged patience semantics;
- forced policy does not break on patience exhaustion and reaches epoch 100;
- the matched forced trace prefix equals the standard trace through the standard stopping epoch;
- behavior holds for both frozen architecture flow modes.

## 17. CLI behavior

Official local execution remains single-worker and sequential.

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

`--resume` is explicit. The tool never silently resumes an existing output directory.

No cloud/multi-GPU/shard arguments are added.

## 18. Required TDD coverage

Implementation is not accepted unless RED → GREEN tests establish:

1. failure after N committed cells followed by `--resume` executes only remaining cells;
2. failure during staging creates no authoritative final cell;
3. abandoned same-run staging is discarded and its cell reruns;
4. corrupt authoritative bundle fails closed and is not silently rerun;
5. execution SHA mismatch fails before any cell callback;
6. Phase 0.7 spec/config/protocol/plan mismatch fails before callbacks;
7. loaded Phase 0.5 config hash mismatch fails before callbacks;
8. loaded simulator config hash mismatch fails before callbacks;
9. dirty worktree fails before device resolution/cell callbacks;
10. unexpected persisted cell ID fails closed;
11. existing valid cells without `--resume` fail closed;
12. valid completed cells are never passed to `run_phase07_cell()` again on resume;
13. completion cannot be produced with anything other than exact 200 expected cells;
14. aggregate metrics are rebuilt from authoritative bundles;
15. official analysis refuses missing/invalid `COMPLETE` or invalid bundles;
16. interruption + resume and uninterrupted execution produce semantically identical scientific evidence for a controlled deterministic fixture;
17. a crossed-bootstrap run with 9,999/10,000 valid `R_SD` replicates cannot pass heterogeneity;
18. invalid `R_SD` replicates produce NaN `R_SD` CI rather than a percentile over a filtered subset;
19. both `none` and `time_scaled` pass the 100-epoch/patience-12 policy invariant fixture;
20. existing Phase 0.5/0.6 tests remain green;
21. non-official CPU smoke remains green;
22. final candidate requires a user-run CUDA smoke after all changes are complete.

Persistence/orchestration tests may inject a small test-only cell set or fake callback through internal Python interfaces. That injection point must not be exposed through the production CLI and must not weaken `plan_phase07_cells()` / `phase07_plan_sha256()` or the production requirement of exactly 200 official cells. No test or CI path may execute the official 200-cell matrix.

## 19. Component boundaries

Expected implementation boundaries:

- **New:** `src/afmc_fm/phase07/store.py`
  - dedicated Phase 0.7 bundle persistence, integrity and resume validation;
  - modeled on Phase 0.6 semantics;
  - no historical store/evidence changes.

- **Modify:** `src/afmc_fm/phase07/execution.py`
  - complete run-identity construction;
  - dependency config hashing;
  - clean-worktree guard;
  - deterministic sequential store-backed orchestration;
  - failure provenance and exact completion boundary;
  - official-analysis loader/gate or a focused helper consumed by CLI.

- **Modify:** `src/afmc_fm/phase07/cli.py`
  - add `--resume` to `official`;
  - replace direct non-atomic cell persistence with `Phase07Store`;
  - add an official analysis/adjudication command only if needed to expose completion-gated analysis; pure statistical APIs remain unchanged.

- **Modify:** `src/afmc_fm/phase07/analysis.py`
  - fail closed on any invalid `R_SD` bootstrap replicate;
  - require full valid-replicate count for heterogeneity.

- **Tests:** Phase 0.7 store/execution/CLI/analysis/policy tests for all contracts above.

The frozen planner, protected-seed validation, estimand algebra, primary effect gate and scientific runner remain conceptually unchanged.

## 20. Security / integrity posture

The system is intentionally fail-closed:

- dirty checkout → stop;
- identity mismatch → stop;
- dependency-config mismatch → stop;
- unexpected authoritative cell → stop;
- corrupt authoritative cell → stop;
- conflicting completion marker → stop;
- failed atomic final rename → stop;
- partial staging work → discard + rerun only after identity validation;
- any invalid `R_SD` bootstrap replicate → heterogeneity inference invalid.

There is no automatic evidence repair and no authorization bypass.

## 21. Authorization boundary

This document authorizes implementation and testing only. It approves no official training.

Required sequence after implementation:

1. exact-head PR CI;
2. repeat independent Codex adversarial audit;
3. freeze the resulting new candidate SHA only if that audit is clear of official-execution blockers;
4. run the non-official CUDA smoke locally on that exact SHA;
5. review PR state;
6. merge only after explicit user authorization;
7. create/bind separate official execution authorization against the final execution identity;
8. require a separate explicit GO before the 200-cell Phase 0.7 run.

Neither `c8c32fb42518634ca153f915d8916af4952bf360` nor any intermediate implementation commit is the final execution SHA.