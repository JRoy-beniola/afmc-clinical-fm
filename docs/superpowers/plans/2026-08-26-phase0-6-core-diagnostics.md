# Phase 0.6 Core Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Phase 0.6 D0-D3 diagnostic program without changing Phase 0.5 training semantics, while producing reproducible D1/D2 diagnostic evidence and a machine-readable D3 adjudication result.

**Architecture:** Instrument the exact existing `afmc_fm.phase05.training` loop through an optional observer contract; keep all Phase 0.6 persistence, planning, execution, and analysis in a new `afmc_fm.phase06` package. D1 and D2 reuse the Phase 0.5 data/model/training semantics but have their own protocol lock, seed firewall, artifact store, and analysis pipeline. D4 interventions are intentionally excluded from this plan and require the separately frozen D4 execution addendum mandated by the design spec.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, pandas, SciPy, scikit-learn, PyYAML, pytest, Ruff; existing AFMC simulator/execution/provenance utilities.

**Spec:** `docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md`

## Global Constraints

- Phase 0.5 remains formally terminated at Stage I-A; no Phase 0.6 result may alter that decision.
- D0-D3 are diagnostic/exploratory evidence only.
- Reserved confirmatory seeds are forbidden: cohort `701..710`, subset `801..810`, model `901..910`.
- D1 uses only smooth world, `{none,time_scaled}`, jump `none`, deterministic uncertainty, N `{5,10,20,40}`, and the five original development bundles.
- D2-A uses only smooth world, `{none,time_scaled}`, jump `none`, deterministic uncertainty, N `{5,40}`, and the locked `OA(25,3,5,2)` mapping `k=(i+j) mod 5`.
- D2 bootstrap settings are exactly `bootstrap_resamples=10000`, `bootstrap_seed=20260826`.
- Phase 0.5 optimizer, learning rate, weight decay, patience, objective, update order, early stopping, and production checkpoint rule remain unchanged in D0-D3.
- The shadow-MAE checkpoint is observational through D1-D3 and is not comparatively evaluated on test data.
- Runtime outputs stay under `outputs/phase06_<execution-sha>/` and remain ignored by Git until intentionally archived later.
- Use `./.venv/bin/python` for local execution and `./.venv/bin/python -m pytest` for tests.
- Do not implement D4 model/control interventions in this plan.

---

## File Structure

Create or modify the following responsibilities without unrelated refactoring:

```text
src/afmc_fm/phase05/training.py
    optional observation contract and observation points in the existing trainer only

src/afmc_fm/phase06/__init__.py
    public Phase 0.6 exports

src/afmc_fm/phase06/diagnostics.py
    concrete in-memory recorder, trace/summary serialization helpers

src/afmc_fm/phase06/config.py
    Phase 0.6 diagnostic config loading and validation

src/afmc_fm/phase06/protocol.py
    confirmatory-seed firewall, D1/D2 plan locks, protocol-lock construction

src/afmc_fm/phase06/planning.py
    immutable cell specifications and exact D1/D2-A planners

src/afmc_fm/phase06/store.py
    atomic/hash-bound persistence for cells, traces, summaries, checkpoints, stage markers

src/afmc_fm/phase06/runner.py
    reuse Phase 0.5 cohort preparation/model/training semantics with diagnostic recorder

src/afmc_fm/phase06/execution.py
    one-worker diagnostic stage execution, resume validation, provenance

src/afmc_fm/phase06/analysis.py
    D1 reproduction, D2 decomposition/bootstrap, D3 adjudication

src/afmc_fm/phase06/cli.py
    `d1`, `d2a`, and `adjudicate` command surface only

configs/experiments/phase06.yaml
    diagnostic-only settings; references Phase 0.5 config rather than duplicating training hyperparameters

tests/phase06/
    D0 invariants, firewall/planning, store, runner/execution, D1/D2/D3 analysis

docs/superpowers/validation/2026-08-26-phase0-6-core-validation.md
    final implementation/readiness evidence before experimental execution
```

Do not create D4 control modules, D4 seed banks, or D4 CLI commands in this implementation cycle.

---

### Task 1: Add the D0 observer contract without changing default training semantics

**Files:**
- Modify: `src/afmc_fm/phase05/training.py`
- Modify: `tests/phase05/test_training.py`
- Create: `tests/phase06/__init__.py`
- Create: `tests/phase06/test_d0_instrumentation.py`

**Interfaces:**
- Produces: `TrainingDiagnosticEpochRecord`, `TrainingDiagnosticSummary`, `TrainingDiagnosticObserver` in `afmc_fm.phase05.training`.
- Produces: `fit_phase05_model(..., diagnostics: TrainingDiagnosticObserver | None = None)` while retaining the existing model return value.
- Consumes: existing `Phase05FlowJumpAdapter`, `_core_loss`, `_forward`, `Phase05Config`.

- [ ] **Step 1: Write a diagnostics-off/on equivalence test before modifying the trainer**

Create a fixed CPU test that initializes two byte-identical deterministic models, deep-copies identical batches, runs one with no observer and one with a recorder stub, and asserts exact final parameter bytes and final metrics.

```python
class _CollectingObserver:
    def __init__(self) -> None:
        self.epochs = []
        self.summary = None

    def on_epoch(self, record) -> None:
        self.epochs.append(record)

    def on_training_end(self, summary) -> None:
        self.summary = summary


def test_diagnostics_are_byte_identical_to_default_training_path():
    torch.manual_seed(29)
    baseline = _model("deterministic")
    observed = deepcopy(baseline)
    train = _batch()
    validation = deepcopy(train)
    train["valid"] = torch.ones(3, 4)
    validation["valid"] = torch.ones(3, 4)

    fit_phase05_model(baseline, deepcopy(train), deepcopy(validation), _tiny_config(), torch.device("cpu"))
    recorder = _CollectingObserver()
    fit_phase05_model(
        observed,
        deepcopy(train),
        deepcopy(validation),
        _tiny_config(),
        torch.device("cpu"),
        diagnostics=recorder,
    )

    assert _parameter_bytes(observed) == _parameter_bytes(baseline)
    assert recorder.summary is not None
    assert evaluate_phase05_model(observed, validation, torch.device("cpu")) == evaluate_phase05_model(
        baseline, validation, torch.device("cpu")
    )
```

- [ ] **Step 2: Run the equivalence test and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/phase06/test_d0_instrumentation.py::test_diagnostics_are_byte_identical_to_default_training_path -v
```

Expected: FAIL because `fit_phase05_model` does not yet accept `diagnostics`.

- [ ] **Step 3: Add immutable diagnostic record types and the narrow observer protocol**

Add to `phase05/training.py` without importing `afmc_fm.phase06`:

```python
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TrainingDiagnosticEpochRecord:
    epoch: int
    train_core_loss: float
    validation_core_loss: float
    validation_mae: float
    validation_rmse: float
    gradient_l2_norm: float
    parameter_l2_norm: float
    mean_flow_displacement: float
    median_flow_displacement: float
    p95_flow_displacement: float
    best_validation_core_loss_so_far: float
    best_core_epoch: int
    shadow_best_validation_mae_so_far: float
    shadow_best_mae_epoch: int
    stale_epochs: int


@dataclass(frozen=True, slots=True)
class TrainingDiagnosticSummary:
    epochs_run: int
    stop_epoch: int
    selected_checkpoint_epoch: int
    selected_validation_core_loss: float
    shadow_mae_checkpoint_epoch: int
    shadow_validation_mae: float
    early_stop_reason: str
    production_state_dict: dict[str, torch.Tensor]
    shadow_state_dict: dict[str, torch.Tensor]


class TrainingDiagnosticObserver(Protocol):
    def on_epoch(self, record: TrainingDiagnosticEpochRecord) -> None: ...
    def on_training_end(self, summary: TrainingDiagnosticSummary) -> None: ...
```

Keep checkpoint tensors out of epoch records; only the final two compact snapshots travel in the terminal summary.

- [ ] **Step 4: Add side-effect-free diagnostic helpers**

Implement private helpers in the same file:

```python
def _global_gradient_l2(parameters: list[torch.nn.Parameter]) -> float:
    squared = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().pow(2).sum()
    return float(torch.sqrt(squared).item())


def _global_parameter_l2(parameters: list[torch.nn.Parameter]) -> float:
    squared = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        squared = squared + parameter.detach().pow(2).sum()
    return float(torch.sqrt(squared).item())


def _validation_point_metrics(output, batch: dict[str, torch.Tensor]) -> tuple[float, float]:
    selected = batch["target_masks"].bool()
    error = output.value_mean[selected] - batch["target_values"][selected]
    mae = error.abs().mean()
    rmse = torch.sqrt(error.pow(2).mean())
    return float(mae.item()), float(rmse.item())


def _flow_displacement(output, batch: dict[str, torch.Tensor]) -> tuple[float, float, float]:
    valid = batch["valid"].bool()
    previous = torch.cat(
        [
            torch.zeros_like(output.post_event_states[:, :1]),
            output.post_event_states[:, :-1],
        ],
        dim=1,
    )
    values = torch.linalg.vector_norm(output.pre_event_states - previous, dim=-1)[valid]
    return (
        float(values.mean().item()),
        float(values.median().item()),
        float(torch.quantile(values, 0.95).item()),
    )
```

If diagnostics are enabled and `valid` is absent, raise `ValueError("diagnostic training requires batch['valid']")` before the first optimizer step.

- [ ] **Step 5: Instrument the exact `_fit_core` loop at the locked observation points**

Do not extract or duplicate the loop. Extend `_fit_core(..., diagnostics=None)` and `fit_phase05_model(..., diagnostics=None)` only.

Required ordering inside each epoch:

```python
loss = _core_loss(model, train, config)
train_loss = float(loss.item())
loss.backward()
gradient_norm = _global_gradient_l2(parameters) if diagnostics is not None else 0.0
optimizer.step()
parameter_norm = _global_parameter_l2(parameters) if diagnostics is not None else 0.0

model.eval()
with torch.no_grad():
    validation_loss = float(_core_loss(model, validation, config).item())
    if diagnostics is not None:
        diagnostic_output = _forward(model, validation)
        validation_mae, validation_rmse = _validation_point_metrics(
            diagnostic_output, validation
        )
        flow_mean, flow_median, flow_p95 = _flow_displacement(
            diagnostic_output, validation
        )
```

The existing `validation_loss < best_loss` condition must remain the only production checkpoint selector and the only stale-epoch reset condition.

Maintain a separate `shadow_best_mae`, `shadow_best_epoch`, and `shadow_best_state` only for observation.

- [ ] **Step 6: Emit records after the existing checkpoint/early-stop decision state has been updated**

Use 1-based epochs. Wrap observer failures clearly:

```python
try:
    diagnostics.on_epoch(record)
except Exception as error:
    raise RuntimeError("training diagnostic observer failed during on_epoch") from error
```

At termination, call `on_training_end` once with `early_stop_reason` equal to `patience_exhausted` or `max_epochs_reached`.

- [ ] **Step 7: Add D0 invariant tests**

Add tests for:

```python
def test_none_flow_reports_exact_zero_displacement(): ...
def test_observer_gets_one_record_per_completed_epoch(): ...
def test_norms_are_finite_and_non_negative(): ...
def test_shadow_checkpoint_never_changes_returned_production_checkpoint(): ...
def test_observer_error_fails_the_diagnostic_run(): ...
def test_phase05_training_does_not_import_phase06(): ...
```

The import-boundary test reads `src/afmc_fm/phase05/training.py` and asserts `"afmc_fm.phase06" not in source`.

- [ ] **Step 8: Run focused tests and the existing Phase 0.5 training regression tests**

```bash
./.venv/bin/python -m pytest tests/phase06/test_d0_instrumentation.py tests/phase05/test_training.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/afmc_fm/phase05/training.py tests/phase05/test_training.py tests/phase06/__init__.py tests/phase06/test_d0_instrumentation.py
git commit -m "feat: add non-invasive Phase 0.6 training diagnostics"
```

---

### Task 2: Implement the concrete Phase 0.6 recorder and serialization contract

**Files:**
- Create: `src/afmc_fm/phase06/__init__.py`
- Create: `src/afmc_fm/phase06/diagnostics.py`
- Create: `tests/phase06/test_diagnostics.py`

**Interfaces:**
- Consumes: `TrainingDiagnosticEpochRecord`, `TrainingDiagnosticSummary`.
- Produces: `Phase06DiagnosticRecorder`, `trace_frame()`, `summary_payload()`, `production_state_dict`, `shadow_state_dict`.

- [ ] **Step 1: Write RED tests for recorder behavior**

```python
def test_recorder_preserves_epoch_order_and_scalar_summary():
    recorder = Phase06DiagnosticRecorder()
    recorder.on_epoch(_epoch_record(epoch=1))
    recorder.on_epoch(_epoch_record(epoch=2))
    recorder.on_training_end(_summary())
    assert recorder.trace_frame()["epoch"].tolist() == [1, 2]
    assert recorder.summary_payload()["selected_checkpoint_epoch"] == 1


def test_recorder_rejects_duplicate_or_out_of_order_epochs():
    recorder = Phase06DiagnosticRecorder()
    recorder.on_epoch(_epoch_record(epoch=2))
    with pytest.raises(ValueError, match="strictly increasing"):
        recorder.on_epoch(_epoch_record(epoch=2))
```

- [ ] **Step 2: Run RED**

```bash
./.venv/bin/python -m pytest tests/phase06/test_diagnostics.py -v
```

Expected: import failure because `afmc_fm.phase06.diagnostics` does not exist.

- [ ] **Step 3: Implement the recorder as an in-memory sink only**

```python
class Phase06DiagnosticRecorder:
    def __init__(self) -> None:
        self._epochs: list[TrainingDiagnosticEpochRecord] = []
        self._summary: TrainingDiagnosticSummary | None = None

    def on_epoch(self, record: TrainingDiagnosticEpochRecord) -> None:
        if self._epochs and record.epoch <= self._epochs[-1].epoch:
            raise ValueError("diagnostic epochs must be strictly increasing")
        self._epochs.append(record)

    def on_training_end(self, summary: TrainingDiagnosticSummary) -> None:
        if self._summary is not None:
            raise RuntimeError("diagnostic training summary already recorded")
        self._summary = summary
```

`trace_frame()` must use the exact field order from the spec. `summary_payload()` must exclude tensor state dicts and return only JSON-safe scalars. Expose defensive CPU clones of the two final checkpoint state dicts.

- [ ] **Step 4: Add validation for finite required fields**

Reject non-finite losses/norms/displacements in `on_epoch`; reject invalid `early_stop_reason`; reject summary epochs outside `[1, epochs_run]`.

- [ ] **Step 5: Run tests and commit**

```bash
./.venv/bin/python -m pytest tests/phase06/test_diagnostics.py -q
git add src/afmc_fm/phase06 tests/phase06/test_diagnostics.py
git commit -m "feat: add Phase 0.6 diagnostic recorder"
```

---

### Task 3: Lock Phase 0.6 config, protocol identity, and confirmatory-seed firewall

**Files:**
- Create: `configs/experiments/phase06.yaml`
- Create: `src/afmc_fm/phase06/config.py`
- Create: `src/afmc_fm/phase06/protocol.py`
- Create: `tests/phase06/test_config.py`
- Create: `tests/phase06/test_protocol.py`

**Interfaces:**
- Produces: `Phase06Config`, `load_phase06_config(path)`, `validate_development_seed_triplet(cohort_seed, subset_seed, model_seed)`, `build_phase06_protocol_lock(...)`.
- Consumes: `load_phase05_config`, `canonical_config_hash`, exact spec/config files.

- [ ] **Step 1: Add the locked YAML without duplicating Phase 0.5 optimizer settings**

Use exactly:

```yaml
phase05_config: configs/experiments/phase05.yaml
phase06_spec: docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md
simulator_config: configs/simulator/full.yaml
world: smooth
d1_train_sizes: [5, 10, 20, 40]
d2_train_sizes: [5, 40]
d1_flow_modes: [none, time_scaled]
d2_flow_modes: [none, time_scaled]
jump_mode: none
uncertainty_mode: deterministic
bootstrap_resamples: 10000
bootstrap_seed: 20260826
forbidden_cohort_seeds: [701, 702, 703, 704, 705, 706, 707, 708, 709, 710]
forbidden_subset_seeds: [801, 802, 803, 804, 805, 806, 807, 808, 809, 810]
forbidden_model_seeds: [901, 902, 903, 904, 905, 906, 907, 908, 909, 910]
```

- [ ] **Step 2: Write config/firewall RED tests**

Test exact tuples, bootstrap values, and rejection of each reserved seed namespace independently.

```python
@pytest.mark.parametrize("seed", range(701, 711))
def test_reserved_confirmatory_cohort_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="confirmatory cohort seed"):
        validate_development_seed_triplet(seed, 501, 601)
```

Repeat for subset and model ranges.

- [ ] **Step 3: Implement `Phase06Config` as frozen dataclass validation**

Reject any flow mode other than `none`/`time_scaled`, any world other than `smooth`, any D1 sizes other than `(5,10,20,40)`, and any D2 sizes other than `(5,40)`.

- [ ] **Step 4: Implement protocol lock construction**

`build_phase06_protocol_lock` must persist at least:

```python
{
    "schema_version": 1,
    "phase06_spec_sha256": spec_sha256,
    "phase06_config_sha256": phase06_config_sha256,
    "phase05_config_sha256": canonical_config_hash(phase05_config),
    "phase05_protocol_sha256": phase05_protocol_sha256,
    "phase05_execution_sha": "50a94c06bc1c419ca55738f15f074cc06ccc3f36",
    "phase05_official_status": "PHASE-0.5 TERMINATED AT STAGE I-A — FLOW MECHANISM GATE NOT ESTABLISHED",
    "execution_commit": execution_commit,
    "forbidden_seed_sets": {...},
    "d1_development_bundles": [[401,501,601], ..., [405,505,605]],
    "d2_mapping": "model_index=(cohort_index+subset_index)%5",
    "bootstrap_resamples": 10000,
    "bootstrap_seed": 20260826,
}
```

Read the Phase 0.5 protocol hash from the archived `docs/results/phase05/raw/official_output/protocol_lock.json` bytes or an explicitly supplied path; do not invent it.

- [ ] **Step 5: Run tests and commit**

```bash
./.venv/bin/python -m pytest tests/phase06/test_config.py tests/phase06/test_protocol.py -q
git add configs/experiments/phase06.yaml src/afmc_fm/phase06/config.py src/afmc_fm/phase06/protocol.py tests/phase06/test_config.py tests/phase06/test_protocol.py
git commit -m "feat: lock Phase 0.6 protocol and seed firewall"
```

---

### Task 4: Implement exact D1 and D2-A planners

**Files:**
- Create: `src/afmc_fm/phase06/planning.py`
- Create: `tests/phase06/test_planning.py`

**Interfaces:**
- Produces: `Phase06CellSpec`, `plan_d1_cells(config, phase05_config)`, `plan_d2a_cells(config)`.
- Consumes: `SeedBundle` from Phase 0.5 config and `validate_development_seed_triplet`.

- [ ] **Step 1: Write RED tests for exact plan cardinalities and balance**

```python
def test_d1_plan_is_exactly_40_locked_cells():
    cells = plan_d1_cells(_phase06_config(), _phase05_config())
    assert len(cells) == 40
    assert {cell.n_train for cell in cells} == {5, 10, 20, 40}
    assert {cell.flow_mode for cell in cells} == {"none", "time_scaled"}


def test_d2a_is_strength_two_orthogonal_array():
    cells = plan_d2a_cells(_phase06_config())
    seed_triples = {
        (cell.cohort_seed, cell.subset_seed, cell.model_seed)
        for cell in cells
    }
    assert len(seed_triples) == 25
```

Also assert each factor level appears five times among the 25 unique triples and every cohort-subset, cohort-model, and subset-model pair appears exactly once.

- [ ] **Step 2: Implement immutable `Phase06CellSpec`**

```python
@dataclass(frozen=True, slots=True)
class Phase06CellSpec:
    stage: Literal["d1", "d2a"]
    world: Literal["smooth"]
    cohort_seed: int
    subset_seed: int
    model_seed: int
    n_train: int
    flow_mode: Literal["none", "time_scaled"]
    jump_mode: Literal["none"] = "none"
    uncertainty_mode: Literal["deterministic"] = "deterministic"

    @property
    def cell_id(self) -> str:
        return (
            f"{self.stage}__smooth__cohort{self.cohort_seed}__"
            f"subset{self.subset_seed}__model{self.model_seed}__"
            f"n{self.n_train}__{self.flow_mode}__none__deterministic"
        )
```

Call the seed firewall in `__post_init__`.

- [ ] **Step 3: Implement D1 from Phase 0.5 development bundles only**

Iterate the exact five `phase05_config.development_bundles`, four N values, and two flow modes. Assert the resulting ID set has cardinality 40 before returning.

- [ ] **Step 4: Implement D2-A using the locked Latin-square mapping**

```python
cohorts = (401, 402, 403, 404, 405)
subsets = (501, 502, 503, 504, 505)
models = (601, 602, 603, 604, 605)
triples = [
    (cohort, subset, models[(i + j) % 5])
    for i, cohort in enumerate(cohorts)
    for j, subset in enumerate(subsets)
]
```

Expand those 25 triples over N `{5,40}` and flow `{none,time_scaled}` to exactly 100 cells.

- [ ] **Step 5: Run tests and commit**

```bash
./.venv/bin/python -m pytest tests/phase06/test_planning.py -q
git add src/afmc_fm/phase06/planning.py tests/phase06/test_planning.py
git commit -m "feat: add locked Phase 0.6 diagnostic planners"
```

---

### Task 5: Implement hash-bound Phase 0.6 artifact storage

**Files:**
- Create: `src/afmc_fm/phase06/store.py`
- Create: `tests/phase06/test_store.py`

**Interfaces:**
- Produces: `Phase06Store.write_protocol_lock`, `write_cell_bundle`, `validate_resume`, `mark_stage_complete`, `load_stage_metrics`.
- Consumes: `Phase06CellSpec`, recorder trace/summary/checkpoints.

- [ ] **Step 1: Write RED persistence tests around one synthetic diagnostic cell**

Verify the exact paths:

```text
stages/d1/cells/<cell_id>.json
stages/d1/traces/<cell_id>.csv
stages/d1/summaries/<cell_id>.json
stages/d1/checkpoints/<cell_id>__production.pt
stages/d1/checkpoints/<cell_id>__shadow_mae.pt
```

Assert conflicting rewrites fail, identical rewrites are idempotent, and resume validation rejects unexpected cell IDs.

- [ ] **Step 2: Implement atomic byte writers and SHA-256 helper**

Reuse the same temp-file + `os.replace` approach used by `Phase05Store`; do not import private Phase 0.5 store helpers.

- [ ] **Step 3: Implement checkpoint serialization**

Serialize CPU state dict clones with `torch.save` to a temporary file, read final bytes for SHA-256, then atomically rename. The summary manifest stores the actual saved-byte hashes; deterministic byte reproduction across machines is not assumed.

- [ ] **Step 4: Implement `write_cell_bundle` as all-or-conflict persistence**

The summary JSON must include:

```python
"artifact_sha256": {
    "cell_metrics": cell_hash,
    "training_trace": trace_hash,
    "production_checkpoint": production_hash,
    "shadow_mae_checkpoint": shadow_hash,
},
"identity": {
    "protocol_lock_sha256": self.protocol_hash,
    "phase06_config_sha256": self.config_hash,
    "execution_commit": self.execution_commit,
}
```

It also includes the scalar `training_summary` payload. No stage can be marked complete unless every expected cell has all five artifacts and all hashes revalidate.

- [ ] **Step 5: Add exact COMPLETE marker validation**

The marker contains sorted expected cell IDs and the store identity. A stale marker with missing/extra cells is an error.

- [ ] **Step 6: Run tests and commit**

```bash
./.venv/bin/python -m pytest tests/phase06/test_store.py -q
git add src/afmc_fm/phase06/store.py tests/phase06/test_store.py
git commit -m "feat: persist hash-bound Phase 0.6 diagnostic artifacts"
```

---

### Task 6: Reuse the exact Phase 0.5 data/model/training path for diagnostic cells

**Files:**
- Create: `src/afmc_fm/phase06/runner.py`
- Create: `tests/phase06/test_runner.py`

**Interfaces:**
- Produces: `run_phase06_cell(prepared, phase05_config, cell, device) -> Phase06CellRun`.
- `Phase06CellRun` contains `metrics: pd.DataFrame`, `trace: pd.DataFrame`, `summary: dict[str, object]`, and the two checkpoint state dicts.
- Consumes: `prepare_phase05_cohort`, `padded_phase05_batch`, `Phase05FlowJumpAdapter`, `fit_phase05_model`, `evaluate_phase05_model`, `select_low_n_budget` semantics already used by Phase 0.5.

- [ ] **Step 1: Write a RED runner test using a small simulated smooth cohort**

The test checks that the output variant is exactly `none__none__deterministic` or `time_scaled__none__deterministic`, trace is non-empty, summary has selected/shadow epochs, and final metrics include `mae`, `rmse`, and `latent_aligned_r2`.

- [ ] **Step 2: Implement D1/D2 cohort preparation as a thin Phase 0.5 reuse**

```python
def prepare_phase06_cohort(
    simulator_config: SimulatorConfig,
    cell: Phase06CellSpec,
) -> PreparedPhase05Cohort:
    cohort = simulate_world("smooth", simulator_config, seed=cell.cohort_seed)
    return prepare_phase05_cohort(cohort, include_historical=False)
```

Do not add a new simulator path.

- [ ] **Step 3: Implement the diagnostic variant runner without calling `run_phase05_variant`**

Reproduce its exact split/budget/model construction in a Phase 0.6 wrapper only because diagnostics must be passed into `fit_phase05_model` and the two final checkpoints must be retained. Import and reuse the same Phase 0.5 helpers/data structures rather than reimplementing sequence encoding.

The critical model construction remains:

```python
model = Phase05FlowJumpAdapter(
    representation_dim=16,
    value_dim=len(prepared.task.value_codes),
    event_dim=len(EventType),
    state_dim=24,
    flow_mode=cell.flow_mode,
    jump_mode="none",
    uncertainty_mode="deterministic",
    time_scale_days=phase05_config.time_scale_days,
)
```

Set NumPy and Torch model seeds exactly as Phase 0.5 does before model construction.

- [ ] **Step 4: Pass `valid` in the training and validation batches only for the diagnostic observer**

`padded_phase05_batch` already produces `valid`; do not modify sequence semantics.

- [ ] **Step 5: Evaluate only the production-selected model on test data**

Call `evaluate_phase05_model(model, test_batch, device)` after training returns. Do not load or evaluate the shadow checkpoint on the test split.

- [ ] **Step 6: Verify output row schema matches Phase 0.5 scientific rows plus Phase 0.6 stage metadata**

Retain `world`, three seeds, N, model, variant, split, metric, value, trainable_parameters, backend. Set `stage` to `d1` or `d2a` from the cell.

- [ ] **Step 7: Run tests and commit**

```bash
./.venv/bin/python -m pytest tests/phase06/test_runner.py tests/phase05/test_runner.py -q
git add src/afmc_fm/phase06/runner.py tests/phase06/test_runner.py
git commit -m "feat: add Phase 0.6 diagnostic cell runner"
```

---

### Task 7: Implement one-worker D1/D2 execution, resume, and provenance

**Files:**
- Create: `src/afmc_fm/phase06/execution.py`
- Create: `src/afmc_fm/phase06/cli.py`
- Modify: `pyproject.toml`
- Create: `tests/phase06/test_execution.py`
- Create: `tests/phase06/test_cli.py`

**Interfaces:**
- Produces: `run_phase06_stage(cells, store, phase05_config, simulator_config, device="cuda", resume=False)`.
- Produces CLI commands `afmc-phase06 d1`, `afmc-phase06 d2a`, `afmc-phase06 adjudicate`.
- Consumes existing runtime metadata helpers from `afmc_fm.execution.manifest` and device resolution from `afmc_fm.execution.device`.

- [ ] **Step 1: Write RED execution tests using injected prepare/run callables**

Test exact planned/completed counts, fail-fast on cell exception, resume skipping only fully hash-valid cells, and rejection of duplicate IDs.

- [ ] **Step 2: Implement sequential execution first and only**

The Phase 0.6 design intends CUDA with one worker for D1 reproduction. Do not add multiprocessing in this plan.

Execution order is deterministic: sorted by `(cohort_seed, subset_seed, model_seed, n_train, flow_mode)`.

- [ ] **Step 3: Persist execution provenance**

Write `execution_provenance.json` with:

```python
{
    "stage": stage,
    "planned_cells": len(cells),
    "completed_before": completed_before,
    "completed_after": completed_after,
    "failures": failures,
    "device": str(device),
    "workers": 1,
    "started_at": started_at,
    "ended_at": ended_at,
    "wall_seconds": wall_seconds,
    "execution_commit": execution_commit,
    "protocol_lock_sha256": store.protocol_hash,
    "forbidden_seed_validation": "passed",
    "runtime": collect_runtime_metadata(...),
}
```

A scientific `not_reproduced` classification is not an execution failure and therefore does not belong in `failures`.

- [ ] **Step 4: Add a dedicated console script**

Modify `pyproject.toml`:

```toml
[project.scripts]
afmc-phase0 = "afmc_fm.cli:main"
afmc-phase06 = "afmc_fm.phase06.cli:main"
```

The CLI supports:

```text
afmc-phase06 d1 --config configs/experiments/phase06.yaml --output outputs/phase06_<sha> --device cuda [--resume]
afmc-phase06 d2a --config configs/experiments/phase06.yaml --output outputs/phase06_<sha> --device cuda [--resume]
afmc-phase06 adjudicate --output outputs/phase06_<sha>
```

There are deliberately no `confirmation`, `robustness`, or D4 commands.

- [ ] **Step 5: Make CLI startup bind protocol/config identity before planning cells**

The CLI must load Phase 0.6 config, then referenced Phase 0.5 config, build/validate protocol lock, construct `Phase06Store`, and only then call the D1/D2 planner. Any hash mismatch exits non-zero before simulation/training.

- [ ] **Step 6: Run tests and commit**

```bash
./.venv/bin/python -m pytest tests/phase06/test_execution.py tests/phase06/test_cli.py -q
git add src/afmc_fm/phase06/execution.py src/afmc_fm/phase06/cli.py pyproject.toml tests/phase06/test_execution.py tests/phase06/test_cli.py
git commit -m "feat: add Phase 0.6 diagnostic execution CLI"
```

---

### Task 8: Implement D1 reproduction analysis exactly as specified

**Files:**
- Create: `src/afmc_fm/phase06/analysis.py`
- Create: `tests/phase06/test_d1_analysis.py`

**Interfaces:**
- Produces: `analyze_d1(metrics, summaries) -> tuple[pd.DataFrame, dict[str, object]]`.
- The table is persisted as `analysis/phase06_d1_reproduction.csv`.

- [ ] **Step 1: Write synthetic fixtures for all three D1 classifications**

Construct 5-bundle fixture frames whose paired MAE effects satisfy:

```text
reproduced: N5<=0, N10<=0, N20<=0, N40>0 with >=4/5 N40 wins
not_reproduced: N40<=0 or <=2/5 N40 wins
ambiguous: every other case
```

- [ ] **Step 2: Implement effect orientation once**

```python
Delta_MAE = MAE_none - MAE_time_scaled
Delta_R2 = R2_time_scaled - R2_none
```

Never infer orientation from metric names elsewhere.

- [ ] **Step 3: Build the required consolidated D1 table**

Each row corresponds to one bundle/N pair and includes control/candidate MAE, `Delta_MAE`, winner, selected epochs, shadow epochs, and stop epochs for both variants.

- [ ] **Step 4: Implement the exact reproduction classifier**

Return one of `reproduced`, `not_reproduced`, `ambiguous` with the five condition values included in the decision payload.

- [ ] **Step 5: Run tests and commit**

```bash
./.venv/bin/python -m pytest tests/phase06/test_d1_analysis.py -q
git add src/afmc_fm/phase06/analysis.py tests/phase06/test_d1_analysis.py
git commit -m "feat: add Phase 0.6 D1 reproduction analysis"
```

---

### Task 9: Implement D2 additive decomposition and fixed-seed bootstrap

**Files:**
- Modify: `src/afmc_fm/phase06/analysis.py`
- Create: `tests/phase06/test_d2_analysis.py`

**Interfaces:**
- Produces: `analyze_d2a(metrics, config) -> D2AnalysisResult` with effect rows, variance-component rows, bootstrap diagnostics, and N5-to-N40 paired shifts.

- [ ] **Step 1: Write a fixture with a known dominant model-seed main effect**

Generate 25 OA paired effects using additive values such as:

```python
cohort_effect = {401: 0.00, 402: 0.01, 403: -0.01, 404: 0.00, 405: 0.00}
subset_effect = {501: 0.00, 502: 0.00, 503: 0.01, 504: -0.01, 505: 0.00}
model_effect = {601: -0.20, 602: -0.10, 603: 0.00, 604: 0.10, 605: 0.20}
```

Verify model has the largest main-effect share and qualifies as diagnostically dominant.

- [ ] **Step 2: Implement the additive least-squares decomposition**

Use `sklearn.preprocessing.OneHotEncoder(drop="first", sparse_output=False)` plus `numpy.linalg.lstsq`, or an equivalent deterministic design matrix. Compute total SS, each factor's partial/main-effect SS by reduced-model comparison, and residual SS. Normalize by total SS when positive.

- [ ] **Step 3: Implement the 10,000-resample fixed RNG bootstrap**

```python
rng = np.random.default_rng(config.bootstrap_seed)
for _ in range(config.bootstrap_resamples):
    indices = rng.integers(0, len(effect_rows), size=len(effect_rows))
```

Refit the additive model on each resample and count which named main effect is largest. Persist the largest-component frequency for cohort/subset/model.

- [ ] **Step 4: Apply diagnostic dominance and weak rules exactly**

Dominant requires largest share, >=2x next-largest main-effect share, and >=0.80 bootstrap-largest frequency. Weak requires smallest main effect and <=0.20 bootstrap-largest frequency.

- [ ] **Step 5: Compute paired N shift for the same 25 OA combinations**

```python
T = Delta_MAE_N40 - Delta_MAE_N5
```

Persist mean T, count `T>0`, and a fixed-seed 95% bootstrap interval for mean T.

- [ ] **Step 6: Run tests and commit**

```bash
./.venv/bin/python -m pytest tests/phase06/test_d2_analysis.py -q
git add src/afmc_fm/phase06/analysis.py tests/phase06/test_d2_analysis.py
git commit -m "feat: add Phase 0.6 D2 variance decomposition"
```

---

### Task 10: Implement predeclared D3 adjudication and escalation routing

**Files:**
- Modify: `src/afmc_fm/phase06/analysis.py`
- Create: `tests/phase06/test_d3_adjudication.py`

**Interfaces:**
- Produces: `adjudicate_phase06(d1_result, d2_result, traces) -> dict[str, object]`.
- Canonical output: `analysis/phase06_d3_adjudication.json`.

- [ ] **Step 1: Write one fixture per classification branch before implementation**

Cover:

```text
H1/H2/H3 strengthened, weakened, unresolved
H4 strong, partial, absent, unresolved
H5 strengthened, weakened, unresolved
H7 strengthened, weakened, unresolved
D2B trigger on interaction ambiguity
STOP route
D4_CHECKPOINT route
D4_OPTIMIZATION route
D4_DATA_REGIME route
D4_CAPACITY_TIME route
```

- [ ] **Step 2: Implement H5 from validation checkpoint gap only**

For each D1 N40 cell:

```python
G_ckpt = validation_mae_at_core_epoch - shadow_best_validation_mae
D_ckpt = G_ckpt_time_scaled - G_ckpt_none
```

H5 strengthened iff mean `D_ckpt>0`, >=4/5 D_ckpt values are positive, and time-scaled `G_ckpt>0` in >=4/5 bundles. Weakened iff mean `D_ckpt<=0` and <=2/5 are positive; else unresolved.

- [ ] **Step 3: Implement H1/H2/H3 directly from D2 factor classifications**

Do not reinterpret residual variance as a named factor. Large residual/unstable ranking produces a D2B escalation flag.

- [ ] **Step 4: Implement H4 exactly**

`strong` requires D1 reproduced, D2 mean N5 <=0, mean N40 >0, mean T >0, >=20/25 positive T values, and the 95% bootstrap interval wholly above zero. Implement `partial`, `absent`, and `unresolved` exactly as the spec states.

- [ ] **Step 5: Implement H7 from D1 N40 predictive and latent effects**

Strengthened iff mean `Delta_MAE>0`, MAE wins >=4/5, mean `Delta_R2<=0`, and latent-R2 wins <=2/5. Weakened iff both effects favor time-scaled on average and each wins >=4/5; otherwise unresolved.

- [ ] **Step 6: Keep H6 exactly `not_yet_tested` in D3**

Parameter count differences must not change this value.

- [ ] **Step 7: Implement deterministic next-stage routing**

Use this priority order so the same evidence cannot route two ways:

```text
1. interaction ambiguity requiring D2-B -> D2B
2. H5 strengthened -> D4_CHECKPOINT
3. H1 strengthened -> D4_OPTIMIZATION
4. H2 or H3 strengthened -> D4_DATA_REGIME
5. H4 strong/partial or H7 strengthened -> D4_CAPACITY_TIME
6. otherwise -> STOP
```

The output records all `triggered_escalations` even though `next_required_stage` is singular.

- [ ] **Step 8: Run tests and commit**

```bash
./.venv/bin/python -m pytest tests/phase06/test_d3_adjudication.py -q
git add src/afmc_fm/phase06/analysis.py tests/phase06/test_d3_adjudication.py
git commit -m "feat: add Phase 0.6 D3 adjudication"
```

---

### Task 11: Wire analysis persistence into the CLI without auto-running later scientific stages

**Files:**
- Modify: `src/afmc_fm/phase06/cli.py`
- Modify: `src/afmc_fm/phase06/store.py`
- Modify: `tests/phase06/test_cli.py`
- Create: `tests/phase06/test_end_to_end_smoke.py`

**Interfaces:**
- D1 command produces D1 artifacts and D1 classification only.
- D2A command produces D2 artifacts only.
- `adjudicate` reads completed hash-bound D1/D2 artifacts and writes D3 JSON.

- [ ] **Step 1: Write a smoke test with injected tiny cell runners**

Use temporary output, fake valid cell bundles, and verify D1/D2 COMPLETE markers plus analysis artifact names. Do not execute 140 real training cells in unit tests.

- [ ] **Step 2: Make D1 analysis run only after all 40 D1 cells are complete**

If the D1 stage is incomplete, analysis must fail with an explicit message rather than classify partial evidence.

- [ ] **Step 3: Make D2 analysis run only after all 100 D2-A cells are complete**

Likewise reject partial D2 evidence.

- [ ] **Step 4: Make `adjudicate` require both stage completion markers and revalidated hashes**

`adjudicate` must never launch D2-B or any D4 experiment automatically. It only writes `phase06_d3_adjudication.json` and prints `next_required_stage`.

- [ ] **Step 5: Run tests and commit**

```bash
./.venv/bin/python -m pytest tests/phase06/test_cli.py tests/phase06/test_end_to_end_smoke.py -q
git add src/afmc_fm/phase06/cli.py src/afmc_fm/phase06/store.py tests/phase06/test_cli.py tests/phase06/test_end_to_end_smoke.py
git commit -m "feat: complete Phase 0.6 D1-D3 diagnostic pipeline"
```

---

### Task 12: Full regression, scientific-boundary audit, and implementation validation record

**Files:**
- Create: `docs/superpowers/validation/2026-08-26-phase0-6-core-validation.md`
- Modify only if a failing test proves necessary: files already introduced in Tasks 1-11

**Interfaces:**
- Produces: a human-readable implementation-readiness record; does not execute official D1/D2 CUDA science runs.

- [ ] **Step 1: Run Ruff on the full repository**

```bash
./.venv/bin/python -m ruff check .
```

Expected: PASS. Do not weaken existing lint rules to hide new-code problems.

- [ ] **Step 2: Run all Phase 0.6 tests**

```bash
./.venv/bin/python -m pytest tests/phase06 -q
```

Expected: PASS.

- [ ] **Step 3: Run Phase 0.5 regression tests**

```bash
./.venv/bin/python -m pytest tests/phase05 -q
```

Expected: PASS with no changed scientific expectations.

- [ ] **Step 4: Run the complete repository suite**

```bash
./.venv/bin/python -m pytest -q
```

Expected: PASS.

- [ ] **Step 5: Run explicit scientific-boundary audits**

```bash
! grep -R "701\|702\|703\|704\|705\|706\|707\|708\|709\|710" -n src/afmc_fm/phase06 tests/phase06 configs/experiments/phase06.yaml | grep -v forbidden
! grep -R "confirmation\|robustness" -n src/afmc_fm/phase06/cli.py
! grep -R "afmc_fm.phase06" -n src/afmc_fm/phase05/training.py
```

The first command may be replaced by a small Python audit if comments/tests legitimately contain forbidden values; the invariant is that no executable D1/D2 plan contains reserved values.

- [ ] **Step 6: Run a CPU non-interference proof test in isolation and record its exact output**

```bash
./.venv/bin/python -m pytest tests/phase06/test_d0_instrumentation.py::test_diagnostics_are_byte_identical_to_default_training_path -vv
```

Expected: PASS.

- [ ] **Step 7: Write the validation record with exact commands and observed results**

The document must state:

```text
implementation status: READY FOR D1 EXECUTION or NOT READY
branch/head SHA
spec SHA-256
phase06 config SHA-256
phase05 config SHA-256
Ruff result
Phase06 test count/result
Phase05 regression count/result
full-suite count/result
D0 non-interference result
seed-firewall result
D1 planned cells = 40
D2-A planned cells = 100
D4 implementation = intentionally absent pending D3/addendum
```

Do not claim readiness until the commands have actually completed successfully.

- [ ] **Step 8: Commit the validation record only after verification**

```bash
git add docs/superpowers/validation/2026-08-26-phase0-6-core-validation.md
git commit -m "docs: validate Phase 0.6 diagnostic implementation"
```

- [ ] **Step 9: Push the branch and leave PR #3 in draft state**

```bash
git push origin phase0-6-diagnostics
```

Do not mark PR #3 ready yet if official D1/D2 execution or any requested code review remains pending.

---

## Plan Self-Review Result

**Spec coverage:** D0 non-interference and observations are covered by Tasks 1-2; config/protocol/firewall by Task 3; exact D1/D2 planning by Task 4; hash-bound artifact contract by Task 5; Phase 0.5 semantic reuse by Task 6; execution/provenance/CLI by Task 7; D1/D2 analysis by Tasks 8-9; D3 adjudication by Task 10; stage gating by Task 11; verification/readiness by Task 12. D4 is intentionally excluded because the approved spec requires a conditional D4 execution addendum after D3.

**Placeholder scan:** No `TBD`, `TODO`, unspecified error-handling step, or undefined future component is required to execute Tasks 1-12.

**Type/interface consistency:** `Phase06CellSpec` is created in Task 4 and consumed by Tasks 5-7; the Phase 0.5 diagnostic record types are created in Task 1 and consumed by Task 2/6; D1/D2 result artifacts are created in Tasks 8-9 and consumed by Task 10/11. `fit_phase05_model` retains its historical return type in every task.

**Execution boundary:** This plan ends with implementation readiness. It does not authorize official D1/D2 CUDA execution until the implementation verification gate is green, and it never authorizes confirmatory seeds or D4 interventions.