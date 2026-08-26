# Phase-0.5 Observation Diagnostic Companion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. This companion plan is intentionally separate from the Phase-0.5 core implementation because the observation mechanism is conditional and may never advance beyond diagnostics.

**Goal:** Implement the optional, decoupled observation-intensity diagnostic specified for Phase-0.5 without letting observation-loss gradients modify the frozen core latent dynamics and without allowing the observation challenger to rescue a failed Stage-III core hypothesis.

**Architecture:** Load the already frozen Phase-0.5 candidate, extract detached pre-event states, and train a small observation-intensity model against measurement opportunities/masks. First establish whether informative observation is predictable beyond marginal-rate and context-only baselines on `informative_observation`; only if that diagnostic passes may a separate future protocol evaluate a propensity-aware correction under `site_shift`.

**Spec:** `docs/superpowers/specs/2026-08-24-phase0-5-mechanistic-redesign-design.md`

## Global Constraints

- This plan begins only after the core Phase-0.5 implementation is complete and a frozen candidate exists.
- The core candidate parameters are always frozen; observation diagnostics consume `stopgrad(z_t^-)`.
- The old Phase-0 BCE observation head is not reintroduced into the core.
- Observation predictability and propensity/missingness correction are separate scientific questions. This plan implements predictability only.
- The diagnostic uses development bundles for mechanism development and the already defined confirmatory bundles only for final held-out description after the diagnostic rule is frozen.
- A positive observation diagnostic cannot change the Stage-III primary gate result.
- If the diagnostic fails, stop; do not implement a correction layer under the same protocol.

---

### Task 1: Define observation-diagnostic examples and context-only controls

**Files:**
- Create: `src/afmc_fm/phase05/observation.py`
- Create: `tests/phase05/test_observation.py`

**Interfaces:**
- `ObservationDiagnosticBatch`.
- `build_observation_diagnostic_examples(...)`.

- [ ] Write tests proving targets correspond to observed lab masks at each valid measurement opportunity and that the feature set contains detached pre-event state plus explicit non-site context only.
- [ ] Define the context-only baseline from elapsed time, prior observation mask/count summaries, and value-code indicator; do not include site identity.
- [ ] Ensure examples are grouped/evaluated by patient/seed bundle rather than treating events as independent replications.
- [ ] Run focused tests and commit.

---

### Task 2: Add detached observation-intensity model and gradient barrier

**Files:**
- Modify: `src/afmc_fm/phase05/observation.py`
- Modify: `tests/phase05/test_observation.py`

**Interfaces:**
- `ObservationIntensityModel(state_dim, context_dim, value_dim)`.
- `fit_observation_intensity(...)`.

- [ ] Write a failing gradient test: after observation BCE backward, every frozen core parameter must have `grad is None` while observation-model gradients are finite/non-zero.
- [ ] Implement a small MLP/logistic head on detached state/context; keep capacity fixed in config before any diagnostic results.
- [ ] Add marginal-rate and context-only baselines.
- [ ] Run CPU/CUDA smoke tests and commit.

---

### Task 3: Implement predictability gate

**Files:**
- Modify: `src/afmc_fm/phase05/observation.py`
- Create: `tests/phase05/test_observation_gate.py`

**Interfaces:**
- `evaluate_observation_predictability(...)`.

- [ ] Compute Brier score and log loss as primary calibration-sensitive diagnostics; ROC-AUC is secondary.
- [ ] Compare latent+context against both marginal-rate and context-only baselines using paired development seed-bundle effects.
- [ ] Require positive mean improvement and >=4/5 development bundles favoring latent+context against each baseline on both Brier and log loss.
- [ ] Do not invent a correction experiment if the gate fails.
- [ ] Persist `robustness/observation_challenger_metrics.csv` with status `predictability_pass` or `predictability_fail` and exact paired effects.
- [ ] Run tests and commit.

---

### Task 4: Add read-only CLI/report integration

**Files:**
- Modify: `src/afmc_fm/cli.py`
- Modify: `src/afmc_fm/phase05/reporting.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/phase05/test_reporting.py`

- [ ] Add `afmc-phase0 phase05 observation-diagnostic` that requires `frozen_candidate.json` and refuses to run before the core freeze.
- [ ] The command must never modify `frozen_candidate.json`, `protocol_lock.json`, or `confirmation/primary_gate_summary.csv`.
- [ ] Include diagnostic plots/tables only as a separate appendix section in reports.
- [ ] Run full relevant tests and Ruff, then commit.

**Stop after predictability. A propensity-aware robustness correction requires a new reviewed design/plan if and only if this diagnostic passes.**
