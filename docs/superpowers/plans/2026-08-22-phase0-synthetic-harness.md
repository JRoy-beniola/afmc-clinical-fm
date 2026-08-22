# Phase 0 Synthetic Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python research harness that generates longitudinal synthetic patient cohorts with known latent dynamics, converts them into a common event representation, trains strong low-data baselines and a compact flow-jump latent-state model, and evaluates sample efficiency, uncertainty, latent-state recovery, and observation-shift robustness before any scarce AFMC cohort is used for architecture development.

**Architecture:** The codebase is organized around four contracts: a MEDS-like event stream, a simulator that exposes latent ground truth, sequence builders that convert event histories into patient-level tensors, and model/evaluation modules that never depend on simulator internals. The first proposed learner uses a low-capacity elapsed-time-conditioned flow plus event-conditioned jumps; explicit observation-process modelling is added only after the core adapter works. A fixed synthetic history encoder is used in Phase 0 only to exercise the generic-representation interface; later CLMBR/EHRSHOT or another EHR foundation model can replace it without changing the downstream model API.

**Tech Stack:** Python 3.11+, NumPy, pandas, SciPy, scikit-learn, PyTorch, PyYAML, matplotlib, pytest, pytest-cov, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-22-phase0-synthetic-design.md`

## Global Constraints

- No real patient record, identifier, clinical value, or derived export may be committed to the repository.
- Patient-level train/validation/test separation is mandatory; events from one patient may never cross those boundaries.
- Synthetic truth generation and learner architecture must remain distinct enough to avoid trivial equation matching.
- The project must not make causal claims from intervention-conditioned observational transitions.
- The initial transition model is a simple elapsed-time-conditioned flow; Neural ODE/CDE variants are ablations, not the default implementation.
- The explicit observation model is a secondary hypothesis and must be removable without breaking the core model.
- Low-N experiments must include `N_train in {5, 10, 20, 40, 80, 100}` where cohort size permits.
- All stochastic components must accept explicit seeds and produce deterministic outputs under the same seed.
- Synthetic benchmark results are methodological evidence only; they are not clinical-validation evidence.

---

## File Map

The completed Phase 0 implementation should create or modify the following files:

```text
.gitignore
README.md
pyproject.toml
.github/workflows/ci.yml
configs/simulator/smoke.yaml
configs/simulator/full.yaml
configs/experiments/low_n.yaml
examples/real_patient_template.csv
src/afmc_fm/__init__.py
src/afmc_fm/config.py
src/afmc_fm/schema/__init__.py
src/afmc_fm/schema/events.py
src/afmc_fm/simulator/__init__.py
src/afmc_fm/simulator/config.py
src/afmc_fm/simulator/dynamics.py
src/afmc_fm/simulator/observation.py
src/afmc_fm/simulator/cohort.py
src/afmc_fm/data/__init__.py
src/afmc_fm/data/encoding.py
src/afmc_fm/data/sequences.py
src/afmc_fm/data/splits.py
src/afmc_fm/models/__init__.py
src/afmc_fm/models/baselines.py
src/afmc_fm/models/flow_jump.py
src/afmc_fm/models/losses.py
src/afmc_fm/metrics/__init__.py
src/afmc_fm/metrics/forecasting.py
src/afmc_fm/metrics/latent.py
src/afmc_fm/experiments/__init__.py
src/afmc_fm/experiments/runner.py
src/afmc_fm/cli.py
tests/test_config.py
tests/schema/test_events.py
tests/simulator/test_dynamics.py
tests/simulator/test_observation.py
tests/simulator/test_cohort.py
tests/data/test_encoding.py
tests/data/test_sequences.py
tests/data/test_splits.py
tests/models/test_baselines.py
tests/models/test_flow_jump.py
tests/models/test_losses.py
tests/metrics/test_forecasting.py
tests/metrics/test_latent.py
tests/experiments/test_runner.py
tests/test_cli.py
```

The boundaries are intentional: simulator truth must not leak into learner code; event representation must not depend on the simulator; experiments must call public model/data APIs rather than importing private helper state.

---

### Task 1: Project scaffold, dependency manifest, privacy guardrails, and configuration loading

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.github/workflows/ci.yml`
- Create: `src/afmc_fm/__init__.py`
- Create: `src/afmc_fm/config.py`
- Create: `configs/simulator/smoke.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_yaml(path: str | Path) -> dict[str, Any]`
- Produces: installable `afmc_fm` package under `src/`
- Produces: a CI job running Ruff and pytest on Python 3.11

- [ ] **Step 1: Write the failing configuration test**

```python
# tests/test_config.py
from pathlib import Path

from afmc_fm.config import load_yaml


def test_load_yaml_reads_mapping(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 7\ncohort_size: 12\n", encoding="utf-8")

    config = load_yaml(config_path)

    assert config == {"seed": 7, "cohort_size": 12}
```

- [ ] **Step 2: Run the test and verify it fails before the package exists**

Run:

```bash
pytest tests/test_config.py -v
```

Expected: import failure for `afmc_fm.config`.

- [ ] **Step 3: Add the package manifest and dependencies**

Use this initial `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=70"]
build-backend = "setuptools.build_meta"

[project]
name = "afmc-clinical-fm"
version = "0.1.0"
description = "Synthetic low-data longitudinal clinical modelling research harness"
requires-python = ">=3.11"
dependencies = [
  "numpy>=2.0",
  "pandas>=2.2",
  "scipy>=1.13",
  "scikit-learn>=1.5",
  "torch>=2.3",
  "PyYAML>=6.0",
  "matplotlib>=3.9",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2",
  "pytest-cov>=5.0",
  "ruff>=0.5",
]

[project.scripts]
afmc-phase0 = "afmc_fm.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 4: Implement YAML loading**

```python
# src/afmc_fm/config.py
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data
```

`src/afmc_fm/__init__.py` should expose only a version constant initially:

```python
__version__ = "0.1.0"
```

- [ ] **Step 5: Add repository privacy exclusions**

The `.gitignore` must contain at least:

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
.venv/
venv/

# Secrets and local configuration
.env
.env.*
!.env.example

# Real/private clinical data: never commit
/data/
/clinical_data/
/patient_data/
/private_data/
/private/

# Large or derived artifacts
*.parquet
*.feather
*.pkl
*.joblib
*.ckpt
*.pt
*.pth
/outputs/
/runs/
/wandb/

# Notebook state
.ipynb_checkpoints/
```

Do **not** globally ignore CSV files because committed synthetic fixtures and `examples/real_patient_template.csv` need to remain possible.

- [ ] **Step 6: Add a smoke simulator config**

```yaml
# configs/simulator/smoke.yaml
seed: 7
cohort_size: 24
n_sites: 2
latent_dim: 4
followup_days: 180.0
mean_event_interval_days: 14.0
observation_regime: mnar
```

- [ ] **Step 7: Add GitHub Actions CI**

```yaml
# .github/workflows/ci.yml
name: ci

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install --upgrade pip
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: pytest -q
```

- [ ] **Step 8: Run tests and lint**

Run:

```bash
pip install -e ".[dev]"
ruff check src tests
pytest tests/test_config.py -v
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore .github/workflows/ci.yml src/afmc_fm/__init__.py src/afmc_fm/config.py configs/simulator/smoke.yaml tests/test_config.py
git commit -m "chore: scaffold Phase 0 research package"
```

---

### Task 2: MEDS-like event schema and patient timeline validation

**Files:**
- Create: `src/afmc_fm/schema/__init__.py`
- Create: `src/afmc_fm/schema/events.py`
- Create: `examples/real_patient_template.csv`
- Test: `tests/schema/test_events.py`

**Interfaces:**
- Produces: `EventType(str, Enum)` with `OBSERVATION`, `INTERVENTION`, `ENCOUNTER`
- Produces: `ClinicalEvent`
- Produces: `PatientTimeline`
- Produces: `events_to_frame(events: Sequence[ClinicalEvent]) -> pandas.DataFrame`
- Produces: `frame_to_events(frame: pandas.DataFrame) -> list[ClinicalEvent]`

- [ ] **Step 1: Write schema tests**

```python
# tests/schema/test_events.py
from datetime import datetime, timezone

import pandas as pd
import pytest

from afmc_fm.schema.events import (
    ClinicalEvent,
    EventType,
    PatientTimeline,
    events_to_frame,
    frame_to_events,
)


def _event(day: int, code: str = "LAB_A") -> ClinicalEvent:
    return ClinicalEvent(
        patient_id="p1",
        start_time=datetime(2026, 1, 1 + day, tzinfo=timezone.utc),
        code=code,
        value=1.5,
        unit="arb",
        event_type=EventType.OBSERVATION,
        source="synthetic",
        metadata={},
    )


def test_patient_timeline_sorts_events_chronologically():
    timeline = PatientTimeline("p1", [_event(2), _event(0), _event(1)])
    assert [e.start_time.day for e in timeline.events] == [1, 2, 3]


def test_patient_timeline_rejects_mixed_patient_ids():
    bad = _event(0)
    bad = ClinicalEvent(**{**bad.__dict__, "patient_id": "p2"})
    with pytest.raises(ValueError, match="same patient"):
        PatientTimeline("p1", [_event(1), bad])


def test_event_frame_round_trip_preserves_core_fields():
    events = [_event(0), _event(1, "LAB_B")]
    frame = events_to_frame(events)
    recovered = frame_to_events(frame)
    assert [e.code for e in recovered] == ["LAB_A", "LAB_B"]
    assert all(e.patient_id == "p1" for e in recovered)
    assert isinstance(frame, pd.DataFrame)
```

- [ ] **Step 2: Verify failure**

Run:

```bash
pytest tests/schema/test_events.py -v
```

Expected: import failure for `afmc_fm.schema.events`.

- [ ] **Step 3: Implement schema objects**

Use dataclasses and timezone-aware datetimes:

```python
# src/afmc_fm/schema/events.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence

import pandas as pd


class EventType(str, Enum):
    OBSERVATION = "observation"
    INTERVENTION = "intervention"
    ENCOUNTER = "encounter"


@dataclass(frozen=True)
class ClinicalEvent:
    patient_id: str
    start_time: datetime
    code: str
    value: float | str | None
    unit: str | None
    event_type: EventType
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_time.tzinfo is None:
            raise ValueError("start_time must be timezone-aware")
        if not self.patient_id:
            raise ValueError("patient_id must be non-empty")
        if not self.code:
            raise ValueError("code must be non-empty")


@dataclass
class PatientTimeline:
    patient_id: str
    events: list[ClinicalEvent]

    def __post_init__(self) -> None:
        if any(event.patient_id != self.patient_id for event in self.events):
            raise ValueError("all events must belong to the same patient")
        self.events = sorted(self.events, key=lambda event: event.start_time)
```

Implement `events_to_frame` and `frame_to_events` using explicit columns and ISO/UTC timestamps. Serialize `metadata` as a JSON string so the representation survives CSV export.

- [ ] **Step 4: Add a synthetic-only real-record template**

`examples/real_patient_template.csv` must contain only demonstrative synthetic values:

```csv
patient_id,start_time,code,value,unit,event_type,source,metadata
example_patient,2026-01-01T09:00:00+00:00,ENCOUNTER,, ,encounter,example,"{}"
example_patient,2026-01-01T09:15:00+00:00,LAB_A,1.5,arb,observation,example,"{}"
example_patient,2026-01-21T10:00:00+00:00,THERAPY_A,1.0,dose,intervention,example,"{}"
```

The README later must explicitly state that this file is a schema example, not a clinical record.

- [ ] **Step 5: Run tests**

```bash
pytest tests/schema/test_events.py -v
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/afmc_fm/schema tests/schema examples/real_patient_template.csv
git commit -m "feat: add longitudinal clinical event schema"
```

---

### Task 3: Simulator configuration and latent physiological dynamics

**Files:**
- Create: `src/afmc_fm/simulator/__init__.py`
- Create: `src/afmc_fm/simulator/config.py`
- Create: `src/afmc_fm/simulator/dynamics.py`
- Test: `tests/simulator/test_dynamics.py`

**Interfaces:**
- Produces: `SimulatorConfig`
- Produces: `PatientParameters`
- Produces: `LatentTrajectory`
- Produces: `sample_patient_parameters(config, rng) -> PatientParameters`
- Produces: `simulate_latent_trajectory(config, params, rng) -> LatentTrajectory`
- Produces: `apply_intervention_jump(state, strength, params, rng) -> np.ndarray`

- [ ] **Step 1: Write deterministic and heterogeneity tests**

```python
# tests/simulator/test_dynamics.py
import numpy as np

from afmc_fm.simulator.config import SimulatorConfig
from afmc_fm.simulator.dynamics import sample_patient_parameters, simulate_latent_trajectory


def test_same_seed_produces_same_latent_trajectory():
    config = SimulatorConfig(cohort_size=4, latent_dim=4, followup_days=90.0)
    rng_a = np.random.default_rng(11)
    params_a = sample_patient_parameters(config, rng_a)
    traj_a = simulate_latent_trajectory(config, params_a, rng_a)

    rng_b = np.random.default_rng(11)
    params_b = sample_patient_parameters(config, rng_b)
    traj_b = simulate_latent_trajectory(config, params_b, rng_b)

    np.testing.assert_allclose(traj_a.times, traj_b.times)
    np.testing.assert_allclose(traj_a.states, traj_b.states)


def test_patient_parameters_are_heterogeneous_across_draws():
    config = SimulatorConfig(cohort_size=4, latent_dim=4)
    rng = np.random.default_rng(3)
    a = sample_patient_parameters(config, rng)
    b = sample_patient_parameters(config, rng)
    assert not np.allclose(a.baseline_state, b.baseline_state)


def test_latent_trajectory_is_finite_and_monotonic_in_time():
    config = SimulatorConfig(cohort_size=4, latent_dim=4, followup_days=120.0)
    rng = np.random.default_rng(5)
    params = sample_patient_parameters(config, rng)
    trajectory = simulate_latent_trajectory(config, params, rng)
    assert np.all(np.diff(trajectory.times) > 0)
    assert np.isfinite(trajectory.states).all()
    assert trajectory.states.shape[1] == 4
```

- [ ] **Step 2: Verify tests fail**

```bash
pytest tests/simulator/test_dynamics.py -v
```

- [ ] **Step 3: Implement simulator dataclasses**

`SimulatorConfig` should include explicit defaults:

```python
@dataclass(frozen=True)
class SimulatorConfig:
    cohort_size: int = 1000
    n_sites: int = 2
    latent_dim: int = 4
    followup_days: float = 365.0
    mean_event_interval_days: float = 14.0
    process_noise: float = 0.03
    measurement_noise: float = 0.10
    intervention_rate: float = 0.04
    observation_regime: str = "mnar"
```

Validate positive dimensions/times and allowed observation regimes `{"mcar", "mar", "mnar", "site_shift"}`.

`PatientParameters` should contain:

```python
baseline_state: np.ndarray
progression_scale: np.ndarray
response_scale: np.ndarray
noise_scale: float
site_id: int
```

`LatentTrajectory` should contain:

```python
times: np.ndarray       # shape [T], elapsed days
states: np.ndarray      # shape [T, latent_dim]
```

- [ ] **Step 4: Implement simulator-side nonlinear dynamics**

Use a stable stochastic update that is intentionally not identical to the learner's transition:

```python
def _drift(state: np.ndarray, progression: np.ndarray, t_days: float) -> np.ndarray:
    coupling = 0.08 * np.tanh(np.roll(state, 1) - state)
    seasonal = 0.01 * np.sin(t_days / 30.0)
    return -0.03 * progression * state + coupling + seasonal
```

Advance with irregular exponentially distributed intervals capped so the final state does not exceed `followup_days`:

```python
next_state = state + dt * _drift(state, params.progression_scale, t) / 30.0
next_state += np.sqrt(max(dt, 1e-6) / 30.0) * params.noise_scale * rng.normal(size=d)
```

- [ ] **Step 5: Implement intervention jump utility**

```python
def apply_intervention_jump(
    state: np.ndarray,
    strength: float,
    params: PatientParameters,
    rng: np.random.Generator,
) -> np.ndarray:
    direction = np.zeros_like(state)
    direction[0] = 1.0
    if state.size > 1:
        direction[1] = -0.35
    noise = 0.02 * rng.normal(size=state.shape)
    return state + strength * params.response_scale * direction + noise
```

This utility is simulator-side ground truth only; the learner must not import it.

- [ ] **Step 6: Run tests**

```bash
pytest tests/simulator/test_dynamics.py -v
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add src/afmc_fm/simulator/config.py src/afmc_fm/simulator/dynamics.py tests/simulator/test_dynamics.py
git commit -m "feat: add stochastic latent patient dynamics"
```

---

### Task 4: Observation mechanisms, clinical emissions, intervention events, and cohort generation

**Files:**
- Create: `src/afmc_fm/simulator/observation.py`
- Create: `src/afmc_fm/simulator/cohort.py`
- Create: `configs/simulator/full.yaml`
- Test: `tests/simulator/test_observation.py`
- Test: `tests/simulator/test_cohort.py`

**Interfaces:**
- Produces: `observation_probability(...) -> float`
- Produces: `emit_labs(state, rng, noise_scale) -> dict[str, float]`
- Produces: `SimulatedPatient`
- Produces: `SimulatedCohort`
- Produces: `simulate_patient(patient_id, config, rng) -> SimulatedPatient`
- Produces: `simulate_cohort(config, seed) -> SimulatedCohort`

- [ ] **Step 1: Write observation-regime tests**

```python
# tests/simulator/test_observation.py
import numpy as np

from afmc_fm.simulator.observation import observation_probability


def test_mcar_probability_does_not_depend_on_latent_state():
    p_low = observation_probability("mcar", np.array([-2.0, 0.0]), None, 0)
    p_high = observation_probability("mcar", np.array([2.0, 0.0]), None, 0)
    assert p_low == p_high


def test_mnar_probability_increases_with_latent_severity():
    p_low = observation_probability("mnar", np.array([-2.0, 0.0]), None, 0)
    p_high = observation_probability("mnar", np.array([2.0, 0.0]), None, 0)
    assert p_high > p_low


def test_site_shift_changes_measurement_probability():
    state = np.array([0.5, -0.2])
    p_a = observation_probability("site_shift", state, None, 0)
    p_b = observation_probability("site_shift", state, None, 1)
    assert p_a != p_b
```

- [ ] **Step 2: Write cohort invariants**

```python
# tests/simulator/test_cohort.py
from afmc_fm.schema.events import EventType
from afmc_fm.simulator.config import SimulatorConfig
from afmc_fm.simulator.cohort import simulate_cohort


def test_cohort_contains_unique_patients_and_sorted_events():
    config = SimulatorConfig(cohort_size=12, followup_days=120.0, latent_dim=4)
    cohort = simulate_cohort(config, seed=17)
    assert len(cohort.patients) == 12
    assert len({p.patient_id for p in cohort.patients}) == 12
    for patient in cohort.patients:
        times = [e.start_time for e in patient.timeline.events]
        assert times == sorted(times)


def test_cohort_contains_observations_and_interventions():
    config = SimulatorConfig(cohort_size=32, followup_days=180.0, intervention_rate=0.15)
    cohort = simulate_cohort(config, seed=19)
    types = [e.event_type for p in cohort.patients for e in p.timeline.events]
    assert EventType.OBSERVATION in types
    assert EventType.INTERVENTION in types
```

- [ ] **Step 3: Verify failure**

```bash
pytest tests/simulator/test_observation.py tests/simulator/test_cohort.py -v
```

- [ ] **Step 4: Implement emissions and observation probabilities**

Use three neutral synthetic lab channels so the simulator does not masquerade as validated thalassemia physiology:

```python
LAB_CODES = ("LAB_FAST", "LAB_SLOW", "LAB_BURDEN")
```

Suggested emissions:

```python
def emit_labs(state, rng, noise_scale):
    return {
        "LAB_FAST": 1.4 * state[0] + 0.25 * np.tanh(state[1]) + rng.normal(0, noise_scale),
        "LAB_SLOW": 0.6 * state[1] ** 2 + 0.3 * state[2 % len(state)] + rng.normal(0, noise_scale),
        "LAB_BURDEN": 0.5 * state.mean() + 0.2 * state[0] * state[-1] + rng.normal(0, noise_scale),
    }
```

Implement probabilities with a numerically stable sigmoid:

```python
mcar: sigmoid(-0.2)
mar: sigmoid(-0.6 + 0.8 * previous_observed_value)
mnar: sigmoid(-0.8 + 1.1 * latent_state[0])
site_shift: sigmoid(site_intercept[site_id] + 0.7 * latent_state[0])
```

Clamp probabilities to `[0.02, 0.98]`.

- [ ] **Step 5: Implement cohort orchestration**

`SimulatedPatient` must expose:

```python
patient_id: str
site_id: int
parameters: PatientParameters
latent: LatentTrajectory
timeline: PatientTimeline
```

`SimulatedCohort` must expose:

```python
patients: list[SimulatedPatient]
config: SimulatorConfig
seed: int
```

For each latent timepoint:

1. optionally generate an intervention according to a probability depending on state and `intervention_rate`;
2. if an intervention occurs, apply the simulator-side jump to subsequent state propagation;
3. create an `INTERVENTION` event;
4. emit candidate lab values;
5. sample each lab's observation mask from the configured observation regime;
6. create only those `OBSERVATION` events actually measured;
7. retain full latent truth inside `SimulatedPatient.latent`, not in the exported clinical event table.

- [ ] **Step 6: Add the full simulator config**

```yaml
# configs/simulator/full.yaml
seed: 42
cohort_size: 5000
n_sites: 3
latent_dim: 4
followup_days: 365.0
mean_event_interval_days: 14.0
process_noise: 0.03
measurement_noise: 0.10
intervention_rate: 0.06
observation_regime: site_shift
```

- [ ] **Step 7: Run simulator tests**

```bash
pytest tests/simulator -v
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add src/afmc_fm/simulator/observation.py src/afmc_fm/simulator/cohort.py configs/simulator/full.yaml tests/simulator
git commit -m "feat: generate observed longitudinal synthetic cohorts"
```

---

### Task 5: Fixed generic history encoder, sequence construction, and leakage-safe patient splits

**Files:**
- Create: `src/afmc_fm/data/__init__.py`
- Create: `src/afmc_fm/data/encoding.py`
- Create: `src/afmc_fm/data/sequences.py`
- Create: `src/afmc_fm/data/splits.py`
- Test: `tests/data/test_encoding.py`
- Test: `tests/data/test_sequences.py`
- Test: `tests/data/test_splits.py`

**Interfaces:**
- Produces: `HistoryEncoder` protocol
- Produces: `SummaryHistoryEncoder`
- Produces: `PatientSequence`
- Produces: `build_patient_sequence(patient, encoder) -> PatientSequence`
- Produces: `split_patient_ids(patient_ids, seed, train_fraction, val_fraction)`

- [ ] **Step 1: Write patient split tests first**

```python
# tests/data/test_splits.py
from afmc_fm.data.splits import split_patient_ids


def test_patient_split_has_no_overlap_and_is_deterministic():
    ids = [f"p{i}" for i in range(100)]
    a = split_patient_ids(ids, seed=8, train_fraction=0.7, val_fraction=0.1)
    b = split_patient_ids(ids, seed=8, train_fraction=0.7, val_fraction=0.1)
    assert a == b
    train, val, test = map(set, a)
    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == set(ids)
```

- [ ] **Step 2: Write encoder causality test**

The representation at time `t_i` must use only events at or before `t_i`.

```python
# tests/data/test_encoding.py
from datetime import datetime, timedelta, timezone

import numpy as np

from afmc_fm.data.encoding import SummaryHistoryEncoder
from afmc_fm.schema.events import ClinicalEvent, EventType, PatientTimeline


def test_encoder_does_not_use_future_events():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    base = [
        ClinicalEvent("p", origin, "LAB_FAST", 1.0, "arb", EventType.OBSERVATION, "synthetic"),
        ClinicalEvent("p", origin + timedelta(days=10), "LAB_FAST", 9.0, "arb", EventType.OBSERVATION, "synthetic"),
    ]
    encoder = SummaryHistoryEncoder(representation_dim=16, seed=4)
    first_only = PatientTimeline("p", [base[0]])
    full = PatientTimeline("p", base)
    a = encoder.encode(first_only, cutoff_time=origin)
    b = encoder.encode(full, cutoff_time=origin)
    np.testing.assert_allclose(a, b)
```

- [ ] **Step 3: Write sequence-shape tests**

```python
# tests/data/test_sequences.py
import numpy as np

from afmc_fm.data.encoding import SummaryHistoryEncoder
from afmc_fm.data.sequences import build_patient_sequence
from afmc_fm.simulator.config import SimulatorConfig
from afmc_fm.simulator.cohort import simulate_cohort


def test_patient_sequence_contains_aligned_time_and_mask_arrays():
    patient = simulate_cohort(SimulatorConfig(cohort_size=1, followup_days=120), seed=2).patients[0]
    encoder = SummaryHistoryEncoder(representation_dim=16, seed=9)
    sequence = build_patient_sequence(patient, encoder)
    t = len(sequence.times)
    assert sequence.representations.shape == (t, 16)
    assert sequence.values.shape[0] == t
    assert sequence.masks.shape == sequence.values.shape
    assert np.all(np.diff(sequence.times) >= 0)
```

- [ ] **Step 4: Verify failure**

```bash
pytest tests/data -v
```

- [ ] **Step 5: Implement `SummaryHistoryEncoder`**

The encoder is a controlled stand-in for a frozen general EHR representation. It is **not** presented as a foundation model.

Build a causal summary vector from history up to `cutoff_time` containing, for each known synthetic lab:

- last observed value;
- time since last observation;
- observation count;
- intervention count;
- elapsed time since first event;
- site indicator.

Map that deterministic summary through a fixed seeded random projection:

```python
projected = np.tanh(summary @ projection + bias)
```

The projection is initialized once from `seed` and never trained.

Expose:

```python
class HistoryEncoder(Protocol):
    representation_dim: int
    def encode(self, timeline: PatientTimeline, cutoff_time: datetime) -> np.ndarray: ...
```

- [ ] **Step 6: Implement `PatientSequence`**

Use a dataclass with:

```python
patient_id: str
site_id: int
times: np.ndarray                 # [T] elapsed days
representations: np.ndarray       # [T, H]
values: np.ndarray                # [T, V], zero where unobserved
masks: np.ndarray                 # [T, V]
event_features: np.ndarray        # [T, E], includes intervention/encounter indicators
target_next_values: np.ndarray    # [T, V]
target_next_masks: np.ndarray     # [T, V]
target_event_within_horizon: np.ndarray  # [T]
```

Group events by timestamp. The representation at each group must be generated causally from events no later than the group timestamp. Targets must look strictly forward.

- [ ] **Step 7: Implement leakage-safe patient splitting**

```python
def split_patient_ids(
    patient_ids: Sequence[str],
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
```

Validate fractions and shuffle unique patient IDs with NumPy RNG.

- [ ] **Step 8: Run data tests**

```bash
pytest tests/data -v
```

Expected: pass.

- [ ] **Step 9: Commit**

```bash
git add src/afmc_fm/data tests/data
git commit -m "feat: build causal patient sequence representations"
```

---

### Task 6: Strong low-data baselines

**Files:**
- Create: `src/afmc_fm/models/__init__.py`
- Create: `src/afmc_fm/models/baselines.py`
- Test: `tests/models/test_baselines.py`

**Interfaces:**
- Produces: `ProbeRegressor`
- Produces: `ProbeClassifier`
- Produces: `GradientBoostingRegressorBaseline`
- Produces: `GRUBaseline`
- Produces: `flatten_sequence_examples(sequences) -> tuple[np.ndarray, ...]`

- [ ] **Step 1: Write simple baseline fit/predict tests**

```python
# tests/models/test_baselines.py
import numpy as np

from afmc_fm.models.baselines import ProbeClassifier, ProbeRegressor


def test_probe_regressor_fits_small_signal():
    x = np.arange(40, dtype=float).reshape(20, 2)
    y = x[:, 0] * 0.5 - x[:, 1] * 0.1
    model = ProbeRegressor().fit(x, y)
    pred = model.predict(x)
    assert np.mean((pred - y) ** 2) < 1e-6


def test_probe_classifier_returns_probabilities():
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    model = ProbeClassifier().fit(x, y)
    prob = model.predict_proba(x)
    assert prob.shape == (4,)
    assert np.all((prob >= 0.0) & (prob <= 1.0))
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/models/test_baselines.py -v
```

- [ ] **Step 3: Implement static probes and gradient boosting**

Use scikit-learn:

```text
ProbeRegressor -> Ridge(alpha=1.0)
ProbeClassifier -> LogisticRegression(max_iter=1000, class_weight="balanced")
GradientBoostingRegressorBaseline -> HistGradientBoostingRegressor
```

Wrapper methods must expose only `fit` and `predict`/`predict_proba` so experiment code is model-family agnostic.

- [ ] **Step 4: Implement a minimal GRU temporal baseline**

`GRUBaseline` should consume padded patient sequences with input channels:

```text
representation
current observed values * mask
mask
event features
log1p(delta_t)
```

Use a single-layer GRU with configurable hidden size, default `32`, followed by:

- Gaussian-style continuous head returning mean and log-scale;
- binary event head returning logits.

This baseline intentionally has no explicit flow/jump decomposition.

- [ ] **Step 5: Add shape/gradient tests for GRU baseline**

Add a test using a two-patient synthetic tensor batch and assert output shapes and finite gradients after one backward pass.

- [ ] **Step 6: Run baseline tests**

```bash
pytest tests/models/test_baselines.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/afmc_fm/models/__init__.py src/afmc_fm/models/baselines.py tests/models/test_baselines.py
git commit -m "feat: add low-data baseline models"
```

---

### Task 7: Core flow-jump latent-state adapter

**Files:**
- Create: `src/afmc_fm/models/flow_jump.py`
- Create: `src/afmc_fm/models/losses.py`
- Test: `tests/models/test_flow_jump.py`
- Test: `tests/models/test_losses.py`

**Interfaces:**
- Produces: `FlowJumpAdapter`
- Produces: `FlowJumpOutput`
- Produces: `gaussian_nll(...)`
- Produces: `masked_gaussian_nll(...)`
- Produces: `event_bce(...)`

- [ ] **Step 1: Write structural flow-jump tests**

```python
# tests/models/test_flow_jump.py
import torch

from afmc_fm.models.flow_jump import FlowJumpAdapter


def _model():
    return FlowJumpAdapter(
        representation_dim=16,
        value_dim=3,
        event_dim=3,
        state_dim=24,
        site_dim=2,
    )


def test_elapsed_time_changes_pre_event_state():
    model = _model()
    z = torch.zeros(2, 24)
    short = model.flow(z, torch.tensor([1.0, 1.0]))
    long = model.flow(z, torch.tensor([30.0, 30.0]))
    assert not torch.allclose(short, long)


def test_event_jump_changes_state():
    model = _model()
    z = torch.zeros(2, 24)
    rep = torch.randn(2, 16)
    values = torch.zeros(2, 3)
    masks = torch.zeros(2, 3)
    event_a = torch.zeros(2, 3)
    event_b = event_a.clone()
    event_b[:, 1] = 1.0
    a = model.jump(z, rep, values, masks, event_a)
    b = model.jump(z, rep, values, masks, event_b)
    assert not torch.allclose(a, b)


def test_forward_returns_pre_and_post_event_states():
    model = _model()
    batch, steps = 3, 5
    out = model(
        representations=torch.randn(batch, steps, 16),
        values=torch.randn(batch, steps, 3),
        masks=torch.ones(batch, steps, 3),
        event_features=torch.zeros(batch, steps, 3),
        times=torch.arange(steps).float().repeat(batch, 1),
        site_context=torch.zeros(batch, steps, 2),
    )
    assert out.pre_event_states.shape == (batch, steps, 24)
    assert out.post_event_states.shape == (batch, steps, 24)
    assert out.value_mean.shape == (batch, steps, 3)
    assert out.event_logits.shape == (batch, steps)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/models/test_flow_jump.py -v
```

- [ ] **Step 3: Implement the time-conditioned flow**

The initial flow must remain low capacity:

```python
flow_input = torch.cat([z, torch.log1p(delta_t).unsqueeze(-1)], dim=-1)
gate = torch.sigmoid(self.flow_gate(flow_input))
candidate = torch.tanh(self.flow_candidate(flow_input))
z_minus = z + gate * candidate
```

Do not use a Neural ODE solver in the initial implementation.

- [ ] **Step 4: Implement the event-conditioned jump**

Use a `GRUCell` as the jump operator:

```python
jump_input = torch.cat(
    [representation, values * masks, masks, event_features], dim=-1
)
z_plus = self.jump_cell(jump_input, z_minus)
```

Observation and intervention semantics are represented in `event_features`; they are not collapsed into one untyped scalar.

- [ ] **Step 5: Implement full sequence forward pass**

For each time index `i`:

1. compute `delta_t = times[:, i] - times[:, i - 1]`, with zero at the first event;
2. compute `z_minus = flow(z_plus_previous, delta_t)`;
3. store `z_minus` as `pre_event_states[:, i]`;
4. compute `z_plus = jump(z_minus, representation_i, values_i, masks_i, event_features_i)`;
5. store `z_plus` as `post_event_states[:, i]`;
6. produce future-value and event-risk outputs from `z_plus`.

Return a `FlowJumpOutput` dataclass containing:

```python
pre_event_states: torch.Tensor
post_event_states: torch.Tensor
value_mean: torch.Tensor
value_log_scale: torch.Tensor
event_logits: torch.Tensor
observation_logits: torch.Tensor | None
```

- [ ] **Step 6: Write and implement masked probabilistic losses**

Test first:

```python
# tests/models/test_losses.py
import torch

from afmc_fm.models.losses import masked_gaussian_nll


def test_masked_gaussian_nll_ignores_unobserved_targets():
    mean = torch.tensor([[0.0, 100.0]])
    log_scale = torch.zeros_like(mean)
    target = torch.tensor([[0.0, -100.0]])
    mask = torch.tensor([[1.0, 0.0]])
    loss = masked_gaussian_nll(mean, log_scale, target, mask)
    assert loss.item() < 1.0
```

Implement Gaussian NLL with scale clamped to a safe range and normalize by the number of observed targets, not by total tensor size. Implement binary cross-entropy for the event target.

- [ ] **Step 7: Verify finite gradients**

Add a test performing one synthetic forward pass, summing continuous and event losses, calling `backward()`, and asserting all trainable gradients are finite.

- [ ] **Step 8: Run model tests**

```bash
pytest tests/models/test_flow_jump.py tests/models/test_losses.py -v
```

- [ ] **Step 9: Commit**

```bash
git add src/afmc_fm/models/flow_jump.py src/afmc_fm/models/losses.py tests/models/test_flow_jump.py tests/models/test_losses.py
git commit -m "feat: add flow-jump latent-state adapter"
```

---

### Task 8: Explicit observation-process head without target leakage

**Files:**
- Modify: `src/afmc_fm/models/flow_jump.py`
- Modify: `src/afmc_fm/models/losses.py`
- Modify: `tests/models/test_flow_jump.py`
- Modify: `tests/models/test_losses.py`

**Interfaces:**
- Extends: `FlowJumpAdapter(..., model_observation_process: bool = False)`
- Produces observation logits from **pre-event states** and site context only
- Produces: `observation_bce(logits, target_mask) -> torch.Tensor`

- [ ] **Step 1: Write the leakage-prevention test**

The current observation mask must not be an input to the current observation head.

```python
def test_observation_logits_are_computed_from_pre_event_state():
    model = FlowJumpAdapter(
        representation_dim=16,
        value_dim=3,
        event_dim=3,
        state_dim=24,
        site_dim=2,
        model_observation_process=True,
    )
    pre = torch.randn(4, 24)
    site = torch.zeros(4, 2)
    logits = model.predict_observation(pre, site)
    assert logits.shape == (4, 3)
```

Do not define any API that accepts the current target mask in `predict_observation`.

- [ ] **Step 2: Implement observation head**

```python
self.observation_head = nn.Sequential(
    nn.Linear(state_dim + site_dim, state_dim),
    nn.Tanh(),
    nn.Linear(state_dim, value_dim),
)
```

`predict_observation(pre_event_state, site_context)` returns logits for which variables will be observed at that event time.

- [ ] **Step 3: Add observation BCE loss**

```python
def observation_bce(logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target_mask.float())
```

The training objective used by experiment code will be:

```text
L_total = L_value + lambda_event * L_event + lambda_obs * L_observation
```

with `lambda_obs=0` for the core model and positive only for the observation-aware variant.

- [ ] **Step 4: Run tests**

```bash
pytest tests/models -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/afmc_fm/models/flow_jump.py src/afmc_fm/models/losses.py tests/models
git commit -m "feat: add leakage-safe observation-process modelling"
```

---

### Task 9: Forecasting, calibration, latent-state recovery, and shift metrics

**Files:**
- Create: `src/afmc_fm/metrics/__init__.py`
- Create: `src/afmc_fm/metrics/forecasting.py`
- Create: `src/afmc_fm/metrics/latent.py`
- Test: `tests/metrics/test_forecasting.py`
- Test: `tests/metrics/test_latent.py`

**Interfaces:**
- Produces: `regression_metrics(...) -> dict[str, float]`
- Produces: `binary_metrics(...) -> dict[str, float]`
- Produces: `interval_coverage(...) -> float`
- Produces: `aligned_latent_r2(z_true, z_hat) -> float`
- Produces: `procrustes_latent_error(z_true, z_hat) -> float`

- [ ] **Step 1: Write known-answer metric tests**

```python
# tests/metrics/test_forecasting.py
import numpy as np

from afmc_fm.metrics.forecasting import interval_coverage, regression_metrics


def test_perfect_regression_has_zero_error():
    y = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


def test_interval_coverage_known_case():
    y = np.array([0.0, 2.0, 5.0, 10.0])
    lower = np.array([-1.0, 1.0, 6.0, 8.0])
    upper = np.array([1.0, 3.0, 7.0, 9.0])
    assert interval_coverage(y, lower, upper) == 0.5
```

- [ ] **Step 2: Write latent alignment test**

```python
# tests/metrics/test_latent.py
import numpy as np

from afmc_fm.metrics.latent import aligned_latent_r2


def test_linear_transform_of_truth_is_recoverable_after_alignment():
    rng = np.random.default_rng(0)
    z_true = rng.normal(size=(200, 3))
    transform = np.array([[2.0, 0.5, 0.0], [0.0, 1.5, 0.3], [0.2, 0.0, 1.0]])
    z_hat = z_true @ transform
    assert aligned_latent_r2(z_true, z_hat) > 0.99
```

- [ ] **Step 3: Implement regression and probability metrics**

Use scikit-learn for MAE, RMSE, ROC-AUC when both classes are present, Brier score, and log loss. Return `NaN` for class-dependent metrics when a test subset contains only one class rather than crashing.

- [ ] **Step 4: Implement latent-state alignment**

Fit a linear map using least squares from `z_hat` to `z_true`, then compute explained variance/R² on the aligned state. Keep this metric clearly labeled as alignment-dependent; do not interpret raw latent axes as identifiable physiological factors.

- [ ] **Step 5: Run metrics tests**

```bash
pytest tests/metrics -v
```

- [ ] **Step 6: Commit**

```bash
git add src/afmc_fm/metrics tests/metrics
git commit -m "feat: add Phase 0 evaluation metrics"
```

---

### Task 10: Low-N experiment runner with patient-level sampling and model ablations

**Files:**
- Create: `src/afmc_fm/experiments/__init__.py`
- Create: `src/afmc_fm/experiments/runner.py`
- Create: `configs/experiments/low_n.yaml`
- Test: `tests/experiments/test_runner.py`

**Interfaces:**
- Produces: `ExperimentConfig`
- Produces: `sample_low_n_train_ids(train_pool, n, seed) -> list[str]`
- Produces: `run_low_n_benchmark(...) -> pandas.DataFrame`
- Produces one tidy result row per `(model, n_train, seed, split, metric)`

- [ ] **Step 1: Write low-N sampling test**

```python
# tests/experiments/test_runner.py
from afmc_fm.experiments.runner import sample_low_n_train_ids


def test_low_n_sampling_is_exact_and_reproducible():
    pool = [f"p{i}" for i in range(200)]
    a = sample_low_n_train_ids(pool, n=40, seed=3)
    b = sample_low_n_train_ids(pool, n=40, seed=3)
    assert a == b
    assert len(a) == 40
    assert len(set(a)) == 40
```

- [ ] **Step 2: Write the fixed-test-set leakage test**

Create a small synthetic cohort, obtain a patient-level split, sample low-N training IDs only from the training pool, and assert no sampled ID occurs in validation or test.

- [ ] **Step 3: Define experiment configuration**

```python
@dataclass(frozen=True)
class ExperimentConfig:
    train_sizes: tuple[int, ...] = (5, 10, 20, 40, 80, 100)
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    max_epochs: int = 100
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    lambda_event: float = 1.0
    lambda_obs: float = 0.2
```

- [ ] **Step 4: Implement training loop with early stopping on validation loss**

Training must:

1. create patient-level batches;
2. never tune on test data;
3. save only in-memory best model state during a run;
4. use the same low-N patient subset for competing models under the same seed;
5. seed NumPy and PyTorch explicitly;
6. report trainable parameter count for each neural model.

- [ ] **Step 5: Implement model registry**

Initial experiment names:

```text
probe_linear
gradient_boosting
gru_from_scratch
flow_jump
flow_jump_observation
```

The result table must include at least:

```text
model
n_train
seed
site_or_shift
metric
value
trainable_parameters
```

- [ ] **Step 6: Add explicit ablation flags**

The flow-jump runner must be able to evaluate:

```text
no_flow
no_jump
no_observation_head
no_probabilistic_scale
```

Implement these as constructor/config options rather than copied model classes.

- [ ] **Step 7: Add low-N experiment YAML**

```yaml
# configs/experiments/low_n.yaml
train_sizes: [5, 10, 20, 40, 80, 100]
seeds: [1, 2, 3, 4, 5]
max_epochs: 100
patience: 12
learning_rate: 0.001
weight_decay: 0.0001
batch_size: 16
lambda_event: 1.0
lambda_obs: 0.2
```

- [ ] **Step 8: Write a tiny end-to-end experiment test**

Use a cohort of 36 patients, a train size of 5, one seed, and at most 2 epochs. Assert that the returned DataFrame is non-empty, contains multiple model names, and has no duplicate `(model, n_train, seed, metric)` rows for the same evaluation slice.

- [ ] **Step 9: Run experiment tests**

```bash
pytest tests/experiments/test_runner.py -v
```

- [ ] **Step 10: Commit**

```bash
git add src/afmc_fm/experiments configs/experiments tests/experiments
git commit -m "feat: add leakage-safe low-N benchmark runner"
```

---

### Task 11: Observation-shift experiment and synthetic-world suite

**Files:**
- Modify: `src/afmc_fm/experiments/runner.py`
- Modify: `src/afmc_fm/simulator/cohort.py`
- Create: `configs/experiments/observation_shift.yaml`
- Modify: `tests/experiments/test_runner.py`

**Interfaces:**
- Produces: `simulate_world(world_name, config, seed)`
- Supports worlds: `smooth`, `jumps`, `informative_observation`, `site_shift`, `misspecified`
- Supports train-on-site/test-on-site evaluation for observation-policy shift

- [ ] **Step 1: Write a world-switch test**

Generate `smooth` and `misspecified` worlds under the same seed and assert that their latent/event-generating parameters differ while the public cohort interface remains identical.

- [ ] **Step 2: Implement named simulation worlds**

The worlds must vary simulator-side mechanisms, not learner-side configuration:

```text
smooth: low intervention rate, mcar/mar observation
jumps: higher intervention rate with immediate jump effects
informative_observation: mnar observation tied to latent state
site_shift: shared latent dynamics, site-dependent observation intercept/noise
misspecified: delayed intervention effect + hidden regime switch + stronger heterogeneity
```

The misspecified world should introduce at least one delayed jump effect and one unrecorded regime change so the learner's simple Markov flow-jump assumptions are imperfect.

- [ ] **Step 3: Implement observation-shift evaluation**

For `site_shift`, train on site 0 patients and evaluate on both site 0 and held-out site 1 patients. Compute metric degradation:

```text
delta_metric = metric(site_1) - metric(site_0)
```

For error metrics, positive degradation is worse; for discrimination metrics, negative degradation is worse. Store both raw metrics and degradation rather than collapsing signs.

- [ ] **Step 4: Test observation-aware comparison path**

The smoke test must run both `flow_jump` and `flow_jump_observation` on a tiny site-shift cohort and confirm both produce site-specific metric rows.

- [ ] **Step 5: Run tests**

```bash
pytest tests/experiments/test_runner.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/afmc_fm/simulator/cohort.py src/afmc_fm/experiments/runner.py configs/experiments/observation_shift.yaml tests/experiments/test_runner.py
git commit -m "feat: add synthetic-world and observation-shift benchmarks"
```

---

### Task 12: CLI, plots, README, and final verification

**Files:**
- Create: `src/afmc_fm/cli.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces CLI commands:
  - `afmc-phase0 simulate --config <yaml> --output <dir>`
  - `afmc-phase0 benchmark --sim-config <yaml> --exp-config <yaml> --output <dir>`
- Produces CSV result tables and PNG learning-curve figures under ignored output directories

- [ ] **Step 1: Write CLI smoke test**

```python
# tests/test_cli.py
from pathlib import Path

from afmc_fm.cli import main


def test_simulate_cli_writes_synthetic_outputs(tmp_path: Path):
    rc = main([
        "simulate",
        "--config", "configs/simulator/smoke.yaml",
        "--output", str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / "events.csv").exists()
    assert (tmp_path / "latent_truth.npz").exists()
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_cli.py -v
```

- [ ] **Step 3: Implement CLI with `argparse`**

Do not add a CLI framework dependency. The `simulate` command should export:

```text
events.csv
latent_truth.npz
simulation_manifest.json
```

The manifest must state `synthetic: true`, seed, simulator config, and generation timestamp.

The `benchmark` command should export:

```text
metrics.csv
learning_curves.png
run_manifest.json
```

- [ ] **Step 4: Add learning-curve plotting**

Use matplotlib and produce one figure with `n_train` on the x-axis and the selected primary metric on the y-axis, with one line per model and uncertainty across seeds. Do not hard-code publication claims into plot titles.

- [ ] **Step 5: Rewrite README around the research contract**

The README must include:

1. project purpose and non-claims;
2. install instructions;
3. synthetic simulator quick start;
4. benchmark quick start;
5. explanation of the event schema;
6. warning that `examples/real_patient_template.csv` is synthetic-only;
7. privacy rule forbidding real clinical data commits;
8. explanation of the flow-jump architecture;
9. low-N evaluation design;
10. roadmap: synthetic -> public EHR -> AFMC.

Do not describe the system as a clinically validated model or as an Indian healthcare foundation model at this stage.

- [ ] **Step 6: Run the complete verification suite**

Run:

```bash
ruff check src tests
pytest -q
python -m afmc_fm.cli simulate --config configs/simulator/smoke.yaml --output outputs/smoke
python -m afmc_fm.cli benchmark --sim-config configs/simulator/smoke.yaml --exp-config configs/experiments/low_n.yaml --output outputs/benchmark_smoke
```

Expected:

- Ruff exits 0.
- All tests pass.
- `outputs/smoke/events.csv`, `latent_truth.npz`, and manifest exist.
- Benchmark smoke output contains `metrics.csv`, `learning_curves.png`, and manifest.
- No file beneath ignored private-data paths is staged or tracked.

- [ ] **Step 7: Inspect tracked files for accidental sensitive-data paths**

Run:

```bash
git ls-files | grep -E '(^|/)(data|clinical_data|patient_data|private_data|private)/' && exit 1 || true
```

Expected: no output.

- [ ] **Step 8: Commit final Phase 0 harness**

```bash
git add README.md src/afmc_fm/cli.py tests/test_cli.py
git commit -m "docs: complete Phase 0 synthetic benchmark workflow"
```

---

## Post-Implementation Experiment Gate

After Tasks 1–12 are complete, do **not** immediately move to AFMC data. Run the following research gate first:

```text
Worlds:
- smooth
- jumps
- informative_observation
- site_shift
- misspecified

N_train:
- 5
- 10
- 20
- 40
- 80
- 100

Models:
- probe_linear
- gradient_boosting
- gru_from_scratch
- flow_jump
- flow_jump_observation

Ablations:
- no_flow
- no_jump
- no_observation_head
- no_probabilistic_scale
```

Advance to public real-EHR work only if the flow-jump model shows a reproducible low-N advantage over both simple probing and temporal-from-scratch baselines in at least the worlds where irregular dynamics or jumps should matter, without materially worse calibration, and remains competitive under the misspecified world.

Retain the observation-process head only if it improves robustness/calibration under informative observation or site shift beyond simple mask/time-since-observation features.

If those conditions fail, revise or kill the relevant architectural component before requesting EHRSHOT/MIMIC/eICU integration.
