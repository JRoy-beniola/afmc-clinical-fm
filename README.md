# AFMC Clinical FM — Phase 0 Synthetic Research Harness

This repository tests whether compact flow-jump adaptation improves low-data longitudinal forecasting and calibration under controlled synthetic dynamics and observation-policy shift. Phase 0 is a methodology-development benchmark, not a clinically validated model, healthcare foundation model, or causal treatment-effect system.

## Install and runtime diagnostics

WSL/Linux is the canonical research runtime. Create and use a native Linux
virtual environment there with the project interpreter:

```bash
cd ~/afmc-clinical-fm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
which python
python -m afmc_fm.cli diagnostics --device auto --workers 1
```

Inside WSL, use `.venv/bin/python` and the native Linux console scripts. Do
not invoke `.venv/Scripts/python.exe` or `.venv/Scripts/afmc-phase0.exe`; those
are Windows virtual-environment entry points and can mix Windows Python with
the WSL runtime. Windows-native execution remains a supported fallback. For
example, from Windows PowerShell:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

To verify CUDA from the same project interpreter, run:

```bash
python - <<'PY'
import torch

print("torch.cuda.is_available()", torch.cuda.is_available())
print("torch.version.cuda", torch.version.cuda)
if torch.cuda.is_available():
    print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))
PY
```

For focused WSL checks, use the exact project interpreter commands:

```bash
TMPDIR=/tmp ./.venv/bin/pytest tests/test_readme.py tests/test_cli.py -v
./.venv/bin/ruff check src tests
git diff --check
```

`TMPDIR=/tmp` is a runtime guardrail for the test process in WSL, not a scientific workaround and not a change to benchmark data or metrics.

## Generate a synthetic cohort

```bash
afmc-phase0 simulate --config configs/simulator/smoke.yaml --output outputs/smoke
```

This writes `events.csv`, `latent_truth.npz`, and `simulation_manifest.json`. Outputs are ignored by Git.

## Run the low-N benchmark

```bash
afmc-phase0 benchmark \
  --sim-config configs/simulator/smoke.yaml \
  --exp-config configs/experiments/smoke.yaml \
  --output outputs/benchmark_smoke
```

The command runs the worlds, models, and ablations named in the experiment YAML. It writes tidy `metrics.csv`, `ablation_metrics.csv`, `gate_summary.csv`, `learning_curves.png`, and `run_manifest.json` files. `gate_summary.csv` is a compact primary-MAE comparison table, not an automatic scientific go/no-go verdict. Use `configs/experiments/low_n.yaml` with the full simulator configuration for the complete five-world matrix.

## Research contract

The common event schema represents every timeline as ordered `ClinicalEvent` records with patient ID, timezone-aware timestamp, code, value, unit, event type, source, and metadata. Observation, intervention, and encounter semantics remain distinct. `examples/real_patient_template.csv` is synthetic-only despite its filename; it is an ingestion template, not a real record.

The simulator exposes latent physiological truth while separately generating interventions, nonlinear clinical emissions, and MCAR/MAR/MNAR/site-shift observation processes. Named worlds cover smooth dynamics, intervention jumps, informative observation, cross-site policy shift, and misspecification.

The proposed learner maps a fixed generic causal history representation into a small latent state. A low-capacity elapsed-time flow evolves that state between events, and an event-conditioned GRU jump updates it at observations or interventions. Probabilistic value and event heads support uncertainty-aware forecasting. The optional observation head predicts measurement masks only from pre-event state. Neither the fixed history encoder nor the learner receives raw site identity.

Evaluation splits patients—not events. Fit and validation selection share the total labelled N-patient budget; the final patient test set remains untouched. Repetitions use separate cohort, subset, and model-initialization seeds, so each repeat generates an independent cohort. Training is intentionally full-batch for this small Phase 0 harness.

Executed comparisons include an engineered-history linear model, an engineered-history gradient-boosted tree, a genuinely representation-free temporal GRU, a representation-only linear probe, a representation-only MLP probe, flow-jump adaptation, and observation-aware flow-jump adaptation. Runnable ablations remove the representation, flow, jump, observation head, or probabilistic scale. Where applicable, rows report MAE/RMSE, Gaussian NLL, 90% interval coverage, horizon-risk classification with event ROC-AUC/Brier/log loss, aligned latent-state recovery, and complete-target observation-shift degradation.

## Phase 0.5 preregistered mechanistic protocol

Phase 0.5 is a new protocol layer on top of the historical Phase 0 harness. It does not rewrite the original Phase 0 benchmark or its results. Stage I uses five development bundles to select and audit the flow, jump, and uncertainty mechanisms before a candidate is frozen. Confirmation then uses ten unseen confirmatory bundles across the locked target worlds; those confirmatory bundles are not reused for mechanism selection.

The staged CLI is:

```text
afmc-phase0 phase05 calibrate
afmc-phase0 phase05 develop
afmc-phase0 phase05 freeze
afmc-phase0 phase05 confirm
afmc-phase0 phase05 robustness
afmc-phase0 phase05 report
```

`calibrate` binds the Phase 0.5 configuration and historical Phase 0 calibration evidence into the immutable protocol lock. `develop` runs the preregistered Stage-I sequence and persists `development/mechanism_metrics.csv` together with the flow, jump, uncertainty, and representation-timing gate artifacts. `freeze` writes the selected candidate and its capacity audit. `confirm` runs the frozen candidate and locked comparators on the unseen confirmatory bundles. `robustness` evaluates site shift and misspecification only after confirmation is finalized. `report` is read-only and derives deterministic scientific tables, figures, `protocol_manifest.json`, and `run_manifest.json` from persisted artifacts.

Ordinary CI and smoke runs do not produce official Phase 0.5 scientific results. CI uses reduced synthetic fixtures to verify code paths, invariants, resume behavior, immutability, reporting, and device compatibility. Official Stage-I development must be started explicitly only after the Phase 0.5 validation record declares it ready; the same non-claim boundary applies to confirmation and robustness.

## Privacy and non-claims

Never commit real clinical data, identifiers, derived patient exports, model checkpoints, or private environment files. The repository ignore policy blocks common private-data and output paths. Synthetic patients do not create independent clinical evidence, and intervention-conditioned predictions are not causal estimates.

## Roadmap

The evidence sequence is deliberately gated:

1. controlled synthetic worlds;
2. public or credentialed EHR validation (for example EHRSHOT, MIMIC-IV, or eICU);
3. only then, a largely frozen external-domain evaluation on scarce AFMC data.

Failure to beat simpler baselines reproducibly is a valid stop or redesign result.
