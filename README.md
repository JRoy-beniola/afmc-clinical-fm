# AFMC Clinical FM — Phase 0 Synthetic Research Harness

This repository tests whether compact flow-jump adaptation improves low-data longitudinal forecasting and calibration under controlled synthetic dynamics and observation-policy shift. Phase 0 is a methodology-development benchmark, not a clinically validated model, healthcare foundation model, or causal treatment-effect system.

## Install

```bash
python -m venv .venv
. .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Generate a synthetic cohort

```bash
afmc-phase0 simulate --config configs/simulator/smoke.yaml --output outputs/smoke
```

This writes `events.csv`, `latent_truth.npz`, and `simulation_manifest.json`. Outputs are ignored by Git.

## Run the low-N benchmark

```bash
afmc-phase0 benchmark \
  --sim-config configs/simulator/smoke.yaml \
  --exp-config configs/experiments/low_n.yaml \
  --output outputs/benchmark_smoke
```

The command writes tidy `metrics.csv`, `learning_curves.png`, and `run_manifest.json` files.

## Research contract

The common event schema represents every timeline as ordered `ClinicalEvent` records with patient ID, timezone-aware timestamp, code, value, unit, event type, source, and metadata. Observation, intervention, and encounter semantics remain distinct. `examples/real_patient_template.csv` is synthetic-only despite its filename; it is an ingestion template, not a real record.

The simulator exposes latent physiological truth while separately generating interventions, nonlinear clinical emissions, and MCAR/MAR/MNAR/site-shift observation processes. Named worlds cover smooth dynamics, intervention jumps, informative observation, cross-site policy shift, and misspecification.

The proposed learner maps a fixed generic causal history representation into a small latent state. A low-capacity elapsed-time flow evolves that state between events, and an event-conditioned GRU jump updates it at observations or interventions. Probabilistic value and event heads support uncertainty-aware forecasting. The optional observation head predicts measurement masks only from pre-event state and site context to avoid target leakage.

Evaluation splits patients—not events—and restricts training to low-N cohorts while keeping validation and test patients independent. Comparisons include linear probing, gradient boosting, a GRU trained from scratch, the flow-jump model, observation-aware flow-jump modelling, and component ablations. Reporting covers errors, probabilistic scores, calibration, latent-state recovery, and observation-shift degradation.

## Privacy and non-claims

Never commit real clinical data, identifiers, derived patient exports, model checkpoints, or private environment files. The repository ignore policy blocks common private-data and output paths. Synthetic patients do not create independent clinical evidence, and intervention-conditioned predictions are not causal estimates.

## Roadmap

The evidence sequence is deliberately gated:

1. controlled synthetic worlds;
2. public or credentialed EHR validation (for example EHRSHOT, MIMIC-IV, or eICU);
3. only then, a largely frozen external-domain evaluation on scarce AFMC data.

Failure to beat simpler baselines reproducibly is a valid stop or redesign result.
