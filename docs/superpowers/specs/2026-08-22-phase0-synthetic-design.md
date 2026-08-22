# Phase 0 Synthetic Research Harness — Design Specification

## 1. Purpose

This repository will begin as a research harness for testing the viability of a low-resource longitudinal clinical modelling method before any scarce AFMC cohort is used for architecture development.

The immediate research question is:

> Can a compact disease-specific continuous-time flow-jump adapter extract more useful, better-calibrated latent patient dynamics from pretrained longitudinal clinical representations than conventional probing or fine-tuning in the extreme low-data regime, especially when the clinical observation process is informative or shifts across settings?

Phase 0 does **not** attempt to train a healthcare foundation model, establish clinical utility, infer treatment causality, or manufacture population evidence from one real person. It is a controlled methodology-development stage.

## 2. Scientific hypotheses

### Primary hypothesis

Given a pretrained longitudinal representation `h_t`, a low-capacity disease-state adapter with explicit elapsed-time dynamics and event-conditioned jumps will achieve better sample efficiency and calibration than frozen probing, generic nonlinear heads, and temporal models trained from scratch when only tens to hundreds of downstream patients are available.

Conceptually:

```text
history H_{<=t}
      |
      v
pretrained encoder f_FM
      |
      v
 generic representation h_t
      |
      v
 disease-state adapter A_phi
      |
      v
 latent state z_t
      |
      +---- flow Phi_theta(z_t, Delta t) ---->
      |
      +---- event jump J_psi(z_t, e_t) ------>
      |
      +---- probabilistic clinical heads ---->
```

### Secondary hypothesis

An explicit observation-process model

```text
p(m_t | z_t, c_t)
```

will improve robustness when the measurement process changes, compared with merely supplying missingness masks or elapsed-time features to the predictor.

This is a secondary hypothesis. Failure of observation-process modelling does not invalidate the core dynamical-adaptation hypothesis.

### Null hypothesis

A frozen pretrained clinical representation with a simple linear or nonlinear downstream head is sufficient, and the additional dynamical structure does not provide a reproducible low-data benefit.

The project must be willing to accept this outcome.

## 3. Scope of Phase 0

Phase 0 has four isolated components:

1. a MEDS-like longitudinal event schema shared by synthetic and later real data;
2. a transparent ground-truth patient simulator;
3. baseline and proposed low-capacity temporal models;
4. a reproducible low-N experimental harness.

No real patient data will be committed to GitHub. The repository may contain a schema/template illustrating how a real longitudinal record would be represented, but actual personal or clinical records must remain outside version control.

## 4. Event schema

All timelines should be representable as an ordered event stream. The minimum event record is:

```text
patient_id
start_time
code
value
unit
event_type
source
metadata
```

`event_type` must distinguish at least:

- `observation`: laboratory value, vital sign, imaging-derived measurement, diagnosis observation;
- `intervention`: transfusion, medication start/stop/change, procedure;
- `encounter`: visit/admission/discharge or other care-context event.

The schema should remain compatible with a future MEDS-style representation rather than using a thalassemia-specific bespoke format.

### Real-record template

A template file may be included under `examples/`, but it must contain only synthetic/example rows. Its role is to validate timestamp semantics, units, event ordering, sparse observations, and ingestion logic.

## 5. Synthetic patient simulator

The simulator must expose the latent ground truth so that the learning algorithm can be tested against known dynamics.

### 5.1 Latent physiological state

Each synthetic patient has a latent state:

```text
z(t) in R^d
```

with a small configurable dimension, initially expected to be between 3 and 8.

Between recorded events, the state evolves according to a simulator-side process:

```text
z(t + Delta) = Phi_true(z(t), Delta, theta_i) + process_noise
```

where `theta_i` introduces patient heterogeneity.

The simulator's transition function must not be architecturally identical to the learner's transition function. The synthetic benchmark should not reward the model simply for matching the simulator's equation class.

### 5.2 Patient heterogeneity

Patient-specific parameters are sampled from population distributions:

```text
theta_i ~ p(theta)
```

These may govern baseline state, progression rate, response magnitude, noise level, and observation intensity.

### 5.3 Interventions as jumps

Interventions can produce discrete changes:

```text
z(t+) = J_true(z(t-), a_t, theta_i) + intervention_noise
```

Intervention effects may be immediate, delayed, nonlinear, or patient-specific depending on the simulation regime.

The learner may condition on interventions, but Phase 0 must not interpret such conditioning as causal effect estimation.

### 5.4 Clinical emissions

Observed clinical values arise from noisy functions of latent state:

```text
y_j(t) ~ p_j(y | z(t))
```

Emission functions should include nonlinear relationships and measurement noise. At least one continuous target and one event/time-to-event target should be generated.

### 5.5 Observation process

For each variable `j`, an observation indicator is generated:

```text
m_j(t) ~ Bernoulli(pi_j(t))
```

with configurable mechanisms.

Required regimes:

- **MCAR-like:** observation probability independent of latent state and history;
- **MAR-like:** observation probability depends on observed history/context;
- **informative/MNAR-like:** observation probability depends partly on latent state;
- **site shift:** multiple sites share similar underlying physiology but have different measurement policies.

The simulator should preserve the distinction between:

```text
latent physiology
clinical measurement process
clinical intervention process
```

## 6. Simulation worlds

A single synthetic world is insufficient. Phase 0 must include multiple worlds that vary the assumptions under which the proposed model should succeed or fail.

### World A — smooth stochastic physiology

Continuous, noisy latent evolution with irregular observations.

### World B — intervention-driven jumps

The same base physiology plus discrete intervention effects.

### World C — informative observation

Measurement probability depends partly on latent severity.

### World D — site observation shift

Underlying state dynamics are shared while observation frequency, variable choice, and measurement noise differ by site.

### World E — model misspecification

Introduce violations such as hidden regime changes, delayed intervention effects, nonlinear emissions, stronger patient heterogeneity, or latent confounding.

This world exists to determine whether the proposed architecture is brittle when its inductive assumptions are imperfect.

## 7. Learner architecture

The first learner should be intentionally simple. Neural ODEs are not the starting point.

### 7.1 Generic representation interface

The model receives a generic representation:

```text
h_i = f_FM(H_{<=t_i})
```

During pure synthetic testing, `f_FM` can first be represented by a fixed synthetic encoder or controlled history representation. The interface must later allow CLMBR/EHRSHOT or another pretrained longitudinal EHR encoder to be inserted without changing the downstream model contract.

### 7.2 Disease-state adapter

A compact adapter maps generic representation plus local observations into a low-dimensional disease state:

```text
z_i = A_phi(h_i, x_i, m_i)
```

The initial state dimension should remain small, approximately 16–64 for the learned representation, with parameter count explicitly tracked.

### 7.3 Continuous-time flow

Between events:

```text
z_{i+1}^- = Phi_theta(z_i^+, Delta t_i)
```

The first implementation should use a low-capacity elapsed-time-conditioned transition, such as a gated residual transition or simple state-space-like transition.

Later ablations may compare:

- no continuous transition;
- elapsed-time embedding only;
- decay transition;
- Delta-conditioned MLP/residual transition;
- Neural ODE;
- Neural CDE or structured continuous-time state-space variants.

The conceptual contribution is the flow-jump adaptation structure, not any one numerical solver.

### 7.4 Event-conditioned jump

When an event occurs:

```text
z_{i+1}^+ = J_psi(z_{i+1}^-, e_{i+1})
```

Observation events and intervention events must remain semantically distinct.

### 7.5 Output heads

Initial output heads should support:

- continuous-value forecasting using probabilistic outputs, e.g. predicted location and scale;
- recurrent or time-to-event prediction using a discrete-time hazard or another stable survival objective;
- optional latent-state decoding for synthetic evaluation only.

### 7.6 Observation-process model

The stronger observation-aware model adds:

```text
p_eta(m_i | z_i, c_i)
```

and optimizes a joint objective such as:

```text
L = L_clinical + lambda_obs * L_observation + lambda_reg * L_regularization
```

The exact weight `lambda_obs` is a tunable hyperparameter and must not be selected using final test data.

## 8. Baselines

Phase 0 must compare at least the following model classes where applicable:

1. engineered-history linear/logistic model;
2. gradient-boosted tree baseline;
3. conventional temporal model trained from scratch;
4. missingness-aware temporal baseline;
5. fixed generic representation + linear probe;
6. fixed generic representation + nonlinear MLP head;
7. generic representation + flow-jump adapter;
8. generic representation + flow-jump adapter + explicit observation likelihood.

When a real pretrained EHR backbone is introduced, ordinary parameter-efficient adaptation should also be included where technically feasible.

The key decomposition is:

```text
representation alone
vs
dynamics alone
vs
representation + structured dynamics
```

## 9. Low-data protocol

The synthetic benchmark should generate a large held-out evaluation population while deliberately restricting the training cohort.

Initial training sizes:

```text
N_train in {5, 10, 20, 40, 80, 100}
```

Each value must be repeated across multiple seeds and independently sampled patient cohorts.

Evaluation units are patients, not events. No event from a held-out patient may appear in the training set.

## 10. Metrics

### Clinical forecasting metrics

For continuous outcomes:

- MAE/RMSE where useful;
- negative log-likelihood;
- CRPS where implemented;
- prediction-interval coverage;
- calibration diagnostics.

For event/time-to-event outcomes:

- concordance or time-dependent discrimination as appropriate;
- Brier score;
- calibration of event probability;
- event-time error where simulation permits exact evaluation.

### Sample efficiency

Primary reporting must include performance-versus-number-of-training-patients curves.

A principal question is whether the proposed method reaches a given performance/calibration threshold using fewer target-domain patients than competing adaptation strategies.

### Latent-state recovery

Because synthetic latent truth is known, evaluate whether learned states recover useful structure in `z_true(t)` after accounting for representational non-identifiability.

Candidate metrics include:

- linear-probe decoding of latent dimensions;
- Procrustes-aligned error;
- CCA-style similarity;
- state-dependent downstream decoding.

No raw coordinate-wise equality between `z_true` and `z_hat` should be expected.

### Observation-shift robustness

Measure degradation in discrimination, probabilistic score, and calibration when the test measurement policy differs from training.

## 11. Required ablations

The model must be evaluated with the following components removed or simplified:

```text
- generic pretrained/history representation
- continuous-time transition
- event jump operator
- observation likelihood
- probabilistic uncertainty head
```

Transition-function complexity must also be ablated so that Neural ODEs or other advanced machinery are retained only if they provide reproducible value over simpler alternatives.

## 12. Phase 0A — one-real-record pipeline check

One real longitudinal record may be used locally to verify:

- event-schema adequacy;
- timestamp ordering;
- unit handling;
- irregular spacing;
- missingness representation;
- ingestion robustness;
- end-to-end model interface compatibility.

It must **not** be used to estimate population parameters or create pseudo-independent patients by perturbation.

No real record, identifier, clinical value, or derived export is committed to the repository.

## 13. Phase 0B — controlled synthetic benchmark

The simulator generates multiple worlds and a large independent test universe. The training cohort is restricted to low-N subsets.

The purpose is to answer:

- can the model recover known dynamics;
- does the flow-jump structure help under irregular time;
- does explicit observation modelling help under informative missingness and observation shift;
- does the advantage survive model misspecification;
- how quickly does performance saturate with patient count.

## 14. Phase 0C — transition to real public EHR data

Synthetic success is necessary but not sufficient.

If the method survives Phase 0B, the next stage should insert a real pretrained longitudinal EHR encoder and validate on public/credentialed clinical datasets such as EHRSHOT, MIMIC-IV, and/or eICU.

Expected roles:

```text
EHRSHOT  -> few-shot representation/adaptation benchmark
MIMIC-IV -> irregular longitudinal value/event forecasting
eICU     -> cross-site observation-policy shift
```

Only after public-data validation should the architecture be frozen as much as possible and applied to the AFMC cohort.

## 15. AFMC continuity

The synthetic harness is not a throwaway simulator. Its interfaces should intentionally match the later AFMC pipeline.

The same conceptual event interface should support future targets such as:

- time to next transfusion;
- future/pre-transfusion haemoglobin;
- ferritin trajectory if longitudinal density is adequate.

The AFMC cohort will be treated as a low-resource external domain, not as the dataset used to invent every architectural choice.

## 16. Privacy and repository policy

The repository must ignore and exclude at minimum:

```text
data/
clinical_data/
patient_data/
private_data/
*.parquet
*.feather
*.pkl
*.joblib
*.ckpt
*.pt
*.pth
.env
.env.*
outputs/
runs/
wandb/
```

A later implementation plan should define an allow-list approach for synthetic example fixtures so that small committed synthetic test data remain possible without creating a path for accidental patient-data commits.

## 17. Non-claims

Phase 0 must not claim:

- that a foundation model has been trained on ~100 patients;
- that synthetic patients create additional independent clinical evidence;
- that one real patient is representative of a population;
- clinical efficacy or safety;
- causal treatment effects from observational intervention-conditioned transitions;
- generalization to Indian healthcare systems;
- superiority of Neural ODEs before ablation against simpler transitions.

The permitted claim, if supported, is methodological:

> Under controlled low-data longitudinal settings, structured flow-jump adaptation of generic patient representations improves sample efficiency and/or robustness relative to simpler adaptation strategies.

Real clinical validity requires subsequent evaluation on independent real datasets.

## 18. Go / no-go criteria

### Proceed to public real-EHR experiments only if

- the flow-jump adapter reproducibly improves at least one primary low-N objective over both simple probing and temporal-from-scratch baselines in worlds where irregular dynamics matter;
- the gain is not explained solely by parameter count;
- probabilistic calibration is not materially worse;
- the method remains competitive under at least one misspecified simulation world.

### Retain the explicit observation model only if

- it improves robustness or calibration under informative observation or observation-policy shift;
- the improvement is reproducible across seeds/worlds;
- it provides benefit beyond simply supplying missingness masks and elapsed-time features.

### Retain Neural ODE/CDE machinery only if

- it materially and reproducibly outperforms simpler Delta-conditioned transitions after accounting for compute and complexity.

### Stop or redesign if

- fixed generic representations with simple heads perform equivalently across low-N regimes;
- the proposed dynamics help only when the simulator exactly matches learner assumptions;
- gains disappear under patient heterogeneity or modest model misspecification;
- observation modelling does not survive comparison to simpler missingness-aware baselines.

## 19. Intended repository structure

The implementation plan should target approximately:

```text
afmc-clinical-fm/
├── README.md
├── pyproject.toml
├── docs/
│   └── superpowers/
│       ├── specs/
│       └── plans/
├── src/
│   └── afmc_fm/
│       ├── schema/
│       ├── simulator/
│       ├── models/
│       ├── experiments/
│       └── metrics/
├── configs/
│   ├── simulator/
│   └── experiments/
├── tests/
├── notebooks/
├── examples/
└── scripts/
```

Modules must remain small and independently testable. Simulator truth generation, learner models, evaluation metrics, and experiment orchestration should not be coupled through notebook-only code.

## 20. Phase 0 success definition

Phase 0 is successful when the repository can reproducibly:

1. generate a synthetic longitudinal cohort with known latent dynamics and configurable observation policies;
2. export it through the common event schema;
3. train simple baselines and the flow-jump model on patient cohorts of size 5–100;
4. evaluate forecasting, uncertainty, latent-state recovery, and observation-shift robustness on independent patients;
5. produce low-N learning curves and ablation tables;
6. determine, before using scarce AFMC data for architecture design, whether the proposed structural hypothesis is worth carrying into public real-EHR experiments.
