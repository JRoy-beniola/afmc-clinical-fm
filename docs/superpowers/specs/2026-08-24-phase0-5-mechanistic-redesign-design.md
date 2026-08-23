# Phase-0.5 Mechanistic Redesign and Confirmatory Protocol

Date: 2026-08-24
Status: Approved design, pending implementation plan
Phase-0 scientific protocol anchor: `be5a66b2e45362f60c90844e4e25673fb7bb3e21`
Phase-0 official execution commit: `d6f105eee73fcb8e9cc5987d292b1bb98a687382`

## 1. Purpose

Phase-0 validated the synthetic benchmark harness and produced a complete official result set, but it did not validate the strongest intended scientific claim: that a structured flow-jump adapter provides a sample-efficiency advantage over conventional probing and generic temporal learning in the extreme low-data regime.

The official Phase-0 result is therefore treated as a diagnostic result, not as evidence to be optimized away post hoc. Phase-0.5 is a controlled, moderate redesign that preserves the core flow -> assimilation -> jump idea while allowing the specific mechanisms exposed as weak by Phase-0 to be redesigned and retested under predeclared positive controls and unseen confirmatory simulator seeds.

This design follows the selected redesign boundary **B**:

- preserve the core continuous/discontinuous latent-dynamics concept;
- permit redesign of the flow, jump, uncertainty, and observation-process mechanisms;
- initially keep assimilation unchanged;
- introduce strict capacity controls;
- freeze the architecture before confirmatory evaluation;
- reserve a substantially different model family, redesign boundary **C**, as a last resort only if Phase-0.5 fails its confirmatory gate.

No AFMC clinical data should be used to rescue, tune, or validate a Phase-0.5 architecture before this synthetic protocol is complete.

## 2. Phase-0 result motivating the redesign

The Phase-0 official run completed successfully with 25/25 shards, 2730/2730 cells, zero failed/cancelled/incomplete cells, 23,520 metric rows, no duplicate scientific metric keys, no NaN/infinite metric values, and complete manifest-level execution provenance.

The scientific result was mixed.

### 2.1 Primary low-N finding

Across the three targeted dynamics worlds (`informative_observation`, `jumps`, `site_shift`) and six train sizes, the original `flow_jump` model beat both `representation_linear` and `gru_from_scratch` on mean MAE in only 4/18 world/N settings:

- `informative_observation`, N=100;
- `jumps`, N=80;
- `site_shift`, N=80;
- `site_shift`, N=100.

For the intended extreme-low-N regime, N <= 40, the simultaneous win rate was 0/12.

This means the original hypothesis of extreme-low-N superiority was not supported by Phase-0.

### 2.2 Capacity ambiguity

Trainable parameter counts were:

| Model | Trainable parameters |
| --- | ---: |
| `engineered_linear` | 17 |
| `representation_linear` | 20 |
| `representation_mlp` | 673 |
| `gru_from_scratch` | 4,455 |
| `flow_jump` | 6,967 |
| `flow_jump_observation` | 7,642 |

The original flow-jump model was approximately 1.56x the size of the scratch GRU. Its later-N crossover therefore cannot yet be attributed specifically to the structured flow-jump inductive bias.

### 2.3 Mechanistic warnings

Phase-0 ablations exposed three important issues.

1. **Flow positive control was weak.** In the `smooth` world, removing flow changed MAE only slightly and did not yield a reproducible mechanistic separation.
2. **Jump semantics were not cleanly validated.** In the `jumps` world, `no_jump` slightly improved mean MAE, despite the world being designed to contain discrete discontinuities. The current jump is a broad event-conditioned GRU update, and the frozen history representation can include current-event information.
3. **Probabilistic scale training interfered with point prediction.** Removing the probabilistic scale substantially improved MAE in difficult worlds, indicating either multi-objective interference or an unnecessarily coupled uncertainty head.

### 2.4 Observation-head finding

The current observation-process auxiliary head did not reproducibly improve site-shift robustness or calibration. It therefore failed its Phase-0 retention criterion and is not part of the Phase-0.5 core model.

## 3. Phase-0.5 scientific objective

The Phase-0.5 objective is hierarchical:

```text
mechanistic validity
        |
        v
low-N sample-efficiency advantage
        |
        v
robustness under shift and misspecification
```

A model cannot count as scientifically successful merely because it has the best aggregate MAE. Each mechanism must first pass a world in which its inductive bias is expected to be useful, and the frozen resulting architecture must then demonstrate a low-N advantage on unseen simulator realizations.

The central target is to move the effective crossover from the Phase-0 regime near N=80-100 toward the intended regime N <= 40, with at least one direct matched-control win at N <= 20.

## 4. World roles

Each simulator world has a predeclared role. Worlds are not averaged into one grand score.

| World | Role | What it is allowed to establish |
| --- | --- | --- |
| `smooth` | Flow positive control | Whether explicit elapsed-time continuous evolution adds value |
| `jumps` | Jump positive control | Whether an explicit discontinuity operator adds value |
| `informative_observation` | Targeted dynamics world | Whether the frozen core remains useful under informative sampling; optional observation modelling is tested separately |
| `site_shift` | Robustness test | Whether performance/calibration degrade less under observation/site shift |
| `misspecified` | Stress test | Whether the selected inductive bias degrades gracefully when simulator assumptions do not favor it |

`site_shift` and `misspecified` cannot rescue a Stage-III core low-N failure.

## 5. Phase-0.5 core architecture boundary

The Phase-0.5 core remains:

```text
strict pre-event history representation
              |
              v
       continuous flow
              |
              v
         assimilation
              |
              v
       discrete jump
              |
              v
      forecast readout
```

Formally:

\[
h_{t^-} = f_{\mathrm{FM}}(H_{<t})
\]

\[
z_t^- = F_\theta(z_{t-1}^+, \Delta t, c_t)
\]

\[
\tilde z_t = A_\phi(z_t^-, h_{t^-}, x_t, m_t)
\]

\[
z_t^+ = J_\psi(\tilde z_t, e_t, q_t)
\]

\[
\hat y_t = R_\omega(z_t^+).
\]

The representation timing is deliberately changed from the Phase-0 implementation to strict pre-event history. Current-event semantics must not be able to bypass the explicit jump pathway through `h_t`.

### 5.1 Assimilation remains fixed initially

The existing assimilation pathway is held fixed during Stage I. Phase-0 did not isolate a clear assimilation-specific failure, and changing it alongside flow, jump, and uncertainty would destroy attribution.

Assimilation may only be redesigned in a later protocol revision after a documented Phase-0.5 result.

## 6. Flow candidates

Stage I compares three flow conditions.

### 6.1 Existing gated residual flow

Retain the existing time-aware gated residual operator as a historical candidate:

\[
z_t^- = z_{t-1}^+ + g_\theta(z_{t-1}^+, \log(1+\Delta t)) \odot
\tanh r_\theta(z_{t-1}^+, \log(1+\Delta t)).
\]

### 6.2 Time-scaled residual flow

Introduce an explicitly elapsed-time-scaled residual evolution:

\[
\delta_t = \frac{\Delta t}{\tau + \Delta t}
\]

\[
z_t^- = z_{t-1}^+ + \delta_t f_\theta(z_{t-1}^+, c_t).
\]

The key mechanistic constraint is:

\[
\Delta t \to 0 \Longrightarrow z_t^- \to z_{t-1}^+.
\]

A numerically stable alternative scaling may be used if specified before Stage-I results are inspected, but a full adaptive Neural ODE solver is out of scope for Phase-0.5.

### 6.3 No-flow control

\[
z_t^- = z_{t-1}^+.
\]

The `smooth` world is the positive-control environment for selecting whether either learned flow earns retention.

## 7. Jump candidates

Stage I compares three jump conditions using strict pre-event representation timing.

### 7.1 Existing GRU jump

Retain the Phase-0 event-conditioned GRU jump as a historical control.

### 7.2 Explicit residual jump

Introduce a local residual discontinuity:

\[
\Delta z_t = \tanh W_\Delta[\tilde z_t, e_t] + b_\Delta
\]

\[
g_t = \sigma(W_g[\tilde z_t, e_t] + b_g)
\]

\[
z_t^+ = \tilde z_t + q_t\, g_t \odot \Delta z_t.
\]

`q_t` is a simulator-semantic eligibility indicator:

\[
q_t \in \{0,1\}.
\]

Only event classes that are defined by the simulator as structurally jump-eligible may set `q_t=1`. Measurement-only or encounter-only events may not freely invoke the residual jump. The eligibility map is determined from simulator semantics and frozen before any Phase-0.5 performance result is examined.

### 7.3 No-jump control

\[
z_t^+ = \tilde z_t.
\]

The `jumps` world is the positive-control environment. If neither jump operator reproducibly outperforms no-jump, the explicit jump mechanism fails Stage I and must not be retained merely because one candidate has a slightly better mean.

## 8. Uncertainty candidates

Uncertainty modelling is tested only after the latent-dynamics mechanisms are selected.

### 8.1 Joint Gaussian head

Retain the current mean/scale joint likelihood formulation as a historical candidate.

### 8.2 Decoupled mean/scale head

Use a point-prediction mean head:

\[
\mu_t = F_\omega(z_t)
\]

with a point objective such as MAE or MSE, and train scale separately from detached residuals:

\[
r_t = \operatorname{stopgrad}(y_t - \mu_t),
\]

\[
\log\sigma_t = S_\psi(\operatorname{stopgrad}(z_t), \ldots),
\]

\[
\mathcal L_\sigma = \frac{1}{2}\frac{r_t^2}{\sigma_t^2} + \log\sigma_t.
\]

The defining requirement is:

\[
\nabla_{\theta,\omega}\mathcal L_\sigma = 0
\]

for the core latent/mean pathway. The uncertainty objective therefore cannot improve likelihood by moving the mean predictor.

### 8.3 Deterministic control

No learned scale head. This condition is allowed to win Stage I-C. Probabilistic modelling is not retained merely because uncertainty estimates are desirable in principle.

## 9. Observation-process policy

The Phase-0 BCE observation head is removed from the Phase-0.5 core and retained only as a historical negative control if needed.

Any new observation-aware model is a separate challenger and follows a two-step ladder.

### 9.1 Diagnostic observation-intensity model

\[
\lambda_t^{\mathrm{obs}} = g_\eta(\operatorname{stopgrad}(z_t^-), c_t).
\]

The auxiliary observation objective must not update the core latent dynamics:

\[
\nabla_\theta \mathcal L_{\mathrm{obs}} = 0.
\]

It first has to demonstrate that informative observation is predictable from latent state/context better than marginal-rate and context-only baselines.

### 9.2 Optional robustness correction

Only if the diagnostic model passes its positive control may a separate challenger use the estimated observation propensity/intensity for a controlled missingness or training-weight correction.

Predictability and correction efficacy must not be established in the same experiment. The observation mechanism earns retention only if it reproducibly improves robustness/calibration beyond the no-observation core.

## 10. Capacity-control suite

Phase-0.5 must distinguish structured inductive bias from generic parameter capacity.

The comparator suite is:

1. original `representation_linear`;
2. original `representation_mlp` (~673 parameters);
3. capacity-matched representation MLP;
4. original `gru_from_scratch` (~4,455 parameters);
5. capacity-matched GRU;
6. original Phase-0 `flow_jump` as a historical reference;
7. frozen Phase-0.5 candidate.

The capacity-matched GRU and representation MLP are sized only after the selected Phase-0.5 core parameter count is known. Their trainable parameter counts should be as close as reasonably possible to the candidate without changing their model-family semantics.

A Phase-0.5 low-N claim requires beating both capacity-matched controls. Beating only the original smaller GRU or the 673-parameter MLP is not sufficient.

## 11. Four-stage falsification protocol

Phase-0.5 is deliberately staged to avoid combinatorial architecture search.

```text
Stage I
Mechanism isolation on development seeds
   |  flow -> jump -> uncertainty -> timing audit
   v
Stage II
Freeze one core architecture
   |
   |  no architecture changes after this point
   v
Stage III
Unseen-seed confirmatory low-N trial
   |
   |  must pass core hypothesis gate
   v
Stage IV
Site-shift, misspecification, optional observation challenger
```

Stage-IV results cannot rescue Stage III.

## 12. Seed isolation

Architecture development and confirmatory evidence use disjoint matched seed bundles.

Let a bundle be:

\[
s=(s_{\mathrm{cohort}},s_{\mathrm{subset}},s_{\mathrm{model}}).
\]

Use:

- **5 development bundles** for Stages I-II;
- **10 fresh confirmatory bundles** for Stages III-IV.

The sets must satisfy:

\[
\mathcal S_{\mathrm{dev}} \cap \mathcal S_{\mathrm{confirm}} = \varnothing.
\]

The confirmatory seeds must not be evaluated during architecture development. Once Stage III begins, its results may not be used to alter Phase-0.5.

## 13. Stage I: mechanism isolation

Stage I is architecture development, not confirmatory hypothesis testing.

### 13.1 General Stage-I gate

For a mechanism-vs-control paired effect on a loss metric:

\[
\Delta = L_{\mathrm{control}} - L_{\mathrm{mechanism}}.
\]

Positive values favor the mechanism.

A candidate mechanism advances only when all are satisfied on the relevant positive-control analysis:

1. mean paired effect > 0;
2. at least 4/5 matched development seed bundles favor the mechanism;
3. provisional relative improvement >= 2%.

The 2% threshold is provisional until implementation-time analysis quantifies the natural Phase-0 paired seed variation. That sanity check must occur before Phase-0.5 mechanism results are examined. If 2% lies below the effective Phase-0 noise floor, the threshold must be raised and documented before Stage I begins; it may not be lowered after seeing Phase-0.5 results.

### 13.2 Stage I-A: flow isolation

World: `smooth`.

Compare:

- existing gated flow;
- time-scaled residual flow;
- no-flow.

Hold assimilation, jump, and forecast-head choice fixed.

Primary metric: MAE.

Secondary mechanistic metrics: RMSE and aligned latent R2.

If neither learned flow passes the positive-control gate over no-flow, the continuous-flow mechanism is classified as failed for this redesign cycle.

### 13.3 Stage I-B: jump isolation

World: `jumps`.

Use the selected flow from Stage I-A and strict `H_<t` representation timing.

Compare:

- historical GRU jump;
- residual jump;
- no-jump.

Primary metric: MAE.

Secondary mechanistic metric: aligned latent R2.

The residual jump may activate only for frozen simulator-defined jump-eligible events.

If neither jump candidate passes over no-jump, the explicit jump mechanism is classified as failed for this redesign cycle.

### 13.4 Stage I-C: uncertainty isolation

Use the selected latent dynamics. Compare:

- joint Gaussian;
- decoupled mean/scale;
- deterministic.

Development worlds: `smooth`, `jumps`, and `informative_observation`.

Metrics:

- MAE;
- RMSE;
- Gaussian NLL where applicable;
- 90% coverage calibration error.

Define:

\[
CE_{90}=|\mathrm{coverage}_{90}-0.90|.
\]

A probabilistic formulation advances only if it improves probabilistic quality while keeping relative point-error degradation within a provisional 2% MAE tolerance. The same pre-Stage-I noise-floor sanity check applies to this tolerance.

If neither probabilistic formulation earns retention, the deterministic forecast head becomes the frozen core and uncertainty estimation is deferred to a later calibration layer.

### 13.5 Stage I-D: representation-timing audit

Quantify the effect of changing:

\[
h_t=f(H_{\le t})
\]

to:

\[
h_{t^-}=f(H_{<t}).
\]

This is an audit, not a model-selection contest. Strict pre-event timing is mandatory for the Phase-0.5 scientific core regardless of the numerical direction. The result is reported to clarify how much current-event representation access may have influenced Phase-0.

## 14. Stage II: freeze one Phase-0.5 core

Selection is sequential, not a full Cartesian architecture search.

| Component | Candidate set | Selection source |
| --- | --- | --- |
| Flow | gated / time-scaled / none | Stage I-A |
| Jump | GRU / residual / none | Stage I-B |
| Representation timing | strict `H_<t` | mandatory |
| Uncertainty | joint / decoupled / deterministic | Stage I-C |
| Assimilation | existing implementation | fixed |
| Observation head | none in core | fixed |

When two candidates are practically indistinguishable under the predeclared gate, choose the simpler/lower-capacity candidate.

At the end of Stage II the architecture, parameterization rules, optimizer settings, loss weights, simulator configuration, train sizes, development/confirmatory seed lists, and comparator definitions are frozen in a versioned protocol artifact.

No architectural changes are permitted between this freeze and Stage III.

## 15. Stage III: confirmatory extreme-low-N trial

Stage III evaluates the frozen candidate on 10 unseen matched seed bundles.

### 15.1 Training sizes

Retain:

\[
N \in \{5,10,20,40,80,100\}.
\]

But distinguish:

- primary extreme-low-N regime: N in {5,10,20,40};
- secondary continuity/crossover regime: N in {80,100}.

N=80 and N=100 do not contribute to the primary endpoint.

### 15.2 Targeted worlds

Primary Stage-III worlds:

- `smooth`;
- `jumps`;
- `informative_observation`.

Each world is evaluated separately.

### 15.3 Primary endpoint: normalized low-N learning-curve area

For each world `w`, confirmatory seed bundle `s`, and model `m`, define:

\[
\operatorname{nAULC}_{w,s}(m)
=
\frac{
\int_{\log 5}^{\log 40}
\operatorname{MAE}_{w,s,m}(N)\,d\log N
}{
\log 40-\log 5
}.
\]

Compute the integral by trapezoidal integration over N={5,10,20,40} in log-N space.

The normalized quantity remains in MAE units and is the primary sample-efficiency endpoint.

### 15.4 Paired primary contrasts

For each capacity-matched comparator `c`:

\[
\Delta_{w,s}^{(c)}
=
\operatorname{nAULC}_{w,s}(c)
-
\operatorname{nAULC}_{w,s}(FJ_{0.5}).
\]

Positive values favor Phase-0.5.

The two primary comparator families are:

- capacity-matched GRU;
- capacity-matched representation MLP.

The original GRU, original representation MLP, representation-linear probe, and Phase-0 flow-jump are secondary/contextual controls.

### 15.5 World-level success

A targeted world passes only if, against **each** capacity-matched comparator:

1. mean paired nAULC effect is positive;
2. at least 8/10 confirmatory seed bundles favor Phase-0.5.

### 15.6 Headline Phase-0.5 success gate

The core low-N hypothesis passes only if all of the following hold:

1. the selected flow mechanism passed its Stage-I positive control;
2. the selected jump mechanism passed its Stage-I positive control;
3. at least 2/3 targeted Stage-III worlds pass the matched-GRU nAULC gate;
4. at least the same 2/3 targeted worlds pass the matched-MLP nAULC gate;
5. there exists at least one N <= 20 where Phase-0.5 has lower mean MAE than both matched controls and at least 8/10 paired seed bundles favor Phase-0.5 against each matched control at that N.

A later-N crossover at N=80 or N=100 cannot satisfy the headline low-resource claim.

## 16. Stage IV: robustness and stress tests

Stage IV is evaluated only after the Stage-III architecture is already frozen.

### 16.1 Site-shift degradation

For MAE:

\[
D_{\mathrm{site}}^{\mathrm{MAE}}
=
\mathrm{MAE}_{\mathrm{site1}}
-
\mathrm{MAE}_{\mathrm{site0}}.
\]

Also report normalized degradation:

\[
R_{\mathrm{site}}^{\mathrm{MAE}}
=
\frac{
\mathrm{MAE}_{\mathrm{site1}}-\mathrm{MAE}_{\mathrm{site0}}
}{
\mathrm{MAE}_{\mathrm{site0}}
}.
\]

Analogous degradation measures are reported for NLL and `CE_90` when the frozen core is probabilistic.

A site-shift robustness claim requires smaller mean degradation than both capacity-matched controls with at least 8/10 paired confirmatory bundles favoring Phase-0.5.

### 16.2 Misspecification

`misspecified` is a graceful-degradation stress test, not a positive-control world.

Let the better matched control be the one with smaller low-N nAULC. Phase-0.5 remains acceptable under misspecification when:

\[
\frac{
\operatorname{nAULC}_{FJ}
-
\min(\operatorname{nAULC}_{GRU},\operatorname{nAULC}_{MLP})
}{
\min(\operatorname{nAULC}_{GRU},\operatorname{nAULC}_{MLP})
}
\le 0.05.
\]

The 5% tolerance is predeclared in this design and is not changed after Phase-0.5 results are inspected.

### 16.3 Observation-aware challenger

The optional decoupled observation-intensity model is tested separately after the core has been frozen.

It first must demonstrate observation predictability on `informative_observation`. Only then may a propensity/intensity-aware correction be evaluated under `site_shift`.

The observation mechanism is retained only if it reproducibly improves robustness or calibration beyond the no-observation frozen core. It cannot rescue Stage III.

## 17. Statistical analysis protocol

### 17.1 Unit of analysis

The paired seed bundle is the inferential unit. Events, observations, timesteps, and records are not treated as independent replications.

For each primary contrast report:

- mean paired effect;
- median paired effect;
- standard deviation;
- win fraction;
- mean relative improvement.

For comparator `c`:

\[
RI_{w,s}^{(c)}
=
100\frac{
\operatorname{nAULC}(c)-\operatorname{nAULC}(FJ_{0.5})
}{
\operatorname{nAULC}(c)
}.
\]

### 17.2 Bootstrap intervals

Use a paired nonparametric bootstrap over the 10 confirmatory seed bundles with B=10,000 resamples. Report a 95% bootstrap interval for the mean paired effect.

The bootstrap interval is supporting evidence. It does not override the preregistered mechanistic/consistency gates, and a favorable interval cannot rescue a failed gate.

### 17.3 Exact sign evidence

Report the exact binomial sign-test probability for the observed number of positive paired effects under `P(delta>0)=0.5`.

The 8/10 requirement is a predeclared reproducibility criterion, not a claim of conventional statistical significance. In particular, an 8/10 one-sided sign result is approximately p=0.055 and must not be described as p<0.05 evidence.

### 17.4 Multiple comparisons

The confirmatory multiplicity family contains six primary contrasts:

- Phase-0.5 vs matched GRU in each of three targeted worlds;
- Phase-0.5 vs matched representation MLP in each of three targeted worlds.

If formal p-values are presented, apply Holm correction across these six contrasts.

Secondary comparators do not enter this primary correction family.

### 17.5 Calibration

For 90% interval coverage, higher coverage is not automatically better. Report:

\[
CE_{90}=|\mathrm{coverage}_{90}-0.90|.
\]

Probabilistic variants are judged jointly on MAE/RMSE, NLL, and `CE_90`.

### 17.6 Latent recovery

Aligned latent R2 remains a secondary mechanistic endpoint, not part of the primary predictive success gate.

A model may recover simulator latent state more faithfully without producing better forecasts, or vice versa. Both outcomes should be reported rather than collapsed into one score.

## 18. No post-confirmation tuning

Once any Stage-III result from the 10 confirmatory seed bundles has been inspected, Phase-0.5 is frozen permanently.

If Stage III fails, it is recorded as a failed confirmatory experiment. The same confirmatory seeds may not be used to redesign Phase-0.5 and then claimed as independent evidence.

Any subsequent redesign becomes Phase-0.6 or redesign boundary C and requires a new confirmatory seed set.

## 19. Expected outputs

Phase-0.5 should produce versioned machine-readable and human-readable evidence sufficient to reconstruct every decision.

At minimum:

```text
outputs/<phase0.5-run>/
  protocol_manifest.json
  development/
    mechanism_metrics.csv
    flow_gate.csv
    jump_gate.csv
    uncertainty_gate.csv
    representation_timing_audit.csv
  frozen_candidate.json
  confirmation/
    metrics.csv
    learning_curves.csv
    paired_naulc_effects.csv
    primary_gate_summary.csv
    bootstrap_intervals.csv
    sign_tests.csv
    capacity_audit.csv
  robustness/
    site_shift_metrics.csv
    misspecification_metrics.csv
    observation_challenger_metrics.csv   # when applicable
  figures/
    mechanism_controls.png
    low_n_learning_curves.png
    paired_naulc_effects.png
    site_shift_degradation.png
    calibration.png                       # when applicable
  run_manifest.json
```

The implementation may reuse the Phase-0 atomic-cell execution engine, resume semantics, deterministic aggregation, and manifest machinery, but Phase-0.5 protocol identity must be distinct from the frozen Phase-0 protocol anchor.

## 20. Reporting diagrams

The final technical report should include at least the following diagrams.

### 20.1 Architecture mechanism diagram

```text
H_<t --> representation h_t-
                    |
                    v
z_(t-1)+ --> FLOW --> z_t- --> ASSIMILATE(x_t,m_t,h_t-) --> z~_t
                                                          |
                                                          v
                                       q_t,e_t --> JUMP --> z_t+
                                                          |
                                                          v
                                                       forecast
```

### 20.2 Development/confirmation isolation

```text
5 DEVELOPMENT BUNDLES
       |
       v
flow positive control
       |
       v
jump positive control
       |
       v
uncertainty selection
       |
       v
ARCHITECTURE FREEZE
============================== no tuning across boundary
       |
       v
10 UNSEEN CONFIRMATORY BUNDLES
       |
       v
low-N nAULC primary test
       |
       +--> pass --> robustness / observation challenger
       |
       +--> fail --> record Phase-0.5 failure; new protocol required
```

### 20.3 Evidence hierarchy

```text
Mechanism works where it should
           AND
Low-N curve beats capacity-matched controls
           AND
Result reproduces on unseen seed bundles
           THEN
Robustness may strengthen the claim
```

## 21. Interpretation rules

A successful Phase-0.5 permits the claim that, in controlled synthetic environments, the selected structured continuous/discontinuous adapter shows a reproducible low-data advantage over capacity-matched generic temporal and representation baselines.

It does **not** permit claims of clinical efficacy, causal treatment-effect estimation, physiological truth, or external validity to AFMC patients.

A partial result must be described narrowly. Examples:

- flow positive control passes but jump fails -> evidence for continuous dynamics only;
- mechanisms pass but Stage III fails -> mechanistic validity without demonstrated low-N sample-efficiency advantage;
- Stage III passes but site-shift fails -> low-N advantage without robustness to observation/site shift;
- observation challenger fails -> retain masks/elapsed time only and do not force an observation head into the core.

## 22. Scope boundaries

Phase-0.5 does not introduce:

- a full adaptive Neural ODE solver;
- changes to the assimilation operator during Stage I;
- a new foundation-model encoder trained on the synthetic cohort;
- clinical AFMC data for architecture tuning;
- causal intervention claims;
- hyperparameter sweeps over the confirmatory seed set;
- a grand score averaging all simulator worlds;
- a mandatory observation-process head;
- rescue tuning after Stage III;
- redesign boundary C unless Phase-0.5 fails and a new protocol is explicitly approved.

## 23. Implementation invariants

The implementation plan must preserve these design-level invariants:

1. strict `H_<t` representation timing for the Phase-0.5 scientific core;
2. simulator-semantic jump eligibility frozen before Stage-I results;
3. sequential Stage-I component selection rather than Cartesian architecture search;
4. assimilation fixed during Stage I;
5. disjoint development and confirmatory seed bundles;
6. architecture freeze before Stage III;
7. capacity-matched GRU and representation-MLP controls;
8. nAULC over N={5,10,20,40} as the primary low-N endpoint;
9. 8/10 paired consistency against each matched comparator for world-level confirmation;
10. at least 2/3 targeted worlds plus a direct N<=20 matched-control win for the headline success claim;
11. Stage-IV results cannot rescue Stage III;
12. no post-confirmatory tuning within Phase-0.5.

## 24. Acceptance criteria for the implementation plan

The later implementation plan is acceptable only if it specifies how to:

1. add strict pre-event representation encoding without silently changing Phase-0 historical outputs;
2. add flow variants behind explicit Phase-0.5 configuration identifiers;
3. add residual jump semantics and frozen event eligibility tests;
4. add joint/decoupled/deterministic uncertainty modes and verify gradient isolation for the decoupled head;
5. retain the current assimilation implementation unchanged during Stage I;
6. construct parameter-matched GRU and MLP comparators deterministically;
7. encode development and confirmatory seed sets separately and prevent accidental cross-use;
8. implement Stage-I gate calculations and the pre-Stage-I Phase-0 noise-floor sanity check;
9. serialize the frozen Stage-II candidate before any confirmatory run;
10. compute normalized log-N AULC and paired confirmatory effects;
11. produce 10,000-resample paired bootstrap intervals and exact sign-test summaries;
12. compute site-shift absolute/relative degradation and misspecification tolerance;
13. keep optional observation-intensity modelling decoupled from the core;
14. preserve atomic/resumable execution and complete provenance;
15. generate the machine-readable tables and figures listed above;
16. include tests that make it impossible to tune or overwrite the frozen candidate after confirmatory execution begins.

## 25. Decision after Phase-0.5

The decision tree is predeclared:

```text
Stage-I mechanism gates fail
    -> record which mechanism failed
    -> do not manufacture a full core win

Stage-I passes, Stage-III fails
    -> Phase-0.5 hypothesis fails
    -> consider Phase-0.6 or redesign boundary C with new confirmatory seeds

Stage-III passes, Stage-IV weak
    -> retain narrow low-N claim only

Stage-III and Stage-IV pass
    -> synthetic methodological evidence supports progression toward public-EHR and later AFMC validation
```

The purpose of Phase-0.5 is not to guarantee a favorable result. Its purpose is to make the next result scientifically diagnostic: either the structured flow-jump idea begins to demonstrate the intended low-resource inductive advantage under clean controls, or the project obtains a principled reason to move to a substantially different architecture.