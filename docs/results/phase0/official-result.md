# AFMC Phase 0 — Official Scientific Result

## 1. Purpose

Phase 0 was designed as a synthetic methodological stress test for the AFMC programme before scarce clinical data are introduced.

The primary research question was:

> Can a compact continuous-time Flow-Jump adapter transform frozen longitudinal clinical representations into disease-specific latent states more sample-efficiently than conventional representation probing or scratch temporal learning?

The experiment was not intended to establish clinical validity, transportability, safety, utility, or effectiveness on real patients.

---

## 2. Experimental scope

The official benchmark covered:

- five synthetic worlds:
  - `smooth`
  - `jumps`
  - `informative_observation`
  - `site_shift`
  - `misspecified`
- six training-patient budgets:
  - N = 5
  - N = 10
  - N = 20
  - N = 40
  - N = 80
  - N = 100
- five matched cohort/subset/model-seed bundles;
- multiple baseline and Flow-Jump variants;
- point forecasting;
- probabilistic forecasting;
- event prediction;
- latent-state recovery;
- ablations;
- observation-process robustness.

The principal point-forecasting metric was test MAE.

The intended scarce-data regime was N <= 40.

---

## 3. Execution result

The official run completed successfully.

- 25 / 25 shards completed.
- 2,730 / 2,730 cells completed.
- 23,520 metric rows were persisted.
- 0 failed cells.
- 0 cancelled cells.
- 0 incomplete cells.
- 0 duplicate scientific keys.
- 0 NaN or infinite metric values.

The negative scientific result is therefore not attributable to incomplete execution.

---

## 4. Primary low-N result

The strongest intended claim was that Flow-Jump would demonstrate a meaningful sample-efficiency advantage in the extreme-low-resource regime.

That claim was not validated.

For N <= 40:

**Flow-Jump beat both the representation-linear probe and scratch GRU in 0 / 12 dynamics-world/N settings.**

This is the central Phase-0 result.

The architecture therefore did not demonstrate the intended extreme-low-N superiority.

---

## 5. Higher-N signal

Flow-Jump was not uniformly poor.

It achieved simultaneous mean-MAE superiority over both principal comparators in four dynamics-world/N settings:

| World | N | Flow-Jump MAE | Representation-linear MAE | Scratch GRU MAE |
|---|---:|---:|---:|---:|
| informative_observation | 100 | 0.447 | 0.668 | 0.478 |
| jumps | 80 | 1.048 | 1.062 | 1.064 |
| site_shift | 80 | 0.496 | 0.681 | 0.510 |
| site_shift | 100 | 0.449 | 0.703 | 0.487 |

All simultaneous wins occurred at N = 80 or N = 100.

This establishes a late learning-curve crossover rather than the intended early low-N crossover.

---

## 6. Learning-curve interpretation

The dominant learning-curve pattern was:

- Flow-Jump generally inferior to scratch GRU at N = 5–40;
- Flow-Jump becoming competitive or superior in selected structured or shifted worlds at N = 80–100.

This is consistent with a structured architecture whose inductive bias becomes exploitable only after sufficient disease-specific observations are available.

It is inconsistent with the intended claim that the structural prior should provide its strongest advantage when data are extremely scarce.

The relevant Phase-0.5 research question therefore became:

> Can the useful crossover be moved from approximately N = 80–100 toward N = 10–40?

---

## 7. Capacity as an alternative explanation

The official Flow-Jump implementation had:

- Flow-Jump: 6,967 trainable parameters
- scratch GRU: 4,455 trainable parameters

Flow-Jump therefore had approximately 1.56 times the trainable capacity of the scratch GRU.

Because the positive crossover appeared primarily at larger N, parameter count remained a plausible alternative explanation for at least part of the observed higher-N advantage.

Phase 0 therefore did **not** establish that the higher-N gains were specifically caused by the Flow-Jump inductive bias.

A capacity-matched recurrent control was required in the next iteration.

---

## 8. Continuous-flow ablation

Ablation effects were interpreted as:

`Delta MAE = MAE(ablated) - MAE(full)`

Positive values mean that removing the component worsened MAE and therefore support a useful contribution from that component.

The continuous-flow pathway showed modest positive contributions in several difficult worlds, including:

- jumps: approximately +0.0289
- misspecified: approximately +0.0523
- site_shift: approximately +0.0168

This indicated that the continuous-flow pathway was not entirely decorative.

However, these ablation effects did not establish robust extreme-low-N superiority.

---

## 9. Jump-pathway result

The explicit jump pathway produced an ambiguous result.

Removing the jump mechanism:

- worsened MAE in the informative-observation world by approximately +0.0761;
- slightly improved MAE in the dedicated jumps world by approximately -0.0155;
- slightly improved MAE in the smooth world by approximately -0.0177.

This meant that the explicit event-conditioned jump mechanism was not validated as the source of performance in the world specifically intended to require discrete jumps.

An important architectural confound was identified:

the frozen representation could still contain current-event information.

Therefore the no-jump ablation did not necessarily eliminate all current-event semantics.

Phase 0.5 required a stricter pre-event representation:

`h_(t-) = f(H_<t)`

so that current-event information could only enter through the explicit jump pathway.

---

## 10. Representation pathway

The frozen representation pathway contributed useful signal in several difficult worlds.

This supported retaining a representation-conditioned architecture rather than treating the learned latent dynamics as completely independent of the pretrained history representation.

However, Phase 0 did not establish that the specific Flow-Jump transformation was the optimal way to exploit those representations.

---

## 11. Probabilistic-scale warning

One of the strongest ablation findings concerned the probabilistic-scale pathway.

Removing probabilistic scale improved MAE in every synthetic world, with a particularly large improvement under misspecification.

The misspecified-world point-prediction effect was approximately:

`Delta MAE = -0.578`

where the negative value means the ablated model improved MAE.

This created an explicit optimization/calibration trade-off.

The scale pathway could only be justified if its likelihood and coverage benefits were large enough to compensate for its point-forecasting cost.

Phase 0 therefore did not justify retaining the probabilistic-scale mechanism unchanged.

---

## 12. Observation-process head

The observation-aware model failed to show reproducible robustness benefit in the dedicated site-shift benchmark.

Across site-shift metrics:

- MAE was slightly worse;
- RMSE was slightly worse;
- event Brier was slightly worse;
- event ROC-AUC was slightly worse;
- latent recovery was slightly worse;
- coverage and event log-loss improved only marginally.

The predefined observation-head retention criterion was not met.

Decision:

> Drop or redesign the current observation-process head. Do not retain it merely because informative observation processes are scientifically plausible.

This result concerns the implemented head, not the general importance of observation processes in clinical data.

---

## 13. Misspecification robustness

In the misspecified world, Flow-Jump lost to the representation-linear probe at every training size.

It remained competitive with scratch GRU in isolated settings, including approximately:

- N = 20: Flow-Jump 1.602 vs scratch GRU 1.618
- N = 80: Flow-Jump 1.471 vs scratch GRU 1.483

The appropriate conclusion is:

> Competitive in isolated settings, not robustly superior.

The representation-linear probe remained an important strong control under misspecification.

---

## 14. Gate interpretation

The literal preregistered Criterion A threshold was technically satisfied because Flow-Jump improved at least one low-N objective in at least one dynamics setting.

However, this was only a weak formal pass.

It must not be confused with validation of the stronger intended scientific claim.

The stronger claim was extreme-low-N superiority over both major comparators.

That claim failed:

**0 / 12 simultaneous wins at N <= 40.**

---

## 15. What Phase 0 established

Phase 0 supports the following claims:

1. The synthetic harness executed successfully and reproducibly.
2. Flow-Jump contains nontrivial predictive structure.
3. Continuous flow contributes modestly in several difficult worlds.
4. Flow-Jump can outperform both primary comparators in selected higher-N settings.
5. The current architecture's useful crossover occurs too late for the intended low-resource objective.
6. The current jump pathway is not mechanistically validated.
7. The current observation-process head does not provide the intended robustness benefit.
8. The probabilistic-scale pathway creates a substantial point-prediction penalty.
9. Capacity remains an unresolved alternative explanation for higher-N gains.
10. Seed-level paired analysis is required for stronger claims.

---

## 16. What Phase 0 did not establish

Phase 0 does **not** establish:

- extreme-low-N superiority;
- clinical utility;
- clinical validity;
- real-world transportability;
- safety;
- causal treatment effects;
- that Flow-Jump higher-N gains are caused specifically by its structural inductive bias;
- that the explicit jump mechanism is correctly recovering discrete clinical transitions;
- that the probabilistic-scale pathway is justified;
- that the observation-process head should be retained.

---

## 17. Official scientific verdict

**PHASE 0 COMPLETE — EXTREME-LOW-N HYPOTHESIS NOT VALIDATED.**

The implementation showed sufficient structured signal to justify continued mechanistic research, but not enough evidence to support the original low-resource superiority claim.

The correct decision was therefore:

> Preserve the official Phase-0 execution as an immutable methodological result and initiate a focused Phase-0.5 redesign.

---

## 18. Phase-0.5 agenda generated by Phase 0

Phase 0 required the next iteration to address:

### Capacity matching

Construct a recurrent comparator with approximately the same trainable parameter count as Flow-Jump.

### Jump-pathway isolation

Prevent current-event semantics from bypassing the explicit jump mechanism.

### Uncertainty redesign

Determine whether probabilistic uncertainty improves likelihood or calibration enough to justify its MAE cost.

### Observation-head redesign

Remove the current observation-process head from the default architecture unless a new mechanism can demonstrate reproducible benefit.

### Low-N crossover

Treat N = 5, 10, 20, and 40 as the primary scientific target.

### Seed-level inference

Use paired seed-bundle effects, uncertainty intervals, and consistency criteria rather than relying primarily on aggregate mean curves.

---

## 19. Evidence boundary

All results in this record are synthetic methodological evidence.

They do not constitute evidence of clinical effectiveness or safety.

The canonical machine-readable evidence is stored in `raw/` and `tables/`, with figures under `figures/` and cryptographic integrity records under `integrity/`.

The full official output is archived separately and protected by a complete SHA-256 manifest.
