Phase-0 Synthetic Validation
Results and Decision Record

Low-resource dynamical adaptation of longitudinal EHR foundation representations

Official execution

Protocol anchor: be5a66b2e45362f60c90844e4e25673fb7bb3e21

Execution SHA: d6f105eee73fcb8e9cc5987d292b1bb98a687382

Runtime: Native WSL2 · CUDA 13.0 · NVIDIA GeForce RTX 4060 Laptop GPU · 2 workers

Execution time: 03 h 40 m 02.49 s

Benchmark scope: 25/25 shards · 2,730/2,730 cells · 23,520 metric rows

Integrity: 0 failed · 0 cancelled · 0 incomplete · 0 duplicate keys · 0 NaN/Inf

Date: 23 August 2026

| Decision in one sentence<br>The structured Flow-Jump adapter shows real signal in higher-N and shift regimes, but the current implementation does not validate the intended extreme-low-N superiority claim; Phase-0 therefore supports a focused Phase-0.5 redesign rather than immediate clinical deployment. |
| --- |

<!-- blank -->




# 1. Executive Summary

The official Phase-0 synthetic benchmark completed successfully and passed execution, persistence, provenance, and metric-integrity audits. The result is scientifically mixed: Flow-Jump is capable of outperforming both a frozen-representation linear probe and a scratch GRU, but those simultaneous wins occur only in 4 of 18 dynamics-world/N settings and only at N = 80 or 100.

| Central result<br>For the genuinely scarce-data regime N ≤ 40, Flow-Jump beats both primary comparators in 0 of 12 dynamics-world settings. This directly weakens the strongest version of the original sample-efficiency hypothesis. |
| --- |

<!-- blank -->

At the same time, the model often outperforms the representation-linear probe, particularly under site shift, and begins to overtake the GRU at larger training budgets. The synthetic harness therefore identifies a potentially useful dynamical inductive bias whose crossover currently occurs too late for the intended extreme-low-resource setting.

The ablations provide actionable diagnosis: the continuous-flow pathway is modestly useful in several difficult worlds; the explicit jump pathway is useful in informative-observation settings but is not clearly useful in the dedicated jumps world; the observation-process head provides negligible or negative robustness benefit; and removing the probabilistic scale pathway improves MAE substantially, raising an optimization/calibration trade-off that must be examined before retaining that component.

| Gate | Criterion | Assessment | Evidence |
| --- | --- | --- | --- |
| A | Improve primary low-N objective | PASS, weak | 4/18 dynamics-world/N mean-MAE wins; 0/12 for N ≤ 40 |
| B | Gain not explained by parameter count | UNRESOLVED | 6,967 Flow-Jump parameters vs 4,455 GRU parameters |
| C | Calibration not materially worse | PENDING | Requires complete NLL and 90% coverage interpretation |
| D | Competitive under misspecification | PARTIAL | Occasional GRU wins; representation probe remains better |
| E | Observation head reproducibly helps | FAIL | Dedicated site-shift benchmark shows near-zero / negative benefit |

Table 1. Preliminary Phase-0 go/no-go assessment. “PASS, weak” refers to the literal preregistered threshold, not to the stronger scientific claim.

# 2. Experimental Context and Evidence Boundary

Phase-0 was designed as a methodological stress test before scarce clinical data are used. The central question was whether a compact continuous-time flow–jump adapter could transform frozen longitudinal clinical representations into disease-specific latent states more sample-efficiently than conventional probing or scratch temporal learning.

<!-- blank -->

<!-- image rel=rId11 -->

Figure 1. Phase-0 evidence path and decision logic.

The benchmark spans five synthetic worlds (smooth, jumps, informative observation, site shift, and misspecified), six training-patient budgets N ∈ {5, 10, 20, 40, 80, 100}, and five matched cohort/subset/model seed bundles. The principal point-forecasting metric is test MAE. The scientific interpretation below is restricted to the outputs supplied by the official run and does not treat synthetic evidence as clinical validation.

# 3. Primary Low-N Result

Flow-Jump beats both the representation-linear probe and the scratch GRU in only four dynamics-world/N settings: informative observation at N=100; jumps at N=80; and site shift at N=80 and N=100. There are no simultaneous wins at N ≤ 40.

<!-- blank -->

<!-- image rel=rId12 -->

Figure 2. Win map for simultaneous mean-MAE superiority over both primary comparators.

| World | N | Flow-Jump | Rep. linear | GRU scratch |
| --- | --- | --- | --- | --- |
| informative_observation | 100 | 0.447 | 0.668 | 0.478 |
| jumps | 80 | 1.048 | 1.062 | 1.064 |
| site_shift | 80 | 0.496 | 0.681 | 0.510 |
| site_shift | 100 | 0.449 | 0.703 | 0.487 |

Table 2. The four dynamics-world/N settings in which Flow-Jump has the lowest mean MAE among the two primary comparators.

# 4. Learning-Curve Crossover

The dominant pattern is a late crossover: Flow-Jump is usually inferior to the scratch GRU at N = 5–40, but becomes competitive or superior in specific structured/shifted worlds at N = 80–100. This is inconsistent with the intended “extreme-low-N first” behavior, but it is consistent with a structured model whose inductive bias becomes exploitable once enough disease-specific observations are available.

<!-- blank -->

<!-- image rel=rId13 -->

Figure 3. Mean test MAE learning curve in the informative observation world.

<!-- blank -->

<!-- image rel=rId14 -->

Figure 4. Mean test MAE learning curve in the jumps world.

<!-- blank -->

<!-- image rel=rId15 -->

Figure 5. Mean test MAE learning curve in the site shift world.

# 5. Capacity and Alternative Explanation

A capacity-matched control is necessary before attributing the higher-N wins specifically to the flow–jump inductive bias. Flow-Jump has 6,967 trainable parameters, approximately 1.56× the scratch GRU’s 4,455 parameters. Because the crossover toward Flow-Jump occurs primarily at N = 80–100, additional capacity remains a plausible explanation for part of the observed advantage.

<!-- blank -->

<!-- image rel=rId16 -->

Figure 6. Trainable parameter counts for the principal benchmark models.

| Required control<br>Add a capacity-matched GRU around ~7k trainable parameters before claiming that the performance gain is structural rather than a consequence of parameter count. |
| --- |

<!-- blank -->

# 6. Ablation Diagnosis

Ablation deltas are defined as ΔMAE = MAE(ablated) − MAE(full). Positive values mean that removing the component worsened MAE and therefore support a useful contribution from the component; negative values mean that the ablated model improved point forecasting.

<!-- blank -->

<!-- image rel=rId17 -->

Figure 7. Flow-Jump ablation effect for no flow.

<!-- blank -->

<!-- image rel=rId18 -->

Figure 8. Flow-Jump ablation effect for no jump.




<!-- blank -->

<!-- image rel=rId19 -->

Figure 9. Flow-Jump ablation effect for no representation.

<!-- blank -->

<!-- image rel=rId20 -->

Figure 10. Flow-Jump ablation effect for no prob scale.

Continuous flow is modestly beneficial in the jumps (+0.0289), misspecified (+0.0523), and site-shift (+0.0168) worlds. The representation pathway also helps in several difficult worlds. These patterns indicate that the structured architecture is not entirely decorative.

The explicit jump pathway is more ambiguous. It is clearly useful in the informative-observation world (+0.0761), but removing it slightly improves MAE in the dedicated jumps world (−0.0155) and smooth world (−0.0177). Because the current no-jump ablation removes the explicit event-conditioned recurrent jump while the frozen representation may still contain current-event information, this finding does not prove that discrete-event information is useless; it does show that the present explicit jump mechanism is not yet validated as the source of performance in the jump-dominated world.

| Major warning: probabilistic scale<br>Removing the probabilistic scale pathway improves MAE in every world, dramatically so under misspecification (Δ = −0.578). The scale head may still be justified by likelihood/calibration, but its point-forecasting cost must be explicitly balanced against NLL and coverage before it is retained. |
| --- |

<!-- blank -->

# 7. Observation-Process Head

The observation-aware model does not show reproducible robustness benefit in the dedicated site-shift benchmark. Mean benefit is defined so that positive values favor the full observation-head model. Most shifted-site effects are extremely close to zero, while MAE, RMSE, event Brier, event ROC-AUC, and latent recovery are slightly negative. Coverage and event log-loss improve only marginally.

<!-- blank -->

<!-- image rel=rId21 -->

Figure 11. Observation-head benefit across site-shift robustness metrics. Effects cluster around zero.

<!-- blank -->

<!-- image rel=rId22 -->

Figure 12. NLL effect of the observation head. Mean benefit is negative with very large dispersion, especially on the shifted site.

| Decision on observation head<br>The predeclared retention criterion is not met. The current observation-process head should be dropped or redesigned for Phase-0.5 rather than defended on the basis of this run. |
| --- |

<!-- blank -->

# 8. Misspecification Robustness

In the misspecified world, Flow-Jump loses to the representation-linear probe at every training size. It is nevertheless competitive with the scratch GRU in isolated settings, including N=20 (1.602 vs 1.618) and N=80 (1.471 vs 1.483). The appropriate conclusion is therefore “competitive in isolated settings, not robustly superior.”

<!-- blank -->

<!-- image rel=rId23 -->

Figure 13. Mean test MAE under simulator misspecification.

# 9. Phase-0 Decision

<!-- blank -->

<!-- image rel=rId24 -->

Figure 14. Current Phase-0 decision state.

The literal Criterion A threshold (“improve at least one primary low-N objective in a dynamics world”) is satisfied, but only weakly. That formal pass should not be confused with validation of the stronger intended claim. In the extreme-low-N regime N ≤ 40, there are zero simultaneous Flow-Jump wins over both primary comparators across 12 dynamics-world/N settings.

| Scientific verdict<br>The current implementation does not validate extreme-low-N superiority. It does provide enough structured, reproducible signal to justify a focused Phase-0.5 iteration aimed at moving the crossover from N≈80–100 toward N≈10–40. |
| --- |

<!-- blank -->

# 10. Phase-0.5 Research Agenda

### Capacity-matched control

Construct a GRU with approximately 7k trainable parameters and rerun the same seed/world/N matrix. This is necessary to separate inductive bias from capacity.

### Jump-pathway diagnosis

Instrument event-conditioned updates and build a stricter pre-event representation path h_{t−}=f(H_{<t}) so the jump ablation cannot inherit current-event semantics through h_t.

### Uncertainty redesign

Evaluate whether the probabilistic-scale head’s MAE penalty purchases better NLL/coverage. If not, decouple point and scale optimization or redesign the uncertainty parameterization.

### Observation-head redesign

Remove the current observation-process head from the default model. Reintroduce only after a mechanism can show reproducible site-shift robustness/calibration benefit.

### Low-N crossover target

Treat N=5,10,20,40 as the primary optimization target. The next architecture should be judged by paired seed-level gains in this region, not by N=80/100 improvements alone.

### Seed-level inference

Report paired differences, confidence intervals / bootstrap intervals, and consistency across the five matched seed bundles rather than relying only on mean learning curves.

# 11. Execution Integrity and Provenance

| Field | Value |
| --- | --- |
| Protocol anchor | be5a66b2e45362f60c90844e4e25673fb7bb3e21 |
| Execution SHA | d6f105eee73fcb8e9cc5987d292b1bb98a687382 |
| CUDA runtime | 13.0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Workers | 2 |
| Wall time | 03 h 40 m 02.49 s |
| Shards | 25 / 25 complete |
| Cells | 2,730 / 2,730 complete |
| Metric rows | 23,520 |
| Failures | 0 |
| Cancelled | 0 |
| Incomplete | 0 |
| Duplicate scientific keys | 0 |
| NaN / infinite metric values | 0 / 0 |
| Backend provenance | PASS (manifest authoritative; row-level backend explicit for representation MLP) |

Table 3. Official run provenance and completeness record.




| Artifact | SHA-256 |
| --- | --- |
| run_record.json | 93de1112ad855388a4d422a06249575cbc5de569a7b9a498e4d7a25f18a87b82 |
| run_manifest.json | 1a85f24e64b18c1728ef1e811a3e35ae16507b66373da29d8004b1961e9c8a08 |
| metrics.csv | f829502bc3489b3608c1ba06c9b4ca0c0910872c88ee628325339232c6816dcf |
| ablation_metrics.csv | 0ea780e62ef39d4024627eddbe1a83f5c48ba838dd763802d69a834e7b4416f6 |
| gate_summary.csv | a8d68248da47c7803b5cc6aaa2465ea682802c85a49a61b05c9ce0f5eecf7b06 |
| learning_curves.png | 888f5cec421e0e5341d90fd62b8ba82df8b8c498c87e25e69620f57fa96ad729 |

Table 4. Checksums captured immediately after the audit-passed official execution.

# 12. Interpretation Limits

Synthetic evidence is methodological evidence only; it does not establish clinical validity, transportability, safety, or utility on real patients.

The “beats both” map is based on mean MAE across matched seed bundles. Seed-level paired effects and uncertainty intervals should accompany any publication-grade claim.

Criterion C remains pending until the complete NLL and 90% coverage tables are interpreted; this report does not invent a post-hoc threshold for “materially worse.”

The observation-head site-shift analysis is strong evidence against retaining the current head, but it is not evidence that informative observation processes are unimportant in clinical data.

The no-jump ablation is a module-level ablation. Because the frozen history representation may contain current-event information, it is not equivalent to deleting all discrete-event information from the model.

| Recommended status<br>PHASE-0 COMPLETE → PHASE-0.5 REQUIRED. Preserve this execution as the immutable first official synthetic result. Do not relabel it as a successful low-N validation; use it as the diagnostic baseline against which the redesigned adapter must improve. |
| --- |

<!-- blank -->
