Phase-0.5 Mechanistic Validation

Results and Decision Record

Stage I-A flow-mechanism falsification under extreme low-N constraints

<!-- blank -->

| Official execution | Stage I-A completed; pipeline stopped at preregistered flow gate |
| --- | --- |
| Repository HEAD | 50a94c06bc1c419ca55738f15f074cc06ccc3f36 |
| Validated execution code | 3fb62ff71e3d9f9750b6dbc1ebf5525db3f71e71 |
| Phase-0 evidence anchor | d6f105eee73fcb8e9cc5987d292b1bb98a687382 |
| Runtime | Native WSL2 · CUDA 13.0 · NVIDIA GeForce RTX 4060 Laptop GPU · 1 worker |
| Launcher elapsed | 00 h 09 m 11 s |
| Executed scope | 60/60 flow cells · 0 jump · 0 uncertainty · 0 timing-audit cells |
| Protocol lock | Phase-0 noise floor = locked effect floor = 4.1567% |
| Date | 26 August 2026 |

| Decision in one sentence<br>Neither learned flow candidate passed the preregistered Stage I-A gate. Time-scaled flow shows a late N=40 predictive crossover, but not improved latent-state recovery; Phase-0.5 therefore terminates at Stage I-A and must not proceed to jump selection or confirmation. |
| --- |

<!-- blank -->




# 1. Executive Summary

The official Phase-0.5 Stage I-A run completed all 60 planned flow-isolation cells successfully. The run then stopped exactly at the preregistered mechanism gate because neither learned flow candidate demonstrated a sufficiently large and sufficiently consistent low-N advantage over the no-flow control.

The locked minimum relative effect was 4.1567%, derived from the Phase-0 empirical noise floor. Gated flow lost on average (-3.62% relative improvement) and won only 2/5 matched development bundles. Time-scaled flow was essentially neutral in aggregate (+0.056% relative improvement), also winning only 2/5 bundles. Both therefore failed the joint requirements of positive mean effect, at least 4/5 wins, and effect size above the locked floor.

Postmortem analysis reveals a structured but non-confirmatory pattern: time-scaled flow is neutral or harmful at N=5, 10, and 20, then improves value forecasting at N=40 with 4/5 bundle wins. However, latent aligned R² is worse at every N, and event calibration metrics do not improve. The N=40 benefit is therefore predictive, not evidence of improved recovery of the simulator-defined latent mechanism.

| Central result<br>Phase-0.5 did not move the desired sample-efficiency crossover into the extreme-low-N regime. The time-scaled variant only becomes predictively favorable at N=40, while the preregistered low-N nAULC gate remains negative. |
| --- |

<!-- blank -->

## Gate Criterion Assessment

| Gate | Question | Status | Evidence |
| --- | --- | --- | --- |
| A | Flow mechanism establishes low-N benefit | FAIL | Neither candidate passes; both 2/5 bundle wins |
| B | Jump mechanism isolation | NOT REACHED | Blocked by failed flow prerequisite |
| C | Uncertainty mechanism selection | NOT REACHED | Blocked by failed flow prerequisite |
| D | Strict-vs-inclusive timing audit | NOT REACHED | Blocked by failed flow prerequisite |
| E | Freeze / confirmation / robustness | NOT EXECUTED | No candidate may be frozen or confirmed |

<!-- blank -->

# 2. Experimental Context and Evidence Boundary

Phase-0.5 was designed as a staged falsification protocol rather than an architecture sweep. Stage I-A isolates continuous latent flow in the smooth positive-control world while jump is disabled and uncertainty is deterministic. Only a flow mechanism that clears the development gate is permitted to advance to jump isolation.

<!-- blank -->

<!-- image rel=rId11 -->

Figure 1. Phase-0.5 staged evidence path. The official run terminated after Stage I-A because the flow gate failed.

The development evidence uses five preregistered seed bundles (401/501/601 through 405/505/605) and the primary low-N budgets N = {5, 10, 20, 40}. Confirmatory bundles 701-710 were not touched. No post-hoc threshold change, seed substitution, or manual mechanism selection was performed.

# 3. Stage I-A Flow Gate Result

The gate compares each learned flow candidate against the no-flow control using paired normalized log-N AULC across the four primary train sizes. A candidate passes only if all three conditions hold: mean paired improvement > 0, at least 4/5 bundles favor the candidate, and mean relative improvement is at least the locked minimum effect.

PASS ⇔ mean(Δ) > 0  AND  wins ≥ 4/5  AND  mean(Δ / control) ≥ 4.1567%

| Candidate | Params | Wins | Mean Δ | Median Δ | Mean rel. | Cand. nAULC | Control nAULC | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated | 4804 | 2/5 | -0.010545 | -0.020105 | -3.620% | 0.337302 | 0.326757 | FAIL |
| time_scaled | 4156 | 2/5 | +0.000978 | -0.002238 | +0.056% | 0.325778 | 0.326757 | FAIL |

<!-- blank -->

<!-- blank -->

<!-- image rel=rId12 -->

Figure 2. Mean paired relative improvement for the two learned flow candidates against the locked 4.1567% effect floor.

| Decision on Stage I-A<br>No flow candidate may advance. Preserve the official output and do not resume this run. |
| --- |

<!-- blank -->

# 4. Seed-Bundle Heterogeneity

Aggregate means conceal substantial heterogeneity. Gated flow has several large negative bundle effects and no stable pattern. Time-scaled flow is more structured: one bundle shows a strong +8.11% nAULC benefit, one a +1.90% benefit, two are slightly negative, and one is strongly negative at -7.64%.

| Bundle (cohort/subset/model) | Gated relative effect | Time-scaled relative effect |
| --- | --- | --- |
| 401/501/601 | -12.799% | -0.729% |
| 402/502/602 | +7.075% | +8.106% |
| 403/503/603 | +2.789% | +1.900% |
| 404/504/604 | -8.661% | -1.352% |
| 405/505/605 | -6.502% | -7.644% |

<!-- blank -->

<!-- blank -->

<!-- image rel=rId13 -->

Figure 3. Time-scaled flow paired nAULC effect by development bundle. The sign changes across matched repetitions.

Because each bundle changes cohort, low-N subset, and model-initialization seeds together, the current artifact cannot attribute this heterogeneity to one source. Cohort variation, subset composition, optimization initialization, or their interactions remain plausible.

# 5. Low-N Learning-Regime Diagnosis

Unfolding nAULC into the individual train sizes reveals a late predictive crossover. Time-scaled flow is negative on average at N=5, 10, and 20, then becomes strongly positive at N=40 with 4/5 bundle wins. Gated flow is more erratic and only weakly positive at N=40.

| N | Gated mean rel. | Gated wins | Time-scaled mean rel. | Time-scaled wins |
| --- | --- | --- | --- | --- |
| 5 | -6.59% | 1/5 | -2.57% | 1/5 |
| 10 | -4.50% | 1/5 | -2.07% | 1/5 |
| 20 | -5.91% | 2/5 | -1.02% | 2/5 |
| 40 | +4.50% | 3/5 | +9.70% | 4/5 |

<!-- blank -->

<!-- blank -->

<!-- image rel=rId14 -->

Figure 4. Mean relative MAE effect across N. Time-scaled flow only shows a clear positive regime at N=40.

| Interpretation<br>The Stage I-A failure is not equivalent to "flow never helps." The data instead suggest that the current time-scaled flow requires more effective data or more reliable model selection before its predictive capacity becomes useful. That does not satisfy the intended extreme-low-N claim. |
| --- |

<!-- blank -->

# 6. Secondary Metrics and Mechanistic Interpretation

The N=40 result must be interpreted across secondary metrics, not MAE alone. Oriented deltas below are defined so that positive values favor time-scaled flow.

| Metric | N=5 | N=10 | N=20 | N=40 |
| --- | --- | --- | --- | --- |
| MAE | -0.009270 | -0.006848 | -0.000566 | +0.029968 |
| RMSE | -0.009791 | -0.011036 | +0.000679 | +0.048872 |
| Latent aligned R² | -0.017946 | -0.007767 | -0.027479 | -0.012063 |
| Event Brier | -0.000884 | -0.002585 | -0.002951 | -0.002222 |
| Event log-loss | -0.023400 | -0.014158 | -0.011536 | -0.007666 |
| Event ROC-AUC | +0.006571 | +0.007873 | +0.012929 | +0.017878 |

<!-- blank -->

<!-- blank -->

<!-- image rel=rId15 -->

Figure 5. N=40 oriented secondary-metric effects. Point prediction improves, but latent-state alignment and probability calibration do not.

At N=40, time-scaled flow improves MAE and RMSE with 4/5 wins and also improves event ROC-AUC. However, latent aligned R² remains worse (1/5 wins), while event Brier score and event log-loss are worse on average. This supports a predictive transformation effect rather than improved reconstruction of the known simulator latent process.

| Mechanistic caution<br>Do not describe the N=40 crossover as successful latent-mechanism recovery. The predictive gain and latent-alignment result point in opposite directions. |
| --- |

<!-- blank -->

# 7. Low-N Model-Selection Constraint

The nominal low-N budget is split internally into fit and validation patients. At the smallest budgets, early stopping is therefore based on extremely small validation sets.

| Nominal N | Fit patients | Validation patients |
| --- | --- | --- |
| 5 | 4 | 1 |
| 10 | 8 | 2 |
| 20 | 16 | 4 |
| 40 | 32 | 8 |

<!-- blank -->

<!-- blank -->

<!-- image rel=rId16 -->

Figure 6. Training/validation allocation. N=5 uses four fit patients and a single validation patient.

The training loop selects the best checkpoint using a multitask validation objective (value loss plus event loss), while the scientific flow gate is ultimately based on test MAE nAULC. This mismatch is not itself a protocol violation, but it makes checkpoint-selection variance a plausible contributor to the low-N instability, especially when N=5 uses one validation patient and N=10 uses two.

| Evidence boundary<br>This is a diagnostic hypothesis, not an explanation proven by the official artifacts. Epoch-level train/validation trajectories and best-epoch metadata were not persisted in this run. |
| --- |

<!-- blank -->

# 8. Phase-0.5 Decision

The official Phase-0.5 run terminates at Stage I-A. Since flow is a prerequisite mechanism in the staged protocol, jump isolation, uncertainty selection, timing audit, freeze, confirmatory low-N testing, and robustness testing are not scientifically authorized within this run.

| Protocol component | Execution | Status | Decision |
| --- | --- | --- | --- |
| Stage I-A flow | 60/60 cells | FAIL | Completed, no candidate selected |
| Stage I-B jump | 0/60 | BLOCKED | Not started |
| Stage I-C uncertainty | 0/180 | BLOCKED | Not started |
| Stage I-D timing audit | 0/120 | BLOCKED | Not started |
| Stage II freeze | Not executed | BLOCKED | No candidate to freeze |
| Stage III confirmation | Untouched | PROTECTED | Confirmatory bundles 701-710 not evaluated |
| Stage IV robustness | Untouched | PROTECTED | Cannot rescue a failed Stage III / Stage I |

<!-- blank -->

| Scientific verdict<br>The current Phase-0.5 implementation does not validate a reproducible extreme-low-N continuous-flow advantage. The time-scaled mechanism shows a potentially useful predictive regime at N=40, but this signal is too late, too heterogeneous across bundles, and not accompanied by improved latent recovery. Preserve the run as an immutable negative result. |
| --- |

<!-- blank -->

# 9. Recommended Follow-up Agenda (Outside Official Phase-0.5)

Any further experimentation should be explicitly separated from the failed official run and restricted to development information. The goal is diagnosis before redesign, not continuation of the stopped protocol.

| Training instrumentation<br>Persist epoch-level train/validation core loss, value loss, event loss, best epoch, stopping epoch, gradient norms, and flow-state/update magnitude. |
| --- |

<!-- blank -->

| Exact development reproduction<br>Re-run only no-flow versus time-scaled flow on the same five development bundles and N={5,10,20,40} in a separate diagnostic output to verify the crossover while capturing trajectories. |
| --- |

<!-- blank -->

| Variance decomposition<br>Hold two seed sources fixed while varying the third to separate cohort, low-N subset, and initialization variance. Do not touch confirmatory bundles. |
| --- |

<!-- blank -->

| Mechanism redesign only after diagnosis<br>If model-seed variance dominates, address optimization/regularization; if subset variance dominates, address low-N statistical stability; if cohort variance dominates, inspect world/mechanism heterogeneity. |
| --- |

<!-- blank -->

| Latent-vs-predictive objective<br>Investigate why MAE/RMSE improve at N=40 while latent aligned R² worsens. A future mechanism claim should require predictive and mechanistic evidence to agree. |
| --- |

<!-- blank -->

# 10. Execution Integrity and Provenance

| Field | Value |
| --- | --- |
| Repository HEAD | 50a94c06bc1c419ca55738f15f074cc06ccc3f36 |
| Validated code commit | 3fb62ff71e3d9f9750b6dbc1ebf5525db3f71e71 |
| Phase-0 execution anchor | d6f105eee73fcb8e9cc5987d292b1bb98a687382 |
| Runtime | Linux WSL2 · Python project venv · PyTorch 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU · 8188 MiB |
| Workers | 1 |
| Observed launcher elapsed | 00 h 09 m 11 s |
| Flow cells | 60 / 60 complete |
| Subsequent Stage-I cells | 0 / 360 executed |
| Phase-0 noise floor | 0.041566851216 (4.1567%) |
| Locked minimum relative effect | 0.041566851216 (4.1567%) |
| Flow gate artifact | development/flow_gate.csv |
| Official output | outputs/phase05_official_50a94c06bc1c419ca55738f15f074cc06ccc3f36 |

<!-- blank -->

The launcher returned exit code 1 because the application intentionally raises a runtime exception when the flow development gate fails. All 60 planned flow cells were already persisted, and the gate artifact was written before the stop. This report therefore treats the event as a protocol-governed scientific termination rather than a computational failure.

# 11. Interpretation Limits

The result is synthetic methodological evidence only; it does not establish clinical validity, utility, safety, or transportability.

The N=40 postmortem pattern is exploratory because Stage I-A was preregistered around low-N nAULC across N={5,10,20,40}; it must not be relabeled as a Phase-0.5 success criterion.

Bundle-level heterogeneity cannot yet be attributed uniquely to cohort seed, subset seed, or model seed because all three vary together in each matched bundle.

The apparent checkpoint-selection explanation is plausible but unproven because epoch-level training and validation trajectories were not persisted.

No confirmatory seed bundle was evaluated. This protected boundary should remain intact until a new protocol is explicitly designed and frozen.

The observed N=40 predictive benefit should not be described as mechanistic latent recovery because latent aligned R² is worse on average at every N.

| Recommended status<br>PHASE-0.5 OFFICIAL RESULT: TERMINATED AT STAGE I-A (FLOW GATE FAIL). Preserve as the immutable second synthetic decision record. Do not resume, freeze, confirm, or reinterpret the N=40 signal as preregistered success. |
| --- |

<!-- blank -->




# 12. Evidence Inventory and Reproducibility Map

| EVIDENCE CLASS: ARCHIVAL / REPRODUCIBILITY |
| --- |

The original nine-page decision record is retained intact as the scientific narrative. This appendix maps the persisted official execution to the deterministic repository-side derivations used to audit the gate and support the exploratory postmortem.

| Evidence item | Pinned value |
| --- | --- |
| Implementation SHA | 50a94c06bc1c419ca55738f15f074cc06ccc3f36 |
| Protocol-lock SHA-256 | 33b25ebe46620626ae3d1bbbd6152fc3aacdb0e6200dd87e03c4fdec99e70db9 |
| Phase-0.5 config SHA-256 | befd7140cbf68cb981418cdc0c8880bb0652c2a6db614d05f61ad267297d519b |
| Specification SHA-256 | c32dae3d702991e47387e1523004c9c02141ae331891a6b7a7e422079d72283e |
| Simulator config SHA-256 | f093115ae47274680e8b7cf1c7fea440a739079881099227e5b8c4abc0320aa8 |
| Flow invocation wall time | 521.2366377040016 s (~8 m 41 s) |
| End-to-end launcher elapsed | 00 h 09 m 11 s |
| Official persisted scope | 60/60 Stage I-A flow cells; 360 metric rows; zero recorded execution failures |

Repository evidence lineage

| Layer | Artifact | Contents / transformation | Scientific role |
| --- | --- | --- | --- |
| Official raw output | raw/official_output/ | 60 cell JSONs + flow gate + protocol/provenance | Immutable source evidence |
| Raw checksum manifest | integrity/full_output_checksums.sha256 | SHA-256 over original output tree | Byte-equivalence audit |
| Analysis generator | tools/analysis/phase05/generate_tables.py | Deterministic flattening + paired recomputation | No manual spreadsheet arithmetic |
| Canonical metric surface | tables/flow_cell_metrics.csv | 360 persisted test-metric rows | Complete Stage I-A metric surface |
| Gate verification | tables/flow_gate_verified.csv | Independent recomputation of both flow candidates | Official decision integrity |
| Exploratory tables | tables/flow_bundle_effects.csv; flow_n_effects.csv; flow_metric_summary.csv | Bundle/N/metric paired effects | Postmortem only |
| External archive | phase05_official_50a94c.tar.zst | zstd integrity test passed; digest recorded separately | Durable off-Git copy |

| Closeout note: the repository evidence manifest should be generated only after this final DOCX is placed in docs/results/phase05/, so the human decision record is itself covered by the final hash set. |
| --- |




# 13. Complete Exploratory Metric Matrix

| EVIDENCE CLASS: EXPLORATORY POSTMORTEM — NOT A PREREGISTERED SUCCESS CRITERION |
| --- |

The table below expands the time-scaled-flow postmortem across every persisted metric and every primary training budget. Effects are oriented so that positive values always favor time-scaled flow. The value in parentheses is the number of development bundles, out of five, in which the candidate was better.

| Metric | Better | N=5 effect | N=10 effect | N=20 effect | N=40 effect |
| --- | --- | --- | --- | --- | --- |
| MAE | lower | -0.009270 (1/5) | -0.006848 (1/5) | -0.000566 (2/5) | +0.029968 (4/5) |
| RMSE | lower | -0.009791 (1/5) | -0.011036 (1/5) | +0.000679 (2/5) | +0.048872 (4/5) |
| Latent aligned R² | higher | -0.017946 (2/5) | -0.007767 (1/5) | -0.027479 (1/5) | -0.012063 (1/5) |
| Event Brier | lower | -0.000884 (1/5) | -0.002585 (2/5) | -0.002951 (1/5) | -0.002222 (2/5) |
| Event log-loss | lower | -0.023400 (0/5) | -0.014158 (0/5) | -0.011536 (1/5) | -0.007666 (2/5) |
| Event ROC-AUC | higher | +0.006571 (3/5) | +0.007873 (4/5) | +0.012929 (4/5) | +0.017878 (4/5) |

N=40 predictive-versus-mechanistic divergence

| Metric | Candidate mean | Control mean | Oriented effect | Wins | Interpretation |
| --- | --- | --- | --- | --- | --- |
| MAE | 0.284985 | 0.314953 | +0.029968 | 4/5 | Predictive error improves |
| Latent aligned R² | 0.303137 | 0.315199 | -0.012063 | 1/5 | Latent recovery worsens |

The N=40 pattern is therefore internally asymmetric: value prediction becomes favorable, while the known simulator latent state is not recovered more faithfully and probability calibration remains unfavorable. This supports the narrower description "late predictive crossover" and argues against labeling the signal as successful mechanistic recovery.

| Interpretation guardrail: Stage I-A remains failed because the official decision statistic is paired MAE nAULC across N={5,10,20,40}, with the locked 4.1567% effect floor and 4/5 consistency requirement. No N-specific postmortem row can retrospectively replace that criterion. |
| --- |




# 14. Phase-0 → Phase-0.5 → Phase-0.6 Reasoning Chain

| STATUS: PHASE-0.6 IS A SEPARATE DIAGNOSTIC PROGRAM; CONFIRMATORY BUNDLES REMAIN PROTECTED |
| --- |

The negative Phase-0.5 result narrows the next scientific question. The next step is not to resume the stopped protocol or try another mechanism opportunistically; it is to identify why the time-scaled model exhibits seed- and N-dependent predictive behavior without corresponding latent recovery.

| Phase | Falsified / unresolved claim | Key evidence | Authorized consequence |
| --- | --- | --- | --- |
| Phase 0 | Extreme-low-N advantage not established | 0/12 primary N≤40 simultaneous mean-MAE wins; crossover appeared only around N≈80–100 | Mechanistic redesign required |
| Phase 0.5 | Flow mechanism gate not established | gated: 2/5 wins, -3.62% mean relative; time_scaled: 2/5 wins, +0.056%; both below 4.1567% | Stop at Stage I-A; no freeze/confirmation |
| Phase 0.6 | Diagnose the structured failure | Reproduce no-flow vs time_scaled with trajectory instrumentation; decompose variance sources before redesign | Development-only diagnostics; confirmatory seeds untouched |

Proposed Phase-0.6 diagnostic sequence

| D0 — instrumentation<br>Persist epoch-level train/validation core, value and event losses; validation MAE/RMSE; gradient and parameter norms; flow-state displacement/update magnitude; stopping epoch; production best-composite epoch; shadow best-validation-MAE epoch. No behavior change. |
| --- |

| D1 — exact 40-cell reproduction<br>Five development bundles × four N values × {none, time_scaled}. First characterize CUDA/reproduction tolerance; then test whether the observed N-dependent crossover reproduces while the original nAULC gate remains failed. |
| --- |

| D2 — orthogonal variance screen<br>Use a balanced cohort/subset/model-seed design to estimate marginal main effects and identify whether cohort, subset composition, or initialization is the dominant source of heterogeneity. Do not claim interaction identification from a strength-2 array. |
| --- |

| D3 — hypothesis adjudication<br>Evaluate optimization/init variance, subset variance, cohort variance, sample-complexity threshold, checkpoint-objective mismatch, capacity effects, and predictive-versus-mechanistic mismatch. |
| --- |

| D4 — evidence-driven redesign<br>Change architecture, optimization, selection objective, or evaluation criteria only after D0–D3 identify the dominant failure mode; then freeze a new falsification protocol before any confirmatory work. |
| --- |

| Protected boundary: bundles 701/801/901 through 710/810/910 remain untouched. Phase-0.6 must use only development information until a new architecture and confirmation protocol are explicitly frozen. |
| --- |
