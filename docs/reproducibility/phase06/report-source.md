AFMC Phase-0.6

Diagnostic Validation Report

From Instrumented Reproduction to Initialization-Stability Stop

<!-- blank -->

| Field | Value |
| --- | --- |
| Final Phase-0.6 status | COMPLETE - D4-B AMBIGUOUS; STOP |
| Core parent execution | 1718402df1d6ef344168677e6d26ea664708e1bc |
| D2-B execution HEAD | 516c9e3c0e965582fa5cce976e9d8ebf32ea8404 |
| D4-B execution HEAD | 6ef4d506e5a6b96b15eb58225b95ebf64d3247ea |
| Validated D4-B implementation | 18f391fa89d80687f38f6c50a60a062e4524edd1 |
| Runtime | Native WSL2 \| Python 3.14.4 \| PyTorch 2.13.0+cu130 \| CUDA 13.0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU \| 8188 MiB |
| Diagnostic science executed | 340 cells total: D1 40 + D2-A 100 + D2-B 100 + D4-B 100 |
| Confirmatory seeds | 0 used; reserved namespaces remained protected |
| Date | 27 August 2026 |

<!-- blank -->

| Final decision in one sentence<br>Phase-0.6 reproduced the late N=40 predictive crossover, independently localized its dominant named variance source to model initialization, and then failed to establish that the time-scaled advantage is robust across a broader initialization bank. The frozen D4-B adjudicator classified the result AMBIGUOUS and terminated the diagnostic chain (STOP). |
| --- |

<!-- blank -->

<!-- blank -->

<!-- image rel=rId11 -->

Figure 1. Complete Phase-0.6 evidence path. Each stage narrowed the explanation before the final initialization-stability falsification.

<!-- blank -->

# Contents

1. Executive Summary

2. Scientific Scope and Evidence Boundary

3. Protocol Architecture and Provenance Model

4. D0 - Diagnostic Instrumentation

5. D1 - Exact Instrumented Reproduction

6. D2-A - Orthogonal Variance Screen

7. D3 - Hypothesis Adjudication

8. D2-B - Complementary Orthogonal Replication

9. Cross-Array Adjudication and Sufficiency

10. D4-B - Frozen Initialization/Optimization Stability Test

11. D4-B Implementation, Review, and Readiness

12. D4-B CUDA Execution and Resume Event

13. D4-B Results

14. Final D4-B Adjudication

15. Scientific Interpretation and Limits

16. Final Evidence Inventory and Phase-0.6 Closeout

| Evidence boundary<br>This report is Phase-0.6 only. Earlier Phase-0/Phase-0.5 results are referenced only as the reason Phase-0.6 existed; they are not re-adjudicated here. Phase-0.6 remained development-only and never touched the reserved confirmatory seed namespaces. |
| --- |

<!-- blank -->

<!-- blank -->

# 1. Executive Summary

Phase-0.6 was a diagnostic program created to explain a structured failure rather than to rescue a failed mechanism claim. Its target was the reproducible observation that time-scaled flow could become predictively favorable around N=40 while remaining heterogeneous across matched seed bundles. The program therefore instrumented training, reproduced the behavior under an auditable output identity, decomposed variance sources, required complementary replication before causal narrowing, and finally subjected the resulting initialization hypothesis to a frozen stress test.

The sequence completed as planned: D0 instrumentation, D1 exact reproduction, D2-A orthogonal variance screening, D3 adjudication, D2-B complementary orthogonal replication, cross-array sufficiency adjudication, and D4-B initialization/optimization stability falsification. D2-A and D2-B independently agreed that model initialization was the unique named dominant factor at N=40, while N=5 had no stable named dominant factor. The five overlapping seed triples between D2-A and D2-B reproduced exactly at both N values, supporting execution fidelity.

D4-B then held five cohort/subset contexts fixed and expanded only model initialization to ten new diagnostic seeds (1001-1010). The official 100-cell CUDA run completed successfully after one interruption at 95/100 cells was resumed into the same hash-bound child store. D4-B produced 50 paired effects, a positive grand mean ΔMAE of +0.008944, six of ten positive model-seed means, three of five positive context means, and a model-seed-cluster 95% bootstrap interval of [-0.001637, +0.019872].

| Final scientific verdict<br>The robust temporal-flow advantage was NOT established. The effect remains slightly positive on average, but its sign and magnitude are too dependent on initialization and context to satisfy the preregistered stability gate. The formal classification is AMBIGUOUS, and the frozen next state is STOP. |
| --- |

<!-- blank -->

| Stage | Executed scope | Decision / consequence |
| --- | --- | --- |
| D0 | Instrumentation only | Non-interfering diagnostics established |
| D1 | 40 cells | Reproduction surface established; proceed to variance diagnosis |
| D2-A | 100 cells | Model dominant at N=40; complementary array required |
| D3 | No new science cells | next_required_stage = D2B |
| D2-B | 100 cells | Complementary evidence available |
| Cross-array | Adjudication only | Evidence sufficient; next = D4_OPTIMIZATION |
| D4-B | 100 cells | AMBIGUOUS; next_required_stage = STOP |

<!-- blank -->

<!-- blank -->

# 2. Scientific Scope and Evidence Boundary

Phase-0.6 was explicitly separated from the stopped Phase-0.5 protocol. Its purpose was diagnostic: determine why a late N=40 predictive crossover appeared and why that behavior varied across seed bundles. The program did not reinterpret the earlier mechanism-gate failure, did not claim latent-mechanism recovery, and did not authorize confirmatory evaluation.

Development-only evidence throughout; no reserved confirmatory cohort seeds 701-710, subset seeds 801-810, or model seeds 901-910 were used.

No post-hoc seed substitution, threshold change, or outcome-dependent alteration of the D4-B classification rule was permitted.

D4-B was constrained to the smallest evidence-driven intervention: broaden model initialization only while holding the training recipe and data contexts fixed.

D4 capacity/time controls were a conditional future route, not part of Phase-0.6 unless D4-B first classified as stable. Because D4-B was ambiguous, that route was never authorized.

All evidence is synthetic methodological evidence; it does not establish clinical validity, safety, utility, or transportability.

# 3. Protocol Architecture and Provenance Model

Phase-0.6 used hash-bound output stores and staged authorization. Execution and adjudication were deliberately separate operations. A later stage could consume earlier evidence only after deep validation of protocol identity, completion, seed firewall, and parent-child linkage. This design prevented a failed or altered parent from silently propagating into a later diagnostic.

| Evidence layer | Identity / role |
| --- | --- |
| Core Phase-0.6 parent | Execution 1718402df1d6ef344168677e6d26ea664708e1bc; contains D1, D2-A, D3 |
| Core protocol lock SHA-256 | c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51 |
| Core D3 artifact SHA-256 | 6b9fffed7503fae6beeac2314238ae3d10ffdebfd27ea71ad952ab1f87916460 |
| D2-B child | Execution HEAD 516c9e3c0e965582fa5cce976e9d8ebf32ea8404 |
| D2-B canonical protocol SHA-256 | 33f7cb1f6e71560f547a746cb7f5eb41130f794ab1e3a6fb1928c40e05b29f12 |
| D2-B canonical adjudication SHA-256 | ea28fd4d5f6490a10fad20d5d3f3e76a1de08bf6b1be3b9805c9cf6c519e845f |
| D4-B child | Execution HEAD 6ef4d506e5a6b96b15eb58225b95ebf64d3247ea; separate writable child |

<!-- blank -->

| Hash-domain note<br>Raw-file SHA-256 and canonical parsed-object/config hashes are intentionally different domains. The report labels them explicitly rather than treating those differences as provenance mismatches. |
| --- |

<!-- blank -->

<!-- blank -->

# 4. D0 - Diagnostic Instrumentation

D0 added the observability missing from the earlier run while preserving training behavior. Persisted diagnostics included epoch-level training and validation losses, validation MAE/RMSE, production checkpoint selection, a shadow best-validation-MAE checkpoint, stopping epoch and reason, gradient and parameter norms, flow displacement/update magnitude, and execution provenance.

The central non-interference requirement was verified: the diagnostic path was byte-identical to the default training path for the tested behavior. Instrumentation was therefore treated as observation, not a training intervention.

| Readiness check | Observed result |
| --- | --- |
| Phase-0.6 tests | 146 passed |
| Phase-0.5 regression tests | 154 passed |
| Full repository suite | 537 passed |
| Ruff | All checks passed |
| D0 non-interference smoke | 1 passed; byte-identical diagnostics/default path |
| Executable planning audit | D1=40 cells; D2-A=100 cells; no reserved-seed intersections |

<!-- blank -->

# 5. D1 - Exact Instrumented Reproduction

D1 replayed the exact five development bundles across N={5,10,20,40} for only the no-flow and time-scaled variants, producing 40 diagnostic cells. Architecture, optimizer, objective, data generation, split semantics, and the earlier decision boundary were held fixed. The purpose was not to rerun a failed gate but to verify that the N-dependent pattern survived under instrumentation and a separate Phase-0.6 identity.

D1 completed and established an auditable reproduction surface. The late N=40 predictive behavior was sufficiently reproduced to justify causal attribution work, but D1 alone could not identify whether heterogeneity came from cohort realization, low-N subset composition, model initialization, checkpoint selection, or interactions among them.

| D1 consequence<br>Proceed to variance diagnosis. No redesign was justified yet, and no confirmatory seed was exposed. |
| --- |

<!-- blank -->

<!-- blank -->

# 6. D2-A - Orthogonal Variance Screen

The five original matched bundles changed cohort seed, subset seed, and model seed together. D2-A therefore recombined the five exposed development levels in a balanced OA(25,3,5,2) strength-2 design. For cohort index i and subset index j, the model index was k=(i+j) mod 5. Each factor level appeared five times and every factor pair occurred once.

D2-A evaluated N={5,40} and flow={none,time_scaled}, giving exactly 100 cells. The paired estimand remained ΔMAE = MAE_none - MAE_time_scaled, so positive values favor time-scaled flow. The design efficiently estimates marginal main effects, but it does not identify arbitrary interactions; that limitation was built into the adjudication logic.

| Training N | Named dominant factor | Interpretation | Consequence |
| --- | --- | --- | --- |
| 5 | none | No named factor satisfied the frozen dominance rule | Do not attribute the effect to cohort/subset/model |
| 40 | model | Model initialization emerged as the named dominant factor | Independent complementary evidence required |

<!-- blank -->

# 7. D3 - Hypothesis Adjudication

D3 did not treat the D2-A model-seed result as final because a single strength-2 orthogonal array can alias main effects with unmodeled interactions. The adjudicator therefore required a second, independently structured complementary array before model-initialization dominance could be considered sufficient evidence.

| Hypothesis family | D3 status |
| --- | --- |
| Optimization / initialization variance | Leading explanation at N=40; replication required |
| Subset composition variance | Not leading named explanation |
| Cohort realization variance | Not leading named explanation |
| Checkpoint-objective mismatch | Plausible diagnostic contributor, not isolated as dominant |
| Capacity/time semantics | Not tested; deferred |
| Predictive vs mechanistic mismatch | Retained as caution; no latent-recovery claim |

<!-- blank -->

| D3 route<br>next_required_stage = D2B. The project deliberately refused to jump directly from one orthogonal screen to mechanism redesign. |
| --- |

<!-- blank -->

<!-- blank -->

# 8. D2-B - Complementary Orthogonal Replication

D2-B changed only the orthogonal mapping to k=(i+2j) mod 5. Cohort/subset levels, N values, flow modes, training semantics, and seed firewall remained unchanged. All 100 cells were recomputed in a new child store, including five seed triples overlapping D2-A so that reproducibility itself became observable evidence rather than inherited data.

| Execution field | Observed value |
| --- | --- |
| Launcher status | COMPLETE; exit code 0 |
| Execution HEAD | 516c9e3c0e965582fa5cce976e9d8ebf32ea8404 |
| Validated implementation | cb61c3eb3f39a00980033e6b61108a0834e5bfca |
| Cells | 100/100; 50 at N=5; 50 at N=40 |
| Flow modes | 50 none; 50 time_scaled |
| Runtime | Python 3.14.4; PyTorch 2.13.0+cu130; CUDA |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU; 8188 MiB |
| Confirmatory seeds | None |

<!-- blank -->

<!-- blank -->

<!-- image rel=rId12 -->

Figure 2. D2-B variance decomposition. At N=40, model initialization explains 59.8% of the additive decomposition and is the unique named dominant factor.

| N | Cohort | Subset | Model | Residual | Largest-factor bootstrap frequency (model) |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.79% | 10.63% | 40.18% | 48.40% | 74.35% - below 80% dominance threshold |
| 40 | 3.34% | 6.56% | 59.80% | 30.30% | 93.00% - stable dominant |

<!-- blank -->

<!-- blank -->

## D2-B N-dependent crossover

Across the 25 D2-B seed triples, mean ΔMAE was -0.00546 at N=5 and +0.02546 at N=40. The paired shift T = ΔMAE_N40 - ΔMAE_N5 averaged +0.03093; 19/25 shifts were positive, and the fixed 10,000-resample 95% bootstrap interval [0.01383, 0.04739] lay wholly above zero. This independently reproduced the late predictive crossover.

<!-- blank -->

<!-- image rel=rId13 -->

Figure 3. D2-B reproduces the N-dependent crossover: slightly unfavorable at N=5, favorable at N=40, with a positive paired shift.

| Mechanistic caution<br>The D2-B result is still a predictive MAE phenomenon. Phase-0.6 does not reinterpret it as improved latent-state recovery or calibrated event modeling. |
| --- |

<!-- blank -->

# 9. Cross-Array Adjudication and Sufficiency

After the D2-B child was audited, a separate adjudicator compared D2-A and D2-B under the frozen sufficiency rule. Both arrays returned the same named-factor state: none at N=5 and model at N=40. Complementary-array evidence was therefore classified sufficient, removing the need for a full 5x5x5 development factorial.

| Adjudication field | Frozen result |
| --- | --- |
| D2-A dominant factor, N=5 | none |
| D2-B dominant factor, N=5 | none |
| D2-A dominant factor, N=40 | model |
| D2-B dominant factor, N=40 | model |
| Complementary-array evidence | sufficient |
| Next required stage | D4_OPTIMIZATION |
| Remaining escalation queue | D4_OPTIMIZATION -> D4_CAPACITY_TIME |

<!-- blank -->

The five overlapping seed triples supplied an execution-fidelity check. At both N=5 and N=40, independently rerun ΔMAE values agreed exactly: mean absolute difference 0.0, maximum absolute difference 0.0, and Pearson r=1.0 across all five overlap pairs.

| Cross-array scientific consequence<br>The N=40 time-scaled effect was now specifically associated with model initialization/optimization dependence. Cohort and subset composition were not the leading named explanations in either array. |
| --- |

<!-- blank -->

<!-- blank -->

# 10. D4-B - Frozen Initialization/Optimization Stability Test

The cross-array result did not justify changing the optimizer. It justified first asking whether the observed N=40 advantage survives broader model-initialization variation when cohort/subset context and the entire training recipe are held fixed. D4-B was therefore a falsification/stability test, not an optimization sweep.

| Dimension | Frozen value |
| --- | --- |
| Fixed contexts | (401,501), (402,502), (403,503), (404,504), (405,505) |
| Diagnostic model seeds | 1001-1010 |
| Training budget | N=40 only |
| Flow modes | none, time_scaled |
| Jump / uncertainty | none / deterministic |
| World | smooth |
| Cardinality | 5 contexts x 10 model seeds x 2 flows = 100 cells |
| Paired effects | 50 |
| Bootstrap | 10,000 model-seed-cluster resamples; seed 20260827 |

<!-- blank -->

Primary estimand: ΔMAE(c,m) = MAE_none(c,m) - MAE_time_scaled(c,m). Positive values favor time-scaled flow.

| Class | Frozen condition | Next state |
| --- | --- | --- |
| stable | Δoverall > 0 AND >=8/10 positive model-seed means AND >=4/5 positive contexts AND lower 95% cluster-bootstrap CI > 0 | D4_CAPACITY_TIME |
| fragile | Δoverall <= 0 OR <=5/10 positive model-seed means OR <=2/5 positive contexts | STOP |
| ambiguous | Every other result | STOP |

<!-- blank -->

| Training invariants<br>The first D4-B experiment could not change optimizer, learning rate, weight decay, patience, epoch limit, training objective, checkpoint objective, architecture, representation/state dimensions, time scale, data generation, split semantics, or low-N selection semantics. Only model seed changed within each fixed context. |
| --- |

<!-- blank -->

<!-- blank -->

# 11. D4-B Implementation, Review, and Readiness

D4-B was implemented behind a strict execution-readiness gate. The child protocol bound two immutable parents: the core D1/D2-A/D3 root and the completed D2-B root. Before child creation, the implementation deeply validated both parents, recomputed D2-A and D2-B analyses, recomputed the cross-array adjudication, required equality with the persisted frozen decision, and required the route to remain D4_OPTIMIZATION.

| Hardening checkpoint | Observed evidence |
| --- | --- |
| Planner RED | CI #317 exposed absent D4-B planning surface |
| Protocol/provenance RED | CI #319 exposed absent D4-B protocol boundary |
| Deep two-parent RED | CI #321: exactly 2 intended failures; 571 passed; 5 CUDA skips |
| Tooling RED | CI #323: exactly 6 intended failures; 586 passed; 5 CUDA skips |
| Persistence RED | CI #324: exactly 1 failure; 592 passed; 5 CUDA skips; store initially rejected stage d4b |
| Minimal persistence fix | Added d4b to the allowed Phase06Store stages; hash-bound semantics unchanged |
| End-to-end fake smoke | 100-cell child, five analysis artifacts, immutable parent snapshots, separate adjudication |
| Independent review | Critical: None; Important: None |
| Exact candidate CI #326 | Ruff clean; 594 passed, 5 CUDA-unavailable skips |
| Final docs-only CI #327 | Ruff clean; 594 passed, 5 CUDA-unavailable skips |

<!-- blank -->

The validated execution-critical implementation SHA was 18f391fa89d80687f38f6c50a60a062e4524edd1. The readiness-record-only branch head later became 6ef4d506e5a6b96b15eb58225b95ebf64d3247ea; execution-critical paths were verified byte-identical to the validated implementation before CUDA launch.

| Readiness boundary<br>READY FOR D4-B EXECUTION authorized only the frozen 100-cell initialization-stability diagnostic. It did not authorize D4 capacity/time, D4-D, a full factorial, Phase-0.5 continuation, or confirmatory seeds. |
| --- |

<!-- blank -->

<!-- blank -->

# 12. D4-B CUDA Execution and Resume Event

The official D4-B child was created at outputs/phase06_d4b_6ef4d506e5a6b96b15eb58225b95ebf64d3247ea under execution HEAD 6ef4d506e5a6b96b15eb58225b95ebf64d3247ea. CUDA preflight passed on the RTX 4060 Laptop GPU with PyTorch 2.13.0+cu130 and CUDA 13.0.

The first fresh run was interrupted when the WSL session terminated after 95 of 100 cells had persisted. At that point there was no COMPLETE marker and no analysis artifact. The existing child was preserved; nothing was deleted or restarted fresh. The frozen resume path revalidated the existing child protocol, both immutable parents, execution identity, and CUDA environment, skipped already persisted cells, executed only the five missing cells, and then wrote the required analysis artifacts and COMPLETE marker.

| Execution event | Observed state |
| --- | --- |
| Fresh run interruption | 95/100 cells persisted; no COMPLETE marker; no analysis |
| Recovery action | Resume same hash-bound child; no fresh rerun |
| Final progress | 100/100 cells |
| N distribution | 100/100 at N=40 |
| Flow distribution | 50 none; 50 time_scaled |
| Runner exit | 0 |
| Stage marker | COMPLETE |
| Analysis | Written |
| Automatic adjudication | Not run; launcher hard-stopped before adjudication |

<!-- blank -->

| Integrity interpretation<br>The interruption was operational, not scientific. Resume preserved the same child identity and completed only the missing work; the final run therefore remains one official D4-B execution. |
| --- |

<!-- blank -->

| D4-B protocol evidence | Value |
| --- | --- |
| Protocol lock SHA-256 | 5017075fb686f1eb0e0d90973f09a861c47c661521051d660346902bbaef8e7e |
| COMPLETE marker SHA-256 | ed1c20b3764241eb837c2bfdb6faa17a44c972f3e5dd8ba7df6e9786fbaac4ac |
| D4-B addendum SHA-256 | 55bb4e3e3b07f8ff12fe0c49d648a6ab884a46e4c2d7ce4a106a0f4ee449c471 |
| Raw phase06 config SHA-256 | b07a87ff2958b6b5e91a2a93cde81fc7d78d93910d5e1769978437f0bdc951f2 |
| Canonical phase06 config hash in protocol | 5db07d24a74050016790b691c377f7723a721c406fa29a402278d20cf74feb50 |

<!-- blank -->

<!-- blank -->

# 13. D4-B Results

The completed D4-B matrix contained exactly 100 unique cells: stage d4b, world smooth, N=40, 50 no-flow cells, 50 time-scaled cells, five fixed contexts, and model seeds 1001-1010. The primary analysis produced exactly 50 paired effects. No adjudication artifact existed during the post-execution audit.

| Primary statistic | Observed result |
| --- | --- |
| Overall mean ΔMAE | +0.0089443761 |
| Median ΔMAE | +0.0017172247 |
| SD across 50 paired effects | 0.0373800492 |
| Positive paired effects | 26/50 |
| Negative paired effects | 24/50 |
| Minimum paired ΔMAE | -0.0739861429 |
| Maximum paired ΔMAE | +0.0632497966 |
| Positive model-seed means | 6/10 |
| Positive fixed-context means | 3/5 |
| Model-seed-cluster 95% CI | [-0.0016367579, +0.0198715107] |

<!-- blank -->

<!-- blank -->

<!-- image rel=rId14 -->

Figure 4. D4-B grand paired effect and 95% model-seed-cluster bootstrap interval. The mean is positive, but the interval crosses zero.

<!-- blank -->

## Model-seed effects

<!-- blank -->

<!-- image rel=rId15 -->

Figure 5. Mean D4-B ΔMAE by model seed. Six seed means are positive and four are negative.

| Model seed | Mean ΔMAE | Median ΔMAE | SD across 5 contexts |
| --- | --- | --- | --- |
| 1001 | +0.007566 | -0.020297 | 0.050095 |
| 1002 | +0.011676 | -0.003930 | 0.027869 |
| 1003 | +0.022402 | +0.029038 | 0.035974 |
| 1004 | +0.036842 | +0.043219 | 0.026761 |
| 1005 | +0.020241 | +0.001519 | 0.037389 |
| 1006 | -0.005308 | -0.024751 | 0.038200 |
| 1007 | -0.001772 | +0.027627 | 0.051788 |
| 1008 | -0.016549 | -0.027551 | 0.032868 |
| 1009 | -0.014497 | -0.016066 | 0.021039 |
| 1010 | +0.028843 | +0.047508 | 0.031559 |

<!-- blank -->

## Fixed-context effects

<!-- blank -->

<!-- image rel=rId16 -->

Figure 6. Mean D4-B ΔMAE by fixed context. Three context means are positive and two are negative.

| Context | Mean ΔMAE | Median ΔMAE | SD across 10 seeds |
| --- | --- | --- | --- |
| 401/501 | +0.012501 | +0.015808 | 0.033671 |
| 402/502 | +0.043797 | +0.049279 | 0.018636 |
| 403/503 | -0.015780 | -0.012137 | 0.013382 |
| 404/504 | -0.027317 | -0.026299 | 0.030197 |
| 405/505 | +0.031521 | +0.031482 | 0.031065 |

<!-- blank -->

<!-- blank -->

## Optimization-dispersion diagnostics

The optimization traces were strongly context-dependent. The no-flow baseline frequently trained to epoch 100, whereas time-scaled flow often selected earlier checkpoints and exhausted patience in specific contexts. These diagnostics do not change the frozen classification, but they support the interpretation that the candidate changes optimization dynamics in a non-uniform way.

<!-- blank -->

<!-- image rel=rId17 -->

Figure 7. Mean production-selected checkpoint epoch by context and flow variant.

| Context | Variant | Selected epoch mean | Stop epoch mean | Patience exhausted | Max epochs reached |
| --- | --- | --- | --- | --- | --- |
| 401/501 | none | 100.0 | 100.0 | 0/10 | 10/10 |
| 401/501 | time_scaled | 71.3 | 77.3 | 5/10 | 5/10 |
| 402/502 | none | 100.0 | 100.0 | 0/10 | 10/10 |
| 402/502 | time_scaled | 100.0 | 100.0 | 0/10 | 10/10 |
| 403/503 | none | 76.8 | 86.1 | 7/10 | 3/10 |
| 403/503 | time_scaled | 37.4 | 49.4 | 10/10 | 0/10 |
| 404/504 | none | 100.0 | 100.0 | 0/10 | 10/10 |
| 404/504 | time_scaled | 44.8 | 55.6 | 9/10 | 1/10 |
| 405/505 | none | 100.0 | 100.0 | 0/10 | 10/10 |
| 405/505 | time_scaled | 89.4 | 91.8 | 2/10 | 8/10 |

<!-- blank -->

| Interpretation<br>The candidate does not simply shift MAE by a consistent amount. In several contexts it substantially alters checkpoint-selection and early-stopping behavior, while in others it behaves like the baseline. This heterogeneity is consistent with initialization/optimization sensitivity rather than a robust architectural gain. |
| --- |

<!-- blank -->

<!-- blank -->

# 14. Final D4-B Adjudication

After the post-execution audit confirmed the exact 100-cell matrix, required analysis files, hashes, completion marker, and absence of any pre-existing adjudication artifact, the separate adjudicate-d4b action was run with both immutable parent roots explicitly supplied. It exited successfully with code 0 and wrote the final adjudication artifact.

| Frozen criterion | Required for STABLE | Observed | Pass? |
| --- | --- | --- | --- |
| Grand mean | Δoverall > 0 | +0.008944 | YES |
| Model-seed consistency | >=8/10 positive means | 6/10 | NO |
| Context consistency | >=4/5 positive means | 3/5 | NO |
| Cluster-bootstrap uncertainty | Lower 95% CI > 0 | -0.001637 | NO |

<!-- blank -->

The result also did not meet the frozen FRAGILE definition: the grand mean was positive, positive model-seed means were greater than five, and positive context means were greater than two. It therefore fell into the deliberately predeclared middle region: AMBIGUOUS.

| Formal adjudication<br>classification = AMBIGUOUS \| Delta_overall = +0.00894437611103 \| positive_model_seed_count = 6/10 \| positive_context_count = 3/5 \| 95% cluster-bootstrap CI = [-0.00163675785065, +0.0198715107441] \| next_required_stage = STOP |
| --- |

<!-- blank -->

| Adjudication input artifact | Hash recorded by adjudicator |
| --- | --- |
| effect_rows | adbbbc7ecab2e56bee83491a586f5dd76fc4f086daff5898eabbc71a4c85a961 |
| model_seed_summary | 97e4542225917ff0c0a18a293f1edc9c2b9b23835d5e62624bd7ef7a59b478d8 |
| context_summary | 236de57bc0039184533abdcefb980018ad5e337a3278a0594f927fc0da60c3be |
| bootstrap_diagnostics (canonical JSON) | 4ec70a3d2a48352c6e251bd3c51008f4032dab23c956b04a5bc801acfb23013b |
| optimization_dispersion | 8a6d7ba9806e4bb1b19e8d84175db1bbf75e885ee36fdbac42cdd0195e5dfabb |

<!-- blank -->

## Why the verdict is AMBIGUOUS

AMBIGUOUS does not mean that Phase-0.6 learned nothing. It means the evidence is insufficient to certify a robust positive effect, but it is also insufficient to classify the candidate as consistently non-beneficial under the stricter fragile rule. The scientific claim that matters is binary at a higher level: D4-B does not support a robust time-scaled advantage. Because the frozen protocol required stability before capacity/time testing, the correct action is STOP rather than threshold revision or further opportunistic search.

<!-- blank -->

# 15. Scientific Interpretation and Limits

## What Phase-0.6 established

The late N=40 predictive crossover is reproducible across independently structured development diagnostics.

Model initialization is the unique named dominant factor at N=40 in both D2-A and D2-B under the frozen dominance rule.

N=5 has no stable named dominant factor and remains diffuse/residual-heavy.

D2-A/D2-B overlap reruns are exactly reproducible, supporting execution fidelity.

Expanded model-seed testing shows that the N=40 time-scaled effect is not stable enough across initialization and fixed contexts to clear the preregistered robustness gate.

Optimization traces show material context-dependent differences in checkpoint selection and early stopping for time-scaled flow.

## What Phase-0.6 did not establish

A robust time-scaled architectural advantage at N=40.

Improved recovery of the simulator-defined latent mechanism.

That elapsed-time semantics, rather than capacity or another architectural property, caused the observed predictive gain.

Any result on capacity/time controls: D4_CAPACITY_TIME was never authorized after the ambiguous D4-B result.

Any result on reserved confirmatory seed bundles.

Clinical validity, safety, transportability, or real-world utility.

| Paper-level conclusion<br>An apparent moderate-low-N predictive gain survived reproduction and independent variance screening, but its dominant source was model initialization and the effect failed a broader preregistered initialization-stability test. The resulting evidence supports a negative robustness conclusion, not a positive mechanism claim. |
| --- |

<!-- blank -->

## Why STOP is scientifically meaningful

The STOP boundary protects the scientific value of the program. Continuing directly into capacity/time testing after a non-stable D4-B result would convert a falsification sequence into post-hoc hypothesis rescue. Phase-0.6 instead closes the branch at the point defined before D4-B results were observed. Any future optimization redesign or new temporal mechanism must therefore begin as a new study with a newly frozen hypothesis and protocol.

<!-- blank -->

# 16. Final Evidence Inventory and Phase-0.6 Closeout

| Evidence / artifact | Pinned status or identity |
| --- | --- |
| Phase-0.6 diagnostic design | docs/superpowers/specs/2026-08-26-phase0-6-diagnostics-design.md |
| Core execution root | outputs/phase06_1718402df1d6ef344168677e6d26ea664708e1bc |
| Core execution SHA | 1718402df1d6ef344168677e6d26ea664708e1bc |
| Core protocol SHA-256 | c001bc278cc0c41793ef21d972f851b1ccd7a6d1b2adc8d2f45060660f709a51 |
| Core D3 SHA-256 | 6b9fffed7503fae6beeac2314238ae3d10ffdebfd27ea71ad952ab1f87916460 |
| D2-B child root | outputs/phase06_d2b_516c9e3c0e965582fa5cce976e9d8ebf32ea8404 |
| D2-B execution HEAD | 516c9e3c0e965582fa5cce976e9d8ebf32ea8404 |
| D2-B adjudication canonical SHA-256 | ea28fd4d5f6490a10fad20d5d3f3e76a1de08bf6b1be3b9805c9cf6c519e845f |
| D4-B addendum | docs/superpowers/specs/2026-08-27-phase0-6-d4b-execution-addendum.md |
| Validated D4-B implementation SHA | 18f391fa89d80687f38f6c50a60a062e4524edd1 |
| D4-B readiness / execution HEAD | 6ef4d506e5a6b96b15eb58225b95ebf64d3247ea |
| D4-B child root | outputs/phase06_d4b_6ef4d506e5a6b96b15eb58225b95ebf64d3247ea |
| D4-B protocol-lock SHA-256 | 5017075fb686f1eb0e0d90973f09a861c47c661521051d660346902bbaef8e7e |
| D4-B COMPLETE marker SHA-256 | ed1c20b3764241eb837c2bfdb6faa17a44c972f3e5dd8ba7df6e9786fbaac4ac |
| D4-B final classification | AMBIGUOUS |
| D4-B final route | STOP |
| Confirmatory seed use | NONE |

<!-- blank -->

## Evidence inventory and final program state

| Program component | Final status | Scientific consequence |
| --- | --- | --- |
| D0 instrumentation | COMPLETE | Diagnostics available; behavior preserved |
| D1 reproduction | COMPLETE | N-dependent diagnostic surface reproduced |
| D2-A variance screen | COMPLETE | Model dominant at N=40; complement required |
| D3 adjudication | COMPLETE | D2-B required |
| D2-B complementary array | COMPLETE | Independent replication available |
| Cross-array adjudication | COMPLETE / SUFFICIENT | D4_OPTIMIZATION authorized |
| D4-B implementation/review | COMPLETE / VALIDATED | 100-cell CUDA test authorized |
| D4-B execution | COMPLETE | 100/100 cells; analysis written |
| D4-B adjudication | COMPLETE / AMBIGUOUS | STOP |
| D4 capacity/time | NOT AUTHORIZED | Not executed |
| Confirmatory seeds | PROTECTED | Untouched |

| FINAL PHASE-0.6 STATUS<br>PHASE-0.6 COMPLETE. The diagnostic chain terminates at D4-B with AMBIGUOUS -> STOP. Preserve the core parent, D2-B child, D4-B child, and final adjudication as immutable evidence. Any subsequent optimization or mechanism redesign is a new study, not continuation of this stopped Phase-0.6 chain. |
| --- |
