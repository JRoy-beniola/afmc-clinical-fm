# AFMC Phase 0.7 Evidence Archive

This directory is the canonical repository-side archival record for the completed AFMC Phase 0.7 prospective optimization-horizon intervention.

## Final result

**Official adjudication: `P07_OPTIMIZATION_HORIZON_NOT_ESTABLISHED`**

Phase 0.7 prospectively tested whether preventing premature stopping would establish the architecture-relative optimization-horizon effect suggested by the post-Phase-0.6 retrospective diagnostics.

It did not.

The official primary interaction estimate was:

- mean G = 0.0006074243783950794
- crossed-bootstrap 95% CI = [-0.0022108983993530317, 0.0047499954700469926]
- positive context means = 1 / 5
- positive model-seed means = 1 / 10
- positive paired G effects = 1 / 50

The preregistered causal gate therefore failed.

Observed residual heterogeneity ratio:

- R_SD = 0.6163983694824
- valid bootstrap replicates = 9989 / 10000

Because 11 bootstrap replicates were invalid under the frozen fail-closed rule, the R_SD confidence interval is NaN and the heterogeneity gate cannot be claimed.

## Exploratory post-hoc diagnostic

A post-adjudication intervention-exposure audit found that the stopping-policy intervention was behaviorally active in only:

- 2 / 100 matched architecture cells
- activation fraction = 0.02

The other 98 cells never exhausted patience before the fixed 100-epoch horizon, so standard early stopping and forced-horizon execution were behaviorally identical in those cells.

Both activated cells improved when continued:

- control (`none`), context 407/507, model seed 1101:
  H_control = +0.0276362299919128
- `time_scaled`, context 407/507, model seed 1103:
  H_time_scaled = +0.0580074489116668

This diagnostic is exploratory and does not modify the official Phase 0.7 adjudication.

## Contents

- `AFMC_Phase0_7_Prospective_Optimization_Horizon_Intervention_Report_FINAL.docx`
  - complete standalone narrative report
- `official-result.md`
  - canonical scientific result and interpretation boundary
- `provenance.md`
  - execution and archival provenance
- `raw/`
  - frozen official adjudication and consolidated metrics
- `tables/`
  - exploratory post-hoc paired-effect and intervention-exposure tables
- `provenance/`
  - protocol, plan, authorization, manifest, and execution provenance
- `integrity/`
  - file manifests, freeze records, and post-hoc checksums

## Scientific boundary

The official Phase 0.7 decision remains:

`P07_OPTIMIZATION_HORIZON_NOT_ESTABLISHED`

The supported conclusion is:

> Under the frozen Phase 0.7 intervention, preventing premature termination did not establish the preregistered architecture-relative optimization-horizon effect.

The post-hoc exposure audit supports the narrower exploratory observation that premature stopping materially affected the two runs in which it occurred, but it was too rare in the unseen Phase 0.7 population to serve as a sufficient population-level explanation for the previously observed architecture heterogeneity.

Phase 0.7 does not establish that optimization horizon caused the Phase 0.6 heterogeneity.

Later work may reinterpret the broader methodological implications, but it must not rewrite the official Phase 0.7 result.
