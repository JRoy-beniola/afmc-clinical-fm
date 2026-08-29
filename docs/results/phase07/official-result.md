# Phase 0.7 Official Result

## Scientific question

Phase 0.7 prospectively tested whether the optimization-horizon association observed in exploratory post-Phase-0.6 analysis could be established as a sufficient causal explanation for architecture-relative performance heterogeneity.

The intervention compared two optimization policies:

- `standard_early_stop`
- `forced_horizon`

for two architectures:

- control: `none__none__deterministic`
- candidate: `time_scaled__none__deterministic`

under a frozen smooth-world, N=40 design.

## Frozen design

The official experiment contained exactly 200 cells:

- 5 cohort/subset contexts
- 10 new model seeds
- 2 architectures
- 2 optimization policies

Contexts:

- (406, 506)
- (407, 507)
- (408, 508)
- (409, 509)
- (410, 510)

Model seeds:

- 1101 through 1110

Training configuration:

- maximum epochs: 100
- patience: 12
- learning rate: 1e-3
- weight decay: 1e-4

Protected historical confirmatory seeds remained untouched.

## Estimands

For each matched architecture/context/model-seed cell:

`Delta_policy = MAE_control,policy - MAE_time_scaled,policy`

The primary architecture-by-policy interaction was:

`G = Delta_forced_horizon - Delta_standard_early_stop`

Equivalent continuation benefits were:

`H_time_scaled = MAE_time_scaled,standard - MAE_time_scaled,forced`

`H_control = MAE_control,standard - MAE_control,forced`

with:

`G = H_time_scaled - H_control`

Positive H means that forced continuation improved that architecture's test MAE.

## Frozen primary causal gate

Support required all of:

1. mean(G) > 0
2. lower crossed-bootstrap 95% CI for mean(G) > 0
3. at least 4 / 5 context means positive
4. at least 8 / 10 model-seed means positive

## Official result

Official adjudication:

`P07_OPTIMIZATION_HORIZON_NOT_ESTABLISHED`

Observed statistics:

- mean G = 0.0006074243783950794
- mean G CI lower = -0.0022108983993530317
- mean G CI upper = 0.0047499954700469926
- positive context means = 1
- positive model-seed means = 1
- positive paired G effects = 1

The primary causal gate failed.

The mean interaction was slightly positive, but its confidence interval crossed zero and the effect did not reproduce across contexts or model seeds.

## Heterogeneity gate

Observed:

- R_SD = 0.6163983694824
- bootstrap resamples = 10000
- valid R_SD replicates = 9989

The frozen analysis contract required all bootstrap replicates to produce a finite, nonzero standard-policy residual SD.

Because 11 replicates violated that condition:

- R_SD CI lower = NaN
- R_SD CI upper = NaN

Under the preregistered fail-closed rule, the heterogeneity gate therefore cannot be claimed.

No bootstrap replicate was dropped, retried, regularized, or epsilon-stabilized after observing the result.

## Official interpretation

The Phase 0.6 optimization-horizon association was diagnostically structured but did not prospectively establish the prespecified positive architecture-by-policy effect as a sufficient causal explanation under the locked Phase 0.7 intervention.

The specific stopping-horizon explanation is therefore not established and cannot be rescued by changing thresholds, seeds, endpoints, bootstrap handling, or intervention definitions after seeing the result.

## Exploratory post-hoc exposure audit

After the official adjudication was frozen, an exploratory read-only audit examined whether the stopping-policy intervention actually created different optimization trajectories.

Across 100 matched architecture cells:

- standard patience exhausted = 2
- forced arm recorded corresponding patience event = 2
- selected checkpoint changed = 2
- test MAE changed = 2
- patience-event mismatches = 0

Activation fraction:

`2 / 100 = 0.02`

Thus, 98 / 100 architecture cells had no behavioral treatment contrast: the standard policy already reached the full horizon.

### Activated control cell

Context:

- cohort seed 407
- subset seed 507
- model seed 1101
- architecture `none`

Standard:

- stop epoch = 54
- selected checkpoint epoch = 42
- test MAE = 0.3294597864151001

Forced:

- selected checkpoint epoch = 100
- test MAE = 0.3018235564231872

Continuation benefit:

`H_control = +0.0276362299919128`

### Activated time-scaled cell

Context:

- cohort seed 407
- subset seed 507
- model seed 1103
- architecture `time_scaled`

Standard:

- stop epoch = 59
- selected checkpoint epoch = 47
- test MAE = 0.323786199092865

Forced:

- selected checkpoint epoch = 100
- test MAE = 0.2657787501811981

Continuation benefit:

`H_time_scaled = +0.0580074489116668`

## Post-hoc interpretation boundary

The exposure audit does not alter the official adjudication.

It supports the exploratory statement:

> Premature stopping can materially hurt individual low-data runs, but in the unseen Phase 0.7 population it activated too rarely to establish the preregistered architecture-relative effect or serve as a sufficient explanation for the observed architecture heterogeneity.

It does not establish that early stopping or optimization horizon caused the Phase 0.6 result.

## Final decision

`P07_OPTIMIZATION_HORIZON_NOT_ESTABLISHED`

Phase 0.7 is terminal for repeated rescue attempts on this specific 200-cell stopping-horizon hypothesis.
