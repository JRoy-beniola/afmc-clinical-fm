# Phase 0.5 Official Result

## Formal status

**PHASE-0.5 TERMINATED AT STAGE I-A — FLOW MECHANISM GATE NOT ESTABLISHED**

## Objective

Phase 0.5 was designed to test whether mechanistically constrained
architectural changes could move the structured model's useful operating
regime toward the extreme-low-N setting that Phase 0 failed to validate.

Stage I was deliberately sequential:

1. continuous-flow mechanism,
2. jump mechanism,
3. uncertainty formulation,
4. representation-timing audit.

Progression to each later mechanism required the preceding development
gate to pass.

## Stage I-A design

The flow stage used the smooth synthetic world as the positive-control
development setting.

Three variants were evaluated:

- `none__none__deterministic`
- `gated__none__deterministic`
- `time_scaled__none__deterministic`

The development design contained:

- five locked development seed bundles,
- four train sizes: N = 5, 10, 20, 40,
- three flow variants,

for exactly 60 cells.

The gate used paired normalized log-N MAE AULC across the five development
bundles.

A candidate passed only if all of the following held:

1. mean paired improvement was positive;
2. at least 4 of 5 bundles favored the candidate;
3. mean relative improvement met or exceeded the locked minimum effect.

The locked minimum relative effect was:

`0.04156685121568672`

or approximately 4.1567%.

## Official gate result

| Candidate | Trainable parameters | Wins | Mean relative improvement | Candidate mean AULC | Control mean AULC | Passed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gated | 4,804 | 2 / 5 | -0.0361967999 | 0.3373018121 | 0.3267566741 | No |
| time_scaled | 4,156 | 2 / 5 | 0.0005612234 | 0.3257782608 | 0.3267566741 | No |

Neither candidate satisfied the preregistered development gate.

The gated mechanism was worse than control on average.

The time-scaled mechanism was approximately neutral on average and its
mean relative improvement was far below the locked 4.1567% threshold.
It also favored the candidate in only 2 of 5 development bundles.

No flow candidate was selected.

## Formal decision

Phase 0.5 stopped at Stage I-A exactly as specified by the staged protocol.

The official conclusion is:

> Neither gated nor time-scaled continuous latent flow demonstrated a
> development effect distinguishable from the Phase-0-derived noise floor
> or sufficiently consistent across the five preregistered development
> bundles.

More narrowly:

> Phase 0.5 Stage I-A did not establish that either gated or time-scaled
> continuous latent flow provides a reproducible low-N advantage over
> no-flow dynamics in the smooth development world.

This is a scientific gate failure, not an engineering execution failure.

## Unentered stages

Because Stage I-A did not pass:

- Stage I-B jump development was not entered;
- Stage I-C uncertainty development was not entered;
- the representation-timing stage was not entered;
- no Phase 0.5 architecture was frozen;
- confirmatory development was not started;
- robustness evaluation was not started.

The locked confirmatory bundles remained outside the official execution.

## Exploratory postmortem

The persisted Stage I-A cells permit exploratory inspection of the
N-specific metrics. These analyses are diagnostic and are not part of
the preregistered gate decision.

One notable pattern appears for the time-scaled flow at N=40:

| Metric | Candidate mean | Control mean | Mean effect | Wins |
| --- | ---: | ---: | ---: | ---: |
| MAE | 0.284985 | 0.314953 | +0.029968 | 4 / 5 |
| latent aligned R2 | 0.303137 | 0.315199 | -0.012063 | 1 / 5 |

Here positive effect is defined as candidate improvement.

Thus predictive MAE shows a late N=40 improvement while latent-state
recovery does not. This pattern must not be described as mechanistic
recovery.

The complete exploratory evidence is preserved in:

- `tables/flow_bundle_effects.csv`
- `tables/flow_n_effects.csv`
- `tables/flow_metric_summary.csv`

## Interpretation boundary

Phase 0.5 does not establish a successful flow mechanism.

It does provide evidence that the failure is structured rather than
uninformative: performance is strongly heterogeneous across development
bundles and train sizes, and predictive improvement can occur without
corresponding latent-state recovery.

These observations motivate diagnostic work rather than confirmatory
testing.
