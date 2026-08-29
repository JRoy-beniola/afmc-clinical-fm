# AFMC Experimental Results Archive

This directory contains the canonical scientific result record for each formal AFMC experimental phase.

Each completed phase should preserve:

- a substantial human-readable report;
- a canonical Markdown result record;
- execution and archival provenance;
- machine-readable raw result data;
- derived analysis tables;
- relevant figures;
- explicit scientific decisions;
- the evidence boundary governing the next phase.

## Phase history

| Phase | Status | Primary decision |
|---|---|---|
| Phase 0 | COMPLETE | Extreme-low-N superiority not validated; focused redesign required |
| Phase 0.5 | COMPLETE | Flow mechanism gate failed; later preregistered stages were not entered |
| Phase 0.6 | COMPLETE | `D4-B AMBIGUOUS -> STOP`; model initialization dominated the N=40 variance diagnosis |
| Post-Phase-0.6 | EXPLORATORY | Optimization-conditioned retrospective signal identified; no causal claim |
| Phase 0.7 | COMPLETE | `P07_OPTIMIZATION_HORIZON_NOT_ESTABLISHED` |

## Directory convention

~~~text
docs/results/
├── README.md
├── phase0/
├── phase05/
├── phase06/
├── phase06_posthoc_optimization/
├── phase07/
└── ...
~~~

Each phase should normally contain:

~~~text
<phase>/
├── README.md
├── official-result.md
├── provenance.md
├── <full narrative report>.docx
├── raw/
├── tables/
├── figures/
└── integrity/
~~~

## Evidence model

The preferred evidence chain is:

~~~text
official raw outputs
        ↓
machine-readable result data
        ↓
derived comparison and validation tables
        ↓
figures
        ↓
canonical Markdown interpretation
        ↓
full narrative report
        ↓
formal scientific decision
~~~

The purpose of this structure is to ensure that every phase remains understandable and auditable at several levels.

A future reader should be able to:

1. inspect the actual consolidated result data;
2. verify seed-level and metric-level effects;
3. reproduce the major comparison tables;
4. inspect the main figures;
5. understand the scientific interpretation without rerunning the full experiment;
6. trace every conclusion back to an official execution and source commit.

## Raw result policy

Synthetic experiment results should be retained in Git whenever practical.

This includes, where available:

- consolidated metric CSV files;
- ablation result CSV files;
- gate summaries;
- seed-level results;
- run manifests;
- run records;
- split manifests;
- protocol locks;
- execution metadata;
- comparison tables;
- validation summaries;
- calibration summaries;
- latent-state recovery summaries;
- robustness summaries;
- learning-curve data.

Large numbers of per-cell intermediate files do not need to be committed individually if the same scientific evidence is preserved in consolidated tables and an externally archived raw execution.

## Derived table policy

Each phase should contain explicit derived CSV tables for the major scientific comparisons made in its report.

Examples include:

- MAE comparisons by world, model, N, and seed;
- RMSE comparisons;
- low-N win matrices;
- simultaneous-comparator win tables;
- learning-curve crossover summaries;
- seed-level paired differences;
- normalized AULC or nAULC effects;
- calibration metrics;
- NLL summaries;
- coverage summaries;
- event Brier score comparisons;
- event log-loss comparisons;
- event ROC-AUC comparisons;
- latent aligned R² comparisons;
- site-shift summaries;
- misspecification summaries;
- component-ablation summaries;
- capacity-matched control summaries;
- checkpoint-selection diagnostics where relevant.

These tables should be generated from the official persisted outputs rather than manually transcribed from narrative reports.

## Figure policy

Important figures used in the scientific interpretation should be stored in each phase's `figures/` directory.

Examples include:

- learning curves;
- low-N win maps;
- paired-effect plots;
- ablation plots;
- calibration plots;
- site-shift comparisons;
- latent-state recovery comparisons;
- bundle-level effect plots;
- N-wise crossover plots;
- checkpoint-trajectory plots.

Figures are supporting evidence. The underlying tabular data should also be preserved whenever practical.

## Human-readable report policy

Each major experimental phase should conclude with a substantial narrative report, normally in Word format.

The report should include:

- executive summary;
- experimental objective;
- protocol and evidence boundary;
- source-code and execution provenance;
- experimental matrix;
- primary result;
- per-world analysis;
- per-N analysis;
- seed-level analysis;
- ablation findings;
- uncertainty and calibration findings;
- event-prediction findings;
- latent-state recovery findings;
- robustness findings;
- failure analysis;
- alternative explanations;
- interpretation limits;
- formal scientific decision;
- implications for the next phase.

The narrative report should not replace the raw and derived machine-readable evidence.

## Canonical Markdown record

Each phase should also contain an `official-result.md`.

This file should provide the canonical text version of the scientific result.

It should clearly distinguish:

- preregistered or predeclared decision criteria;
- official results;
- exploratory post-hoc analysis;
- interpretation;
- unresolved questions;
- claims that are supported;
- claims that are not supported.

This distinction is especially important when a phase terminates early at a scientific gate.

## Provenance policy

Each phase should contain a `provenance.md` recording, where applicable:

- protocol anchor;
- execution source SHA;
- implementation branch;
- configuration files;
- seed definitions;
- execution command;
- output directory;
- runtime environment;
- Python version;
- PyTorch version;
- CUDA version;
- GPU;
- worker count;
- wall-clock duration;
- shard count;
- cell count;
- metric-row count;
- failures;
- cancellations;
- incomplete cells;
- duplicate-key audit;
- NaN/Inf audit;
- artifact checksums;
- archive checksum;
- archive location.

The purpose is to make every formal result traceable to a specific implementation and execution.

## Integrity and archival policy

The complete raw execution should be preserved separately when it is impractical or undesirable to commit every generated file.

For each such archive:

- create a compressed archive;
- verify archive integrity;
- calculate SHA-256;
- preserve a complete file-level checksum manifest;
- record archive size and source directory;
- record archive identity in `provenance.md`.

The repository should contain enough evidence to verify that the consolidated files committed to Git are byte-identical to the official execution artifacts.

## Phase 0

Phase 0 is the first official synthetic validation cycle.

Its canonical result is:

> The intended extreme-low-N superiority claim was not validated. Flow-Jump showed structured signal and selected higher-N advantages, but the useful crossover occurred later than intended.

Primary low-N result:

- N = 5, 10, 20, 40 treated as the extreme-low-N region;
- 0 / 12 simultaneous mean-MAE wins over both the representation-linear probe and scratch GRU across the primary dynamics-world/N settings.

The result motivated a focused Phase-0.5 mechanistic redesign.

Phase-0 archival materials are stored under:

~~~text
docs/results/phase0/
~~~

## Phase 0.5

Phase 0.5 is the mechanistic redesign cycle following Phase 0.

The official run reached Stage I-A flow mechanism isolation and terminated because neither candidate passed the locked development gate.

The canonical Phase-0.5 record should preserve:

- the protocol lock;
- the Phase-0-derived noise floor;
- the locked minimum relative-effect threshold;
- all 60 executed Stage I-A cells;
- flow gate results;
- bundle-level paired effects;
- N-wise effects;
- MAE and RMSE comparisons;
- latent aligned R² comparisons;
- event Brier results;
- event log-loss results;
- event ROC-AUC results;
- low-N training/validation budget structure;
- the fact that later Phase-0.5 stages were not entered;
- the fact that confirmatory seeds remained untouched;
- exploratory post-gate diagnostics separately from the official decision.

Phase-0.5 archival materials should be stored under:

~~~text
docs/results/phase05/
~~~

## Phase 0.6

Phase 0.6 is the completed diagnostic-decomposition program following the failed Phase 0.5 mechanism gate.

Its principal findings were:

- model initialization was the dominant named source of N=40 variability;
- the terminal D4-B result was heterogeneous across contexts and model seeds;
- the frozen classification was `D4-B AMBIGUOUS -> STOP`;
- no confirmatory rescue was authorized from that result.

The canonical Phase 0.6 archive is stored under:

~~~text
docs/results/phase06/
~~~

A later optimization-conditioned analysis reused the D4-B development pairs only as exploratory evidence. That work remains explicitly post-Phase-0.6 and cannot alter the frozen `AMBIGUOUS -> STOP` decision.

Its archive is stored under:

~~~text
docs/results/phase06_posthoc_optimization/
~~~

## Phase 0.7

Phase 0.7 is the completed prospective optimization-horizon intervention motivated by the exploratory post-Phase-0.6 diagnostics.

The frozen experiment tested whether preventing premature stopping would establish the preregistered positive architecture-by-policy interaction and reduce initialization-linked heterogeneity.

The official result was:

`P07_OPTIMIZATION_HORIZON_NOT_ESTABLISHED`

Primary evidence:

- mean G = 0.0006074243783950794;
- crossed-bootstrap 95% CI = [-0.0022108983993530317, 0.0047499954700469926];
- 1 / 5 positive context means;
- 1 / 10 positive model-seed means;
- 1 / 50 positive paired G effects.

The observed residual heterogeneity ratio was below one, but 11 of 10,000 bootstrap replicates violated the frozen validity condition. Under the preregistered fail-closed rule, the R_SD confidence interval is therefore undefined and the heterogeneity gate cannot be claimed.

A post-adjudication exploratory exposure audit found that the stopping intervention was behaviorally active in only 2 / 100 matched architecture cells. Both activated cells improved under continuation, one in each architecture, but this cannot modify the official Phase 0.7 adjudication.

The supported interpretation is:

> Under the frozen Phase 0.7 intervention, preventing premature termination did not establish the preregistered architecture-relative optimization-horizon effect.

The canonical Phase 0.7 archive is stored under:

~~~text
docs/results/phase07/
~~~

## Research-history principle

Each phase should remain historically immutable once its official scientific decision is recorded.

A failed scientific hypothesis is still a completed and valuable result.

The repository history should therefore preserve:

~~~text
Phase 0
official execution
scientific decision
        ↓
Phase 0.5
official execution
scientific decision
        ↓
Phase 0.6
new diagnostic and redesign cycle
~~~

Later phases may reinterpret earlier evidence, but they must not rewrite the official outcome of the earlier phase.
