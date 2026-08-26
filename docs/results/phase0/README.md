# AFMC Phase 0 — Official Synthetic Validation Record

## Status

**COMPLETE**

Phase 0 is the first official synthetic validation cycle for the AFMC low-resource longitudinal clinical foundation-model adaptation programme.

Its purpose was to test whether a compact continuous-time Flow-Jump adapter applied to frozen longitudinal clinical representations could demonstrate a sample-efficiency advantage over simpler probing and scratch temporal baselines before any scarce real clinical dataset is used.

## Scientific decision

**The intended extreme-low-N superiority claim was not validated.**

Across the primary dynamics worlds and training sizes N = 5, 10, 20, and 40, Flow-Jump achieved:

- **0 / 12** simultaneous mean-MAE wins over both the representation-linear probe and scratch GRU.

Flow-Jump did show structured signal at larger training budgets and in selected shifted or structured worlds, but the crossover occurred later than intended.

The official Phase-0 decision was therefore:

> Preserve Phase 0 as a valid negative/mixed methodological result and proceed to a focused Phase-0.5 mechanistic redesign. Do not relabel Phase 0 as successful low-N validation.

## Official execution

- Protocol anchor: `be5a66b2e45362f60c90844e4e25673fb7bb3e21`
- Execution SHA: `d6f105eee73fcb8e9cc5987d292b1bb98a687382`
- Official output directory:
  `outputs/phase0_full_cuda_d6f105eee73fcb8e9cc5987d292b1bb98a687382`
- Shards: 25 / 25 complete
- Cells: 2,730 / 2,730 complete
- Metric rows: 23,520
- Failed cells: 0
- Cancelled cells: 0
- Incomplete cells: 0
- Duplicate scientific keys: 0
- NaN / infinite metric values: 0 / 0

## Canonical records

### Human-readable report

`AFMC_Phase0_Synthetic_Validation_Report.docx`

This is the full narrative results and decision report.

### Scientific result

`official-result.md`

Canonical text record of the scientific findings, ablations, limitations, and Phase-0 decision.

### Provenance

`provenance.md`

Execution environment, source-code provenance, output identity, integrity checks, and archival information.

### Machine-readable evidence

`raw/`, `tables/`, `figures/`, and `integrity/`

Contains the consolidated official metrics, ablation results, gate summary, run records, learning curves, and checksums required to audit the result.

## Evidence hierarchy

The evidence associated with Phase 0 is interpreted in the following order:

1. immutable official raw output;
2. consolidated official raw and derived machine-readable evidence;
3. provenance and checksum records;
4. canonical Markdown scientific result;
5. narrative Word report.

The Word report and Markdown summaries interpret the official output; they do not replace it.

## Transition to Phase 0.5

Phase 0 identified several specific issues requiring mechanistic redesign:

- the low-N crossover occurred too late;
- model capacity remained a plausible alternative explanation;
- the explicit jump mechanism was not validated in the dedicated jumps world;
- the probabilistic-scale pathway imposed a point-prediction penalty;
- the observation-process head failed its intended robustness role;
- seed-level paired inference was needed rather than reliance on aggregate mean curves.

These findings define the evidence boundary from which Phase 0.5 begins.
