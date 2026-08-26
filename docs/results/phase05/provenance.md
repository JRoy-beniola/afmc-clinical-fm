# Phase 0.5 Execution and Archival Provenance

## Execution identity

Official implementation SHA:

`50a94c06bc1c419ca55738f15f074cc06ccc3f36`

Official output identity:

`phase05_official_50a94c06bc1c419ca55738f15f074cc06ccc3f36`

Phase 0 execution anchor:

`d6f105eee73fcb8e9cc5987d292b1bb98a687382`

Phase 0 metrics SHA-256:

`f829502bc3489b3608c1ba06c9b4ca0c0910872c88ee628325339232c6816dcf`

## Locked protocol identity

Phase 0.5 configuration SHA-256:

`befd7140cbf68cb981418cdc0c8880bb0652c2a6db614d05f61ad267297d519b`

Protocol-lock SHA-256:

`33b25ebe46620626ae3d1bbbd6152fc3aacdb0e6200dd87e03c4fdec99e70db9`

Specification SHA-256:

`c32dae3d702991e47387e1523004c9c02141ae331891a6b7a7e422079d72283e`

Simulator configuration SHA-256:

`f093115ae47274680e8b7cf1c7fea440a739079881099227e5b8c4abc0320aa8`

Locked Phase-0-derived noise floor:

`0.04156685121568672`

The same value was used as the minimum relative flow-gate effect.

## Development bundles

The locked development bundles were:

- `(401, 501, 601)`
- `(402, 502, 602)`
- `(403, 503, 603)`
- `(404, 504, 604)`
- `(405, 505, 605)`

The protocol also locked ten confirmatory bundles from
`(701, 801, 901)` through `(710, 810, 910)`.

Those confirmatory bundles were not entered by the official Phase 0.5
execution.

## Execution environment

The Stage I-A execution used:

- WSL2 Linux
- Python 3.14.4
- PyTorch 2.13.0+cu130
- CUDA runtime 13.0
- NVIDIA GeForce RTX 4060 Laptop GPU
- compute capability 8.9
- 8,188 MiB reported GPU memory
- one worker

Recorded library versions included:

- NumPy 2.5.2
- pandas 3.0.5
- SciPy 1.18.1
- scikit-learn 1.9.0
- matplotlib 3.11.1

## Official execution record

The flow invocation began at:

`2026-08-26T01:55:42.302067+00:00`

and ended at:

`2026-08-26T02:04:23.535641+00:00`

Recorded wall time:

`521.2366377040016 seconds`

Expected cells:

`60`

Completed before invocation:

`0`

Completed after invocation:

`60`

Recorded execution failures:

`0`

The persisted output contains exactly 60 Stage I-A cell JSON files.

Each cell contains six test metrics:

- MAE
- RMSE
- event ROC-AUC
- event Brier score
- event log-loss
- latent aligned R2

The complete flattened metric surface therefore contains 360 rows.

## Launcher exit semantics

The launcher log ends with:

`RuntimeError: flow development gate failed; development stopped`

and returns exit code 1.

This is the implementation of the preregistered scientific stop rule.
It must not be interpreted as an engineering crash.

The Stage I-A flow directory contains a `COMPLETE` marker, the execution
provenance records 60 of 60 cells completed with no failures, and
`development/flow_gate.csv` contains the completed gate evaluation.

## Repository-side raw preservation

The complete official output tree is preserved byte-for-byte under:

`docs/results/phase05/raw/official_output/`

The original-output SHA-256 manifest is:

`docs/results/phase05/integrity/full_output_checksums.sha256`

The official launcher log is separately preserved as:

`docs/results/phase05/raw/official_stage1.log`

Launcher-state evidence is preserved under:

`docs/results/phase05/raw/control/`

## Derived evidence

`tools/analysis/phase05/generate_tables.py` reconstructs the persisted
metric surface and independently recomputes the official flow gate using
the repository's Phase 0.5 nAULC and paired-development-gate definitions.

The resulting repository tables are:

- `flow_cell_metrics.csv`
- `flow_gate_verified.csv`
- `flow_bundle_effects.csv`
- `flow_n_effects.csv`
- `flow_metric_summary.csv`

The gate-verification table is intended as an integrity check.

The other effect tables are exploratory postmortem evidence and do not
change the preregistered decision.

## External archive

The complete original output directory is additionally preserved outside
Git as:

`phase05_official_50a94c.tar.zst`

Its SHA-256 is recorded in:

`docs/results/phase05/integrity/archive_sha256.txt`

The archive was validated with `zstd -t`.

## Reproducibility boundary

Phase 0.5 preserves all persisted Stage I-A result cells, execution
identity, protocol lock, gate output, runtime provenance, launcher log,
derived tables, and operational tooling.

The execution did not persist training trajectories or epoch-level
optimization diagnostics. Therefore Phase 0.5 cannot retrospectively
establish whether the observed seed/N heterogeneity arose primarily from
optimization, subset composition, cohort composition, checkpoint
selection, or another mechanism.

That question belongs to the subsequent diagnostic phase.

## Historical immutability

This Phase 0.5 result is a historical research boundary.

Later diagnostic work may explain the result or motivate a redesigned
mechanism, but it must not rewrite the official Phase 0.5 gate outcome.
