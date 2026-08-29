# Phase 0.7 Provenance

## Execution identity

Phase: `phase07`

Execution source commit:

`43a9f75d69250f6421b5d2e8f35f656665abd62e`

Implementation branch:

`phase0-7-optimization-horizon-intervention`

Archival branch:

`phase0-7-results-archive`

Initial evidence archival commit:

`55c5b9c02674e7ec96a17c2d249d9e8a463b8a40`

Narrative report archival commit:

`906de74a65c16740c5874221156240d4631f7cd3`

## Frozen identities

Phase 0.7 plan SHA-256:

`3dbf88a54b982c7c6e73d71027e4f814dc633d660e3f8804a354b9d0d94fab91`

Phase 0.7 config SHA-256:

`d0b3eaf9bddbf4543c83dce61982c1b13c7ca4c2be831570153d3ce43e6617d3`

Phase 0.7 specification SHA-256:

`01fbefb609ec827001776f3b1be90148fe8d5071e44ecc14331aa2a717dbe48f`

Phase 0.5 config SHA-256:

`befd7140cbf68cb981418cdc0c8880bb0652c2a6db614d05f61ad267297d519b`

Simulator config SHA-256:

`f093115ae47274680e8b7cf1c7fea440a739079881099227e5b8c4abc0320aa8`

Protocol lock SHA-256:

`f937661347a8f8ea2caab603c3e60909430093a9d4cf9851763ee4ad3ca2530e`

Official authorization SHA-256:

`031f705e0f577775506f219b4ae3ef895142aaae7e6dfb640463922f17bb4693`

## Official execution

Command:

    afmc-phase07 official \
      --authorization "$P07_ROOT/prep/authorization.json" \
      --output "$P07_ROOT/run" \
      --device cuda

Planned cells: `200`

Completed cells: `200`

Device: `cuda`

GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`

Runtime environment:

- platform: `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.43`
- Python: `3.14.4`
- PyTorch: `2.13.0+cu130`
- CUDA: `13.0`

Official invocation start:

`2026-08-28T19:42:20.122243+00:00`

Official invocation end:

`2026-08-28T20:18:58.970361+00:00`

Recorded wall time:

`2198.8480909380014 seconds`

Failure: `null`

Last attempted cell:

`p07__smooth__cohort410__subset510__model1110__n40__time_scaled__forced_horizon`

## Official analysis

Analysis command:

    afmc-phase07 analyze --output "$P07_ROOT/run"

Frozen adjudication:

`P07_OPTIMIZATION_HORIZON_NOT_ESTABLISHED`

Frozen adjudication SHA-256:

`44c15c7433cd3847d98679a9b7f9739d93440f204c6ac710ba3c386f1606b522`

Official freeze record SHA-256:

`de45ef1f932492a2dd27c72750ec624088aff0418cca674afceba72a87d13696`

## Repository-preserved official artifacts

### Raw

- `raw/phase07-adjudication-frozen.json`
- `raw/phase07-metrics-frozen.csv`

### Execution provenance

- `provenance/authorization.json`
- `provenance/manifest.json`
- `provenance/plan.json`
- `provenance/protocol.json`
- `provenance/phase07-execution-provenance-frozen.json`

### Integrity

- `integrity/phase07-official-file-manifest.sha256`
- `integrity/phase07-freeze-record.json`

The official file-level manifest records the complete official execution tree.

## Exploratory post-hoc evidence

The post-adjudication diagnostic layer is explicitly separate from the official Phase 0.7 result.

Repository-preserved files:

- `tables/G_matrix.csv`
- `tables/cell_execution_summary.csv`
- `tables/context_means.csv`
- `tables/intervention_exposure.csv`
- `tables/model_means.csv`
- `tables/phase07_pair_table.csv`
- `tables/phase07-posthoc-summary.json`

Integrity records:

- `integrity/POSTHOC_SHA256SUMS.txt`
- `integrity/phase07-posthoc-freeze-record.json`

The post-hoc freeze record explicitly states that these diagnostics cannot modify, rescue, or replace the frozen official Phase 0.7 adjudication.

## Evidence boundary

Official execution evidence, official adjudication, and exploratory post-hoc diagnostics are preserved as separate layers.

No post-hoc analysis may:

- change the official adjudication;
- modify the frozen thresholds;
- alter the bootstrap procedure;
- replace the official endpoints;
- substitute different seeds;
- rerun the official experiment as a rescue;
- reinterpret exploratory findings as preregistered evidence.

The execution source SHA remains distinct from later archival documentation commits.
