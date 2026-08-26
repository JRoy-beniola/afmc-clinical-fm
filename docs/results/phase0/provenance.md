# AFMC Phase 0 — Execution and Archival Provenance

## 1. Official execution identity

| Field | Value |
|---|---|
| Protocol anchor | `be5a66b2e45362f60c90844e4e25673fb7bb3e21` |
| Execution SHA | `d6f105eee73fcb8e9cc5987d292b1bb98a687382` |
| Official output directory | `outputs/phase0_full_cuda_d6f105eee73fcb8e9cc5987d292b1bb98a687382` |
| Execution date | 23 August 2026 |
| Runtime environment | Native WSL2 |
| CUDA runtime | 13.0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Worker count | 2 |
| Wall time | 03 h 40 m 02.49 s |

Phase 0 is treated as an immutable official synthetic execution.

Later archival and analysis utilities may reorganize or derive evidence from the run, but they do not alter the scientific execution or its result.

---

## 2. Completion and integrity audit

| Item | Result |
|---|---:|
| Shards | 25 / 25 |
| Cells | 2,730 / 2,730 |
| Metric rows | 23,520 |
| Failed cells | 0 |
| Cancelled cells | 0 |
| Incomplete cells | 0 |
| Duplicate scientific keys | 0 |
| NaN / infinite metric values | 0 / 0 |

The official run therefore reached a complete and internally consistent terminal state.

The negative Phase-0 scientific result is not attributable to interrupted or incomplete execution.

---

## 3. Repository evidence layout

The Phase-0 evidence package is stored under:

~~~text
docs/results/phase0/
├── README.md
├── official-result.md
├── provenance.md
├── AFMC_Phase0_Synthetic_Validation_Report.docx
├── raw/
├── tables/
├── figures/
└── integrity/
~~~

The directories have distinct roles.

### `raw/`

Contains machine-readable scientific evidence derived directly from, or preserved directly from, the official execution.

### `tables/`

Contains reproducibly derived comparison and validation tables.

### `figures/`

Contains figures used in scientific interpretation.

### `integrity/`

Contains cryptographic checksum and archival-integrity records.

---

## 4. Canonical raw result files

The principal consolidated files copied directly from the official execution are:

- `raw/metrics.csv`
- `raw/ablation_metrics.csv`
- `raw/gate_summary.csv`
- `raw/run_manifest.json`
- `raw/run_record.json`

The following official figure was also copied directly:

- `figures/learning_curves.png`

These files were verified byte-for-byte against the corresponding files in the official execution output before archival.

---

## 5. Original official artifact hashes

The Phase-0 audit recorded the following SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `run_record.json` | `93de1112ad855388a4d422a06249575cbc5de569a7b9a498e4d7a25f18a87b82` |
| `run_manifest.json` | `1a85f24e64b18c1728ef1e811a3e35ae16507b66373da29d8004b1961e9c8a08` |
| `metrics.csv` | `f829502bc3489b3608c1ba06c9b4ca0c0910872c88ee628325339232c6816dcf` |
| `ablation_metrics.csv` | `0ea780e62ef39d4024627eddbe1a83f5c48ba838dd763802d69a834e7b4416f6` |
| `gate_summary.csv` | `a8d68248da47c7803b5cc6aaa2465ea682802c85a49a61b05c9ce0f5eecf7b06` |
| `learning_curves.png` | `888f5cec421e0e5341d90fd62b8ba82df8b8c498c87e25e69620f57fa96ad729` |

---

## 6. Cell-level result preservation

The original execution persisted one JSON payload for every completed scientific cell under:

~~~text
outputs/.../shards/<shard>/cells/<cell>.json
~~~

There are exactly:

- 2,730 persisted cell JSON files;
- 23,520 metric rows across those files.

For repository preservation, the complete set of cell payloads was consolidated into:

- `raw/cell_results.jsonl`

One canonical JSON object is stored per line.

The original cell-to-file mapping is preserved in:

- `raw/cell_index.csv`

The index records, where available:

- original relative path;
- original cell-file SHA-256;
- shard identity;
- world;
- cohort seed;
- subset seed;
- model seed;
- cell ID;
- benchmark;
- training size;
- model;
- ablation;
- metric-row count.

A machine-readable archive description is stored in:

- `raw/cell_archive_summary.json`

---

## 7. Cell-to-metrics equivalence audit

The 2,730 persisted cell payloads were independently read and their metric rows reconstructed.

Recovered metric-row count:

**23,520**

The reconstructed rows were compared against:

- `raw/metrics.csv`

using all scientific columns.

Every non-value scientific key matched uniquely.

The initial comparison using Pandas' default CSV floating-point parser showed 8,135 apparent mismatches caused solely by floating-point parsing differences.

Observed numerical differences under the default parser were extremely small:

- median absolute difference approximately `5.55e-17`;
- maximum absolute difference approximately `1.14e-13`.

The comparison was repeated using:

~~~python
pd.read_csv(..., float_precision="round_trip")
~~~

Under round-trip parsing:

- unmatched scientific keys: 0;
- duplicate non-value keys: 0;
- unequal metric values: 0;
- exact matching metric values: 23,520 / 23,520.

Therefore:

> The archived cell payloads reproduce `metrics.csv` exactly as a multiset of all 23,520 official scientific metric rows.

This is an exact serialization-equivalence result, not a tolerance-based numerical comparison.

---

## 8. Derived analysis tables

The Phase-0 repository contains derived CSV tables generated from the official persisted results.

The generator is:

~~~text
tools/analysis/phase0/generate_tables.py
~~~

The generated tables include:

- complete full-model test results;
- all-metric summaries;
- MAE comparisons by world, N, and model;
- seed-level MAE;
- paired primary-comparator effects;
- low-N win matrices;
- verified gate summaries;
- learning-curve summaries;
- parameter-count comparisons;
- training/validation budget summaries;
- calibration summaries;
- latent-state recovery summaries;
- event-prediction summaries;
- site-shift summaries;
- misspecification summaries;
- seed-level ablation effects;
- aggregate ablation summaries;
- MAE-specific ablation summaries;
- low-N all-metric summaries;
- Phase-0 headline MAE results.

The generator independently verifies the principal Phase-0 low-N result:

- 12 primary dynamics-world/N settings;
- 0 / 12 simultaneous Flow-Jump wins over both primary comparators.

Derived tables are interpretations of official persisted evidence and are not treated as independent executions.

---

## 9. Cell archival utility

The repository-level cell archive is generated by:

~~~text
tools/analysis/phase0/archive_cells.py
~~~

The utility:

1. discovers all original persisted cell JSON files;
2. verifies the expected count of 2,730;
3. verifies the expected 23,520 metric rows;
4. creates `cell_results.jsonl`;
5. creates `cell_index.csv`;
6. checks uniqueness of cell identities;
7. verifies exact equivalence with `metrics.csv`;
8. writes an archival summary;
9. writes cryptographic checksums for the Git-facing result archive.

The resulting checksum record is:

~~~text
integrity/github_result_archive_sha256.txt
~~~

---

## 10. Full raw-output archive

The complete official Phase-0 output directory was additionally archived outside Git as:

~~~text
phase0_official_d6f105.tar.zst
~~~

Source output size:

approximately 20 MiB.

Compressed archive size:

approximately 1.1 MiB.

The archive passed:

~~~text
zstd -t
~~~

integrity verification.

Its SHA-256 is stored in:

~~~text
integrity/archive_sha256.txt
~~~

The compressed archive itself is intentionally not committed to Git.

---

## 11. Complete raw-output checksum manifest

A SHA-256 manifest covering every file in the original Phase-0 output tree is stored as:

~~~text
integrity/full_output_checksums.sha256
~~~

This permits a future extracted or transferred raw execution to be checked against the original official filesystem contents.

---

## 12. Synthetic-data preservation boundary

Phase 0 generated synthetic longitudinal cohorts during execution, but the official run did not persist the generated patient-level synthetic timelines as standalone dataset artifacts.

The official output contains:

- consolidated result CSVs;
- run records;
- run manifests;
- gate summaries;
- per-cell result JSON payloads;
- figures;
- execution metadata.

It does not contain a separately serialized patient-level synthetic cohort dataset, train/test observation table, `.parquet`, `.npy`, `.npz`, or equivalent dataset artifact.

Therefore the Phase-0 repository must not claim byte-for-byte preservation of the generated synthetic patient trajectories themselves.

The preserved scientific evidence consists of the official execution results and provenance required to identify how those results were produced.

Future formal phases should explicitly persist, where practical:

- generated-cohort manifests;
- train/validation/test split manifests;
- synthetic observation data;
- simulator latent truth;
- seed assignments;
- dataset-level checksums.

---

## 13. Storage policy

Git stores the scientifically useful Phase-0 evidence surface, including:

- narrative report;
- canonical result record;
- provenance record;
- consolidated raw result CSVs;
- run manifest;
- run record;
- all 2,730 cell payloads in canonical JSONL form;
- cell index;
- cell archive summary;
- derived comparison tables;
- validation tables;
- figures;
- checksum records;
- reproducible archival and analysis utilities.

Git does not duplicate the complete original raw execution directory file-for-file.

That directory is preserved separately as the compressed raw-output archive.

---

## 14. Reproducibility boundary

A future reproduction should distinguish between:

### Archival equivalence

Whether preserved artifacts are identical to the official execution artifacts.

### Computational reproduction

Whether rerunning the same implementation and protocol recreates equivalent outputs.

### Scientific reproduction

Whether the same qualitative scientific conclusions survive rerunning under appropriately controlled stochastic and computational conditions.

The archived Phase-0 execution remains the authoritative result.

A future run does not replace it.

---

## 15. Historical immutability

Phase 0 should remain permanently identifiable as:

> `PHASE-0 COMPLETE — EXTREME-LOW-N HYPOTHESIS NOT VALIDATED`

Later phases may investigate, explain, or overcome the Phase-0 result.

They must not rewrite the official Phase-0 result itself.
