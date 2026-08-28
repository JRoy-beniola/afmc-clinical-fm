# AFMC Reproducibility Closure Through Post-Phase-0.6

Status: **DRAFT FOR FINAL WRITTEN REVIEW — DESIGN APPROVED IN CHAT; IMPLEMENTATION NOT YET STARTED**

Date: 2026-08-27

## Purpose

Close the reproducibility gap across the AFMC research line from Phase 0 through Phase 0.5, Phase 0.6, and the post-Phase-0.6 optimization-conditioned analysis.

The goal is not merely to preserve code and final reports. The repository must support four distinct reproducibility questions:

1. **Computational** — can the historical experiment be rerun from the historically bound implementation/configuration?
2. **Analytical** — can archived raw evidence deterministically regenerate the reported tables and figures?
3. **Scientific** — does a fresh rerun preserve the same preregistered scientific decision?
4. **Documentary** — can the narrative report, tables, figures, diagrams, captions, and structure be regenerated in essentially the same document form?

This work is infrastructure only. It must not alter any historical scientific result, change the frozen Phase 0.6 terminal decision, or authorize Phase 0.7 execution.

## Historical scientific boundary

The following historical records remain immutable scientific evidence:

- `docs/results/phase0/`
- `docs/results/phase05/`
- `docs/results/phase06/`
- `docs/results/phase06_posthoc_optimization/`

Phase 0.6 remains exactly `AMBIGUOUS -> STOP`.

The post-Phase-0.6 analysis remains exploratory and retains the classification:

`structured optimization-conditioned heterogeneity worth prospective testing`

No reproducibility command may revise those outcomes or promote post-hoc evidence into confirmatory evidence.

Protected confirmatory seeds remain forbidden for any reproduction path that is not an explicitly authorized historical rerun of a phase that originally used them.

## Design principles

### 1. Preserve history, add a thin reproducibility layer

Existing historical tools are provenance-bearing artifacts and must not be rewritten merely to create a cleaner interface:

- `tools/analysis/`
- `tools/execution/`
- `tools/monitoring/`

The new system wraps and records them where appropriate.

### 2. One supported interface

The public interface is a phase-aware CLI:

```bash
afmc-reproduce status

afmc-reproduce verify phase0
afmc-reproduce verify phase05
afmc-reproduce verify phase06
afmc-reproduce verify phase06-posthoc
afmc-reproduce verify all

afmc-reproduce rebuild phase06
afmc-reproduce rebuild all

afmc-reproduce rerun phase06
```

Historical scripts remain directly callable, but `afmc-reproduce` is the supported entry point for future users.

### 3. Separate verification, rebuilding, and rerunning

The verbs have intentionally different semantics:

- `verify`: read-only validation of official archived evidence, manifests, provenance, hashes, expected artifacts, and source bindings.
- `rebuild`: regenerate documentary and analytical artifacts from committed archived evidence without rerunning training.
- `rerun`: execute historical computation in an isolated historical implementation environment and compare new results against the official archive.

No command may silently substitute one mode for another.

### 4. Official archives are never overwritten by reproduction

Historical official archives are read-only inputs.

All regenerated outputs go under:

```text
outputs/reproduction/<phase>/
```

A successful rebuild or rerun does not make its output official.

For future phases, promotion into `docs/results/<phase>/` requires a separate explicit archival/freeze operation with validation and provenance capture. Historical Phase 0, 0.5, and 0.6 records cannot be replaced through this workflow.

## Documentary reproduction standard

The selected standard is **document reproduction**, not byte-for-byte identity.

A historical report rebuild must aim to preserve:

- exact historical narrative wording;
- heading and section ordering;
- table content and logical formatting;
- figure and diagram placement/order;
- captions;
- references to evidence and decisions;
- phase-specific Word styling as closely as practical.

Binary identity of `.docx` files is not required because Word/OpenXML serialization, font engines, renderer versions, and metadata can differ across environments.

## Authoring model

### Historical phases

For Phase 0, Phase 0.5, and Phase 0.6:

- the existing official `.docx` remains the immutable reference rendering;
- the historical wording is preserved exactly;
- a one-time extraction/bootstrap creates version-controlled report source;
- extraction does not change the scientific record;
- the extracted source is audited against the official report and then frozen as reproducibility source.

### Post-Phase-0.6 and future phases

Reports become source-first.

Canonical source components:

```text
report-source.md
report.yaml
artifact bindings
tables/*.csv
figures/*
diagrams/*
template.docx
```

The document builder consumes those sources and creates the candidate `.docx` under `outputs/reproduction/...` or a future pre-archive output location.

## Report source format

Markdown is the narrative source of truth.

A phase manifest (`report.yaml` or equivalent validated structured representation) records:

- phase identifier;
- report title;
- reference report path where applicable;
- reference report SHA-256;
- report-source path and SHA-256;
- implementation commit;
- execution commit where distinct;
- protocol/spec/config paths and hashes;
- official evidence roots;
- table declarations;
- figure declarations;
- diagram declarations;
- expected section/caption ordering;
- template path;
- renderer/tool requirements;
- environment status;
- known historical reconstruction limitations.

The manifest must be machine-validated before rebuild.

## Historical report bootstrapping

Historical `.docx` reports are converted once into canonical reproducibility source.

The bootstrap records at minimum:

- source document SHA-256;
- extraction timestamp;
- extraction tool/version;
- normalization rules;
- heading sequence;
- paragraph text;
- table count, dimensions, and cell text;
- figure/caption ordering;
- section structure where recoverable;
- asset references.

The extracted source is reviewed against the immutable official report.

After acceptance, routine rebuilds are source-to-document only; reverse extraction is not part of normal future report generation.

## Diagram and figure reproducibility

Scientific plots and conceptual diagrams are separate artifact classes.

### Scientific figures

Use Python/Matplotlib or existing historically bound plotting code.

Every generated scientific figure must declare:

- evidence inputs;
- generator entry point;
- generator version/commit;
- output path;
- deterministic parameters where applicable.

### Conceptual diagrams

PlantUML, Mermaid, and Graphviz DOT are all permitted.

Each diagram must keep editable text source in Git. Rendered PNG/SVG/PDF alone is insufficient as canonical source.

The phase manifest records:

- source path;
- renderer type;
- renderer/tool version where known;
- output path(s).

The system must not require one diagram language across all historical/future artifacts.

## Proposed repository layout

```text
docs/
├── results/                         # immutable official evidence
│   ├── phase0/
│   ├── phase05/
│   ├── phase06/
│   └── phase06_posthoc_optimization/
│
└── reproducibility/
    ├── README.md
    ├── research-lineage.md
    ├── artifact-map.yaml
    ├── phase0/
    │   ├── report-source.md
    │   ├── report.yaml
    │   ├── template.docx
    │   └── diagrams/
    ├── phase05/
    ├── phase06/
    └── phase06_posthoc/

tools/
├── analysis/                        # historical — preserve
├── execution/                       # historical — preserve
├── monitoring/                      # historical — preserve
└── reproducibility/
    ├── verify.py
    ├── rebuild.py
    ├── rerun.py
    ├── reports.py
    ├── diagrams.py
    ├── comparison.py
    └── historical_environment.py
```

Exact internal filenames may be adjusted during implementation if the existing package structure makes another placement cleaner, but the separation of historical tools, reproducibility sources, and immutable official evidence is mandatory.

## `verify` semantics

`afmc-reproduce verify <phase>` is read-only.

It validates, as available for the phase:

1. expected official archive paths exist;
2. historical manifests/checksum files validate;
3. protocol/config/spec files referenced by provenance exist;
4. recorded Git SHAs are syntactically valid and resolvable when Git history is available;
5. raw evidence expected by downstream analysis exists;
6. report source and artifact-map references resolve;
7. declared table/figure/diagram sources exist;
8. frozen decision records match the expected historical classification;
9. no rebuild output is being mistaken for official evidence.

The command returns a structured machine-readable report in addition to concise terminal output.

## `rebuild` semantics

`afmc-reproduce rebuild <phase>` must not run model training.

It operates only from the archived official evidence plus committed documentary source.

A rebuild may:

- regenerate deterministic derived tables;
- regenerate scientific figures;
- render conceptual diagrams from committed source;
- assemble the report source with bound artifacts;
- generate a candidate `.docx`;
- compare its normalized structure/content against the official historical `.docx` where a reference exists;
- write a reproduction report.

Outputs are written beneath:

```text
outputs/reproduction/<phase>/rebuild/
```

## `rerun` semantics

Historical reruns use a hybrid reproducibility strategy.

### Historical implementation binding

The recorded historical implementation commit is mandatory.

The rerun runner must execute in an isolated Git worktree or equivalent isolated checkout at that commit. It must not silently use current `HEAD` for a historical rerun.

### Environment binding

If a historically pinned environment/container exists, use it.

If not, reconstruct the environment from recorded project metadata and clearly classify it as **environment-reconstructed**, not environment-identical.

No retroactive reconstruction may be presented as an exact historical environment snapshot.

### Output isolation

Fresh rerun outputs go beneath:

```text
outputs/reproduction/<phase>/rerun/<run-id>/
```

They must never write into `docs/results/`.

## Layered rerun comparison

A rerun is evaluated at four layers.

### Layer 1 — structural identity

Check, where applicable:

- same planned cells;
- same seed sets;
- same configurations;
- same row/key sets;
- same artifact schemas;
- same stopping/gating semantics.

### Layer 2 — exact deterministic identity

Use exact comparison or hashes for deterministic objects such as:

- protocol locks;
- canonical manifests;
- configuration snapshots;
- deterministic adjudication metadata;
- derived files explicitly known to be bitwise deterministic.

### Layer 3 — numerical reproduction

Compare floating-point scientific outputs using phase-defined `atol`/`rtol` rather than a single global tolerance.

Checks may include:

- metric values;
- paired effects;
- signs/orderings where scientifically relevant;
- gate inputs;
- confidence interval endpoints within declared tolerance.

### Layer 4 — scientific equivalence

The preregistered/frozen scientific decision must resolve identically under the rerun evidence.

A rerun therefore receives one of these top-level verdicts:

- `EXACT`
- `NUMERICALLY_REPRODUCED`
- `SCIENTIFICALLY_REPRODUCED`
- `FAILED_REPRODUCTION`

The report must expose lower-layer failures even when the top-level scientific conclusion is preserved.

## Phase-specific adapters

The unified CLI does not require every historical phase to have identical internals.

Each phase gets an adapter that declares:

- archive root;
- official report path;
- historical implementation/execution commits where known;
- configs/specs/protocols;
- verification hooks;
- rebuild hooks;
- rerun hooks;
- comparison policy;
- tolerances;
- expected historical decision.

This allows Phase 0, 0.5, 0.6, and post-0.6 to retain their authentic historical machinery while presenting one consistent user interface.

## Research lineage

`docs/reproducibility/research-lineage.md` must describe the causal/documentary history rather than only list files.

At minimum it must map:

```text
Phase 0
  -> Phase 0.5 mechanistic redesign
  -> Phase 0.6 diagnostic validation
  -> D2-B / D4-B model-seed robustness
  -> Phase 0.6 AMBIGUOUS -> STOP
  -> post-Phase-0.6 optimization-conditioned exploratory analysis
  -> prospective Phase 0.7 design (draft, not authorized)
```

Each transition records the deciding artifact/spec/decision and relevant commit references.

## Artifact map

`docs/reproducibility/artifact-map.yaml` is the cross-phase index.

It maps each phase to:

- official evidence root;
- official report;
- canonical decision record;
- protocol/design/config;
- implementation/execution SHAs;
- historical tools;
- raw evidence;
- derived tables;
- figures;
- report source;
- environment metadata;
- integrity manifests.

The artifact map is descriptive and must not override phase-specific frozen protocol semantics.

## Environment capture going forward

Retroactive historical environments may be incomplete.

For post-0.6/future work, the workflow should capture at official execution time:

- Python version;
- package/dependency snapshot;
- OS/platform information;
- CUDA version where applicable;
- PyTorch/runtime accelerator information;
- GPU model when relevant;
- Git implementation SHA;
- config/protocol hashes;
- command line or invocation metadata.

A stronger lock/container can be added prospectively without pretending that equivalent historical snapshots existed.

## Testing requirements

Implementation follows TDD.

Minimum tests must cover:

1. CLI parser exposes `status`, `verify`, `rebuild`, and `rerun`;
2. unknown phases are rejected;
3. verification is read-only;
4. rebuild never writes under `docs/results/`;
5. rerun never uses current HEAD when historical commit binding is required;
6. rerun output is isolated;
7. historical environment reconstruction is labeled correctly;
8. manifest/schema validation rejects missing or inconsistent artifact bindings;
9. phase decision guards detect a mismatch between archived and expected historical classification;
10. layered comparison produces deterministic verdicts;
11. diagram source declarations require an editable source file;
12. historical official archives remain byte-identical before/after verify/rebuild tests;
13. a small fixture phase can complete verify + rebuild without model training;
14. report structural comparison detects changed narrative/table/caption ordering;
15. existing test suite remains green.

Historical experiment reruns themselves are not required inside ordinary CI if they are computationally expensive. CI should test orchestration against fixtures/smokes.

## Historical archive safety test

Before and after reproducibility implementation, compute or reuse integrity manifests for the frozen historical result roots.

At minimum, implementation verification must prove that no file under:

```text
docs/results/phase0/
docs/results/phase05/
docs/results/phase06/
```

was modified by the reproducibility work.

The post-Phase-0.6 archive may gain reproducibility metadata only outside its already archived analysis payload, and only if explicitly designed; the safer default is to keep all new reproducibility metadata under `docs/reproducibility/`.

## Scope exclusions

This project does not:

- change Phase 0/0.5/0.6 scientific conclusions;
- rerun official experiments merely to populate the reproducibility source tree;
- redesign historical analysis methods;
- reformat historical reports into a new common visual style;
- claim byte-identical Word reproduction;
- claim historical environment identity where no snapshot exists;
- implement or execute Phase 0.7;
- merge or mark the post-Phase-0.6 PR ready as a side effect;
- alter protected confirmatory seed firewalls.

## Implementation sequence

Implementation should be staged so that useful closure arrives early:

1. add schemas/phase registry and read-only `status`/`verify`;
2. add cross-phase artifact map and research lineage;
3. add historical report-source bootstrap/extraction tooling;
4. bootstrap and audit Phase 0, 0.5, and 0.6 report sources;
5. add deterministic table/figure/diagram rebuild adapters using existing tools where possible;
6. add DOCX assembly and normalized structural comparison;
7. add `rebuild` orchestration;
8. add isolated historical `rerun` orchestration and environment capture/reconstruction labeling;
9. add layered numerical/scientific comparison;
10. close post-Phase-0.6 documentary source and make source-first reporting the default for future phases.

No official Phase 0.7 training may begin merely because some subset of this sequence is complete; Phase 0.7 retains its separate review/freeze/authorization gate.

## Acceptance criteria

Reproducibility closure through post-Phase-0.6 is complete when:

1. `afmc-reproduce verify all` validates the historical archive chain without modifying it;
2. every phase has an artifact-map entry with evidence, decision, provenance, implementation/configuration references, and documentary source status;
3. Phase 0, Phase 0.5, and Phase 0.6 have canonical historical `report-source.md` files whose wording matches the official reports under the declared normalization rules;
4. historical reports can be rebuilt into `outputs/reproduction/...` with structural/content comparison reports;
5. existing scientific figures/tables are regenerable where historical code/evidence supports them, with gaps explicitly recorded rather than fabricated;
6. conceptual diagrams used by reproducible reports have committed editable source;
7. `afmc-reproduce rerun <supported-phase>` uses the historically bound implementation in isolation and produces a layered reproduction report;
8. historical environment status is truthful (`historically pinned` vs `environment-reconstructed`);
9. frozen archives remain unchanged;
10. the full existing test suite plus reproducibility tests is green;
11. post-Phase-0.6/future reporting has a source-first path so this reverse-bootstrap process does not need to be repeated.

## Non-negotiable scientific invariant

Reproducibility infrastructure may explain, regenerate, validate, and rerun the historical research record. It may not rewrite that record.
