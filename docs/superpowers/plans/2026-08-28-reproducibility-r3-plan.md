# Reproducibility R3: Deterministic Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic analytical/documentary rebuild support that consumes immutable archived evidence plus R2 report source, writes only under `outputs/reproduction/<phase>/rebuild/`, assembles candidate DOCX reports, and compares their normalized structure against the immutable reference reports.

**Architecture:** R3 introduces explicit rebuild manifests and adapter modes rather than pretending every historical artifact has the same generator. Historical artifacts with a validated deterministic generator use that generator; reference-only legacy artifacts are copied into the reproduction workspace and labeled `reference-copy`, never presented as regenerated. A source-driven DOCX builder consumes the canonical R2 Markdown and artifact declarations, and a structural comparator uses the R2 OOXML extractor for normalized candidate/reference comparison.

**Tech Stack:** Python 3.11, PyYAML, python-docx, existing pandas/matplotlib tools, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-27-reproducibility-closure-design.md`

## Global Constraints

- Historical `docs/results/...` roots remain immutable and are read-only inputs.
- Every rebuild output must be nested beneath `outputs/reproduction/<phase>/rebuild/`.
- `rebuild` never runs model training.
- A reference-copy artifact must never be labeled regenerated.
- Official DOCX reports remain the immutable documentary references.
- Document reproduction is structural/content reproduction, not byte-for-byte DOCX identity.
- Phase 0.6 remains `D4-B AMBIGUOUS -> STOP`; no rebuild operation may change a scientific conclusion.

---

### Task 1: Rebuild manifest schema and output firewall

**Files:**
- Create: `src/afmc_fm/reproducibility/rebuild_models.py`
- Create: `tests/reproducibility/test_rebuild_models.py`

**Interfaces:**
- Produces: `ArtifactDeclaration(kind, source, output, mode, generator=None)`
- Produces: `RebuildManifest(phase_id, report_source, reference_report, artifacts, document_output)`
- Produces: `load_rebuild_manifest(path: Path) -> RebuildManifest`
- Produces: `validate_rebuild_destination(root: Path, phase_id: str, destination: Path) -> Path`

- [ ] **Step 1: Write failing destination-firewall tests**

```python
@pytest.mark.parametrize(
    "bad",
    [Path("docs/results/phase0"), Path("docs/reproducibility/phase0"), Path("outputs/phase0")],
)
def test_rebuild_destination_must_be_isolated(tmp_path, bad):
    with pytest.raises(ValueError):
        validate_rebuild_destination(tmp_path, "phase0", tmp_path / bad)


def test_default_rebuild_destination_is_reproduction_tree(tmp_path):
    assert validate_rebuild_destination(tmp_path, "phase0", None) == (
        tmp_path / "outputs/reproduction/phase0/rebuild"
    )
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_rebuild_models.py -q`

- [ ] **Step 3: Implement strict schema/loading**

Allowed artifact modes are exactly `generate`, `reference-copy`, and `source-render`. `generate` requires a dotted generator entry point; `reference-copy` forbids one; `source-render` is reserved for diagrams/documents generated from committed editable source. Reject absolute artifact paths and any `..` traversal.

- [ ] **Step 4: Add malformed-manifest tests and verify GREEN**

Run: `pytest tests/reproducibility/test_rebuild_models.py -q && ruff check src tests`

- [ ] **Step 5: Commit**

```bash
git add src/afmc_fm/reproducibility/rebuild_models.py tests/reproducibility/test_rebuild_models.py
git commit -m "feat: define reproducibility rebuild manifests"
```

---

### Task 2: Deterministic artifact adapters

**Files:**
- Create: `src/afmc_fm/reproducibility/artifacts.py`
- Create: `tests/reproducibility/test_artifacts.py`

**Interfaces:**
- Produces: `ArtifactResult(output: Path, mode: str, sha256: str, generated: bool)`
- Produces: `materialize_artifact(root: Path, destination: Path, declaration: ArtifactDeclaration) -> ArtifactResult`
- Produces generator protocol: `generator(root: Path, source: Path, output: Path) -> None`

- [ ] **Step 1: Write failing mode tests**

Test that `reference-copy` performs a byte-identical copy into the reproduction directory and returns `generated=False`; test that `generate` loads a fixture dotted entry point, writes deterministic output, and returns `generated=True`; test that generators are rejected if they attempt to escape the declared destination.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_artifacts.py -q`

- [ ] **Step 3: Implement adapters**

Use `importlib` for dotted generator loading, `shutil.copy2` for reference-copy, and SHA-256 over final bytes. For generated output, create a temporary sibling file and atomically replace the destination. Before and after invoking a generator, snapshot `docs/results/` and raise `RuntimeError` if any historical archive byte changes.

- [ ] **Step 4: Add deterministic-repeat test**

Run the same fixture generator twice into separate destinations and require matching SHA-256.

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_artifacts.py -q && ruff check src tests`

```bash
git add src/afmc_fm/reproducibility/artifacts.py tests/reproducibility/test_artifacts.py
git commit -m "feat: add deterministic rebuild artifact adapters"
```

---

### Task 3: Source-driven DOCX builder

**Files:**
- Modify: `pyproject.toml`
- Create: `src/afmc_fm/reproducibility/documents.py`
- Create: `tests/reproducibility/test_documents.py`

**Interfaces:**
- Dependency: `python-docx>=1.1`
- Produces: `build_docx(report_source: Path, output: Path, *, reference: Path | None = None) -> Path`
- Produces: `parse_canonical_markdown(text: str) -> tuple[DocumentNode, ...]`

- [ ] **Step 1: Write failing parser/builder test**

Use canonical Markdown containing headings, paragraphs, blank markers, a Markdown table, and image markers. Build a DOCX and re-extract it with R2 `extract_docx`; assert heading/paragraph/table order matches the canonical nodes. Image markers without declared assets remain explicit text placeholders rather than guessed images.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_documents.py -q`

- [ ] **Step 3: Add python-docx and implement deterministic builder**

When `reference` is supplied, load it only as a style/theme/section template: remove body children except final `w:sectPr`, preserve styles/theme/header/footer relationships, then append canonical source nodes. Do not copy reference narrative/body content. Without a reference, use a fresh document.

Canonical rules:

- Markdown headings map to built-in Heading 1..6;
- normal text becomes one paragraph per canonical source paragraph;
- `<!-- blank -->` becomes an empty paragraph;
- Markdown tables become Word tables with exact cell text;
- image markers are rendered as literal bracketed placeholders unless a declared artifact binding supplies an image path;
- output metadata (`core_properties`) is fixed to deterministic neutral values where python-docx permits.

- [ ] **Step 4: Add source mutation/no-historical-write tests**

Snapshot source Markdown and official reference report before/after build. Assert both unchanged and output lives outside `docs/results/`.

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_documents.py -q && ruff check src tests`

```bash
git add pyproject.toml src/afmc_fm/reproducibility/documents.py tests/reproducibility/test_documents.py
git commit -m "feat: build candidate reports from canonical source"
```

---

### Task 4: Normalized documentary structural comparison

**Files:**
- Create: `src/afmc_fm/reproducibility/document_compare.py`
- Create: `tests/reproducibility/test_document_compare.py`

**Interfaces:**
- Produces: `DocumentComparison(ok, paragraph_order_match, table_order_match, heading_order_match, caption_order_match, differences)`
- Produces: `compare_documents(reference: Path, candidate: Path) -> DocumentComparison`

- [ ] **Step 1: Write failing comparator tests**

Create fixture DOCXs where one paragraph, table-cell, heading order, or caption changes. Require the corresponding comparison field to fail while an equivalent document with metadata differences passes.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_document_compare.py -q`

- [ ] **Step 3: Implement comparison over R2 snapshots**

Normalize only serialization-insensitive details: Unicode whitespace runs, paragraph style aliases that map to the same heading level, and empty-run differences. Do not normalize wording, table-cell values, heading order, or caption text. `ok` requires all required structural/content dimensions to pass.

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_document_compare.py -q && ruff check src tests`

```bash
git add src/afmc_fm/reproducibility/document_compare.py tests/reproducibility/test_document_compare.py
git commit -m "feat: compare rebuilt reports structurally"
```

---

### Task 5: Phase rebuild manifests and orchestration

**Files:**
- Create: `src/afmc_fm/reproducibility/rebuild.py`
- Create: `docs/reproducibility/phase0/rebuild.yaml`
- Create: `docs/reproducibility/phase05/rebuild.yaml`
- Create: `docs/reproducibility/phase06/rebuild.yaml`
- Modify: `src/afmc_fm/reproducibility/registry.py`
- Test: `tests/reproducibility/test_rebuild.py`

**Interfaces:**
- Produces: `rebuild_phase(root: Path, phase_id: str, destination: Path | None = None) -> RebuildReport`
- `RebuildReport` records every artifact result, candidate DOCX, structural comparison, source/reference hashes, and historical-archive before/after digest.

- [ ] **Step 1: Add failing fixture orchestration test**

The fixture phase must complete verify + rebuild without training. It must create a report under `outputs/reproduction/fixture/rebuild/`, produce a machine-readable `rebuild-report.json`, and leave the fixture's official archive unchanged.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_rebuild.py -q`

- [ ] **Step 3: Add historical phase manifests conservatively**

For each historical phase, classify every declared documentary artifact as one of:

- `generate` only if an existing deterministic generator and complete committed inputs are demonstrably available;
- `reference-copy` if only the immutable archived artifact is available;
- `source-render` for report assembly from R2 source.

Never invent a generator to improve reproducibility status. The rebuild report exposes the count of generated versus reference-copy artifacts.

- [ ] **Step 4: Implement orchestration**

Pipeline:

1. validate R1 phase registration and R2 report manifest;
2. snapshot historical archive;
3. create/clean only the requested reproduction destination;
4. materialize declared artifacts;
5. build candidate DOCX;
6. compare candidate/reference structure;
7. write deterministic JSON report;
8. resnapshot historical archive and fail if changed.

- [ ] **Step 5: Mark `rebuild_supported=True` only for phases that pass their manifest validation**

Phase support is evidence-based, not inferred from the existence of historical scripts.

- [ ] **Step 6: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_rebuild.py tests/reproducibility/test_documents.py tests/reproducibility/test_document_compare.py -q && ruff check src tests`

```bash
git add src/afmc_fm/reproducibility docs/reproducibility/phase0/rebuild.yaml docs/reproducibility/phase05/rebuild.yaml docs/reproducibility/phase06/rebuild.yaml tests/reproducibility/test_rebuild.py
git commit -m "feat: orchestrate deterministic historical report rebuilds"
```

---

### Task 6: Public `afmc-reproduce rebuild` CLI

**Files:**
- Modify: `src/afmc_fm/reproducibility/cli.py`
- Modify: `tests/reproducibility/test_cli.py`

**Interfaces:**
- Adds: `afmc-reproduce rebuild <phase>`
- Adds: `afmc-reproduce rebuild all`
- Optional: `--destination <path>` for a single phase only.

- [ ] **Step 1: Write failing CLI tests**

Require parser exposure, rejection of unknown phases, refusal to rebuild unsupported phases, concise generated/reference-copy counts, candidate report path, structural-comparison status, and nonzero exit for a failed rebuild.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_cli.py -q`

- [ ] **Step 3: Implement CLI**

`rebuild all` runs every `rebuild_supported` phase and reports unsupported phases explicitly rather than silently skipping them. No CLI path accepts a destination under `docs/results/`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_cli.py tests/reproducibility/test_rebuild.py -q && ruff check src tests`

```bash
git add src/afmc_fm/reproducibility/cli.py tests/reproducibility/test_cli.py
git commit -m "feat: expose reproducibility rebuild CLI"
```

---

### Task 7: R3 full gate

**Files:**
- Modify: `docs/reproducibility/README.md`
- Modify: `tests/reproducibility/test_immutability.py`

- [ ] **Step 1: Add rebuild archive-immutability regression**

For every supported historical rebuild, snapshot all registered historical evidence roots before and after. Assert byte-identical snapshots and assert every produced path is under `outputs/reproduction/`.

- [ ] **Step 2: Document generated vs reference-copy semantics**

The README must state that documentary rebuild can be supported even when some legacy assets are reference-copy only, and that the rebuild report records that limitation explicitly.

- [ ] **Step 3: Run R3 tests**

Run: `pytest tests/reproducibility -q`

- [ ] **Step 4: Run Ruff**

Run: `ruff check src tests tools/reproducibility`

- [ ] **Step 5: Run full suite**

Run: `pytest -q`

Expected: all tests green; only established CUDA-unavailable skips remain.

- [ ] **Step 6: Confirm historical diff firewall**

Run: `git diff --name-only <R3-base-sha>...HEAD -- docs/results/`

Expected: no output.

- [ ] **Step 7: Commit final gate updates**

```bash
git add docs/reproducibility/README.md tests/reproducibility/test_immutability.py
git commit -m "test: close R3 deterministic rebuild support"
```

## R3 Completion Gate

R3 is complete when the rebuild schema and output firewall are enforced, supported historical phases can rebuild candidate reports from R2 source without training, artifact provenance distinguishes generated from reference-copy bytes, structural comparison is deterministic, all outputs are isolated under `outputs/reproduction/`, official archives remain byte-identical, Ruff is clean, the full suite is green, and no rerun/Phase-0.7/protected-seed activity occurred.
