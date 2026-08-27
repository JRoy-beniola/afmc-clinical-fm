# Reproducibility R2: Historical Report Source Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover, audit, and freeze canonical version-controlled documentary source for the immutable Phase 0, Phase 0.5, and Phase 0.6 historical reports without changing any official archive bytes.

**Architecture:** R2 adds an OOXML-only extractor that reads historical `.docx` files as immutable inputs and emits a normalized report snapshot, Markdown narrative/table source, and a machine-readable report manifest under `docs/reproducibility/<phase>/`. A bootstrap command may write only to an explicitly supplied non-historical destination. Acceptance tests bind every recovered source to the exact SHA-256 of its official report and prove that the historical archives are unchanged.

**Tech Stack:** Python 3.11 standard library (`zipfile`, `xml.etree.ElementTree`, `hashlib`, `json`, `pathlib`), PyYAML, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-27-reproducibility-closure-design.md`

## Global Constraints

- `docs/results/phase0/`, `docs/results/phase05/`, `docs/results/phase06/`, and `docs/results/phase06_posthoc_optimization/` are immutable historical evidence.
- Phase 0.6 remains `D4-B AMBIGUOUS -> STOP`.
- Post-Phase-0.6 evidence remains exploratory.
- No protected confirmatory seeds may be consumed.
- R2 performs no model training and no historical experiment rerun.
- Historical report extraction is one-time bootstrap tooling; routine future report production is source-first.
- Missing historical information must remain explicit; no reconstructed metadata may be presented as exact historical metadata.

---

### Task 1: OOXML report snapshot extractor

**Files:**
- Create: `src/afmc_fm/reproducibility/report_source.py`
- Create: `tests/reproducibility/test_report_source.py`

**Interfaces:**
- Produces: `extract_docx(path: Path) -> ReportSnapshot`
- Produces: `ReportSnapshot.reference_sha256: str`
- Produces: ordered block types `ParagraphBlock`, `TableBlock`, and `ImageBlock`
- Produces: `ReportSnapshot.to_json_dict() -> dict[str, object]`

- [ ] **Step 1: Write fixture-DOCX helpers and failing ordering/hash test**

```python
from pathlib import Path

from afmc_fm.reproducibility.report_source import (
    ParagraphBlock,
    TableBlock,
    extract_docx,
)


def test_extract_docx_preserves_body_order_and_hash(tmp_path: Path):
    docx = write_fixture_docx(
        tmp_path / "fixture.docx",
        paragraphs=[("Heading 1", "Title"), ("Normal", "Intro")],
        table_rows=[["A", "B"], ["1", "2"]],
        trailing_paragraph="After table",
    )
    snapshot = extract_docx(docx)
    assert snapshot.reference_sha256 == sha256(docx)
    assert isinstance(snapshot.blocks[0], ParagraphBlock)
    assert snapshot.blocks[0].style == "Heading 1"
    assert snapshot.blocks[0].text == "Title"
    assert isinstance(snapshot.blocks[2], TableBlock)
    assert snapshot.blocks[2].rows == (("A", "B"), ("1", "2"))
    assert snapshot.blocks[3].text == "After table"
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest tests/reproducibility/test_report_source.py -q`

Expected: import failure because `report_source` does not exist.

- [ ] **Step 3: Implement minimal OOXML extraction**

`extract_docx` must:

1. hash the raw `.docx` bytes with SHA-256;
2. open the ZIP read-only;
3. parse `word/document.xml` and `word/styles.xml`;
4. iterate direct body children in document order;
5. concatenate all text nodes in each paragraph/table cell, preserving tabs and line breaks as text separators;
6. resolve paragraph style IDs to style names where available;
7. record table rows/cells exactly as extracted;
8. record inline/drawing relationship IDs in an `ImageBlock` without inventing captions;
9. never write beside the source document.

Use immutable dataclasses and tuple fields so the snapshot cannot be mutated accidentally.

- [ ] **Step 4: Add edge-case tests**

Tests must cover empty paragraphs, merged text runs, missing style declarations, table-cell line breaks, malformed ZIP input, and repeated image relationships.

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_report_source.py -q && ruff check src tests`

```bash
git add src/afmc_fm/reproducibility/report_source.py tests/reproducibility/test_report_source.py
git commit -m "feat: extract immutable report snapshots from docx"
```

---

### Task 2: Canonical Markdown source and report manifest

**Files:**
- Modify: `src/afmc_fm/reproducibility/report_source.py`
- Create: `src/afmc_fm/reproducibility/report_manifest.py`
- Create: `tests/reproducibility/test_report_manifest.py`

**Interfaces:**
- Produces: `render_markdown(snapshot: ReportSnapshot) -> str`
- Produces: `ReportSourceManifest`
- Produces: `build_manifest(phase: PhaseDefinition, snapshot: ReportSnapshot, source_path: Path, source_bytes: bytes) -> ReportSourceManifest`
- Produces: `load_manifest(path: Path) -> ReportSourceManifest`
- Produces: `validate_manifest(root: Path, manifest: ReportSourceManifest) -> tuple[CheckResult, ...]`

- [ ] **Step 1: Write failing canonical-source tests**

```python
def test_render_markdown_is_deterministic(snapshot):
    first = render_markdown(snapshot)
    second = render_markdown(snapshot)
    assert first == second
    assert first.endswith("\n")


def test_manifest_binds_reference_and_source_hashes(tmp_path):
    source = b"# Title\n\nBody\n"
    manifest = build_manifest(phase, snapshot, Path("report-source.md"), source)
    assert manifest.reference_report_sha256 == snapshot.reference_sha256
    assert manifest.report_source_sha256 == hashlib.sha256(source).hexdigest()
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_report_manifest.py -q`

- [ ] **Step 3: Implement canonical rendering rules**

`render_markdown` uses only a deterministic subset:

- `Heading 1`..`Heading 6` -> Markdown headings;
- other non-empty paragraphs -> plain paragraphs;
- empty paragraphs -> a single explicit `<!-- blank -->` marker so ordering can be audited;
- tables -> deterministic Markdown tables, escaping `|` and backslashes;
- images -> `<!-- image rel=<relationship-id> -->` markers at their body position;
- no generated prose, rewritten scientific language, or inferred captions.

`ReportSourceManifest` stores:

```yaml
schema_version: 1
phase_id: phase06
reference_report: docs/results/phase06/AFMC_Phase0_6_Diagnostic_Validation_Report_FINAL.docx
reference_report_sha256: <64 hex>
report_source: docs/reproducibility/phase06/report-source.md
report_source_sha256: <64 hex>
extraction_snapshot: docs/reproducibility/phase06/extraction.json
extractor: afmc_fm.reproducibility.report_source
extractor_schema_version: 1
block_count: <int>
paragraph_count: <int>
table_count: <int>
image_count: <int>
audit_status: accepted
known_limitations: []
```

- [ ] **Step 4: Add manifest tamper/missing-path tests**

`validate_manifest` must detect a changed official report, changed source Markdown, missing extraction snapshot, phase mismatch, and invalid SHA-256 syntax.

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_report_source.py tests/reproducibility/test_report_manifest.py -q && ruff check src tests`

```bash
git add src/afmc_fm/reproducibility/report_source.py src/afmc_fm/reproducibility/report_manifest.py tests/reproducibility
git commit -m "feat: define canonical historical report source manifests"
```

---

### Task 3: One-time bootstrap command with archive write firewall

**Files:**
- Create: `tools/reproducibility/bootstrap_reports.py`
- Create: `tests/reproducibility/test_bootstrap_reports.py`

**Interfaces:**
- Produces CLI: `python tools/reproducibility/bootstrap_reports.py <phase> --root <repo> --destination <dir>`
- `phase` choices: `phase0`, `phase05`, `phase06`
- Emits exactly `report-source.md`, `report.yaml`, and `extraction.json` beneath the supplied destination.

- [ ] **Step 1: Write failing firewall test**

```python
def test_bootstrap_refuses_historical_destination(tmp_path):
    with pytest.raises(ValueError, match="historical evidence"):
        bootstrap_phase(
            root=tmp_path,
            phase=fixture_phase,
            destination=tmp_path / "docs/results/phase0/reproducibility",
        )
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_bootstrap_reports.py -q`

- [ ] **Step 3: Implement bootstrap**

The command must reject any destination equal to or nested beneath `docs/results/` or `outputs/reproduction/`. It reads the registered `official_report`, extracts the snapshot, writes UTF-8 Markdown/JSON/YAML through temporary sibling files, fsyncs, then atomically replaces destination files. It must refuse overwrite unless `--replace` is passed, and `--replace` is permitted only beneath `docs/reproducibility/`.

- [ ] **Step 4: Test deterministic repeatability and no source mutation**

Snapshot the source DOCX hash before/after bootstrap and assert identical. Bootstrap twice to two temporary destinations and assert all three generated text files are byte-identical.

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_bootstrap_reports.py -q && ruff check src tests tools/reproducibility`

```bash
git add tools/reproducibility/bootstrap_reports.py tests/reproducibility/test_bootstrap_reports.py
git commit -m "feat: add guarded historical report bootstrap"
```

---

### Task 4: Bootstrap and audit the three real historical reports

**Files:**
- Create: `docs/reproducibility/phase0/report-source.md`
- Create: `docs/reproducibility/phase0/report.yaml`
- Create: `docs/reproducibility/phase0/extraction.json`
- Create: `docs/reproducibility/phase05/report-source.md`
- Create: `docs/reproducibility/phase05/report.yaml`
- Create: `docs/reproducibility/phase05/extraction.json`
- Create: `docs/reproducibility/phase06/report-source.md`
- Create: `docs/reproducibility/phase06/report.yaml`
- Create: `docs/reproducibility/phase06/extraction.json`
- Modify: `src/afmc_fm/reproducibility/registry.py`
- Modify: `docs/reproducibility/artifact-map.yaml`
- Test: `tests/reproducibility/test_real_report_sources.py`

**Interfaces:**
- Registry `report_source` becomes non-null for `phase0`, `phase05`, and `phase06` only after the corresponding source passes audit.
- `environment_status`, `rebuild_supported`, and `rerun_supported` remain unchanged in R2.

- [ ] **Step 1: Add failing real-source audit**

```python
@pytest.mark.parametrize("phase_id", ["phase0", "phase05", "phase06"])
def test_real_report_source_is_bound_and_audited(phase_id):
    phase = get_phase(phase_id)
    manifest_path = Path("docs/reproducibility") / phase_id / "report.yaml"
    manifest = load_manifest(manifest_path)
    failures = [check for check in validate_manifest(Path("."), manifest) if not check.ok]
    assert failures == []
    assert phase.report_source == manifest.report_source
```

- [ ] **Step 2: Verify RED before generated sources exist**

Run: `pytest tests/reproducibility/test_real_report_sources.py -q`

- [ ] **Step 3: Generate sources from the real immutable reports**

Run from repository root:

```bash
python tools/reproducibility/bootstrap_reports.py phase0 --root . --destination docs/reproducibility/phase0
python tools/reproducibility/bootstrap_reports.py phase05 --root . --destination docs/reproducibility/phase05
python tools/reproducibility/bootstrap_reports.py phase06 --root . --destination docs/reproducibility/phase06
```

- [ ] **Step 4: Audit extraction against immutable reference**

For each phase assert:

1. the recorded reference SHA-256 equals the official DOCX bytes;
2. rendering the same DOCX again yields byte-identical Markdown and snapshot JSON after removal of no fields (there are no timestamps in canonical output);
3. `block_count`, `paragraph_count`, `table_count`, and `image_count` in `report.yaml` match `extraction.json`;
4. every non-empty extracted paragraph/table-cell string is represented in `report-source.md` after the renderer's documented escaping;
5. the frozen decision phrase for the phase appears in either the recovered narrative or the immutable decision record; no decision text is rewritten during audit.

Set `audit_status: accepted` only when all five checks pass.

- [ ] **Step 5: Update registry/artifact map and verify GREEN**

Run: `pytest tests/reproducibility -q && ruff check src tests tools/reproducibility`

- [ ] **Step 6: Commit**

```bash
git add docs/reproducibility/phase0 docs/reproducibility/phase05 docs/reproducibility/phase06 src/afmc_fm/reproducibility/registry.py docs/reproducibility/artifact-map.yaml tests/reproducibility/test_real_report_sources.py
git commit -m "docs: freeze audited historical report sources"
```

---

### Task 5: R2 archive-immutability and full-suite gate

**Files:**
- Modify: `tests/reproducibility/test_immutability.py`
- Modify: `docs/reproducibility/README.md`

- [ ] **Step 1: Extend archive snapshot coverage**

Add a test that snapshots every file under all four registered historical evidence roots, runs extraction/manifest validation for all three historical reports, and asserts byte-identical snapshots afterward.

- [ ] **Step 2: Document R2 status precisely**

Update the README to state that report source is recovered/audited for Phase 0/0.5/0.6, that the official DOCX remains authoritative, and that R2 still does not imply rebuild or rerun support.

- [ ] **Step 3: Run R2 tests**

Run: `pytest tests/reproducibility -q`

- [ ] **Step 4: Run Ruff**

Run: `ruff check src tests tools/reproducibility`

- [ ] **Step 5: Run complete suite**

Run: `pytest -q`

Expected: all tests green; only already-established CUDA-unavailable skips remain.

- [ ] **Step 6: Confirm historical diff firewall**

Run:

```bash
git diff --name-only <R2-base-sha>...HEAD -- docs/results/
```

Expected: no output.

- [ ] **Step 7: Commit final R2 documentation/test updates**

```bash
git add tests/reproducibility/test_immutability.py docs/reproducibility/README.md
git commit -m "test: close R2 historical report-source recovery"
```

## R2 Completion Gate

R2 is complete only when all three historical official reports have deterministic canonical Markdown plus extraction JSON and validated report manifests, every source is cryptographically bound to the immutable official DOCX, the registry/artifact map agree, archive snapshots remain byte-identical, Ruff is clean, the full pytest suite is green, and no training/rerun/Phase-0.7/protected-seed activity occurred.
