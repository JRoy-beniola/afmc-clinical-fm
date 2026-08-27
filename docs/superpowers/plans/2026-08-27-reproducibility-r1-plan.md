# Reproducibility R1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only, phase-aware reproducibility registry and `afmc-reproduce status|verify` interface for Phase 0, Phase 0.5, Phase 0.6, and the post-Phase-0.6 archive.

**Architecture:** Add a focused `afmc_fm.reproducibility` package without rewriting historical execution or analysis code. Immutable Python phase definitions bind each historical decision to repository evidence; verification emits structured checks and never writes. `docs/reproducibility/artifact-map.yaml` mirrors the registry for humans and is consistency-tested.

**Tech Stack:** Python 3.11, `argparse`, `dataclasses`, `hashlib`, `pathlib`, `re`, `subprocess`, PyYAML, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-27-reproducibility-closure-design.md`

## Global Constraints

- `docs/results/phase0/`, `docs/results/phase05/`, `docs/results/phase06/`, and `docs/results/phase06_posthoc_optimization/` are read-only historical evidence.
- Phase 0.6 remains exactly `D4-B AMBIGUOUS -> STOP` as the normalized R1 label for the frozen `PHASE 0.6 TERMINATED — D4-B AMBIGUOUS → STOP` record.
- The post-Phase-0.6 classification remains `structured optimization-conditioned heterogeneity worth prospective testing` and remains exploratory.
- No protected confirmatory seeds may be consumed.
- R1 implements registry, artifact map, lineage, `status`, and read-only `verify` only. It does not implement rebuild, rerun, or Phase 0.7.
- Verification must never treat `outputs/reproduction/` as official evidence.
- Historical tools under `tools/analysis/`, `tools/execution/`, and `tools/monitoring/` are not refactored.

---

### Task 1: Immutable phase model and registry

**Files:**
- Create: `src/afmc_fm/reproducibility/__init__.py`
- Create: `src/afmc_fm/reproducibility/models.py`
- Create: `src/afmc_fm/reproducibility/registry.py`
- Test: `tests/reproducibility/test_registry.py`

**Interfaces:**
- Produces: `PhaseDefinition`, `EnvironmentStatus`, `PHASES`, `get_phase()`, `iter_phases()`.

- [ ] **Step 1: Write the failing registry test**

```python
from pathlib import Path

import pytest

from afmc_fm.reproducibility.registry import PHASES, get_phase, iter_phases


def test_registry_exposes_exact_historical_phases():
    assert tuple(PHASES) == ("phase0", "phase05", "phase06", "phase06-posthoc")
    assert tuple(item.phase_id for item in iter_phases()) == tuple(PHASES)


def test_phase06_decision_is_frozen():
    phase = get_phase("phase06")
    assert phase.expected_classification == "D4-B AMBIGUOUS -> STOP"
    assert phase.official_evidence_root == Path("docs/results/phase06")


def test_posthoc_is_exploratory():
    phase = get_phase("phase06-posthoc")
    assert phase.result_kind == "exploratory"
    assert phase.expected_classification == (
        "structured optimization-conditioned heterogeneity worth prospective testing"
    )


def test_unknown_phase_is_rejected():
    with pytest.raises(ValueError, match="unknown reproducibility phase"):
        get_phase("phase07")
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_registry.py -q`

Expected: import/collection failure because the package does not exist.

- [ ] **Step 3: Implement the minimal model**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

EnvironmentStatus = Literal["exact", "reconstructed", "unknown"]


@dataclass(frozen=True)
class PhaseDefinition:
    phase_id: str
    official_evidence_root: Path
    official_report: Path | None
    decision_record: Path
    expected_classification: str
    result_kind: Literal["historical", "exploratory"]
    implementation_sha: str
    execution_sha: str | None
    protocol_paths: tuple[Path, ...]
    manifest_paths: tuple[Path, ...]
    raw_evidence_paths: tuple[Path, ...]
    derived_table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    report_source: Path | None
    environment_status: EnvironmentStatus
    rebuild_supported: bool
    rerun_supported: bool
```

Populate the four phase entries only from committed evidence. Keep `rebuild_supported=False` and `rerun_supported=False` in R1.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/reproducibility/test_registry.py -q`

- [ ] **Step 5: Commit**

```bash
git add src/afmc_fm/reproducibility tests/reproducibility/test_registry.py
git commit -m "feat: add reproducibility phase registry"
```

---

### Task 2: Read-only verification primitives

**Files:**
- Create: `src/afmc_fm/reproducibility/integrity.py`
- Create: `src/afmc_fm/reproducibility/verify.py`
- Test: `tests/reproducibility/test_verify.py`

**Interfaces:**
- Produces: `CheckResult`, `VerificationReport`, `verify_phase(root, phase)`, `verify_all(root)`.

- [ ] **Step 1: Write fixture-based failing tests**

```python
import dataclasses
import hashlib
from pathlib import Path

from afmc_fm.reproducibility.models import PhaseDefinition
from afmc_fm.reproducibility.verify import verify_phase


def fixture_phase() -> PhaseDefinition:
    return PhaseDefinition(
        phase_id="fixture",
        official_evidence_root=Path("docs/results/fixture"),
        official_report=None,
        decision_record=Path("docs/results/fixture/decision.md"),
        expected_classification="FIXTURE STOP",
        result_kind="historical",
        implementation_sha="a" * 40,
        execution_sha=None,
        protocol_paths=(Path("docs/results/fixture/protocol.json"),),
        manifest_paths=(Path("docs/results/fixture/MANIFEST.sha256"),),
        raw_evidence_paths=(Path("docs/results/fixture/raw.csv"),),
        derived_table_paths=(),
        figure_paths=(),
        report_source=None,
        environment_status="unknown",
        rebuild_supported=False,
        rerun_supported=False,
    )


def write_valid_fixture(tmp_path: Path) -> None:
    root = tmp_path / "docs/results/fixture"
    root.mkdir(parents=True)
    (root / "decision.md").write_text("FIXTURE STOP\n", encoding="utf-8")
    (root / "protocol.json").write_text("{}\n", encoding="utf-8")
    (root / "raw.csv").write_text("x\n1\n", encoding="utf-8")
    digest = hashlib.sha256((root / "raw.csv").read_bytes()).hexdigest()
    (root / "MANIFEST.sha256").write_text(f"{digest}  raw.csv\n", encoding="utf-8")


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_missing_evidence_is_detected(tmp_path):
    report = verify_phase(tmp_path, fixture_phase())
    assert report.ok is False
    assert any(check.code == "missing_path" for check in report.checks)


def test_bad_manifest_hash_is_detected(tmp_path):
    write_valid_fixture(tmp_path)
    manifest = tmp_path / "docs/results/fixture/MANIFEST.sha256"
    manifest.write_text("0" * 64 + "  raw.csv\n", encoding="utf-8")
    report = verify_phase(tmp_path, fixture_phase())
    assert any(check.code == "sha256_mismatch" for check in report.checks)


def test_classification_mismatch_is_detected(tmp_path):
    write_valid_fixture(tmp_path)
    decision = tmp_path / "docs/results/fixture/decision.md"
    decision.write_text("WRONG\n", encoding="utf-8")
    report = verify_phase(tmp_path, fixture_phase())
    assert any(check.code == "classification_mismatch" for check in report.checks)


def test_verify_writes_nothing(tmp_path):
    write_valid_fixture(tmp_path)
    before = snapshot(tmp_path)
    report = verify_phase(tmp_path, fixture_phase())
    after = snapshot(tmp_path)
    assert report.ok is True
    assert after == before


def test_reproduction_output_cannot_be_official_evidence(tmp_path):
    phase = dataclasses.replace(
        fixture_phase(),
        official_evidence_root=Path("outputs/reproduction/fixture"),
    )
    report = verify_phase(tmp_path, phase)
    assert any(
        check.code == "official_evidence_in_reproduction_output"
        for check in report.checks
    )
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_verify.py -q`

Expected: import failure because verification modules do not exist.

- [ ] **Step 3: Implement minimal verification**

`integrity.py` must validate lowercase 64-character SHA-256 lines, resolve entries relative to each manifest, reject absolute paths and `..` traversal, and compare raw bytes. `verify.py` must return check records rather than writing output or mutating evidence.

```python
@dataclass(frozen=True)
class CheckResult:
    code: str
    ok: bool
    subject: str
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    phase_id: str
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)
```

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/reproducibility/test_verify.py -q`

- [ ] **Step 5: Commit**

```bash
git add src/afmc_fm/reproducibility tests/reproducibility/test_verify.py
git commit -m "feat: add read-only archive verification"
```

---

### Task 3: Public `afmc-reproduce` CLI

**Files:**
- Create: `src/afmc_fm/reproducibility/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/reproducibility/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

```python
import pytest

from afmc_fm.reproducibility.cli import main


def test_status_lists_all_phases(capsys):
    assert main(["status"]) == 0
    output = capsys.readouterr().out
    for phase in ("phase0", "phase05", "phase06", "phase06-posthoc"):
        assert phase in output


def test_unknown_verify_phase_is_argparse_error():
    with pytest.raises(SystemExit) as exc:
        main(["verify", "phase07"])
    assert exc.value.code == 2


def test_verify_all_aggregates_failures(tmp_path, capsys):
    assert main(["verify", "all"], root=tmp_path) == 1
    assert "FAIL" in capsys.readouterr().out
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_cli.py -q`

- [ ] **Step 3: Implement CLI and entry point**

`status` prints archive presence, manifest presence, report-source status, environment status, rebuild/rerun support, and historical decision. `verify <phase|all>` prints concise check results and returns `1` if any requested phase fails.

Add to `pyproject.toml`:

```toml
afmc-reproduce = "afmc_fm.reproducibility.cli:main"
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_cli.py -q`

```bash
git add src/afmc_fm/reproducibility/cli.py tests/reproducibility/test_cli.py pyproject.toml
git commit -m "feat: add reproducibility status and verify CLI"
```

---

### Task 4: Artifact map and research lineage

**Files:**
- Create: `docs/reproducibility/README.md`
- Create: `docs/reproducibility/artifact-map.yaml`
- Create: `docs/reproducibility/research-lineage.md`
- Test: `tests/reproducibility/test_documentation.py`

- [ ] **Step 1: Write failing consistency test**

```python
from pathlib import Path

import yaml

from afmc_fm.reproducibility.registry import PHASES


def test_artifact_map_matches_registry():
    payload = yaml.safe_load(
        Path("docs/reproducibility/artifact-map.yaml").read_text(encoding="utf-8")
    )
    assert tuple(payload["phases"]) == tuple(PHASES)
    for phase_id, phase in PHASES.items():
        item = payload["phases"][phase_id]
        assert item["official_evidence_root"] == phase.official_evidence_root.as_posix()
        assert item["expected_classification"] == phase.expected_classification
        assert item["implementation_sha"] == phase.implementation_sha
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/reproducibility/test_documentation.py -q`

Expected: missing artifact map.

- [ ] **Step 3: Add the three documentation files**

The map records every phase's evidence root, report, decision record, protocol/config/spec, implementation/execution SHAs, historical tools, raw/derived evidence, figures, report-source status, environment status, integrity manifests, and R1 support flags. The lineage preserves:

```text
Phase 0 -> Phase 0.5 -> Phase 0.6 -> D4-B AMBIGUOUS -> STOP
        -> post-Phase-0.6 exploratory optimization-conditioned analysis
        -> Phase 0.7 design only (not authorized for execution)
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest tests/reproducibility/test_documentation.py -q`

```bash
git add docs/reproducibility tests/reproducibility/test_documentation.py
git commit -m "docs: add reproducibility artifact map and lineage"
```

---

### Task 5: Bind all real historical archives

**Files:**
- Modify: `src/afmc_fm/reproducibility/registry.py`
- Modify: `src/afmc_fm/reproducibility/integrity.py`
- Modify: `src/afmc_fm/reproducibility/verify.py`
- Modify: `tests/reproducibility/test_registry.py`
- Modify: `tests/reproducibility/test_verify.py`

- [ ] **Step 1: Add failing real-repository verification test**

```python
from pathlib import Path

from afmc_fm.reproducibility.verify import verify_all


def test_real_repository_has_no_binding_or_classification_errors():
    reports = verify_all(Path("."))
    failures = {
        check.code
        for report in reports
        for check in report.checks
        if not check.ok
    }
    assert "invalid_sha" not in failures
    assert "classification_mismatch" not in failures
    assert "official_evidence_in_reproduction_output" not in failures
```

- [ ] **Step 2: Verify RED where historical manifest layouts require adapters**

Run: `pytest tests/reproducibility/test_registry.py tests/reproducibility/test_verify.py -q`

- [ ] **Step 3: Correct only reproducibility-layer bindings/parsers**

Do not edit historical archives to make verification pass. Missing historical report source is reported as status, not corruption. If a historical manifest uses paths rooted differently from another phase, handle that through a declared manifest base mode in the reproducibility layer.

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest tests/reproducibility -q`

```bash
git add src/afmc_fm/reproducibility tests/reproducibility
git commit -m "feat: bind historical archives to reproducibility verification"
```

---

### Task 6: Archive immutability and full-suite gate

**Files:**
- Create: `tests/reproducibility/test_archive_immutability.py`

- [ ] **Step 1: Add command-level immutability guard**

```python
import hashlib
from pathlib import Path

from afmc_fm.reproducibility.cli import main

ROOTS = (
    Path("docs/results/phase0"),
    Path("docs/results/phase05"),
    Path("docs/results/phase06"),
    Path("docs/results/phase06_posthoc_optimization"),
)


def snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_verify_all_does_not_mutate_historical_archives():
    before = {root: snapshot(root) for root in ROOTS}
    main(["verify", "all"])
    after = {root: snapshot(root) for root in ROOTS}
    assert after == before
```

- [ ] **Step 2: Run R1 tests**

Run: `pytest tests/reproducibility -q`

- [ ] **Step 3: Run Ruff**

Run: `ruff check src tests`

- [ ] **Step 4: Run full suite**

Run: `pytest -q`

Expected: all tests green; only already-established CUDA-unavailable skips remain.

- [ ] **Step 5: Exercise public interface**

```bash
afmc-reproduce status
afmc-reproduce verify phase0
afmc-reproduce verify phase05
afmc-reproduce verify phase06
afmc-reproduce verify phase06-posthoc
afmc-reproduce verify all
```

- [ ] **Step 6: Commit**

```bash
git add tests/reproducibility/test_archive_immutability.py
git commit -m "test: prove reproducibility verification is read-only"
```

---

## R1 Completion Gate

R1 is complete only when `status` and every `verify` command work read-only, all four historical phases are bound, the artifact map and lineage agree with the registry, tampering/missing evidence/classification mismatch/invalid SHAs are detected, archive snapshots remain unchanged, Ruff is clean, the complete pytest suite is green, and no rebuild, rerun, Phase 0.7 execution, PR #7 merge, or protected-seed use has occurred.
