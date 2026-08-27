# Reproducibility R1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only, phase-aware reproducibility registry and `afmc-reproduce status|verify` interface for Phase 0, Phase 0.5, Phase 0.6, and the post-Phase-0.6 archive.

**Architecture:** Add a new importable `afmc_fm.reproducibility` package without rewriting historical execution or analysis tools. Phase truth is encoded as immutable Python phase definitions; `docs/reproducibility/artifact-map.yaml` is the human-readable cross-phase index and is checked for consistency with those definitions. Verification returns structured check records and never writes to the repository or reproduction outputs.

**Tech Stack:** Python 3.11, stdlib `argparse`/`dataclasses`/`hashlib`/`json`/`pathlib`/`subprocess`, PyYAML, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-27-reproducibility-closure-design.md`

## Global Constraints

- `docs/results/phase0/`, `docs/results/phase05/`, `docs/results/phase06/`, and `docs/results/phase06_posthoc_optimization/` are read-only historical evidence.
- Phase 0.6 remains exactly `D4-B AMBIGUOUS -> STOP`.
- The post-Phase-0.6 result remains exploratory with classification `structured optimization-conditioned heterogeneity worth prospective testing`.
- No protected confirmatory seeds may be consumed.
- R1 implements only registry, artifact map, lineage, `status`, and read-only `verify`; no report extraction, rebuild, rerun, or Phase 0.7 execution.
- Verification must never treat `outputs/reproduction/` as official evidence.
- Unknown or malformed phases/manifests must fail explicitly.
- Existing historical tools under `tools/analysis/`, `tools/execution/`, and `tools/monitoring/` are not refactored.

---

### Task 1: Phase model and immutable registry

**Files:**
- Create: `src/afmc_fm/reproducibility/__init__.py`
- Create: `src/afmc_fm/reproducibility/models.py`
- Create: `src/afmc_fm/reproducibility/registry.py`
- Test: `tests/reproducibility/test_registry.py`

**Interfaces:**
- Produces: `PhaseDefinition`, `EnvironmentStatus`, `PHASES`, `get_phase(phase_id)`, `iter_phases()`.
- Consumes: repository-relative paths only; no filesystem writes.

- [ ] **Step 1: Write failing registry tests**

```python
from pathlib import Path

import pytest

from afmc_fm.reproducibility.registry import PHASES, get_phase, iter_phases


def test_registry_exposes_exact_historical_phases():
    assert tuple(PHASES) == ("phase0", "phase05", "phase06", "phase06-posthoc")
    assert tuple(p.phase_id for p in iter_phases()) == tuple(PHASES)


def test_phase06_historical_decision_is_frozen():
    phase = get_phase("phase06")
    assert phase.expected_classification == "D4-B AMBIGUOUS -> STOP"
    assert phase.official_evidence_root == Path("docs/results/phase06")


def test_posthoc_remains_exploratory():
    phase = get_phase("phase06-posthoc")
    assert phase.expected_classification == (
        "structured optimization-conditioned heterogeneity worth prospective testing"
    )
    assert phase.result_kind == "exploratory"


def test_unknown_phase_is_rejected():
    with pytest.raises(ValueError, match="unknown reproducibility phase"):
        get_phase("phase07")
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/reproducibility/test_registry.py -q`

Expected: collection/import failure because `afmc_fm.reproducibility` does not yet exist.

- [ ] **Step 3: Implement minimal immutable models and registry**

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

Populate `PHASES` with only evidence that exists in the repository at implementation time. Keep R1 support flags `False` for rebuild/rerun.

- [ ] **Step 4: Run registry tests and verify GREEN**

Run: `pytest tests/reproducibility/test_registry.py -q`

Expected: all registry tests pass.

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
- Consumes: `PhaseDefinition`, repository root `Path`.
- Produces: `CheckResult`, `VerificationReport`, `verify_phase(root, phase)`, `verify_all(root)`.

- [ ] **Step 1: Write failing verification tests using a temporary fixture tree**

```python
from pathlib import Path

from afmc_fm.reproducibility.models import PhaseDefinition
from afmc_fm.reproducibility.verify import verify_phase


def _fixture_phase() -> PhaseDefinition:
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


def test_verify_detects_missing_evidence(tmp_path):
    report = verify_phase(tmp_path, _fixture_phase())
    assert report.ok is False
    assert any(c.code == "missing_path" for c in report.checks)


def test_verify_detects_bad_manifest_hash(tmp_path):
    root = tmp_path / "docs/results/fixture"
    root.mkdir(parents=True)
    (root / "decision.md").write_text("FIXTURE STOP\n", encoding="utf-8")
    (root / "protocol.json").write_text("{}\n", encoding="utf-8")
    (root / "raw.csv").write_text("x\n1\n", encoding="utf-8")
    (root / "MANIFEST.sha256").write_text("0" * 64 + "  raw.csv\n", encoding="utf-8")
    report = verify_phase(tmp_path, _fixture_phase())
    assert any(c.code == "sha256_mismatch" for c in report.checks)


def test_verify_detects_classification_mismatch(tmp_path):
    root = tmp_path / "docs/results/fixture"
    root.mkdir(parents=True)
    (root / "decision.md").write_text("NOT THE FROZEN DECISION\n", encoding="utf-8")
    (root / "protocol.json").write_text("{}\n", encoding="utf-8")
    (root / "raw.csv").write_text("x\n1\n", encoding="utf-8")
    import hashlib
    digest = hashlib.sha256((root / "raw.csv").read_bytes()).hexdigest()
    (root / "MANIFEST.sha256").write_text(f"{digest}  raw.csv\n", encoding="utf-8")
    report = verify_phase(tmp_path, _fixture_phase())
    assert any(c.code == "classification_mismatch" for c in report.checks)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/reproducibility/test_verify.py -q`

Expected: import failure because verification modules do not exist.

- [ ] **Step 3: Implement manifest parser and report models**

Implement SHA-256 parsing with strict 64-lowercase-hex validation, resolve manifest entries relative to the manifest parent, reject absolute paths and `..` traversal, hash bytes without normalizing file contents, and return explicit check records instead of raising for ordinary verification failures.

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

- [ ] **Step 4: Add explicit read-only guard tests**

```python
def test_verify_writes_nothing(tmp_path):
    # construct a valid fixture, snapshot every relative path + file bytes,
    # execute verify_phase, and assert the snapshot is identical afterward.
    ...


def test_official_evidence_cannot_point_into_reproduction_outputs():
    phase = dataclasses.replace(
        _fixture_phase(),
        official_evidence_root=Path("outputs/reproduction/fixture"),
    )
    report = verify_phase(Path("."), phase)
    assert any(c.code == "official_evidence_in_reproduction_output" for c in report.checks)
```

For the first test, implement a local test helper that returns `{relative_path: bytes}` before and after; do not place snapshot logic in production code.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `pytest tests/reproducibility/test_verify.py -q`

Expected: all verification tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/afmc_fm/reproducibility/integrity.py src/afmc_fm/reproducibility/verify.py tests/reproducibility/test_verify.py
git commit -m "feat: add read-only archive verification"
```

---

### Task 3: `afmc-reproduce status` and `verify` CLI

**Files:**
- Create: `src/afmc_fm/reproducibility/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/reproducibility/test_cli.py`

**Interfaces:**
- Produces console script `afmc-reproduce`.
- Commands: `status`, `verify <phase|all>`.
- Exit code: `0` when requested checks pass; `1` when verification fails; argparse retains `2` for invalid syntax/phase choices.

- [ ] **Step 1: Write failing parser/CLI tests**

```python
from afmc_fm.reproducibility.cli import main


def test_status_lists_all_historical_phases(capsys):
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    for phase in ("phase0", "phase05", "phase06", "phase06-posthoc"):
        assert phase in out


def test_verify_rejects_unknown_phase(capsys):
    try:
        main(["verify", "phase07"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse must reject an unknown phase")


def test_verify_all_aggregates_failures(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["verify", "all"]) == 1
    assert "FAIL" in capsys.readouterr().out
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/reproducibility/test_cli.py -q`

Expected: import failure because CLI does not exist.

- [ ] **Step 3: Implement minimal CLI**

Use `argparse` and the repository root as `Path.cwd()` unless `main(..., root=...)` is supplied by tests. `status` must show, per phase: archive present, integrity manifest present, report source present/missing, environment exact/reconstructed/unknown, rebuild support, rerun support, and historical decision. `verify all` must print each failed check with phase + code + subject.

- [ ] **Step 4: Add console entry point**

```toml
[project.scripts]
afmc-phase0 = "afmc_fm.cli:main"
afmc-phase06 = "afmc_fm.phase06.cli:main"
afmc-reproduce = "afmc_fm.reproducibility.cli:main"
```

- [ ] **Step 5: Run CLI tests and verify GREEN**

Run: `pytest tests/reproducibility/test_cli.py -q`

Expected: all CLI tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/afmc_fm/reproducibility/cli.py tests/reproducibility/test_cli.py pyproject.toml
git commit -m "feat: add reproducibility status and verify CLI"
```

---

### Task 4: Cross-phase artifact map and research lineage

**Files:**
- Create: `docs/reproducibility/README.md`
- Create: `docs/reproducibility/artifact-map.yaml`
- Create: `docs/reproducibility/research-lineage.md`
- Test: `tests/reproducibility/test_documentation.py`

**Interfaces:**
- `artifact-map.yaml` is descriptive; Python registry remains the verification authority.
- Documentation test ensures all four phase IDs, roots, classifications, and implementation SHAs agree with the registry.

- [ ] **Step 1: Write failing documentation consistency test**

```python
from pathlib import Path

import yaml

from afmc_fm.reproducibility.registry import PHASES


def test_artifact_map_matches_registry():
    payload = yaml.safe_load(Path("docs/reproducibility/artifact-map.yaml").read_text())
    assert tuple(payload["phases"]) == tuple(PHASES)
    for phase_id, phase in PHASES.items():
        item = payload["phases"][phase_id]
        assert item["official_evidence_root"] == phase.official_evidence_root.as_posix()
        assert item["expected_classification"] == phase.expected_classification
        assert item["implementation_sha"] == phase.implementation_sha
```

- [ ] **Step 2: Run test and verify RED**

Run: `pytest tests/reproducibility/test_documentation.py -q`

Expected: `FileNotFoundError` because the artifact map is not yet present.

- [ ] **Step 3: Create artifact map and lineage docs from verified repository evidence**

`artifact-map.yaml` must list the official evidence root, official report (or null), decision record, protocol/design/config paths, implementation/execution SHAs, historical tool paths, raw evidence, derived tables, figures, report source status, environment status, integrity manifests, and R1 support flags for each phase.

`research-lineage.md` must preserve this sequence and interpretation boundary:

```text
Phase 0 -> Phase 0.5 -> Phase 0.6 -> D4-B AMBIGUOUS -> STOP
        -> post-Phase-0.6 exploratory optimization-conditioned analysis
        -> Phase 0.7 design only (not authorized for execution)
```

- [ ] **Step 4: Run documentation tests and verify GREEN**

Run: `pytest tests/reproducibility/test_documentation.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add docs/reproducibility tests/reproducibility/test_documentation.py
git commit -m "docs: add reproducibility artifact map and lineage"
```

---

### Task 5: Bind the four real archives and make `verify all` useful

**Files:**
- Modify: `src/afmc_fm/reproducibility/registry.py`
- Modify: `src/afmc_fm/reproducibility/verify.py`
- Modify: `tests/reproducibility/test_registry.py`
- Modify: `tests/reproducibility/test_verify.py`

**Interfaces:**
- Real archive verification must tolerate historically different manifest layouts through phase declarations, not by rewriting old archives.
- Recorded SHAs must be 40 lowercase hex and, when `.git` history is available, `git cat-file -e <sha>^{commit}` must resolve them.

- [ ] **Step 1: Add failing real-repository tests**

```python
from pathlib import Path

from afmc_fm.reproducibility.verify import verify_all


def test_real_repository_verify_all_has_no_registry_or_classification_errors():
    report = verify_all(Path("."))
    bad_codes = {
        check.code
        for phase_report in report
        for check in phase_report.checks
        if not check.ok
    }
    assert "invalid_sha" not in bad_codes
    assert "classification_mismatch" not in bad_codes
    assert "official_evidence_in_reproduction_output" not in bad_codes
```

Add targeted assertions for each historical manifest path actually present in the phase registry.

- [ ] **Step 2: Run focused test and verify RED if any declared binding is wrong**

Run: `pytest tests/reproducibility/test_registry.py tests/reproducibility/test_verify.py -q`

Expected: at least one failure until all real phase paths/manifests are bound correctly.

- [ ] **Step 3: Correct only reproducibility-layer declarations/parsing**

Do not edit historical evidence to make verification pass. If a historical manifest format differs, add a read-only parser branch keyed by the declared manifest path/type. If historical documentary source is absent, report `report_source_present=false`; absence is status, not corruption.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/reproducibility -q`

Expected: all R1 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/afmc_fm/reproducibility tests/reproducibility
git commit -m "feat: bind historical archives to reproducibility verification"
```

---

### Task 6: Archive immutability and full-suite gate

**Files:**
- Create: `tests/reproducibility/test_archive_immutability.py`
- Modify only if required by tests: files under `src/afmc_fm/reproducibility/`

**Interfaces:**
- Produces final R1 proof that `status`/`verify` are read-only and do not mutate official archives.

- [ ] **Step 1: Write failing/guard tests for command-level immutability**

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


def _snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_verify_all_does_not_mutate_historical_archives():
    before = {root: _snapshot(root) for root in ROOTS}
    main(["verify", "all"])
    after = {root: _snapshot(root) for root in ROOTS}
    assert after == before
```

- [ ] **Step 2: Run R1 tests**

Run: `pytest tests/reproducibility -q`

Expected: pass after Task 5; if this guard exposes a write, fix production code before continuing.

- [ ] **Step 3: Run Ruff**

Run: `ruff check src tests`

Expected: clean.

- [ ] **Step 4: Run the complete test suite**

Run: `pytest -q`

Expected: all existing tests plus R1 tests pass; existing CUDA-unavailable skips remain skips only.

- [ ] **Step 5: Exercise the public interface**

Run:

```bash
afmc-reproduce status
afmc-reproduce verify phase0
afmc-reproduce verify phase05
afmc-reproduce verify phase06
afmc-reproduce verify phase06-posthoc
afmc-reproduce verify all
```

Expected: concise phase-by-phase audit output; no command writes beneath `docs/results/` or `outputs/reproduction/`.

- [ ] **Step 6: Commit**

```bash
git add tests/reproducibility/test_archive_immutability.py src/afmc_fm/reproducibility
git commit -m "test: prove reproducibility verification is read-only"
```

---

## R1 Completion Gate

R1 is complete only when all of the following are true:

- `afmc-reproduce status` reports all four historical phases.
- `afmc-reproduce verify <phase>` is read-only and produces structured checks.
- `afmc-reproduce verify all` aggregates failures clearly.
- Manifest tampering, missing evidence, invalid SHAs, classification mismatch, and reproduction-output-as-official-evidence are detected.
- `docs/reproducibility/artifact-map.yaml`, `research-lineage.md`, and `README.md` exist and agree with the registry.
- Historical result roots are byte-identical before and after verification.
- Ruff is clean.
- Full pytest is green aside from already-expected CUDA-unavailable skips.
- No rebuild, rerun, Phase 0.7 execution, PR #7 merge, or protected-seed use has occurred.
