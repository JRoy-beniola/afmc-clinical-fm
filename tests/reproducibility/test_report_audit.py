import zipfile
from pathlib import Path

from afmc_fm.reproducibility.bootstrap import bootstrap_phase
from afmc_fm.reproducibility.models import PhaseDefinition
from afmc_fm.reproducibility.report_audit import audit_report_source
from afmc_fm.reproducibility.report_manifest import load_manifest


def _phase() -> PhaseDefinition:
    return PhaseDefinition(
        phase_id="fixture",
        official_evidence_root=Path("docs/results/fixture"),
        official_report=Path("docs/results/fixture/report.docx"),
        decision_record=Path("docs/results/fixture/decision.md"),
        expected_classification="FIXTURE STOP",
        result_kind="historical",
        implementation_sha="a" * 40,
        execution_sha="a" * 40,
        protocol_paths=(),
        manifests=(),
        raw_evidence_paths=(),
        derived_table_paths=(),
        figure_paths=(),
        report_source=None,
        environment_status="unknown",
        rebuild_supported=False,
        rerun_supported=False,
    )


def _write_fixture(root: Path) -> None:
    report = root / "docs/results/fixture/report.docx"
    report.parent.mkdir(parents=True)
    document = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Fixture report</w:t></w:r></w:p>
    <w:p><w:r><w:t>Recovered narrative.</w:t></w:r></w:p>
    <w:sectPr/>
  </w:body>
</w:document>
"""
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/></w:style>
</w:styles>
"""
    with zipfile.ZipFile(report, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", styles)
    (report.parent / "decision.md").write_text("Final decision: FIXTURE STOP\n", encoding="utf-8")


def test_audit_accepts_only_a_deterministic_bound_extraction(tmp_path: Path):
    phase = _phase()
    _write_fixture(tmp_path)
    destination = tmp_path / "staging"
    bootstrap_phase(root=tmp_path, phase=phase, destination=destination)

    report = audit_report_source(tmp_path, phase, destination, accept=True)

    assert report.ok
    assert all(check.ok for check in report.checks)
    manifest = load_manifest(destination / "report.yaml")
    assert manifest.audit_status == "accepted"


def test_audit_detects_source_tampering(tmp_path: Path):
    phase = _phase()
    _write_fixture(tmp_path)
    destination = tmp_path / "staging"
    bootstrap_phase(root=tmp_path, phase=phase, destination=destination)
    (destination / "report-source.md").write_text("tampered\n", encoding="utf-8")

    report = audit_report_source(tmp_path, phase, destination)

    assert not report.ok
    assert "report_source_mismatch" in {check.code for check in report.checks if not check.ok}


def test_audit_requires_frozen_decision_evidence(tmp_path: Path):
    phase = _phase()
    _write_fixture(tmp_path)
    (tmp_path / phase.decision_record).write_text("different outcome\n", encoding="utf-8")
    destination = tmp_path / "staging"
    bootstrap_phase(root=tmp_path, phase=phase, destination=destination)

    report = audit_report_source(tmp_path, phase, destination)

    assert not report.ok
    assert "classification_unbound" in {check.code for check in report.checks if not check.ok}
