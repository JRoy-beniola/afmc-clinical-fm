import hashlib
from pathlib import Path

from afmc_fm.reproducibility.report_manifest import build_manifest, validate_manifest

from afmc_fm.reproducibility.models import ManifestSpec, PhaseDefinition
from afmc_fm.reproducibility.report_source import (
    ParagraphBlock,
    ReportSnapshot,
    TableBlock,
    render_markdown,
)


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
        manifests=(ManifestSpec(Path("docs/results/fixture/manifest.sha256"), "repository"),),
        raw_evidence_paths=(),
        derived_table_paths=(),
        figure_paths=(),
        report_source=None,
        environment_status="unknown",
        rebuild_supported=False,
        rerun_supported=False,
    )


def _snapshot(reference_sha256: str = "b" * 64) -> ReportSnapshot:
    return ReportSnapshot(
        reference_sha256=reference_sha256,
        blocks=(
            ParagraphBlock("Title", "Heading 1"),
            ParagraphBlock("Body"),
            TableBlock((("A", "B"), ("1", "2"))),
        ),
    )


def test_render_markdown_is_deterministic():
    snapshot = _snapshot()

    first = render_markdown(snapshot)
    second = render_markdown(snapshot)

    assert first == second
    assert first == "# Title\n\nBody\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"


def test_manifest_binds_reference_and_source_hashes(tmp_path: Path):
    phase = _phase()
    source = b"# Title\n\nBody\n"
    manifest = build_manifest(
        phase,
        _snapshot(),
        Path("docs/reproducibility/fixture/report-source.md"),
        source,
    )

    assert manifest.reference_report_sha256 == "b" * 64
    assert manifest.report_source_sha256 == hashlib.sha256(source).hexdigest()
    assert manifest.phase_id == "fixture"


def test_manifest_validation_detects_source_tampering(tmp_path: Path):
    phase = _phase()
    report = tmp_path / phase.official_report
    report.parent.mkdir(parents=True)
    report.write_bytes(b"reference")
    source_path = tmp_path / "docs/reproducibility/fixture/report-source.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# Original\n", encoding="utf-8")
    snapshot_path = source_path.parent / "extraction.json"
    snapshot_path.write_text("{}\n", encoding="utf-8")
    snapshot = _snapshot(hashlib.sha256(report.read_bytes()).hexdigest())
    manifest = build_manifest(
        phase,
        snapshot,
        Path("docs/reproducibility/fixture/report-source.md"),
        source_path.read_bytes(),
    )

    source_path.write_text("# Tampered\n", encoding="utf-8")
    failures = [check.code for check in validate_manifest(tmp_path, manifest) if not check.ok]

    assert "report_source_sha256_mismatch" in failures
