import dataclasses
import hashlib
from pathlib import Path

from afmc_fm.reproducibility.verify import verify_phase
from afmc_fm.reproducibility.models import ManifestSpec, PhaseDefinition


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
        manifests=(
            ManifestSpec(
                Path("docs/results/fixture/MANIFEST.sha256"),
                "manifest_parent",
            ),
        ),
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


def test_invalid_manifest_sha_is_detected(tmp_path):
    write_valid_fixture(tmp_path)
    manifest = tmp_path / "docs/results/fixture/MANIFEST.sha256"
    manifest.write_text("ABC  raw.csv\n", encoding="utf-8")
    report = verify_phase(tmp_path, fixture_phase())
    assert any(check.code == "invalid_sha" for check in report.checks)


def test_manifest_path_traversal_is_rejected(tmp_path):
    write_valid_fixture(tmp_path)
    manifest = tmp_path / "docs/results/fixture/MANIFEST.sha256"
    manifest.write_text("0" * 64 + "  ../outside.csv\n", encoding="utf-8")
    report = verify_phase(tmp_path, fixture_phase())
    assert any(check.code == "unsafe_manifest_path" for check in report.checks)


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
