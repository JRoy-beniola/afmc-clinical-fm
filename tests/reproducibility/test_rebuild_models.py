from pathlib import Path

import pytest
from afmc_fm.reproducibility.rebuild_models import (
    load_rebuild_manifest,
    validate_rebuild_destination,
)


@pytest.mark.parametrize(
    "bad",
    [
        Path("docs/results/phase0"),
        Path("docs/reproducibility/phase0"),
        Path("outputs/phase0"),
        Path("outputs/reproduction/phase05/rebuild"),
    ],
)
def test_rebuild_destination_must_be_phase_isolated(tmp_path: Path, bad: Path):
    with pytest.raises(ValueError, match="outputs/reproduction/phase0/rebuild"):
        validate_rebuild_destination(tmp_path, "phase0", tmp_path / bad)


def test_default_rebuild_destination_is_reproduction_tree(tmp_path: Path):
    assert validate_rebuild_destination(tmp_path, "phase0", None) == (
        tmp_path / "outputs/reproduction/phase0/rebuild"
    ).resolve()


def test_nested_rebuild_destination_is_allowed(tmp_path: Path):
    destination = tmp_path / "outputs/reproduction/phase0/rebuild/manual"
    assert validate_rebuild_destination(tmp_path, "phase0", destination) == destination.resolve()


def test_load_rebuild_manifest_uses_strict_modes_and_relative_paths(tmp_path: Path):
    manifest_path = tmp_path / "rebuild.yaml"
    manifest_path.write_text(
        """schema_version: 1
phase_id: phase0
report_source: docs/reproducibility/phase0/report-source.md
reference_report: docs/results/phase0/report.docx
document_output: report/rebuilt.docx
artifacts:
  - kind: reference-report
    source: docs/results/phase0/report.docx
    output: reference/official-report.docx
    mode: reference-copy
""",
        encoding="utf-8",
    )

    manifest = load_rebuild_manifest(manifest_path)

    assert manifest.phase_id == "phase0"
    assert manifest.document_output == Path("report/rebuilt.docx")
    assert len(manifest.artifacts) == 1
    artifact = manifest.artifacts[0]
    assert artifact.mode == "reference-copy"
    assert artifact.generator is None


@pytest.mark.parametrize(
    "artifact_yaml, match",
    [
        (
            """  - kind: table
    source: docs/source.csv
    output: tables/out.csv
    mode: generate
""",
            "generator",
        ),
        (
            """  - kind: table
    source: docs/source.csv
    output: tables/out.csv
    mode: reference-copy
    generator: package.module:function
""",
            "generator",
        ),
        (
            """  - kind: table
    source: ../escape.csv
    output: tables/out.csv
    mode: reference-copy
""",
            "relative",
        ),
        (
            """  - kind: table
    source: docs/source.csv
    output: /tmp/out.csv
    mode: reference-copy
""",
            "relative",
        ),
        (
            """  - kind: table
    source: docs/source.csv
    output: tables/out.csv
    mode: invented
""",
            "mode",
        ),
    ],
)
def test_load_rebuild_manifest_rejects_invalid_artifacts(
    tmp_path: Path,
    artifact_yaml: str,
    match: str,
):
    manifest_path = tmp_path / "rebuild.yaml"
    manifest_path.write_text(
        """schema_version: 1
phase_id: phase0
report_source: docs/reproducibility/phase0/report-source.md
reference_report: docs/results/phase0/report.docx
document_output: report/rebuilt.docx
artifacts:
"""
        + artifact_yaml,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match):
        load_rebuild_manifest(manifest_path)


def test_load_rebuild_manifest_rejects_unknown_top_level_keys(tmp_path: Path):
    manifest_path = tmp_path / "rebuild.yaml"
    manifest_path.write_text(
        """schema_version: 1
phase_id: phase0
report_source: docs/reproducibility/phase0/report-source.md
reference_report: docs/results/phase0/report.docx
document_output: report/rebuilt.docx
artifacts: []
extra: forbidden
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown"):
        load_rebuild_manifest(manifest_path)
