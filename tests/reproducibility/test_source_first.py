from pathlib import Path

import yaml

from afmc_fm.reproducibility.registry import get_phase

_POSTHOC_ROOT = Path("docs/results/phase06_posthoc_optimization")
_REPORT_SOURCE = Path("docs/reproducibility/phase06_posthoc/report-source.md")
_REPORT_MANIFEST = Path("docs/reproducibility/phase06_posthoc/report.yaml")
_EVIDENCE_MANIFEST = _POSTHOC_ROOT / "MANIFEST.sha256"
_ARCHIVED_ANALYSIS = (
    _POSTHOC_ROOT / "analysis/phase06_posthoc_associations.csv",
    _POSTHOC_ROOT / "analysis/phase06_posthoc_leave_one_out.csv",
    _POSTHOC_ROOT / "analysis/phase06_posthoc_pair_mechanisms.csv",
    _POSTHOC_ROOT / "analysis/phase06_posthoc_screening.json",
)
_SOURCE_INPUTS = (
    _POSTHOC_ROOT / "decision_record.md",
    _POSTHOC_ROOT / "execution_provenance.json",
    *_ARCHIVED_ANALYSIS,
)


def _manifest_paths() -> set[Path]:
    paths: set[Path] = set()
    for raw_line in _EVIDENCE_MANIFEST.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        _, relative = line.split(maxsplit=1)
        paths.add(_POSTHOC_ROOT / relative.removeprefix("*"))
    return paths


def test_posthoc_registry_uses_source_first_documentation_without_inventing_report():
    phase = get_phase("phase06-posthoc")

    assert phase.report_source == _REPORT_SOURCE
    assert phase.official_report is None
    assert phase.result_kind == "exploratory"
    assert not phase.rebuild_supported
    assert not phase.rerun_supported


def test_posthoc_source_manifest_binds_only_frozen_posthoc_inputs():
    phase = get_phase("phase06-posthoc")
    payload = yaml.safe_load(_REPORT_MANIFEST.read_text(encoding="utf-8"))

    assert payload["phase_id"] == phase.phase_id
    assert payload["result_kind"] == "exploratory"
    assert payload["official_report"] is None
    assert Path(payload["report_source"]) == _REPORT_SOURCE
    assert Path(payload["decision_record"]) == phase.decision_record
    assert Path(payload["evidence_manifest"]) == _EVIDENCE_MANIFEST

    inputs = tuple(Path(path) for path in payload["source_inputs"])
    assert inputs == _SOURCE_INPUTS
    assert set(_ARCHIVED_ANALYSIS) == _manifest_paths()

    for path in inputs:
        assert path.is_relative_to(_POSTHOC_ROOT)
        assert path.is_file()
        assert path.parts[:2] != ("outputs", "reproduction")


def test_posthoc_source_preserves_exploratory_and_phase06_stop_boundaries():
    phase = get_phase("phase06-posthoc")
    text = _REPORT_SOURCE.read_text(encoding="utf-8")
    lowered = text.casefold()

    assert phase.expected_classification in text
    assert "D4-B AMBIGUOUS -> STOP" in text
    assert "exploratory" in lowered
    assert "no official historical report" in lowered
    assert "does not authorize" in lowered
    assert "protected confirmatory" in lowered


def test_source_first_policy_requires_prospective_capture_and_explicit_freeze():
    text = Path("docs/reproducibility/source-first-policy.md").read_text(encoding="utf-8")
    lowered = text.casefold()

    for required in ("python", "package", "platform", "cuda", "gpu", "git", "config"):
        assert required in lowered

    assert "editable" in lowered and "diagram" in lowered
    assert "archive/freeze" in lowered
    assert "rebuild" in lowered
    assert "rerun" in lowered
    assert "must not promote" in lowered
