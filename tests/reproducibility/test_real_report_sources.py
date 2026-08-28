from pathlib import Path

import pytest

from afmc_fm.reproducibility.registry import get_phase
from afmc_fm.reproducibility.report_audit import audit_report_source
from afmc_fm.reproducibility.report_manifest import load_manifest


@pytest.mark.parametrize("phase_id", ["phase0", "phase05", "phase06"])
def test_real_historical_report_source_is_registered_and_audited(phase_id: str):
    phase = get_phase(phase_id)
    expected_source = Path("docs/reproducibility") / phase_id / "report-source.md"
    bundle = expected_source.parent

    assert phase.report_source == expected_source
    manifest = load_manifest(bundle / "report.yaml")
    assert manifest.phase_id == phase_id
    assert manifest.audit_status == "accepted"
    assert manifest.report_source == expected_source

    audit = audit_report_source(Path("."), phase, bundle)
    assert audit.ok, [(check.code, check.detail) for check in audit.checks if not check.ok]
