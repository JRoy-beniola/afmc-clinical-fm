from pathlib import Path

import pytest

from afmc_fm.reproducibility.bootstrap import bootstrap_phase
from afmc_fm.reproducibility.models import PhaseDefinition


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


def test_bootstrap_refuses_historical_destination(tmp_path: Path):
    with pytest.raises(ValueError, match="historical evidence"):
        bootstrap_phase(
            root=tmp_path,
            phase=_phase(),
            destination=tmp_path / "docs/results/phase0/reproducibility",
        )
