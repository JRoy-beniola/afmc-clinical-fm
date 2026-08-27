from pathlib import Path

import yaml

from afmc_fm.reproducibility.registry import PHASES


def test_artifact_map_matches_registry():
    payload = yaml.safe_load(
        Path("docs/reproducibility/artifact-map.yaml").read_text(encoding="utf-8")
    )
    assert tuple(payload["phases"]) == tuple(PHASES)

    for phase_id, phase in PHASES.items():
        documented = payload["phases"][phase_id]
        assert documented["official_evidence_root"] == phase.official_evidence_root.as_posix()
        assert documented["expected_classification"] == phase.expected_classification
        assert documented["result_kind"] == phase.result_kind
        assert documented["implementation_sha"] == phase.implementation_sha
        assert documented["execution_sha"] == phase.execution_sha
        assert documented["environment_status"] == phase.environment_status
        assert documented["rebuild_supported"] is phase.rebuild_supported
        assert documented["rerun_supported"] is phase.rerun_supported
        assert documented["manifests"] == [path.as_posix() for path in phase.manifest_paths]
        assert documented["report_source"] == (
            phase.report_source.as_posix() if phase.report_source is not None else None
        )
