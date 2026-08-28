from pathlib import Path

from .models import ManifestSpec, PhaseDefinition

PHASES: dict[str, PhaseDefinition] = {
    "phase0": PhaseDefinition(
        phase_id="phase0",
        official_evidence_root=Path("docs/results/phase0"),
        official_report=Path(
            "docs/results/phase0/AFMC_Phase0_Synthetic_Validation_Report.docx"
        ),
        decision_record=Path("docs/results/phase0/official-result.md"),
        expected_classification=(
            "PHASE 0 COMPLETE — EXTREME-LOW-N HYPOTHESIS NOT VALIDATED."
        ),
        result_kind="historical",
        implementation_sha="d6f105eee73fcb8e9cc5987d292b1bb98a687382",
        execution_sha="d6f105eee73fcb8e9cc5987d292b1bb98a687382",
        protocol_paths=(Path("docs/results/phase0/raw/run_manifest.json"),),
        manifests=(
            ManifestSpec(
                Path("docs/results/phase0/integrity/repository_evidence_manifest.sha256"),
                "repository",
            ),
        ),
        raw_evidence_paths=(
            Path("docs/results/phase0/raw/metrics.csv"),
            Path("docs/results/phase0/raw/run_manifest.json"),
            Path("docs/results/phase0/raw/run_record.json"),
        ),
        derived_table_paths=(
            Path("docs/results/phase0/tables/all_full_model_test_results.csv"),
        ),
        figure_paths=(Path("docs/results/phase0/figures/learning_curves.png"),),
        report_source=Path("docs/reproducibility/phase0/report-source.md"),
        environment_status="unknown",
        rebuild_supported=True,
        rerun_supported=False,
    ),
    "phase05": PhaseDefinition(
        phase_id="phase05",
        official_evidence_root=Path("docs/results/phase05"),
        official_report=Path(
            "docs/results/phase05/AFMC_Phase0_5_Mechanistic_Validation_Report.docx"
        ),
        decision_record=Path("docs/results/phase05/official-result.md"),
        expected_classification=(
            "PHASE-0.5 TERMINATED AT STAGE I-A — FLOW MECHANISM GATE NOT ESTABLISHED"
        ),
        result_kind="historical",
        implementation_sha="50a94c06bc1c419ca55738f15f074cc06ccc3f36",
        execution_sha="50a94c06bc1c419ca55738f15f074cc06ccc3f36",
        protocol_paths=(
            Path("docs/results/phase05/raw/official_output/protocol_lock.json"),
        ),
        manifests=(
            ManifestSpec(
                Path("docs/results/phase05/integrity/repository_evidence_manifest.sha256"),
                "repository",
            ),
        ),
        raw_evidence_paths=(
            Path("docs/results/phase05/raw/official_output/execution_provenance.json"),
            Path("docs/results/phase05/raw/official_output/protocol_lock.json"),
        ),
        derived_table_paths=(Path("docs/results/phase05/tables/flow_gate_verified.csv"),),
        figure_paths=(),
        report_source=Path("docs/reproducibility/phase05/report-source.md"),
        environment_status="unknown",
        rebuild_supported=True,
        rerun_supported=False,
    ),
    "phase06": PhaseDefinition(
        phase_id="phase06",
        official_evidence_root=Path("docs/results/phase06"),
        official_report=Path(
            "docs/results/phase06/AFMC_Phase0_6_Diagnostic_Validation_Report_FINAL.docx"
        ),
        decision_record=Path("docs/results/phase06/decision_record.md"),
        expected_classification="D4-B AMBIGUOUS -> STOP",
        result_kind="historical",
        implementation_sha="18f391fa89d80687f38f6c50a60a062e4524edd1",
        execution_sha="6ef4d506e5a6b96b15eb58225b95ebf64d3247ea",
        protocol_paths=(Path("docs/results/phase06/evidence/d4b/protocol_lock.json"),),
        manifests=(
            ManifestSpec(
                Path("docs/results/phase06/provenance/FINAL_MANIFEST.sha256"),
                "evidence_root",
            ),
        ),
        raw_evidence_paths=(
            Path("docs/results/phase06/provenance/execution_chain.json"),
            Path("docs/results/phase06/evidence/d4b/analysis/phase06_d4b_effects.csv"),
        ),
        derived_table_paths=(
            Path(
                "docs/results/phase06/evidence/d4b/analysis/"
                "phase06_d4b_model_seed_summary.csv"
            ),
            Path(
                "docs/results/phase06/evidence/d4b/analysis/"
                "phase06_d4b_context_summary.csv"
            ),
        ),
        figure_paths=(),
        report_source=Path("docs/reproducibility/phase06/report-source.md"),
        environment_status="unknown",
        rebuild_supported=True,
        rerun_supported=False,
    ),
    "phase06-posthoc": PhaseDefinition(
        phase_id="phase06-posthoc",
        official_evidence_root=Path("docs/results/phase06_posthoc_optimization"),
        official_report=None,
        decision_record=Path(
            "docs/results/phase06_posthoc_optimization/decision_record.md"
        ),
        expected_classification=(
            "structured optimization-conditioned heterogeneity worth prospective testing"
        ),
        result_kind="exploratory",
        implementation_sha="8c9de2aaeee1e26f65e28a1dd682f8ac3effad72",
        execution_sha="8c9de2aaeee1e26f65e28a1dd682f8ac3effad72",
        protocol_paths=(),
        manifests=(
            ManifestSpec(
                Path("docs/results/phase06_posthoc_optimization/MANIFEST.sha256"),
                "manifest_parent",
            ),
        ),
        raw_evidence_paths=(
            Path(
                "docs/results/phase06_posthoc_optimization/analysis/"
                "phase06_posthoc_pair_mechanisms.csv"
            ),
            Path(
                "docs/results/phase06_posthoc_optimization/analysis/"
                "phase06_posthoc_associations.csv"
            ),
            Path(
                "docs/results/phase06_posthoc_optimization/analysis/"
                "phase06_posthoc_leave_one_out.csv"
            ),
            Path(
                "docs/results/phase06_posthoc_optimization/analysis/"
                "phase06_posthoc_screening.json"
            ),
            Path(
                "docs/results/phase06_posthoc_optimization/execution_provenance.json"
            ),
        ),
        derived_table_paths=(),
        figure_paths=(),
        report_source=None,
        environment_status="unknown",
        rebuild_supported=False,
        rerun_supported=False,
    ),
}


def get_phase(phase_id: str) -> PhaseDefinition:
    try:
        return PHASES[phase_id]
    except KeyError as exc:
        raise ValueError(f"unknown reproducibility phase: {phase_id}") from exc


def iter_phases() -> tuple[PhaseDefinition, ...]:
    return tuple(PHASES.values())
