from pathlib import Path

import pytest

from afmc_fm.reproducibility.registry import PHASES, get_phase, iter_phases


def test_registry_exposes_exact_historical_phases():
    assert tuple(PHASES) == ("phase0", "phase05", "phase06", "phase06-posthoc")
    assert tuple(item.phase_id for item in iter_phases()) == tuple(PHASES)


def test_phase0_decision_is_frozen():
    phase = get_phase("phase0")
    assert phase.expected_classification == (
        "PHASE 0 COMPLETE — EXTREME-LOW-N HYPOTHESIS NOT VALIDATED."
    )
    assert phase.official_evidence_root == Path("docs/results/phase0")


def test_phase05_decision_is_frozen():
    phase = get_phase("phase05")
    assert phase.expected_classification == (
        "PHASE-0.5 TERMINATED AT STAGE I-A — FLOW MECHANISM GATE NOT ESTABLISHED"
    )
    assert phase.official_evidence_root == Path("docs/results/phase05")


def test_phase06_decision_is_frozen():
    phase = get_phase("phase06")
    assert phase.expected_classification == "D4-B AMBIGUOUS -> STOP"
    assert phase.official_evidence_root == Path("docs/results/phase06")


def test_posthoc_is_exploratory():
    phase = get_phase("phase06-posthoc")
    assert phase.result_kind == "exploratory"
    assert phase.expected_classification == (
        "structured optimization-conditioned heterogeneity worth prospective testing"
    )


def test_registry_binds_known_execution_identities():
    assert get_phase("phase0").execution_sha == "d6f105eee73fcb8e9cc5987d292b1bb98a687382"
    assert get_phase("phase05").implementation_sha == (
        "50a94c06bc1c419ca55738f15f074cc06ccc3f36"
    )
    assert get_phase("phase06").implementation_sha == (
        "18f391fa89d80687f38f6c50a60a062e4524edd1"
    )
    assert get_phase("phase06").execution_sha == "6ef4d506e5a6b96b15eb58225b95ebf64d3247ea"
    assert get_phase("phase06-posthoc").implementation_sha == (
        "8c9de2aaeee1e26f65e28a1dd682f8ac3effad72"
    )


def test_unknown_phase_is_rejected():
    with pytest.raises(ValueError, match="unknown reproducibility phase"):
        get_phase("phase07")
