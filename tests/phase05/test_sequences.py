from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
from afmc_fm.phase05.sequences import (
    JUMP_ELIGIBLE_EVENT_CODES,
    build_phase05_sequence,
)

from afmc_fm.data.encoding import SummaryHistoryEncoder
from afmc_fm.data.sequences import build_patient_sequence
from afmc_fm.data.tasks import LongitudinalTask
from afmc_fm.schema.events import ClinicalEvent, EventType, PatientTimeline


def _event(
    patient_id: str,
    timestamp: datetime,
    code: str,
    event_type: EventType,
    value: float | None = None,
    *,
    measurement_opportunity: bool = False,
) -> ClinicalEvent:
    return ClinicalEvent(
        patient_id=patient_id,
        start_time=timestamp,
        code=code,
        value=value,
        unit="arb" if value is not None else None,
        event_type=event_type,
        source="test",
        metadata=(
            {"measurement_opportunity": True}
            if measurement_opportunity
            else {}
        ),
    )


def _patient(patient_id: str, events: list[ClinicalEvent]):
    return SimpleNamespace(
        patient_id=patient_id,
        timeline=PatientTimeline(patient_id, events),
    )


def test_strict_representation_excludes_current_intervention_presence():
    patient_id = "strict-history"
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    current = origin + timedelta(days=7)
    task = LongitudinalTask(("LAB_FAST",))
    encoder = SummaryHistoryEncoder(task, representation_dim=8, seed=19)
    shared_history = [
        _event(patient_id, origin, "LAB_FAST", EventType.OBSERVATION, 0.5),
    ]
    with_intervention = _patient(
        patient_id,
        [
            *shared_history,
            _event(
                patient_id,
                current,
                "SYNTHETIC_INTERVENTION",
                EventType.INTERVENTION,
                1.0,
            ),
        ],
    )
    without_intervention = _patient(
        patient_id,
        [
            *shared_history,
            _event(patient_id, current, "FOLLOWUP", EventType.ENCOUNTER),
        ],
    )

    strict_with = build_phase05_sequence(with_intervention, encoder, task)
    strict_without = build_phase05_sequence(without_intervention, encoder, task)
    historical_with = build_patient_sequence(with_intervention, encoder, task)
    historical_without = build_patient_sequence(without_intervention, encoder, task)

    np.testing.assert_allclose(
        strict_with.representations[1],
        strict_without.representations[1],
    )
    assert strict_with.jump_eligible_mask[1] == 1.0
    assert strict_without.jump_eligible_mask[1] == 0.0
    assert not np.allclose(
        historical_with.representations[1],
        historical_without.representations[1],
    )


def test_strict_representation_is_invariant_to_current_intervention_strength():
    patient_id = "strength-invariance"
    origin = datetime(2026, 2, 1, tzinfo=UTC)
    current = origin + timedelta(days=5)
    task = LongitudinalTask(("LAB_FAST",))
    encoder = SummaryHistoryEncoder(task, representation_dim=8, seed=23)

    def build(strength: float):
        return _patient(
            patient_id,
            [
                _event(patient_id, origin, "LAB_FAST", EventType.OBSERVATION, 0.25),
                _event(
                    patient_id,
                    current,
                    "SYNTHETIC_INTERVENTION",
                    EventType.INTERVENTION,
                    strength,
                ),
            ],
        )

    weak = build_phase05_sequence(build(0.5), encoder, task)
    strong = build_phase05_sequence(build(1.5), encoder, task)
    np.testing.assert_allclose(weak.representations[1], strong.representations[1])


def test_jump_eligibility_is_frozen_to_synthetic_interventions_only():
    assert JUMP_ELIGIBLE_EVENT_CODES == frozenset({"SYNTHETIC_INTERVENTION"})
    patient_id = "eligibility"
    origin = datetime(2026, 3, 1, tzinfo=UTC)
    task = LongitudinalTask(("LAB_FAST",))
    encoder = SummaryHistoryEncoder(task, representation_dim=8, seed=29)
    patient = _patient(
        patient_id,
        [
            _event(
                patient_id,
                origin,
                "SYNTHETIC_MEASUREMENT_OPPORTUNITY",
                EventType.ENCOUNTER,
                measurement_opportunity=True,
            ),
            _event(
                patient_id,
                origin + timedelta(days=1),
                "LAB_FAST",
                EventType.OBSERVATION,
                0.2,
            ),
            _event(
                patient_id,
                origin + timedelta(days=2),
                "FOLLOWUP",
                EventType.ENCOUNTER,
            ),
            _event(
                patient_id,
                origin + timedelta(days=3),
                "SYNTHETIC_INTERVENTION",
                EventType.INTERVENTION,
                1.0,
            ),
            _event(
                patient_id,
                origin + timedelta(days=4),
                "OTHER_INTERVENTION",
                EventType.INTERVENTION,
                1.0,
            ),
        ],
    )

    sequence = build_phase05_sequence(patient, encoder, task)

    assert sequence.jump_eligible_mask.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert sequence.update_mask.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]


def test_phase05_sequence_construction_does_not_mutate_phase0_semantics():
    patient_id = "historical-regression"
    origin = datetime(2026, 4, 1, tzinfo=UTC)
    task = LongitudinalTask(("LAB_FAST",))
    encoder = SummaryHistoryEncoder(task, representation_dim=8, seed=31)
    patient = _patient(
        patient_id,
        [
            _event(patient_id, origin, "LAB_FAST", EventType.OBSERVATION, 0.1),
            _event(
                patient_id,
                origin + timedelta(days=3),
                "SYNTHETIC_INTERVENTION",
                EventType.INTERVENTION,
                0.8,
            ),
        ],
    )

    before = build_patient_sequence(patient, encoder, task)
    build_phase05_sequence(patient, encoder, task)
    after = build_patient_sequence(patient, encoder, task)

    for field in (
        "times",
        "representations",
        "values",
        "masks",
        "event_features",
        "update_mask",
        "target_next_values",
        "target_next_masks",
        "target_event_within_horizon",
        "target_event_valid",
    ):
        np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
