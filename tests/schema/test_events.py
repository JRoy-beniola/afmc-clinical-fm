from datetime import UTC, datetime

import pandas as pd
import pytest

from afmc_fm.schema.events import (
    ClinicalEvent,
    EventType,
    PatientTimeline,
    events_to_frame,
    frame_to_events,
)


def _event(day: int, code: str = "LAB_A") -> ClinicalEvent:
    return ClinicalEvent(
        patient_id="p1",
        start_time=datetime(2026, 1, 1 + day, tzinfo=UTC),
        code=code,
        value=1.5,
        unit="arb",
        event_type=EventType.OBSERVATION,
        source="synthetic",
        metadata={},
    )


def test_patient_timeline_sorts_events_chronologically():
    timeline = PatientTimeline("p1", [_event(2), _event(0), _event(1)])
    assert [e.start_time.day for e in timeline.events] == [1, 2, 3]


def test_patient_timeline_rejects_mixed_patient_ids():
    bad = _event(0)
    bad = ClinicalEvent(**{**bad.__dict__, "patient_id": "p2"})
    with pytest.raises(ValueError, match="same patient"):
        PatientTimeline("p1", [_event(1), bad])


def test_event_frame_round_trip_preserves_core_fields():
    events = [_event(0), _event(1, "LAB_B")]
    frame = events_to_frame(events)
    recovered = frame_to_events(frame)
    assert [e.code for e in recovered] == ["LAB_A", "LAB_B"]
    assert all(e.patient_id == "p1" for e in recovered)
    assert isinstance(frame, pd.DataFrame)
