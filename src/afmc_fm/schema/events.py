from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from typing import Any, Mapping, Sequence

import pandas as pd


class EventType(str, Enum):
    OBSERVATION = "observation"
    INTERVENTION = "intervention"
    ENCOUNTER = "encounter"


@dataclass(frozen=True)
class ClinicalEvent:
    patient_id: str
    start_time: datetime
    code: str
    value: float | str | None
    unit: str | None
    event_type: EventType
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_time.tzinfo is None:
            raise ValueError("start_time must be timezone-aware")
        if not self.patient_id:
            raise ValueError("patient_id must be non-empty")
        if not self.code:
            raise ValueError("code must be non-empty")


@dataclass
class PatientTimeline:
    patient_id: str
    events: list[ClinicalEvent]

    def __post_init__(self) -> None:
        if any(event.patient_id != self.patient_id for event in self.events):
            raise ValueError("all events must belong to the same patient")
        self.events = sorted(self.events, key=lambda event: event.start_time)


def events_to_frame(events: Sequence[ClinicalEvent]) -> pd.DataFrame:
    rows = []
    for event in events:
        rows.append(
            {
                "patient_id": event.patient_id,
                "start_time": event.start_time.isoformat(),
                "code": event.code,
                "value": event.value,
                "unit": event.unit,
                "event_type": event.event_type.value,
                "source": event.source,
                "metadata": json.dumps(dict(event.metadata), sort_keys=True),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "patient_id",
            "start_time",
            "code",
            "value",
            "unit",
            "event_type",
            "source",
            "metadata",
        ],
    )


def frame_to_events(frame: pd.DataFrame) -> list[ClinicalEvent]:
    required = {
        "patient_id",
        "start_time",
        "code",
        "value",
        "unit",
        "event_type",
        "source",
        "metadata",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing event columns: {sorted(missing)}")

    events: list[ClinicalEvent] = []
    for row in frame.to_dict(orient="records"):
        timestamp = pd.Timestamp(row["start_time"])
        if timestamp.tzinfo is None:
            raise ValueError("start_time must include timezone information")
        metadata_raw = row.get("metadata")
        metadata = {} if pd.isna(metadata_raw) or metadata_raw == "" else json.loads(str(metadata_raw))
        value = None if pd.isna(row.get("value")) else row.get("value")
        unit = None if pd.isna(row.get("unit")) or row.get("unit") == "" else str(row.get("unit"))
        events.append(
            ClinicalEvent(
                patient_id=str(row["patient_id"]),
                start_time=timestamp.to_pydatetime(),
                code=str(row["code"]),
                value=value,
                unit=unit,
                event_type=EventType(str(row["event_type"])),
                source=str(row["source"]),
                metadata=metadata,
            )
        )
    return events
