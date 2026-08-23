from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from afmc_fm.data.encoding import HistoryEncoder
from afmc_fm.data.sequences import build_patient_sequence
from afmc_fm.data.tasks import LongitudinalTask
from afmc_fm.schema.events import ClinicalEvent, EventType, PatientTimeline

JUMP_ELIGIBLE_EVENT_CODES = frozenset({"SYNTHETIC_INTERVENTION"})


class TimelinePatient(Protocol):
    patient_id: str
    timeline: PatientTimeline


@dataclass(frozen=True)
class Phase05Sequence:
    patient_id: str
    times: np.ndarray
    representations: np.ndarray
    values: np.ndarray
    masks: np.ndarray
    event_features: np.ndarray
    update_mask: np.ndarray
    jump_eligible_mask: np.ndarray
    target_next_values: np.ndarray
    target_next_masks: np.ndarray
    target_event_within_horizon: np.ndarray
    target_event_valid: np.ndarray


def _group_by_time(events: list[ClinicalEvent]) -> list[tuple[datetime, list[ClinicalEvent]]]:
    groups: list[tuple[datetime, list[ClinicalEvent]]] = []
    for event in events:
        if not groups or groups[-1][0] != event.start_time:
            groups.append((event.start_time, [event]))
        else:
            groups[-1][1].append(event)
    return groups


def _strict_representation(
    patient: TimelinePatient,
    encoder: HistoryEncoder,
    timestamp: datetime,
) -> np.ndarray:
    pre_event_timeline = PatientTimeline(
        patient_id=patient.patient_id,
        events=[
            event
            for event in patient.timeline.events
            if event.start_time < timestamp
        ],
    )
    return encoder.encode(pre_event_timeline, timestamp)


def _jump_eligible(events: list[ClinicalEvent]) -> bool:
    return any(
        event.event_type == EventType.INTERVENTION
        and event.code in JUMP_ELIGIBLE_EVENT_CODES
        for event in events
    )


def build_phase05_sequence(
    patient: TimelinePatient,
    encoder: HistoryEncoder,
    task: LongitudinalTask,
) -> Phase05Sequence:
    historical = build_patient_sequence(patient, encoder, task)
    groups = _group_by_time(patient.timeline.events)
    if len(groups) != len(historical.times):
        raise RuntimeError("phase05 grouped timeline does not match historical sequence length")

    representations = np.zeros_like(historical.representations)
    jump_eligible_mask = np.zeros(len(groups), dtype=np.float32)
    for index, (timestamp, events) in enumerate(groups):
        representations[index] = _strict_representation(patient, encoder, timestamp)
        jump_eligible_mask[index] = float(_jump_eligible(events))

    return Phase05Sequence(
        patient_id=historical.patient_id,
        times=np.array(historical.times, copy=True),
        representations=representations,
        values=np.array(historical.values, copy=True),
        masks=np.array(historical.masks, copy=True),
        event_features=np.array(historical.event_features, copy=True),
        update_mask=np.array(historical.update_mask, copy=True),
        jump_eligible_mask=jump_eligible_mask,
        target_next_values=np.array(historical.target_next_values, copy=True),
        target_next_masks=np.array(historical.target_next_masks, copy=True),
        target_event_within_horizon=np.array(
            historical.target_event_within_horizon,
            copy=True,
        ),
        target_event_valid=np.array(historical.target_event_valid, copy=True),
    )


__all__ = [
    "JUMP_ELIGIBLE_EVENT_CODES",
    "Phase05Sequence",
    "build_phase05_sequence",
]
