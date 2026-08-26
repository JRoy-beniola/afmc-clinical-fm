from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np

from afmc_fm.data.encoding import HistoryEncoder
from afmc_fm.data.tasks import LongitudinalTask
from afmc_fm.schema.events import ClinicalEvent, EventType, PatientTimeline


class TimelinePatient(Protocol):
    patient_id: str
    timeline: PatientTimeline


@dataclass(frozen=True)
class PatientSequence:
    patient_id: str
    times: np.ndarray
    representations: np.ndarray
    values: np.ndarray
    masks: np.ndarray
    event_features: np.ndarray
    update_mask: np.ndarray
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


def build_patient_sequence(
    patient: TimelinePatient,
    encoder: HistoryEncoder,
    task: LongitudinalTask,
) -> PatientSequence:
    groups = _group_by_time(patient.timeline.events)
    steps = len(groups)
    value_dim = len(task.value_codes)
    values = np.zeros((steps, value_dim), dtype=np.float32)
    masks = np.zeros_like(values)
    event_features = np.zeros((steps, len(EventType)), dtype=np.float32)
    update_mask = np.zeros(steps, dtype=np.float32)
    representations = np.zeros((steps, encoder.representation_dim), dtype=np.float32)

    for index, (timestamp, events) in enumerate(groups):
        representations[index] = encoder.encode(patient.timeline, timestamp)
        for event in events:
            event_features[index, list(EventType).index(event.event_type)] = 1.0
            is_opportunity_anchor = (
                event.event_type == EventType.ENCOUNTER
                and bool(event.metadata.get("measurement_opportunity", False))
            )
            if not is_opportunity_anchor:
                update_mask[index] = 1.0
            if event.event_type == EventType.OBSERVATION and event.code in task.value_codes:
                lab_index = task.value_codes.index(event.code)
                values[index, lab_index] = float(event.value)
                masks[index, lab_index] = 1.0

    if groups:
        origin = groups[0][0]
        times = np.asarray(
            [(timestamp - origin).total_seconds() / 86400.0 for timestamp, _ in groups],
            dtype=np.float32,
        )
    else:
        times = np.empty(0, dtype=np.float32)

    target_values = np.zeros_like(values)
    target_masks = np.zeros_like(masks)
    target_events = np.zeros(steps, dtype=np.float32)
    target_event_valid = np.zeros(steps, dtype=np.float32)
    if steps > 1:
        target_values[:-1] = values[1:]
        target_masks[:-1] = masks[1:]

    if groups:
        followup_end = groups[-1][0]
        target_event_times = [
            event.start_time
            for event in patient.timeline.events
            if event.event_type == task.event_target_type
        ]
        horizon = timedelta(days=task.event_horizon_days)
        for index, (timestamp, _) in enumerate(groups):
            horizon_end = timestamp + horizon
            if horizon_end > followup_end:
                continue
            target_event_valid[index] = 1.0
            target_events[index] = float(
                any(
                    timestamp < event_time <= horizon_end
                    for event_time in target_event_times
                )
            )

    return PatientSequence(
        patient_id=patient.patient_id,
        times=times,
        representations=representations,
        values=values,
        masks=masks,
        event_features=event_features,
        update_mask=update_mask,
        target_next_values=target_values,
        target_next_masks=target_masks,
        target_event_within_horizon=target_events,
        target_event_valid=target_event_valid,
    )
