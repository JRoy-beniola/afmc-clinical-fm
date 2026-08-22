from dataclasses import dataclass
from datetime import datetime

import numpy as np

from afmc_fm.data.encoding import HistoryEncoder
from afmc_fm.schema.events import ClinicalEvent, EventType
from afmc_fm.simulator.cohort import SimulatedPatient
from afmc_fm.simulator.observation import LAB_CODES


@dataclass(frozen=True)
class PatientSequence:
    patient_id: str
    site_id: int
    times: np.ndarray
    representations: np.ndarray
    values: np.ndarray
    masks: np.ndarray
    event_features: np.ndarray
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
    patient: SimulatedPatient,
    encoder: HistoryEncoder,
) -> PatientSequence:
    groups = _group_by_time(patient.timeline.events)
    steps = len(groups)
    value_dim = len(LAB_CODES)
    values = np.zeros((steps, value_dim), dtype=np.float32)
    masks = np.zeros_like(values)
    event_features = np.zeros((steps, 3), dtype=np.float32)
    representations = np.zeros((steps, encoder.representation_dim), dtype=np.float32)

    for index, (timestamp, events) in enumerate(groups):
        representations[index] = encoder.encode(patient.timeline, timestamp)
        for event in events:
            event_features[index, list(EventType).index(event.event_type)] = 1.0
            if event.event_type == EventType.OBSERVATION and event.code in LAB_CODES:
                lab_index = LAB_CODES.index(event.code)
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
        target_events[:-1] = event_features[1:, 1]
        target_event_valid[:-1] = 1.0

    return PatientSequence(
        patient_id=patient.patient_id,
        site_id=patient.site_id,
        times=times,
        representations=representations,
        values=values,
        masks=masks,
        event_features=event_features,
        target_next_values=target_values,
        target_next_masks=target_masks,
        target_event_within_horizon=target_events,
        target_event_valid=target_event_valid,
    )
