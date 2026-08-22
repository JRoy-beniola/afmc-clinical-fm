from datetime import datetime
from typing import Protocol

import numpy as np

from afmc_fm.data.tasks import LongitudinalTask
from afmc_fm.schema.events import EventType, PatientTimeline


class HistoryEncoder(Protocol):
    representation_dim: int

    def encode(self, timeline: PatientTimeline, cutoff_time: datetime) -> np.ndarray: ...


class SummaryHistoryEncoder:
    def __init__(
        self,
        task: LongitudinalTask,
        representation_dim: int = 16,
        seed: int = 0,
    ) -> None:
        if representation_dim <= 0:
            raise ValueError("representation_dim must be positive")
        self.task = task
        self.representation_dim = representation_dim
        rng = np.random.default_rng(seed)
        self._projection = rng.normal(0.0, 0.25, size=(3 * len(task.value_codes) + 2, representation_dim))
        self._bias = rng.normal(0.0, 0.05, size=representation_dim)

    def encode(self, timeline: PatientTimeline, cutoff_time: datetime) -> np.ndarray:
        history = [event for event in timeline.events if event.start_time <= cutoff_time]
        first_time = history[0].start_time if history else cutoff_time
        elapsed_days = max((cutoff_time - first_time).total_seconds() / 86400.0, 0.0)
        features: list[float] = []
        for code in self.task.value_codes:
            observations = [
                event
                for event in history
                if event.event_type == EventType.OBSERVATION
                and event.code == code
                and isinstance(event.value, (int, float))
            ]
            if observations:
                last = observations[-1]
                last_value = float(last.value)
                time_since = (cutoff_time - last.start_time).total_seconds() / 86400.0
            else:
                last_value = 0.0
                time_since = elapsed_days
            features.extend((last_value, time_since, float(len(observations))))

        intervention_count = sum(
            event.event_type == EventType.INTERVENTION for event in history
        )
        features.extend((float(intervention_count), elapsed_days))
        summary = np.asarray(features, dtype=float)
        return np.tanh(summary @ self._projection + self._bias)
