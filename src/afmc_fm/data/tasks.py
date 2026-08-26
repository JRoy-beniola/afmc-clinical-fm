import math
from dataclasses import dataclass

from afmc_fm.schema.events import EventType


@dataclass(frozen=True)
class LongitudinalTask:
    value_codes: tuple[str, ...]
    event_target_type: EventType = EventType.INTERVENTION
    event_horizon_days: float = 30.0

    def __post_init__(self) -> None:
        if not self.value_codes:
            raise ValueError("value_codes must not be empty")
        if len(set(self.value_codes)) != len(self.value_codes):
            raise ValueError("value_codes must be unique")
        if any(not code for code in self.value_codes):
            raise ValueError("value_codes must not contain empty codes")
        if not math.isfinite(self.event_horizon_days) or self.event_horizon_days <= 0:
            raise ValueError("event_horizon_days must be positive and finite")
