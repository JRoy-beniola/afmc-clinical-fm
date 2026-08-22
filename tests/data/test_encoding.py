from datetime import UTC, datetime, timedelta

import numpy as np

from afmc_fm.data.encoding import SummaryHistoryEncoder
from afmc_fm.schema.events import ClinicalEvent, EventType, PatientTimeline


def test_encoder_does_not_use_future_events():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    base = [
        ClinicalEvent("p", origin, "LAB_FAST", 1.0, "arb", EventType.OBSERVATION, "synthetic"),
        ClinicalEvent(
            "p", origin + timedelta(days=10), "LAB_FAST", 9.0, "arb", EventType.OBSERVATION, "synthetic"
        ),
    ]
    encoder = SummaryHistoryEncoder(representation_dim=16, seed=4)
    first_only = PatientTimeline("p", [base[0]])
    full = PatientTimeline("p", base)
    a = encoder.encode(first_only, cutoff_time=origin)
    b = encoder.encode(full, cutoff_time=origin)
    np.testing.assert_allclose(a, b)
