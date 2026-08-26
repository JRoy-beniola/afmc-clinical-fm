from datetime import UTC, datetime, timedelta

import numpy as np

from afmc_fm.data.encoding import SummaryHistoryEncoder
from afmc_fm.data.tasks import LongitudinalTask
from afmc_fm.schema.events import ClinicalEvent, EventType, PatientTimeline

TASK = LongitudinalTask(("LAB_FAST", "LAB_SLOW", "LAB_BURDEN"))


def test_encoder_does_not_use_future_events():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    base = [
        ClinicalEvent("p", origin, "LAB_FAST", 1.0, "arb", EventType.OBSERVATION, "synthetic"),
        ClinicalEvent(
            "p", origin + timedelta(days=10), "LAB_FAST", 9.0, "arb", EventType.OBSERVATION, "synthetic"
        ),
    ]
    encoder = SummaryHistoryEncoder(TASK, representation_dim=16, seed=4)
    first_only = PatientTimeline("p", [base[0]])
    full = PatientTimeline("p", base)
    a = encoder.encode(first_only, cutoff_time=origin)
    b = encoder.encode(full, cutoff_time=origin)
    np.testing.assert_allclose(a, b)


def test_encoder_ignores_site_metadata():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    site_zero = ClinicalEvent(
        "p",
        origin,
        "LAB_FAST",
        1.0,
        "arb",
        EventType.OBSERVATION,
        "synthetic",
        metadata={"site_id": 0},
    )
    unseen_site = ClinicalEvent(
        "p",
        origin,
        "LAB_FAST",
        1.0,
        "arb",
        EventType.OBSERVATION,
        "synthetic",
        metadata={"site_id": 999},
    )
    encoder = SummaryHistoryEncoder(TASK, representation_dim=16, seed=4)
    a = encoder.encode(PatientTimeline("p", [site_zero]), cutoff_time=origin)
    b = encoder.encode(PatientTimeline("p", [unseen_site]), cutoff_time=origin)
    np.testing.assert_allclose(a, b)
