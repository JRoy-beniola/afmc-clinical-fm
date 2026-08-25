import json
from datetime import datetime
from pathlib import Path

from .test_store import _store


def test_confirmation_start_marker_persists_stable_utc_timestamp(tmp_path: Path):
    store = _store(tmp_path)
    store.write_frozen_candidate(
        {
            "flow_mode": "time_scaled",
            "jump_mode": "residual",
            "uncertainty_mode": "deterministic",
        }
    )

    candidate_hash = store.mark_confirmation_started()
    marker = store.output / "confirmation" / "STARTED"
    first_bytes = marker.read_bytes()
    payload = json.loads(first_bytes)

    assert payload["identity"] == store.identity
    assert payload["frozen_candidate_sha256"] == candidate_hash
    started_at = datetime.fromisoformat(payload["started_at"])
    assert started_at.tzinfo is not None
    assert started_at.utcoffset().total_seconds() == 0

    assert store.mark_confirmation_started() == candidate_hash
    assert marker.read_bytes() == first_bytes
