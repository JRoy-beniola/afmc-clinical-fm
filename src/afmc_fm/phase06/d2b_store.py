from __future__ import annotations

from pathlib import Path

import pandas as pd

from afmc_fm.phase06.store import (
    PHASE06_STORE_SCHEMA_VERSION,
    Phase06Store,
    _atomic_write_bytes,
    _canonical_json_bytes,
    _cell_from_payload,
    _metric_columns_from_payload,
    _validate_expected_cell_ids,
)

_D2B_STAGE = "d2b"


class Phase06D2BStore(Phase06Store):
    """Child store that accepts only the separately authorized D2-B stage."""

    def _stage_dir(self, stage: str) -> Path:
        if stage != _D2B_STAGE:
            raise ValueError("D2-B child store only supports stage d2b")
        return self.output / "stages" / _D2B_STAGE

    def validate_resume(
        self,
        stage: str,
        *,
        expected_cell_ids: set[str] | frozenset[str],
    ) -> frozenset[str]:
        self._validate_protocol_identity()
        self._stage_dir(stage)
        expected = _validate_expected_cell_ids(expected_cell_ids)
        observed = self._observed_cell_ids(stage)
        unexpected = observed - expected
        if unexpected:
            raise ValueError(
                "unexpected persisted cell IDs: " + ", ".join(sorted(unexpected))
            )

        for cell_id in sorted(observed):
            payload = self._read_cell_payload(stage, cell_id)
            cell = _cell_from_payload(payload)
            if cell.stage != _D2B_STAGE:
                raise ValueError(f"persisted child cell is not D2-B: {cell_id}")
            self._validate_persisted_bundle(cell)

        marker = self._stage_dir(stage) / "COMPLETE"
        if marker.exists():
            from afmc_fm.phase06.store import _load_json_object

            marker_payload = _load_json_object(marker)
            expected_marker = {
                "schema_version": PHASE06_STORE_SCHEMA_VERSION,
                "identity": self.identity,
                "stage": stage,
                "cell_ids": sorted(expected),
            }
            if marker_payload != expected_marker or observed != expected:
                raise ValueError("complete marker does not match exact expected cell set")
        return observed

    def mark_stage_complete(
        self,
        stage: str,
        expected_cell_ids: set[str] | frozenset[str],
    ) -> None:
        self._stage_dir(stage)
        expected = _validate_expected_cell_ids(expected_cell_ids)
        if not expected:
            raise ValueError("expected cell IDs must be non-empty")
        observed = self.validate_resume(stage, expected_cell_ids=expected)
        if observed != expected:
            missing = expected - observed
            if missing:
                raise ValueError("missing expected cells: " + ", ".join(sorted(missing)))
            unexpected = observed - expected
            raise ValueError(
                "unexpected persisted cell IDs: " + ", ".join(sorted(unexpected))
            )
        payload = {
            "schema_version": PHASE06_STORE_SCHEMA_VERSION,
            "identity": self.identity,
            "stage": stage,
            "cell_ids": sorted(expected),
        }
        data = _canonical_json_bytes(payload)
        marker = self._stage_dir(stage) / "COMPLETE"
        if marker.exists():
            if marker.is_file() and marker.read_bytes() == data:
                return
            raise ValueError("conflicting stage completion marker")
        _atomic_write_bytes(marker, data)

    def load_stage_metrics(self, stage: str) -> pd.DataFrame:
        self._validate_protocol_identity()
        self._stage_dir(stage)
        frames: list[pd.DataFrame] = []
        for cell_id in sorted(self._observed_cell_ids(stage)):
            payload = self._read_cell_payload(stage, cell_id)
            cell = _cell_from_payload(payload)
            if cell.stage != _D2B_STAGE:
                raise ValueError(f"persisted child cell is not D2-B: {cell_id}")
            self._validate_persisted_bundle(cell)
            rows = payload.get("metric_rows")
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"invalid persisted metric rows: {cell_id}")
            columns = _metric_columns_from_payload(payload, cell_id)
            frames.append(pd.DataFrame(rows, columns=columns))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)


__all__ = ["Phase06D2BStore"]
