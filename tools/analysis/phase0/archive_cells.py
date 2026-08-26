from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

PHASE0 = ROOT / "docs" / "results" / "phase0"
RAW = PHASE0 / "raw"
INTEGRITY = PHASE0 / "integrity"

OFFICIAL = (
    ROOT
    / "outputs"
    / "phase0_full_cuda_d6f105eee73fcb8e9cc5987d292b1bb98a687382"
)

METRICS_PATH = RAW / "metrics.csv"

CELL_JSONL = RAW / "cell_results.jsonl"
CELL_INDEX = RAW / "cell_index.csv"
SUMMARY_PATH = RAW / "cell_archive_summary.json"

EXPECTED_CELLS = 2730
EXPECTED_METRIC_ROWS = 23520


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, numbers.Integral):
        return int(value)

    if isinstance(value, numbers.Real):
        value = float(value)
        if math.isnan(value):
            return None
        return value

    return str(value)


def canonical_metric_counter(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> Counter:
    return Counter(
        tuple(normalize_scalar(row.get(column)) for column in columns)
        for row in rows
    )


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    INTEGRITY.mkdir(parents=True, exist_ok=True)

    if not OFFICIAL.is_dir():
        raise RuntimeError(
            f"Missing official Phase-0 output directory: {OFFICIAL}"
        )

    if not METRICS_PATH.is_file():
        raise RuntimeError(
            f"Missing canonical metrics.csv: {METRICS_PATH}"
        )

    cell_paths = sorted(
        OFFICIAL.glob("shards/*/cells/*.json")
    )

    print(f"Persisted cell JSON files found: {len(cell_paths):,}")

    if len(cell_paths) != EXPECTED_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_CELLS:,} cells, "
            f"found {len(cell_paths):,}"
        )

    run_identity = None
    schema_version = None

    index_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    with CELL_JSONL.open("w", encoding="utf-8") as jsonl:
        for number, path in enumerate(cell_paths, start=1):
            raw_bytes = path.read_bytes()
            payload = json.loads(raw_bytes)

            required_top = {
                "schema_version",
                "run_identity",
                "shard",
                "cell",
                "metric_rows",
            }

            missing = required_top - set(payload)
            if missing:
                raise RuntimeError(
                    f"{path} missing fields: {sorted(missing)}"
                )

            if run_identity is None:
                run_identity = payload["run_identity"]
            elif payload["run_identity"] != run_identity:
                raise RuntimeError(
                    f"Run identity mismatch in {path}"
                )

            if schema_version is None:
                schema_version = payload["schema_version"]
            elif payload["schema_version"] != schema_version:
                raise RuntimeError(
                    f"Cell schema mismatch in {path}"
                )

            cell = payload["cell"]
            shard = payload["shard"]
            rows = payload["metric_rows"]

            if not isinstance(rows, list) or not rows:
                raise RuntimeError(
                    f"Missing metric rows in {path}"
                )

            metric_rows.extend(rows)

            # One canonical JSON object per line.
            jsonl.write(
                json.dumps(
                    payload,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            jsonl.write("\n")

            index_rows.append(
                {
                    "cell_number": number,
                    "original_relative_path": str(
                        path.relative_to(OFFICIAL)
                    ),
                    "original_sha256": hashlib.sha256(
                        raw_bytes
                    ).hexdigest(),
                    "shard_id": shard.get("shard_id"),
                    "world": shard.get("world"),
                    "cohort_seed": shard.get("cohort_seed"),
                    "subset_seed": shard.get("subset_seed"),
                    "model_seed": shard.get("model_seed"),
                    "cell_id": cell.get("cell_id"),
                    "benchmark": cell.get("benchmark"),
                    "n_train": cell.get("n_train"),
                    "model": cell.get("model"),
                    "ablation": cell.get("ablation"),
                    "metric_row_count": len(rows),
                }
            )

    print(f"Metric rows recovered from cells: {len(metric_rows):,}")

    if len(metric_rows) != EXPECTED_METRIC_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_METRIC_ROWS:,} metric rows "
            f"inside cell payloads, found {len(metric_rows):,}"
        )

    # ------------------------------------------------------------------
    # Verify that the per-cell payloads reproduce metrics.csv exactly
    # as a multiset of scientific rows.
    # ------------------------------------------------------------------

    official_metrics = pd.read_csv(METRICS_PATH, float_precision="round_trip")

    if len(official_metrics) != EXPECTED_METRIC_ROWS:
        raise RuntimeError(
            f"metrics.csv contains {len(official_metrics):,} rows, "
            f"expected {EXPECTED_METRIC_ROWS:,}"
        )

    columns = list(official_metrics.columns)

    payload_columns = set().union(
        *(row.keys() for row in metric_rows)
    )

    if set(columns) != payload_columns:
        raise RuntimeError(
            "Metric-row schema mismatch.\n"
            f"metrics.csv columns: {sorted(columns)}\n"
            f"cell payload columns: {sorted(payload_columns)}"
        )

    csv_counter = canonical_metric_counter(
        official_metrics.to_dict("records"),
        columns,
    )

    cell_counter = canonical_metric_counter(
        metric_rows,
        columns,
    )

    if csv_counter != cell_counter:
        missing_from_cells = csv_counter - cell_counter
        extra_in_cells = cell_counter - csv_counter

        raise RuntimeError(
            "Per-cell metric rows do not exactly reproduce "
            "metrics.csv.\n"
            f"Missing rows: {sum(missing_from_cells.values())}\n"
            f"Extra rows: {sum(extra_in_cells.values())}"
        )

    print("PASS: cell metric rows exactly reproduce metrics.csv")

    # ------------------------------------------------------------------
    # Cell index
    # ------------------------------------------------------------------

    index = pd.DataFrame(index_rows)

    if index["cell_id"].duplicated().any():
        duplicates = index.loc[
            index["cell_id"].duplicated(),
            "cell_id",
        ].tolist()

        raise RuntimeError(
            f"Duplicate cell IDs found: {duplicates[:10]}"
        )

    index.to_csv(CELL_INDEX, index=False)

    # ------------------------------------------------------------------
    # Archive summary
    # ------------------------------------------------------------------

    summary = {
        "phase": "phase0",
        "official_execution_sha":
            "d6f105eee73fcb8e9cc5987d292b1bb98a687382",
        "official_output_directory": str(
            OFFICIAL.relative_to(ROOT)
        ),
        "cell_schema_version": schema_version,
        "run_identity": run_identity,
        "persisted_cell_count": len(cell_paths),
        "persisted_metric_row_count": len(metric_rows),
        "metrics_csv_row_count": len(official_metrics),
        "cell_metrics_exactly_match_metrics_csv": True,
        "cell_results_jsonl": str(
            CELL_JSONL.relative_to(ROOT)
        ),
        "cell_results_jsonl_sha256": sha256_file(
            CELL_JSONL
        ),
        "cell_index_csv": str(
            CELL_INDEX.relative_to(ROOT)
        ),
        "cell_index_csv_sha256": sha256_file(
            CELL_INDEX
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Git-facing integrity record
    # ------------------------------------------------------------------

    integrity_path = (
        INTEGRITY / "github_result_archive_sha256.txt"
    )

    archive_files = [
        RAW / "metrics.csv",
        RAW / "ablation_metrics.csv",
        RAW / "gate_summary.csv",
        RAW / "run_manifest.json",
        RAW / "run_record.json",
        CELL_JSONL,
        CELL_INDEX,
        SUMMARY_PATH,
    ]

    with integrity_path.open("w", encoding="utf-8") as handle:
        for path in archive_files:
            if not path.is_file():
                raise RuntimeError(
                    f"Expected archive file missing: {path}"
                )

            handle.write(
                f"{sha256_file(path)}  "
                f"{path.relative_to(ROOT)}\n"
            )

    print()
    print("=" * 72)
    print("PHASE-0 CELL ARCHIVE COMPLETE")
    print("=" * 72)
    print(f"Cells archived:              {len(cell_paths):,}")
    print(f"Metric rows recovered:       {len(metric_rows):,}")
    print("metrics.csv equivalence:     PASS")
    print(f"JSONL:                       {CELL_JSONL}")
    print(f"Index:                       {CELL_INDEX}")
    print(f"Summary:                     {SUMMARY_PATH}")
    print(f"Integrity record:            {integrity_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
