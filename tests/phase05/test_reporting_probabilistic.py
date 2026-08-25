import hashlib
import json
from pathlib import Path

import pandas as pd

from afmc_fm.phase05.reporting import write_phase05_report_artifacts

from .test_reporting import _report_fixture


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_probabilistic_report_writes_deterministic_calibration_figure(tmp_path: Path):
    output = _report_fixture(tmp_path)

    frozen_path = output / "frozen_candidate.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["uncertainty_mode"] = "decoupled"
    frozen_path.write_text(
        json.dumps(frozen, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (output / "confirmation" / "STARTED").write_text(
        json.dumps(
            {
                "frozen_candidate_sha256": hashlib.sha256(
                    frozen_path.read_bytes()
                ).hexdigest()
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    metrics_path = output / "confirmation" / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    calibration_rows = pd.DataFrame(
        [
            {**metrics.iloc[0].to_dict(), "metric": "nll", "value": 0.6},
            {**metrics.iloc[0].to_dict(), "metric": "coverage_90", "value": 0.88},
        ]
    )
    pd.concat([metrics, calibration_rows], ignore_index=True).to_csv(
        metrics_path,
        index=False,
    )

    first = write_phase05_report_artifacts(output)
    calibration = output / "figures" / "calibration.png"
    assert first["calibration"] == calibration
    assert calibration.is_file()
    first_sha = _sha(calibration)

    write_phase05_report_artifacts(output)
    assert _sha(calibration) == first_sha
