from pathlib import Path

import pandas as pd

from afmc_fm import cli
from tests.phase05.test_cli_develop import (
    _calibrated_output,
    _develop_argv,
    _persist_synthetic_jobs,
)


def test_develop_persists_reconstructible_mechanism_metrics(tmp_path: Path, monkeypatch):
    output = _calibrated_output(tmp_path)

    def synthetic_run(jobs, *, store, **_kwargs):
        return _persist_synthetic_jobs(tuple(jobs), store)

    monkeypatch.setattr(cli, "run_phase05_jobs", synthetic_run, raising=False)
    assert cli.main(_develop_argv(output)) == 0

    metrics_path = output / "development" / "mechanism_metrics.csv"
    assert metrics_path.is_file()
    frame = pd.read_csv(metrics_path)
    assert set(frame["stage"]) == {"flow", "jump", "uncertainty"}
    assert "timing_audit" not in set(frame["stage"])
    assert len(frame) == 540
    first_bytes = metrics_path.read_bytes()

    metrics_path.unlink()

    def forbidden_rerun(*_args, **_kwargs):
        raise AssertionError("finalized development cells were rerun")

    monkeypatch.setattr(cli, "run_phase05_jobs", forbidden_rerun, raising=False)
    assert cli.main(_develop_argv(output, resume=True)) == 0
    assert metrics_path.read_bytes() == first_bytes
