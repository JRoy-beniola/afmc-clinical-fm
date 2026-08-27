import pytest

from afmc_fm.reproducibility.cli import main


def test_status_lists_all_phases(capsys):
    assert main(["status"]) == 0
    output = capsys.readouterr().out
    for phase in ("phase0", "phase05", "phase06", "phase06-posthoc"):
        assert phase in output


def test_unknown_verify_phase_is_argparse_error():
    with pytest.raises(SystemExit) as exc:
        main(["verify", "phase07"])
    assert exc.value.code == 2


def test_verify_all_aggregates_failures(tmp_path, capsys):
    assert main(["verify", "all"], root=tmp_path) == 1
    output = capsys.readouterr().out
    for phase in ("phase0", "phase05", "phase06", "phase06-posthoc"):
        assert f"FAIL {phase}" in output
