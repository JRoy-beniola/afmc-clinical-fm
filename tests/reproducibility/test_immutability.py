import hashlib
from pathlib import Path

from afmc_fm.reproducibility.cli import main
from afmc_fm.reproducibility.registry import iter_phases


def tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_verify_all_preserves_historical_archives(capsys):
    repository = Path(".")
    before = {
        phase.phase_id: tree_snapshot(repository / phase.official_evidence_root)
        for phase in iter_phases()
    }

    assert main(["verify", "all"], root=repository) == 0
    output = capsys.readouterr().out
    for phase in iter_phases():
        assert f"PASS {phase.phase_id}" in output

    after = {
        phase.phase_id: tree_snapshot(repository / phase.official_evidence_root)
        for phase in iter_phases()
    }
    assert after == before
    assert not (repository / "outputs/reproduction").exists()
