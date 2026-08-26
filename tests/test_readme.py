from pathlib import Path


def test_readme_matches_executed_benchmark_contract():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "gate_summary.csv" in readme
    assert "ablation_metrics.csv" in readme
    assert "engineered-history" in readme
    assert "representation-only MLP" in readme
    assert "total labelled N-patient budget" in readme
    assert "site context" not in readme
    assert "full-batch" in readme
    assert "cohort, subset, and model-initialization seeds" in readme
    assert "horizon-risk classification" in readme


def test_readme_documents_native_wsl_runtime_and_diagnostics():
    readme = Path("README.md").read_text(encoding="utf-8")

    for command in (
        "cd ~/afmc-clinical-fm",
        "python3 -m venv .venv",
        "source .venv/bin/activate",
        "python -m pip install -U pip",
        'pip install -e ".[dev]"',
        "which python",
        "python -m afmc_fm.cli diagnostics --device auto --workers 1",
    ):
        assert command in readme

    assert ".venv/bin/python" in readme
    assert ".venv/Scripts/python.exe" in readme
    assert ".venv/Scripts/afmc-phase0.exe" in readme
    assert "Windows-native" in readme


def test_readme_documents_cuda_checks_and_tmpdir_runtime_guardrail():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "torch.cuda.is_available()" in readme
    assert "torch.version.cuda" in readme
    assert "torch.cuda.get_device_name(0)" in readme
    assert "TMPDIR=/tmp ./.venv/bin/pytest" in readme
    assert "runtime guardrail" in readme
    assert "not a scientific workaround" in readme
