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
