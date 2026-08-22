from pathlib import Path


def test_ci_uses_cpu_torch_and_deduplicated_branch_triggers():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    cpu_install = (
        "pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )
    editable_install = 'pip install -e ".[dev]"'
    assert cpu_install in workflow
    assert workflow.index(cpu_install) < workflow.index(editable_install)
    assert "pull_request:" in workflow
    assert "branches:\n      - main" in workflow
