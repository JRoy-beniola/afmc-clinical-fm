import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from afmc_fm.reproducibility.environment import (
    capture_current_environment,
    classify_historical_environment,
)

from afmc_fm.reproducibility.models import ManifestSpec, PhaseDefinition
from afmc_fm.reproducibility.registry import get_phase


def _phase(
    *,
    environment_status: str = "unknown",
    raw_evidence_paths: tuple[Path, ...] = (),
    manifests: tuple[ManifestSpec, ...] = (),
) -> PhaseDefinition:
    return PhaseDefinition(
        phase_id="fixture",
        official_evidence_root=Path("docs/results/fixture"),
        official_report=None,
        decision_record=Path("docs/results/fixture/decision.md"),
        expected_classification="FIXTURE",
        result_kind="historical",
        implementation_sha="a" * 40,
        execution_sha="a" * 40,
        protocol_paths=(),
        manifests=manifests,
        raw_evidence_paths=raw_evidence_paths,
        derived_table_paths=(),
        figure_paths=(),
        report_source=None,
        environment_status=environment_status,  # type: ignore[arg-type]
        rebuild_supported=False,
        rerun_supported=False,
    )


def test_exact_requires_explicit_status_registered_lock_and_valid_digest(tmp_path: Path):
    archive = tmp_path / "docs/results/fixture"
    archive.mkdir(parents=True)
    lock = archive / "environment.lock"
    lock.write_text("python=3.11.9\ntorch=2.3.1\n", encoding="utf-8")
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    manifest = archive / "MANIFEST.sha256"
    manifest.write_text(f"{digest}  environment.lock\n", encoding="utf-8")
    phase = _phase(
        environment_status="exact",
        raw_evidence_paths=(Path("docs/results/fixture/environment.lock"),),
        manifests=(
            ManifestSpec(
                Path("docs/results/fixture/MANIFEST.sha256"),
                "manifest_parent",
            ),
        ),
    )

    record = classify_historical_environment(tmp_path, phase)

    assert record.status == "exact"
    assert Path("docs/results/fixture/environment.lock") in record.source_paths
    assert not any("retroactive" in item for item in record.limitations)


def test_invalid_exact_lock_cannot_remain_exact(tmp_path: Path):
    archive = tmp_path / "docs/results/fixture"
    archive.mkdir(parents=True)
    lock = archive / "environment.lock"
    lock.write_text("python=3.11.9\n", encoding="utf-8")
    manifest = archive / "MANIFEST.sha256"
    manifest.write_text(f"{'0' * 64}  environment.lock\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    phase = _phase(
        environment_status="exact",
        raw_evidence_paths=(Path("docs/results/fixture/environment.lock"),),
        manifests=(
            ManifestSpec(
                Path("docs/results/fixture/MANIFEST.sha256"),
                "manifest_parent",
            ),
        ),
    )

    record = classify_historical_environment(tmp_path, phase)

    assert record.status == "reconstructed"
    assert any("exact" in item.casefold() for item in record.limitations)


def test_archived_runtime_metadata_is_reconstructed_not_exact(tmp_path: Path):
    archive = tmp_path / "docs/results/fixture"
    archive.mkdir(parents=True)
    provenance = archive / "execution_provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "runtime_metadata": {
                    "python_version": "3.14.4",
                    "os": "Linux-WSL2",
                    "cuda_runtime": "13.0",
                    "gpu_name": "Fixture GPU",
                    "library_versions": {
                        "numpy": "2.5.2",
                        "torch": "2.13.0+cu130",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    phase = _phase(
        raw_evidence_paths=(Path("docs/results/fixture/execution_provenance.json"),)
    )

    record = classify_historical_environment(tmp_path, phase)

    assert record.status == "reconstructed"
    assert record.python_version == "3.14.4"
    assert record.platform == "Linux-WSL2"
    assert record.torch == "2.13.0+cu130"
    assert record.cuda == "13.0"
    assert record.gpu == "Fixture GPU"
    assert dict(record.packages)["numpy"] == "2.5.2"
    assert any("not an exact" in item.casefold() for item in record.limitations)


def test_retroactive_requirements_file_cannot_create_exact_status(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("torch==2.13.0\n", encoding="utf-8")
    phase = _phase(environment_status="exact")

    record = classify_historical_environment(tmp_path, phase)

    assert record.status == "reconstructed"
    assert Path("requirements.txt") in record.source_paths
    assert any("exact" in item.casefold() for item in record.limitations)


def test_insufficient_evidence_is_unknown(tmp_path: Path):
    record = classify_historical_environment(tmp_path, _phase())

    assert record.status == "unknown"
    assert record.source_paths == ()
    assert record.python_version is None
    assert record.packages == ()


def test_classification_is_read_only(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    classify_historical_environment(tmp_path, _phase())

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_current_capture_is_new_reproduction_environment_not_historical_exact():
    record = capture_current_environment()

    assert record.status == "reconstructed"
    assert record.python_version
    assert record.platform
    assert isinstance(record.packages, tuple)
    assert any("current reproduction environment" in item for item in record.limitations)


@pytest.mark.parametrize("phase_id", ["phase0", "phase05", "phase06"])
def test_registered_historical_environments_are_never_auto_promoted_to_exact(phase_id: str):
    phase = dataclasses.replace(get_phase(phase_id), environment_status="unknown")

    record = classify_historical_environment(Path.cwd(), phase)

    assert record.status in {"reconstructed", "unknown"}
    assert record.status != "exact"
