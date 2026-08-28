from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

phase06_cli = importlib.import_module("afmc_fm.phase06.cli")

_RAW_FILES = (
    "phase06_posthoc_pair_mechanisms.csv",
    "phase06_posthoc_associations.csv",
    "phase06_posthoc_leave_one_out.csv",
    "phase06_posthoc_screening.json",
)


def test_posthoc_cli_exposes_locked_defaults() -> None:
    parser = phase06_cli.build_parser()
    assert "posthoc-optimization" in parser.format_help()

    args = parser.parse_args(["posthoc-optimization"])
    assert args.archive == "docs/results/phase06/evidence/d4b"
    assert args.output == "outputs/phase06_posthoc_optimization"
    assert args.permutation_resamples == 10_000
    assert args.permutation_seed == 20260827


def test_archive_cli_requires_analysis_commit_and_uses_separate_namespace() -> None:
    parser = phase06_cli.build_parser()
    assert "archive-posthoc-optimization" in parser.format_help()

    args = parser.parse_args(
        [
            "archive-posthoc-optimization",
            "--analysis-commit",
            "8c9de2aaeee1e26f65e28a1dd682f8ac3effad72",
        ]
    )
    assert args.source == "outputs/phase06_posthoc_optimization"
    assert args.destination == "docs/results/phase06_posthoc_optimization"
    assert args.permutation_resamples == 10_000
    assert args.permutation_seed == 20260827


def _write_source(source: Path) -> dict[str, bytes]:
    source.mkdir(parents=True)
    payloads = {
        "phase06_posthoc_pair_mechanisms.csv": b"mechanism,value\na,1\n",
        "phase06_posthoc_associations.csv": b"mechanism,two_way_pearson\na,0.5\n",
        "phase06_posthoc_leave_one_out.csv": b"mechanism,omitted_axis\na,context\n",
        "phase06_posthoc_screening.json": (
            json.dumps(
                {
                    "classification": (
                        "structured optimization-conditioned heterogeneity worth prospective testing"
                    ),
                    "passing_primary_mechanisms": ["delta_selected_epoch"],
                    "exploratory_not_confirmatory": True,
                    "phase06_terminal_decision": "AMBIGUOUS -> STOP",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    }
    for name, content in payloads.items():
        (source / name).write_bytes(content)
    return payloads


def test_archive_posthoc_outputs_copies_exact_bytes_and_hashes(tmp_path: Path) -> None:
    posthoc_archive = importlib.import_module("afmc_fm.phase06.posthoc_archive")
    source = tmp_path / "outputs" / "phase06_posthoc_optimization"
    destination = tmp_path / "docs" / "results" / "phase06_posthoc_optimization"
    frozen = tmp_path / "docs" / "results" / "phase06"
    payloads = _write_source(source)

    provenance = posthoc_archive.archive_posthoc_outputs(
        source,
        destination,
        analysis_commit="8c9de2aaeee1e26f65e28a1dd682f8ac3effad72",
        permutation_resamples=10_000,
        permutation_seed=20260827,
        frozen_phase06_root=frozen,
    )

    for name in _RAW_FILES:
        assert (destination / "analysis" / name).read_bytes() == payloads[name]

    assert provenance["analysis_commit"] == "8c9de2aaeee1e26f65e28a1dd682f8ac3effad72"
    assert provenance["permutation_resamples"] == 10_000
    assert provenance["permutation_seed"] == 20260827
    assert provenance["phase06_terminal_decision"] == "AMBIGUOUS -> STOP"
    assert provenance["exploratory_not_confirmatory"] is True

    saved = json.loads((destination / "execution_provenance.json").read_text(encoding="utf-8"))
    assert saved == provenance

    manifest_lines = (destination / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 4
    for name in _RAW_FILES:
        digest = hashlib.sha256(payloads[name]).hexdigest()
        assert f"{digest}  analysis/{name}" in manifest_lines


def test_archive_posthoc_outputs_rejects_frozen_phase06_destination(tmp_path: Path) -> None:
    posthoc_archive = importlib.import_module("afmc_fm.phase06.posthoc_archive")
    source = tmp_path / "source"
    _write_source(source)
    frozen = tmp_path / "docs" / "results" / "phase06"

    with pytest.raises(ValueError, match="frozen Phase 0.6"):
        posthoc_archive.archive_posthoc_outputs(
            source,
            frozen / "posthoc",
            analysis_commit="8c9de2aaeee1e26f65e28a1dd682f8ac3effad72",
            frozen_phase06_root=frozen,
        )
