from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_RAW_FILES = (
    "phase06_posthoc_pair_mechanisms.csv",
    "phase06_posthoc_associations.csv",
    "phase06_posthoc_leave_one_out.csv",
    "phase06_posthoc_screening.json",
)
_ALLOWED_CLASSIFICATIONS = frozenset(
    {
        "structured optimization-conditioned heterogeneity worth prospective testing",
        "no sufficiently coherent mechanism identified",
    }
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _load_screening(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("post-hoc screening artifact is not valid JSON") from error
    if not isinstance(payload, dict):
        raise TypeError("post-hoc screening artifact must be a JSON object")
    if payload.get("classification") not in _ALLOWED_CLASSIFICATIONS:
        raise ValueError("post-hoc screening artifact has an invalid classification")
    if payload.get("exploratory_not_confirmatory") is not True:
        raise ValueError("post-hoc screening artifact must remain exploratory")
    if payload.get("phase06_terminal_decision") != "AMBIGUOUS -> STOP":
        raise ValueError("post-hoc screening artifact changes the frozen Phase 0.6 decision")
    passing = payload.get("passing_primary_mechanisms")
    if not isinstance(passing, list) or not all(isinstance(item, str) for item in passing):
        raise ValueError("post-hoc screening artifact has invalid passing mechanisms")
    return payload


def archive_posthoc_outputs(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    analysis_commit: str,
    permutation_resamples: int = 10_000,
    permutation_seed: int = 20260827,
    frozen_phase06_root: str | Path = "docs/results/phase06",
) -> dict[str, object]:
    """Copy the already-executed post-hoc artifacts byte-for-byte with provenance."""
    if not _COMMIT_PATTERN.fullmatch(analysis_commit):
        raise ValueError("analysis_commit must be a full 40-character lowercase Git SHA")
    if permutation_resamples <= 0:
        raise ValueError("permutation_resamples must be positive")

    source = Path(source_dir).expanduser().resolve()
    destination = Path(destination_dir).expanduser().resolve()
    frozen = Path(frozen_phase06_root).expanduser().resolve()
    if _is_within(destination, frozen):
        raise ValueError("post-hoc archive destination may not be inside frozen Phase 0.6 results")
    if _is_within(destination, source) or _is_within(source, destination):
        raise ValueError("post-hoc source and archive destination must be distinct and non-nested")
    if not source.is_dir():
        raise ValueError("post-hoc source output directory does not exist")

    missing = [name for name in _RAW_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError(f"post-hoc source is missing required artifacts: {', '.join(missing)}")

    screening = _load_screening(source / "phase06_posthoc_screening.json")
    analysis_dir = destination / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    hashes: dict[str, str] = {}
    for name in _RAW_FILES:
        raw = (source / name).read_bytes()
        (analysis_dir / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()

    provenance: dict[str, object] = {
        "analysis_commit": analysis_commit,
        "classification": screening["classification"],
        "passing_primary_mechanisms": screening["passing_primary_mechanisms"],
        "permutation_resamples": int(permutation_resamples),
        "permutation_seed": int(permutation_seed),
        "exploratory_not_confirmatory": True,
        "phase06_terminal_decision": "AMBIGUOUS -> STOP",
        "source_archive": "docs/results/phase06/evidence/d4b",
        "source_output": str(Path(source_dir)),
        "raw_artifact_sha256": hashes,
    }
    (destination / "execution_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = "".join(
        f"{hashes[name]}  analysis/{name}\n" for name in sorted(_RAW_FILES)
    )
    (destination / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    return provenance


__all__ = ["archive_posthoc_outputs"]
