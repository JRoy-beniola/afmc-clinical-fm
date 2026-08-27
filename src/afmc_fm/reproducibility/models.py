from dataclasses import dataclass
from pathlib import Path
from typing import Literal

EnvironmentStatus = Literal["exact", "reconstructed", "unknown"]
ManifestBase = Literal["repository", "evidence_root", "manifest_parent"]
ResultKind = Literal["historical", "exploratory"]


@dataclass(frozen=True)
class ManifestSpec:
    """A checksum manifest and the root used to resolve its entries."""

    path: Path
    base: ManifestBase


@dataclass(frozen=True)
class PhaseDefinition:
    """Immutable repository bindings for one frozen research phase."""

    phase_id: str
    official_evidence_root: Path
    official_report: Path | None
    decision_record: Path
    expected_classification: str
    result_kind: ResultKind
    implementation_sha: str
    execution_sha: str | None
    protocol_paths: tuple[Path, ...]
    manifests: tuple[ManifestSpec, ...]
    raw_evidence_paths: tuple[Path, ...]
    derived_table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    report_source: Path | None
    environment_status: EnvironmentStatus
    rebuild_supported: bool
    rerun_supported: bool

    @property
    def manifest_paths(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.manifests)
