from pathlib import Path

from .models import PhaseDefinition


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validated_destination(root: Path, destination: Path) -> Path:
    repository = Path(root).resolve()
    candidate = Path(destination)
    if not candidate.is_absolute():
        candidate = repository / candidate
    candidate = candidate.resolve()

    historical = (repository / "docs/results").resolve()
    reproduction = (repository / "outputs/reproduction").resolve()
    if _within(candidate, historical):
        raise ValueError("bootstrap destination may not be historical evidence")
    if _within(candidate, reproduction):
        raise ValueError("bootstrap destination may not be reproduction output")
    return candidate


def bootstrap_phase(
    *,
    root: Path,
    phase: PhaseDefinition,
    destination: Path,
) -> Path:
    """Validate the one-time report-bootstrap destination."""

    del phase
    return _validated_destination(root, destination)
