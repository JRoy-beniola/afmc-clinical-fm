import hashlib
from pathlib import Path

import pytest

from afmc_fm.reproducibility.artifacts import materialize_artifact
from afmc_fm.reproducibility.rebuild_models import ArtifactDeclaration


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_copy_is_byte_identical_and_not_generated(tmp_path: Path):
    source = tmp_path / "docs/results/fixture/reference.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable-reference\x00bytes")
    destination = tmp_path / "outputs/reproduction/fixture/rebuild"
    declaration = ArtifactDeclaration(
        kind="reference",
        source=Path("docs/results/fixture/reference.bin"),
        output=Path("reference/copied.bin"),
        mode="reference-copy",
    )

    result = materialize_artifact(tmp_path, destination, declaration)

    output = destination / declaration.output
    assert output.read_bytes() == source.read_bytes()
    assert result.output == output
    assert result.sha256 == _sha256(source)
    assert result.mode == "reference-copy"
    assert not result.generated


def test_generate_loads_entry_point_and_is_deterministic(tmp_path: Path, monkeypatch):
    module = tmp_path / "fixture_generator.py"
    module.write_text(
        """def write_output(root, source, output):
    output.write_bytes(source.read_bytes().upper())
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    source = tmp_path / "inputs/source.txt"
    source.parent.mkdir(parents=True)
    source.write_text("deterministic\n", encoding="utf-8")
    declaration = ArtifactDeclaration(
        kind="generated",
        source=Path("inputs/source.txt"),
        output=Path("generated/output.txt"),
        mode="generate",
        generator="fixture_generator:write_output",
    )

    first = materialize_artifact(
        tmp_path,
        tmp_path / "outputs/reproduction/fixture/rebuild/one",
        declaration,
    )
    second = materialize_artifact(
        tmp_path,
        tmp_path / "outputs/reproduction/fixture/rebuild/two",
        declaration,
    )

    assert first.generated and second.generated
    assert first.sha256 == second.sha256
    assert first.output.read_text(encoding="utf-8") == "DETERMINISTIC\n"


def test_generator_side_effect_outside_declared_output_is_rejected(tmp_path: Path, monkeypatch):
    module = tmp_path / "escaping_generator.py"
    module.write_text(
        """def write_output(root, source, output):
    output.write_text('declared\\n', encoding='utf-8')
    (root / 'escaped.txt').write_text('escape\\n', encoding='utf-8')
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    source = tmp_path / "inputs/source.txt"
    source.parent.mkdir(parents=True)
    source.write_text("source\n", encoding="utf-8")
    declaration = ArtifactDeclaration(
        kind="generated",
        source=Path("inputs/source.txt"),
        output=Path("generated/output.txt"),
        mode="generate",
        generator="escaping_generator:write_output",
    )

    with pytest.raises(RuntimeError, match="outside declared output"):
        materialize_artifact(
            tmp_path,
            tmp_path / "outputs/reproduction/fixture/rebuild",
            declaration,
        )


def test_generator_historical_archive_mutation_is_rejected(tmp_path: Path, monkeypatch):
    historical = tmp_path / "docs/results/fixture/official.txt"
    historical.parent.mkdir(parents=True)
    historical.write_text("official\n", encoding="utf-8")
    module = tmp_path / "mutating_generator.py"
    module.write_text(
        """def write_output(root, source, output):
    output.write_text('generated\\n', encoding='utf-8')
    (root / 'docs/results/fixture/official.txt').write_text('mutated\\n', encoding='utf-8')
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    source = tmp_path / "inputs/source.txt"
    source.parent.mkdir(parents=True)
    source.write_text("source\n", encoding="utf-8")
    declaration = ArtifactDeclaration(
        kind="generated",
        source=Path("inputs/source.txt"),
        output=Path("generated/output.txt"),
        mode="generate",
        generator="mutating_generator:write_output",
    )

    with pytest.raises(RuntimeError, match="historical archive"):
        materialize_artifact(
            tmp_path,
            tmp_path / "outputs/reproduction/fixture/rebuild",
            declaration,
        )
