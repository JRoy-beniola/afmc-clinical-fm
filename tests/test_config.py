from pathlib import Path

import pytest

from afmc_fm.config import load_yaml


def test_load_yaml_reads_mapping(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 7\ncohort_size: 12\n", encoding="utf-8")

    config = load_yaml(config_path)

    assert config == {"seed": 7, "cohort_size": 12}


def test_load_yaml_rejects_non_mapping_root_with_type_error(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(TypeError):
        load_yaml(config_path)
