"""Tests for configuration, version alignment, and revision pinning."""

from __future__ import annotations

import tomllib

import pytest
import yaml

from glassbox import __version__
from glassbox.config import (
    ProjectConfig,
    default_config_path,
    load_config,
    project_config_from_mapping,
    repo_root,
)
from glassbox.reproducibility import require_commit_hash


def test_shipped_config_matches_code_defaults() -> None:
    loaded = load_config()
    assert loaded == ProjectConfig()
    assert loaded.model.name == "Qwen/Qwen2.5-3B-Instruct"
    assert loaded.model.revision == "main"
    assert loaded.phase == "scaffold"
    assert loaded.is_offline()
    assert loaded.artifacts_path() == repo_root() / "artifacts"


def test_version_matches_pyproject() -> None:
    with (repo_root() / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    assert version == __version__ == "0.1.0"


def test_default_config_file_exists() -> None:
    assert default_config_path().is_file()


def test_unknown_top_level_key_is_rejected() -> None:
    payload = _payload()
    payload["extra"] = True
    with pytest.raises(ValueError, match="unknown"):
        project_config_from_mapping(payload)


def test_unknown_dataset_is_rejected() -> None:
    payload = _payload()
    payload["data"]["exploration_dataset"] = "synthetic"
    with pytest.raises(ValueError, match="exploration_dataset"):
        project_config_from_mapping(payload)


def test_duplicate_evaluation_datasets_are_rejected() -> None:
    payload = _payload()
    payload["data"]["evaluation_datasets"] = ["halueval_qa", "halueval_qa"]
    with pytest.raises(ValueError, match="duplicates"):
        project_config_from_mapping(payload)


def test_artifacts_dir_must_stay_relative() -> None:
    payload = _payload()
    payload["artifacts_dir"] = "../outside"
    with pytest.raises(ValueError, match="artifacts_dir"):
        project_config_from_mapping(payload)

    payload = _payload()
    payload["artifacts_dir"] = "/tmp/glassbox"
    with pytest.raises(ValueError, match="artifacts_dir"):
        project_config_from_mapping(payload)


def test_backbone_must_stay_frozen() -> None:
    payload = _payload()
    payload["probe"]["freeze_backbone"] = False
    with pytest.raises(ValueError, match="freeze_backbone"):
        project_config_from_mapping(payload)


def test_boolean_flags_reject_strings() -> None:
    payload = _payload()
    payload["model"]["allow_download"] = "false"
    with pytest.raises(ValueError, match="allow_download"):
        project_config_from_mapping(payload)


def test_commit_pin_rejects_branch_names_and_accepts_hashes() -> None:
    with pytest.raises(ValueError, match="commit hash"):
        require_commit_hash("main", field="model.revision")
    pinned = require_commit_hash("A" * 40, field="git_commit")
    assert pinned == "a" * 40


def _payload() -> dict:
    loaded = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded
