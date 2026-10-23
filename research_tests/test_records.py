"""Tests for run records and the local environment reader."""

from __future__ import annotations

import json
import os

import pytest

from glassbox.constants import REQUIRED_METRICS
from glassbox.env import load_env_file
from glassbox.evaluation.records import build_run_record, write_json

_COMMIT = "abc123def456abc123def456abc123def456abcd"


def test_run_record_allows_a_null_early_warning_only() -> None:
    record = build_run_record(
        run_id="probe-layer-0",
        git_commit=_COMMIT,
        config={"phase": "scaffold"},
        metrics=_metrics(early_warning_lead=None),
        notes="schema check",
    )
    assert record["schema_version"] == 1
    assert record["git_commit"] == _COMMIT
    assert record["metrics"]["precision"] == 1.0
    assert record["metrics"]["early_warning_lead"] is None

    with pytest.raises(ValueError, match="precision"):
        build_run_record(
            run_id="probe-layer-0",
            git_commit=_COMMIT,
            config={},
            metrics=_metrics(precision=None),
        )


def test_run_record_rejects_secrets_unpinned_commits_and_bad_ids() -> None:
    with pytest.raises(ValueError, match="secret-like"):
        build_run_record(
            run_id="run-1",
            git_commit=_COMMIT,
            config={"HF_TOKEN": "hidden"},
            metrics=_metrics(),
        )
    with pytest.raises(ValueError, match="commit hash"):
        build_run_record(
            run_id="run-1",
            git_commit="main",
            config={},
            metrics=_metrics(),
        )
    with pytest.raises(ValueError, match="run_id"):
        build_run_record(
            run_id="../escape",
            git_commit=_COMMIT,
            config={},
            metrics=_metrics(),
        )


def test_json_writer_stays_inside_the_root(tmp_path) -> None:
    record = build_run_record(
        run_id="run-1",
        git_commit=_COMMIT,
        config={"seed": 0},
        metrics=_metrics(),
    )
    destination = write_json(
        tmp_path / "experiments" / "run-1" / "record.json", record, root=tmp_path
    )
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["run_id"] == "run-1"
    with pytest.raises(ValueError, match="outside"):
        write_json(tmp_path.parent / "record.json", record, root=tmp_path)


def test_env_file_preserves_existing_variables(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# local only",
                "HF_TOKEN=from-file",
                'HF_HOME="artifacts/hf_home"',
                "export EMPTY=",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "already-set")

    assert load_env_file(env_file) == ["HF_TOKEN", "HF_HOME", "EMPTY"]
    assert os.environ["HF_TOKEN"] == "already-set"
    assert os.environ["HF_HOME"] == "artifacts/hf_home"
    assert os.environ["EMPTY"] == ""

    assert load_env_file(env_file, override=True) == ["HF_TOKEN", "HF_HOME", "EMPTY"]
    assert os.environ["HF_TOKEN"] == "from-file"
    assert load_env_file(tmp_path / "missing.env") == []

    env_file.write_text("NOT_A_PAIR\n", encoding="utf-8")
    with pytest.raises(ValueError, match="KEY=VALUE"):
        load_env_file(env_file)


def _metrics(**overrides: float | None) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {name: 1.0 for name in REQUIRED_METRICS}
    metrics.update(overrides)
    return metrics
