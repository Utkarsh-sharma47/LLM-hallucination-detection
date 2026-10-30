"""Resume bookkeeping for the missing v2 answers. No Qwen load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glassbox.cli import main
from glassbox.data.generation_resume import (
    append_jsonl,
    assert_resume_directory,
    load_completed_ids,
    manifest_matches,
    pending_question_ids,
    refuse_run_id,
    rejection_reason,
)


def test_pending_ids_skip_cached_and_completed_rows() -> None:
    pending = pending_question_ids(
        ["a", "b", "c", "d"],
        cached_ids={"a"},
        completed_ids={"c"},
    )
    assert pending == ["b", "d"]


def test_invalid_generations_are_not_accepted() -> None:
    vector = np.ones(4)
    assert (
        rejection_reason(
            answer="  ",
            model_revision="a" * 40,
            expected_revision="a" * 40,
            generated_tokens=1,
            feature=vector,
        )
        == "empty answer"
    )
    assert "revision" in rejection_reason(
        answer="Paris.",
        model_revision="b" * 40,
        expected_revision="a" * 40,
        generated_tokens=1,
        feature=vector,
    )
    assert (
        rejection_reason(
            answer="Paris.",
            model_revision="a" * 40,
            expected_revision="a" * 40,
            generated_tokens=0,
            feature=None,
        )
        == "hidden state or generated piece missing"
    )
    assert (
        rejection_reason(
            answer="Paris.",
            model_revision="a" * 40,
            expected_revision="a" * 40,
            generated_tokens=2,
            feature=vector,
        )
        is None
    )


def test_resume_appends_without_repeating_an_id(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    append_jsonl(path, {"question_id": "row1", "answer": "one"})
    append_jsonl(path, {"question_id": "row2", "answer": "two"})
    assert load_completed_ids(path) == {"row1", "row2"}
    text = path.read_text(encoding="utf-8")
    assert text.count("row1") == 1
    with pytest.raises(ValueError, match="repeats"):
        append_jsonl(path, {"question_id": "row1", "answer": "again"})
        load_completed_ids(path)


def test_historical_run_and_a_changed_manifest_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="protected"):
        refuse_run_id("halueval-generation-001")
    with pytest.raises(ValueError, match="protected"):
        refuse_run_id("final-hidden-state")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"dataset_sha256": "abc", "seed": 0, "max_questions": 2400, "model_revision": "a"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset_sha256"):
        manifest_matches(
            manifest,
            {
                "dataset_sha256": "def",
                "seed": 0,
                "max_questions": 2400,
                "model_revision": "a",
            },
        )


def test_torn_last_line_is_ignored_and_an_earlier_break_is_not(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text('{"question_id": "row1"}\n{"question_id":\n', encoding="utf-8")
    assert load_completed_ids(path) == {"row1"}
    path.write_text('{"question_id":\n{"question_id": "row2"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not JSON"):
        load_completed_ids(path)


def test_resume_directory_allows_vectors_and_refuses_a_foreign_file(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    (directory / "vectors").mkdir(parents=True)
    (directory / "answers.jsonl").write_text("", encoding="utf-8")
    (directory / "integrity.json").write_text("{}", encoding="utf-8")
    (directory / "review_queue.json").write_text("{}", encoding="utf-8")
    assert_resume_directory(directory)
    (directory / "probe.joblib").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="probe.joblib"):
        assert_resume_directory(directory)


def test_generate_command_refuses_generation_001_before_loading_a_model(
    monkeypatch, capsys
) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("model backend was constructed")

    monkeypatch.setattr("glassbox.models.backend.HuggingFaceBackend", boom)
    code = main(
        ["--offline", "generate-v2-answers", "--run-id", "halueval-generation-001", "--limit", "1"]
    )
    assert code == 1
    assert "protected" in capsys.readouterr().err.lower()
