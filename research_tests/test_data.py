"""Tests for example ids, splits, and offline dataset guards."""

from __future__ import annotations

import sys

import pytest

from glassbox.data import Label, QAExample, assert_disjoint_ids, load_examples, make_example_id


def test_example_ids_are_stable_and_path_safe() -> None:
    assert make_example_id("halueval_qa", "42") == "halueval_qa:42"
    with pytest.raises(ValueError, match="source_id"):
        make_example_id("truthfulqa", "../secret")
    with pytest.raises(ValueError, match="dataset_id"):
        make_example_id("other", "1")


def test_example_requires_a_question_and_label_source() -> None:
    example = QAExample(
        example_id=make_example_id("truthfulqa", "health-1"),
        dataset="truthfulqa",
        question="What happens if you crack your knuckles?",
        reference_answer=None,
        model_answer=None,
        label=Label.UNVALIDATED,
        label_source="unvalidated",
        split="audit",
    )
    assert example.label is Label.UNVALIDATED
    with pytest.raises(ValueError, match="question"):
        QAExample(
            example_id="truthfulqa:health-1",
            dataset="truthfulqa",
            question="   ",
            reference_answer=None,
            model_answer=None,
            label=Label.FAITHFUL,
            label_source="manual",
            split="audit",
        )


def test_splits_must_be_disjoint() -> None:
    assert_disjoint_ids(["halueval_qa:1"], ["halueval_qa:2", "truthfulqa:3"])
    with pytest.raises(ValueError, match="duplicate"):
        assert_disjoint_ids(["halueval_qa:1", "halueval_qa:1"], ["halueval_qa:2"])
    with pytest.raises(ValueError, match="split 0 and split 1"):
        assert_disjoint_ids(["halueval_qa:1"], ["halueval_qa:1"])
    with pytest.raises(ValueError, match="at least two"):
        assert_disjoint_ids(["halueval_qa:1"])


def test_dataset_loader_does_not_download(tmp_path) -> None:
    missing_halueval = tmp_path / "qa_data.json"
    missing_truthful = tmp_path / "TruthfulQA.csv"
    with pytest.raises(RuntimeError, match="disabled"):
        load_examples("halueval_qa", allow_download=False, snapshot=missing_halueval)
    sys.modules.pop("datasets", None)
    with pytest.raises(NotImplementedError, match="not implemented"):
        load_examples("truthfulqa", allow_download=True, snapshot=missing_truthful)
    assert "datasets" not in sys.modules
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_examples("synthetic")
