"""Filtering and splits for the HaluEval pair protocol."""

from pathlib import Path

import pytest

from glassbox.data.halueval import (
    PINNED_SHA256,
    assign_splits,
    download_snapshot,
    filter_pairs,
    prepare_pairs,
    read_pinned_snapshot,
)
from glassbox.data.splits import assert_disjoint_ids
from glassbox.guards import DownloadDisabledError


def _row(
    question: str, right: str, hall: str, knowledge: str = "A short passage."
) -> dict[str, str]:
    return {
        "knowledge": knowledge,
        "question": question,
        "right_answer": right,
        "hallucinated_answer": hall,
    }


def test_filters_drop_copies_yes_no_and_duplicate_questions():
    rows = [
        _row("What is the capital of France?", "Paris", "Lyon"),
        _row("What is the capital of France?", "Paris", "Marseille"),
        _row("Same fact?", "Paris", "The answer is Paris."),
        _row("Are they the same?", "no", "They differ."),
        _row("Identical?", "Paris", "Paris"),
        _row("What is the capital of Spain?", "Madrid", "Barcelona"),
        _row("What is the capital of Italy?", "Rome", "Milan"),
        _row("What is the capital of Germany?", "Berlin", "Bonn"),
        _row("What is the capital of Canada?", "Ottawa", "Toronto"),
        _row("When was she born?", "9 January 1982", "She was born in 1982."),
        _row("When was he born?", "July 18, 1974", "He was born on July 18th."),
    ]
    kept, report = filter_pairs(rows)
    assert [pair.question for pair in kept] == [
        "What is the capital of France?",
        "What is the capital of Spain?",
        "What is the capital of Italy?",
        "What is the capital of Germany?",
        "What is the capital of Canada?",
    ]
    assert report.dropped["duplicate_question"] == 1
    assert report.dropped["answer_contains_the_other"] == 1
    assert report.dropped["yes_no_answer"] == 1
    assert report.dropped["identical_answers"] == 1
    assert report.dropped["numeric_restatement"] == 2
    assert report.kept == 5


def test_splits_are_deterministic_and_a_subset_does_not_move_ids():
    rows = [_row(f"Where is marker {index}?", "alpha", "beta") for index in range(10)]
    prepared = prepare_pairs(rows, seed=0)
    again = prepare_pairs(rows, seed=0)
    assert prepared.assignment.train == again.assignment.train
    assert_disjoint_ids(
        prepared.assignment.train,
        prepared.assignment.validation,
        prepared.assignment.test,
    )
    subset = prepared.assignment.subset(5)
    assert set(subset.train) <= set(prepared.assignment.train)
    assert set(subset.validation) <= set(prepared.assignment.validation)
    assert set(subset.test) <= set(prepared.assignment.test)
    assert_disjoint_ids(subset.train, subset.validation, subset.test)


def test_unpinned_snapshot_is_refused(tmp_path: Path):
    path = tmp_path / "qa_data.json"
    path.write_text(
        '{"knowledge":"k","question":"q","right_answer":"a","hallucinated_answer":"b"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=PINNED_SHA256):
        read_pinned_snapshot(path)


def test_dataset_download_stays_disabled(tmp_path: Path):
    with pytest.raises(DownloadDisabledError, match="disabled"):
        download_snapshot(tmp_path / "qa_data.json", allow_download=False)


def test_assign_splits_rejects_too_few_questions():
    kept, _report = filter_pairs([_row(f"Q{i}?", f"R{i}", f"H{i}") for i in range(4)])
    with pytest.raises(ValueError, match="five"):
        assign_splits(kept, seed=0)
