"""Review-history integrity. These tests do not load Qwen or invent labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glassbox.cli import main
from glassbox.data.review_log import (
    append_reviews,
    apply_current_labels,
    current_reviews,
    display_queue_item,
    export_snapshot,
    load_history,
    parse_review_event,
    unreviewed_rows,
)
from glassbox.data.schema import Label


def test_history_keeps_revisions_and_does_not_invent_binary_labels(tmp_path: Path) -> None:
    history = tmp_path / "reviews.jsonl"
    queue = [
        {
            "question_id": "row1",
            "split": "validation",
            "question": "Where?",
            "knowledge": "The office is in Delhi.",
            "answer": "Somewhere in India.",
            "right_answer": "Delhi",
            "hallucinated_answer": "Mumbai",
            "label_name": "UNVALIDATED",
            "decision": {"automatic_label": "UNVALIDATED", "detail": "no match"},
        }
    ]
    labeled = apply_current_labels(queue, [])
    assert labeled[0]["current_label"] == "UNVALIDATED"
    assert labeled[0]["human_reviewed"] is False
    first = append_reviews(
        history,
        [
            {
                "question_id": "row1",
                "label": "FAITHFUL",
                "reviewer": "ada",
                "evidence_note": "The passage names Delhi and the answer restates that city.",
                "dataset_id": "halueval_qa",
                "source_example_id": "row1",
            }
        ],
    )
    assert first[0].revision == 1
    second = append_reviews(
        history,
        [
            {
                "question_id": "row1",
                "label": "UNVALIDATED",
                "reviewer": "ada",
                "evidence_note": "On second look the answer never asserted Delhi, only India.",
                "dataset_id": "halueval_qa",
                "source_example_id": "row1",
            }
        ],
    )
    assert second[0].revision == 2
    assert second[0].previous_label == "FAITHFUL"
    events = load_history(history)
    assert len(events) == 2
    assert current_reviews(events)["row1"].label is Label.UNVALIDATED
    snapshot = export_snapshot(events)
    assert snapshot == [
        {
            "question_id": "row1",
            "label": "UNVALIDATED",
            "reviewer": "ada",
            "evidence_note": "On second look the answer never asserted Delhi, only India.",
        }
    ]
    pending = unreviewed_rows(queue, events)
    assert pending == []


def test_invalid_or_missing_review_fields_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid label"):
        parse_review_event(
            {
                "question_id": "row1",
                "label": "WRONG",
                "reviewer": "ada",
                "evidence_note": "This note is long enough to pass the length check.",
                "dataset_id": "halueval_qa",
                "source_example_id": "row1",
            }
        )
    with pytest.raises(ValueError, match="evidence_note"):
        parse_review_event(
            {
                "question_id": "row1",
                "label": "FAITHFUL",
                "reviewer": "ada",
                "evidence_note": "too short",
                "dataset_id": "halueval_qa",
                "source_example_id": "row1",
            }
        )
    with pytest.raises(ValueError, match="reviewer"):
        parse_review_event(
            {
                "question_id": "row1",
                "label": "FAITHFUL",
                "evidence_note": "This note is long enough to pass the length check.",
                "dataset_id": "halueval_qa",
                "source_example_id": "row1",
            }
        )


def test_historical_generation_queue_is_not_rewritten(tmp_path: Path) -> None:
    forbidden = (
        tmp_path / "artifacts" / "experiments" / "halueval-generation-001" / "review_queue.json"
    )
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="historical artifact"):
        append_reviews(
            forbidden,
            [
                {
                    "question_id": "row1",
                    "label": "FAITHFUL",
                    "reviewer": "ada",
                    "evidence_note": "The passage names Delhi and the answer restates that city.",
                    "dataset_id": "halueval_qa",
                    "source_example_id": "row1",
                }
            ],
        )


def test_review_cli_shows_queue_without_assigning_labels(tmp_path: Path, capsys) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "question_id": "row1",
                        "split": "test",
                        "question": "Where?",
                        "knowledge": "The office is in Delhi.",
                        "answer": "I do not know.",
                        "right_answer": "Delhi",
                        "hallucinated_answer": "Mumbai",
                        "label_name": "UNVALIDATED",
                        "decision": {"detail": "no match"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    history = tmp_path / "history.jsonl"
    code = main(
        [
            "review-generations",
            "--queue",
            str(queue),
            "--history",
            str(history),
            "--show",
            "1",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "UNVALIDATED" in output
    assert "wording difference" in output.lower() or "wording" in output.lower()
    assert not history.exists()
    card = display_queue_item(json.loads(queue.read_text())["rows"][0])
    assert "PASSAGE" in card
    assert "GENERATED ANSWER" in card
