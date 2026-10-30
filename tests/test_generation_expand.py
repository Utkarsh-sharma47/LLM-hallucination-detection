"""Generation expansion planning. These tests do not load Qwen or rewrite run 001."""

from __future__ import annotations

import pytest

from glassbox.cli import main
from glassbox.data.generation_expand import (
    HISTORICAL_GENERATION_RUN,
    assert_no_split_leakage,
    cached_answers_for_split,
    plan_expansion,
)
from glassbox.data.generation_splits import assign_generation_splits, subset_preserving_groups
from glassbox.data.halueval import HaluPair, SplitAssignment


def test_expansion_keeps_known_rows_in_the_same_split() -> None:
    pairs = _pairs(24)
    assignment, groups = assign_generation_splits(pairs, seed=0)
    small = subset_preserving_groups(assignment, groups, 12)
    labeled = [
        {
            "question_id": question_id,
            "split": split,
            "label_name": "FAITHFUL" if index % 2 == 0 else "HALLUCINATED",
        }
        for split, index, question_id in _enumerate(small)
    ]
    plan = plan_expansion(pairs, labeled, seed=0, max_questions=18)
    assert plan["rewrites_historical_run"] is False
    for row in labeled:
        split = row["split"]
        assert row["question_id"] in plan["question_ids"][split]
        assert row["question_id"] not in plan["new_question_ids"][split]


def test_split_leakage_is_refused() -> None:
    with pytest.raises(ValueError, match="leakage"):
        assert_no_split_leakage(["a", "b"], ["b"], ["c"])
    assert_no_split_leakage(["a"], ["b"], ["c"])


def test_cached_answers_cannot_move_splits() -> None:
    selected = SplitAssignment(train=("q0",), validation=("q1",), test=("q2",), seed=0)
    with pytest.raises(ValueError, match="move"):
        cached_answers_for_split(
            selected,
            [{"question_id": "q0", "split": "test", "label_name": "UNVALIDATED"}],
        )


def test_plan_cli_refuses_the_historical_run_id(capsys) -> None:
    code = main(
        [
            "plan-generation-expansion",
            "--run-id",
            HISTORICAL_GENERATION_RUN,
            "--max-questions",
            "24",
        ]
    )
    assert code == 1
    assert "historical run id" in capsys.readouterr().err


def test_unvalidated_rows_are_not_counted_as_binary() -> None:
    pairs = _pairs(12)
    labeled = [
        {"question_id": f"q{index}", "split": "train", "label_name": "UNVALIDATED"}
        for index in range(12)
    ]
    # Splits will not match train-only labels for every id; use empty known labels.
    plan = plan_expansion(pairs, [], seed=0, max_questions=12)
    assert plan["known_labeled_in_selected"]["validation"]["FAITHFUL"] == 0
    assert plan["known_labeled_in_selected"]["validation"]["HALLUCINATED"] == 0
    assert plan["target_met_with_known_labels"] is False
    assert labeled[0]["label_name"] == "UNVALIDATED"


def _pairs(count: int) -> list[HaluPair]:
    return [
        HaluPair(
            question_id=f"q{index}",
            source_row=index,
            question=f"Question {index}?",
            knowledge=f"passage {index // 2} about a city",
            right_answer=f"Answer {index}",
            hallucinated_answer=f"Other {index}",
        )
        for index in range(count)
    ]


def _enumerate(assignment: SplitAssignment) -> list[tuple[str, int, str]]:
    rows = []
    for split in ("train", "validation", "test"):
        for index, question_id in enumerate(assignment.ids(split)):
            rows.append((split, index, question_id))
    return rows
