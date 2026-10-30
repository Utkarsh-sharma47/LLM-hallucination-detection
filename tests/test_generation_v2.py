"""Generation dataset v2 assembly. These tests do not load Qwen or invent labels."""

from __future__ import annotations

import numpy as np
import pytest

from glassbox.cli import main
from glassbox.data.generation_expand import plan_expansion
from glassbox.data.generation_splits import assign_generation_splits, subset_preserving_groups
from glassbox.data.generation_v2 import assemble_generation_v2, jsonable_rows
from glassbox.data.halueval import HaluPair


def test_v2_keeps_cached_split_membership_and_does_not_invent_labels() -> None:
    pairs = _pairs(24)
    assignment, groups = assign_generation_splits(pairs, seed=0)
    small = subset_preserving_groups(assignment, groups, 12)
    cached = []
    for split in ("train", "validation", "test"):
        for index, question_id in enumerate(small.ids(split)):
            cached.append(
                {
                    "question_id": question_id,
                    "split": split,
                    "label_name": "FAITHFUL" if index % 2 == 0 else "UNVALIDATED",
                    "label": 0 if index % 2 == 0 else -1,
                    "answer": "cached answer",
                }
            )
    plan = plan_expansion(pairs, cached, seed=0, max_questions=18)
    assembled = assemble_generation_v2(plan, cached)
    assert assembled["rewrites_historical_run"] is False
    assert assembled["invented_human_labels"] is False
    assert assembled["unvalidated_not_auto_labeled"] is True
    assert assembled["target_met"] is False
    cached_ids = {row["question_id"] for row in cached}
    for row in assembled["rows"]:
        if row["question_id"] in cached_ids:
            original = next(item for item in cached if item["question_id"] == row["question_id"])
            assert row["split"] == original["split"]
            assert row["label_name"] == original["label_name"]
    assert assembled["missing_question_count"] > 0
    dumped = jsonable_rows(assembled["rows"])
    assert all("vector" not in row for row in dumped)


def test_v2_refuses_to_replace_a_cached_answer() -> None:
    pairs = _pairs(24)
    cached = [
        {
            "question_id": "q0",
            "split": "train",
            "label_name": "UNVALIDATED",
            "label": -1,
            "answer": "old",
        }
    ]
    plan = plan_expansion(pairs, [], seed=0, max_questions=12)
    generated = [{"question_id": "q0", "split": "train", "label_name": "FAITHFUL", "answer": "new"}]
    # q0 may or may not be in the selected plan; force overlap through cached_answers.
    selected_ids = plan["question_ids"]["train"][:1]
    cached[0]["question_id"] = selected_ids[0]
    generated[0]["question_id"] = selected_ids[0]
    cached[0]["split"] = "train"
    with pytest.raises(ValueError, match="replace a cached"):
        assemble_generation_v2(plan, cached, generated_rows=generated)


def test_v2_cli_refuses_historical_run(capsys) -> None:
    code = main(
        ["build-generation-v2", "--run-id", "halueval-generation-001", "--max-questions", "12"]
    )
    assert code == 1
    assert "protected" in capsys.readouterr().err


def test_v2_cli_refuses_live_pair_probe_name(capsys) -> None:
    code = main(["build-generation-v2", "--run-id", "final-hidden-state", "--max-questions", "12"])
    assert code == 1
    assert "protected" in capsys.readouterr().err


def test_v2_attaches_cached_vectors_without_mean_pool() -> None:
    pairs = _pairs(24)
    assignment, groups = assign_generation_splits(pairs, seed=0)
    small = subset_preserving_groups(assignment, groups, 12)
    cached = []
    vectors = {}
    for split in ("train", "validation", "test"):
        for question_id in small.ids(split):
            cached.append(
                {
                    "question_id": question_id,
                    "split": split,
                    "label_name": "FAITHFUL",
                    "label": 0,
                    "answer": "Paris",
                }
            )
            vectors[question_id] = np.ones(3, dtype=np.float64)
    plan = plan_expansion(pairs, cached, seed=0, max_questions=12)
    assembled = assemble_generation_v2(plan, cached, cached_vectors=vectors)
    assert assembled["mean_pool_rows"] == 0
    assert all(
        row.get("vector") is not None for row in assembled["rows"] if row["question_id"] in vectors
    )


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
