"""Demo-question entries in configs/claim_references.yaml. No probe and no model weights."""

from __future__ import annotations

import pytest

from glassbox.analysis.claims import load_catalog, verify_answer

CATALOG = load_catalog()


@pytest.mark.parametrize(
    ("question", "answer", "expected", "claim_id"),
    [
        (
            "What is the capital of Australia?",
            "The capital of Australia is Canberra.",
            "FAITHFUL",
            "au-capital",
        ),
        (
            "What is the capital of Australia?",
            "Sydney is the capital of Australia.",
            "HALLUCINATED",
            "au-capital",
        ),
        (
            "Why is Sydney the capital of Australia?",
            "Sydney is not the capital of Australia. The capital is Canberra.",
            "FAITHFUL",
            "au-capital",
        ),
        (
            "Why is Sydney the capital of Australia?",
            "Sydney is the capital of Australia because it is the largest city.",
            "HALLUCINATED",
            "au-capital",
        ),
        (
            "What is the capital of Austria?",
            "The capital of Austria is Vienna.",
            "FAITHFUL",
            "at-capital",
        ),
        (
            "What is the capital of Austria?",
            "The capital of Austria is Salzburg.",
            "HALLUCINATED",
            "at-capital",
        ),
        (
            "How many bones does a newborn baby have?",
            "A newborn baby has about 300 bones.",
            "FAITHFUL",
            "newborn-bones",
        ),
        (
            "How many bones does a newborn baby have?",
            "A newborn baby has 206 bones.",
            "HALLUCINATED",
            "newborn-bones",
        ),
        (
            "How many planets are in the solar system?",
            "There are eight planets in the solar system.",
            "FAITHFUL",
            "planets-count",
        ),
        (
            "Name the nine planets of the solar system.",
            "The nine planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto.",
            "HALLUCINATED",
            "planets-count",
        ),
    ],
)
def test_demo_answers_get_a_named_source(question, answer, expected, claim_id) -> None:
    result = verify_answer(answer, CATALOG, question)
    assert result["overall"] == expected
    matched = [item for item in result["claims"] if item["claim_id"] == claim_id]
    assert matched, result
    assert matched[0]["source"] != "none"
    assert "Not retrieved at runtime" in matched[0]["source"]


def test_adult_bone_count_is_not_a_contradiction() -> None:
    result = verify_answer(
        "A newborn baby has about 300 bones. Adults have 206 bones.",
        CATALOG,
        "How many bones does a newborn baby have?",
    )
    assert result["counts"]["HALLUCINATED"] == 0
    assert result["overall"] == "UNVALIDATED"


def test_scoped_claims_ignore_unrelated_questions() -> None:
    result = verify_answer(
        "Sydney is the capital of New South Wales.",
        CATALOG,
        "What is the capital of New South Wales?",
    )
    assert result["overall"] == "UNVALIDATED"
    assert result["counts"]["HALLUCINATED"] == 0


def test_scoped_claims_are_skipped_without_a_question() -> None:
    result = verify_answer("Sydney is the capital of Australia.", CATALOG)
    assert result["counts"]["HALLUCINATED"] == 0


def test_unscoped_claims_still_apply_everywhere() -> None:
    result = verify_answer("Mumbai is the capital of India.", CATALOG)
    assert result["overall"] == "HALLUCINATED"
