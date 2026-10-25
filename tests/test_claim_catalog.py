"""Curated claim types stay out of binary classes. No probe and no model weights."""

from __future__ import annotations

from pathlib import Path

import yaml

from glassbox.config import repo_root
from glassbox.data.schema import Label
from glassbox.verification import (
    assess_answer,
    find_fact,
    load_curated_facts,
    reference_for_question,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "claim_types.yaml"
_BINARY = {Label.FAITHFUL, Label.HALLUCINATED}


def _catalog() -> list[dict[str, str]]:
    payload = yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))
    items = payload["items"]
    assert isinstance(items, list)
    return items


def test_catalog_covers_each_claim_type() -> None:
    expected = {item["expected"] for item in _catalog()}
    assert expected == {"faithful", "hallucinated", "ambiguous", "unvalidated", "not_configured"}


def test_reference_labels_follow_listed_phrases() -> None:
    facts = load_curated_facts(repo_root() / "configs" / "curated_facts.yaml")
    seen: dict[str, str] = {}
    for item in _catalog():
        if item["expected"] == "not_configured":
            result = reference_for_question(item["question"], facts)
            assert result.label is None
            assert result.status == "not_configured"
        else:
            fact = find_fact(item["question"], facts)
            assert fact is not None
            result = assess_answer(fact, item["answer"])
            assert result.label is not None
            assert result.status == item["expected"]
            assert result.label is Label[item["expected"].upper()]
        seen[item["id"]] = result.status
    assert seen["paraphrase"] == "unvalidated"
    assert seen["paraphrase"] != "hallucinated"


def test_ambiguous_and_unvalidated_are_excluded_from_binary_classes() -> None:
    facts = load_curated_facts(repo_root() / "configs" / "curated_facts.yaml")
    eligible: list[Label] = []
    held_out: list[str] = []
    for item in _catalog():
        if item["expected"] == "not_configured":
            held_out.append(item["expected"])
            continue
        fact = find_fact(item["question"], facts)
        assert fact is not None
        label = assess_answer(fact, item["answer"]).label
        assert label is not None
        if label in _BINARY:
            eligible.append(label)
        else:
            held_out.append(item["expected"])
    assert held_out == ["ambiguous", "unvalidated", "not_configured"]
    assert eligible == [Label.FAITHFUL, Label.HALLUCINATED]
    assert Label.AMBIGUOUS not in eligible
    assert Label.UNVALIDATED not in eligible
