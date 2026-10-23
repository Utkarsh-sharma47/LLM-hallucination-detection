"""Curated reference checks.

A match against an accepted alias supports a faithful label for that curated
item. A wording difference is left unvalidated. It is not treated as a
hallucination. This check is separate from any probe score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from glassbox.data.schema import Label

_WHITESPACE = re.compile(r"\s+")
_NEGATIONS = {
    "not",
    "no",
    "never",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "don't",
    "doesn't",
    "didn't",
    "cannot",
    "can't",
    "nor",
}


@dataclass(frozen=True)
class CuratedFact:
    """One short question with accepted aliases and known false claims."""

    item_id: str
    question: str
    accepted: tuple[str, ...]
    false_claims: tuple[str, ...]
    split: str


@dataclass(frozen=True)
class ReferenceResult:
    """The curated check for one question, independent of the probe."""

    configured: bool
    status: str
    detail: str
    label: Label | None = None


def load_curated_facts(path: Path) -> tuple[CuratedFact, ...]:
    """Load the curated fact file. The split field is used only by training."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("Curated facts must be a mapping with an items list")
    facts: list[CuratedFact] = []
    seen: set[str] = set()
    for raw in payload["items"]:
        fact = _parse_fact(raw)
        if fact.item_id in seen:
            raise ValueError(f"Duplicate curated id {fact.item_id}")
        seen.add(fact.item_id)
        facts.append(fact)
    if len(facts) < 2:
        raise ValueError("Curated facts need at least two items")
    return tuple(facts)


def label_answer(fact: CuratedFact, answer: str) -> Label:
    """Label a model answer from asserted aliases and known false claims.

    An answer that asserts a listed false claim is ``HALLUCINATED``. An answer
    that asserts an accepted alias and no false claim is ``FAITHFUL``. Every
    other answer, including a paraphrase that matches neither list, is
    ``UNVALIDATED``.
    """
    false_hit = any(asserts_phrase(answer, claim) for claim in fact.false_claims)
    true_hit = any(asserts_phrase(answer, alias) for alias in fact.accepted)
    if false_hit and true_hit:
        return Label.AMBIGUOUS
    if false_hit:
        return Label.HALLUCINATED
    if true_hit:
        return Label.FAITHFUL
    return Label.UNVALIDATED


def reference_for_question(question: str, facts: tuple[CuratedFact, ...]) -> ReferenceResult:
    """Return the curated check, or an explicit not-configured result."""
    key = normalize_question(question)
    matches = [fact for fact in facts if normalize_question(fact.question) == key]
    if not matches:
        return ReferenceResult(
            configured=False,
            status="not_configured",
            detail=(
                "NO REFERENCE AVAILABLE. This question is outside the curated list, "
                "so no external verification was applied."
            ),
        )
    fact = matches[0]
    return ReferenceResult(
        configured=True,
        status="configured",
        detail=(
            f"Curated item {fact.item_id} is available. "
            "Accepted aliases and known false claims are listed in the verification view. "
            "A wording difference from an accepted alias is not a hallucination label."
        ),
        label=None,
    )


def assess_answer(fact: CuratedFact, answer: str) -> ReferenceResult:
    """Apply the curated rubric and keep the result out of the probe section."""
    label = label_answer(fact, answer)
    if label is Label.FAITHFUL:
        detail = (
            f"Curated item {fact.item_id}: the answer asserts an accepted alias. "
            "This reference check is not a probe score."
        )
        status = "faithful"
    elif label is Label.HALLUCINATED:
        detail = (
            f"Curated item {fact.item_id}: the answer asserts a listed false claim. "
            "This reference check is not a probe score."
        )
        status = "hallucinated"
    elif label is Label.AMBIGUOUS:
        detail = (
            f"Curated item {fact.item_id}: the answer asserts both an accepted alias and a listed false claim. "
            "It is not counted as faithful or hallucinated."
        )
        status = "ambiguous"
    else:
        detail = (
            f"Curated item {fact.item_id}: the answer was left unvalidated. "
            "A wording mismatch with the accepted alias is not a hallucination label."
        )
        status = "unvalidated"
    return ReferenceResult(configured=True, status=status, detail=detail, label=label)


def find_fact(question: str, facts: tuple[CuratedFact, ...]) -> CuratedFact | None:
    """Return the curated item whose question matches, if there is one."""
    key = normalize_question(question)
    for fact in facts:
        if normalize_question(fact.question) == key:
            return fact
    return None


def asserts_phrase(text: str, phrase: str) -> bool:
    """Return whether ``text`` asserts ``phrase`` rather than negating it."""
    normalized = _normalize(text)
    needle = _normalize(phrase)
    if not needle:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)")
    for match in pattern.finditer(normalized):
        window = normalized[max(0, match.start() - 32) : match.start()]
        words = re.findall(r"[a-z0-9']+", window)
        if any(word in _NEGATIONS for word in words[-4:]):
            continue
        return True
    return False


def normalize_question(text: str) -> str:
    """Compare questions without case, extra space, or a trailing mark."""
    return _normalize(text).rstrip("?.!")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text.strip().lower())


def _parse_fact(raw: Any) -> CuratedFact:
    if not isinstance(raw, dict):
        raise ValueError("Each curated item must be a mapping")
    item_id = raw.get("id")
    question = raw.get("question")
    accepted = raw.get("accepted")
    false_claims = raw.get("false_claims")
    split = raw.get("split")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("Curated item id must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{item_id}: question must be a non-empty string")
    if not isinstance(accepted, list) or not all(
        isinstance(item, str) and item for item in accepted
    ):
        raise ValueError(f"{item_id}: accepted must be a list of strings")
    if not isinstance(false_claims, list) or not all(
        isinstance(item, str) and item for item in false_claims
    ):
        raise ValueError(f"{item_id}: false_claims must be a list of strings")
    if split not in {"train", "test"}:
        raise ValueError(f"{item_id}: split must be train or test")
    return CuratedFact(
        item_id=item_id,
        question=question.strip(),
        accepted=tuple(accepted),
        false_claims=tuple(false_claims),
        split=split,
    )
