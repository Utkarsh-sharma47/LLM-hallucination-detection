"""Small deterministic text helpers. No model, no embeddings, no entailment.

Stemming is a crude suffix strip so that "Australia" and "Australian" meet. It is a
documented heuristic, not linguistics. It is why a paraphrase can still be recognized.
"""

from __future__ import annotations

import math
import re

_TOKEN = re.compile(r"[A-Za-z0-9']+")
_SENTENCE = re.compile(r"[^.!?]+[.!?]?")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "as",
        "and",
        "or",
        "but",
        "if",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "why",
        "how",
        "than",
        "then",
        "so",
        "such",
        "can",
        "could",
        "will",
        "would",
        "may",
        "might",
        "also",
        "into",
        "over",
        "under",
        "about",
        "not",
        "no",
        "never",
        "nor",
    ]
)
_NEGATIONS = frozenset(
    {
        "not",
        "no",
        "never",
        "nor",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "doesn't",
        "didn't",
        "don't",
        "cannot",
        "can't",
    }
)
_NEGATED_COPULA = frozenset({"isn't", "aren't", "wasn't", "weren't", "doesn't", "didn't"})
_COPULA = frozenset({"is", "are", "was", "were", "does", "did", "do"})


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def stem(word: str) -> str:
    value = word.lower()
    if value.endswith("'s"):
        value = value[:-2]
    for suffix in ("ian", "ia", "ing", "ed"):
        if value.endswith(suffix) and len(value) - len(suffix) >= 4:
            return value[: -len(suffix)]
    if value.endswith("s") and not value.endswith("ss") and len(value) > 3:
        return value[:-1]
    return value


def stems(text: str) -> list[str]:
    return [stem(word) for word in tokens(text)]


def derive_terms(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (values, anchors). Values are numbers and capitalized names; anchors are the
    remaining content words. Values keep their spelling; anchors are lower-case."""
    values: list[str] = []
    anchors: list[str] = []
    for word in tokens(text):
        lowered = word.lower()
        if lowered in STOPWORDS:
            continue
        if word[0].isdigit() or word[0].isupper():
            if word not in values:
                values.append(word)
        elif lowered not in anchors:
            anchors.append(lowered)
    return tuple(values), tuple(anchors)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences, with surrounding whitespace excluded."""
    spans: list[tuple[int, int]] = []
    for match in _SENTENCE.finditer(text):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        start = match.start() + (len(raw) - len(raw.lstrip()))
        spans.append((start, start + len(stripped)))
    return spans


def anchors_needed(count: int) -> int:
    return math.ceil(count / 2)


def anchors_present(sentence_stems: set[str], anchors: tuple[str, ...]) -> int:
    return sum(1 for anchor in anchors if stem(anchor) in sentence_stems)


def find_phrase(sentence_tokens: list[str], phrase: str) -> int:
    """Index of the first token of ``phrase`` in the sentence, or -1. Stem-based."""
    needle = stems(phrase)
    haystack = [stem(word) for word in sentence_tokens]
    if not needle:
        return -1
    for index in range(len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            return index
    return -1


def is_negated(sentence_tokens: list[str], index: int, length: int = 1) -> bool:
    """Whether the phrase at ``index`` is negated: a negation just before it, or a negated
    copula just after it ("Sydney is not", "Sydney isn't")."""
    before = [word.lower() for word in sentence_tokens[max(0, index - 2) : index]]
    if any(word in _NEGATIONS for word in before):
        return True
    after = [word.lower() for word in sentence_tokens[index + length : index + length + 2]]
    if after and after[0] in _NEGATED_COPULA:
        return True
    return len(after) == 2 and after[0] in _COPULA and after[1] in _NEGATIONS
