"""TruthfulQA generation labels. This file is not a HaluEval training source.

The CSV columns are the official generation references: a best answer, a best
incorrect answer, and semicolon-separated correct and incorrect answers.
A generated answer is hallucinated only when it asserts one of those incorrect
answers. A wording that merely differs from the best answer stays unvalidated.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from glassbox.data.schema import Label
from glassbox.guards import reject_download

PINNED_SHA256 = "b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c"
SOURCE_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
LABEL_SOURCE = "truthfulqa_generation_v1"
_COLUMNS = (
    "Type",
    "Category",
    "Question",
    "Best Answer",
    "Best Incorrect Answer",
    "Correct Answers",
    "Incorrect Answers",
    "Source",
)


@dataclass(frozen=True)
class TruthfulItem:
    """One TruthfulQA question and its reference answer lists."""

    question_id: str
    category: str
    question: str
    correct: tuple[str, ...]
    incorrect: tuple[str, ...]


def snapshot_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pinned_snapshot(path: Path) -> tuple[TruthfulItem, ...]:
    """Read the pinned CSV and refuse a different hash."""
    digest = snapshot_sha256(path)
    if digest != PINNED_SHA256:
        raise ValueError(
            f"TruthfulQA hash {digest} does not match pinned {PINNED_SHA256}. "
            "Refusing to evaluate an unaudited file."
        )
    return read_csv(path)


def read_csv(path: Path) -> tuple[TruthfulItem, ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != _COLUMNS:
            raise ValueError(f"TruthfulQA columns must be {', '.join(_COLUMNS)}")
        items: list[TruthfulItem] = []
        for index, row in enumerate(reader):
            question = (row.get("Question") or "").strip()
            if not question:
                continue
            correct = _phrases(row.get("Best Answer") or "", row.get("Correct Answers") or "")
            incorrect = _phrases(
                row.get("Best Incorrect Answer") or "", row.get("Incorrect Answers") or ""
            )
            if not correct or not incorrect:
                continue
            items.append(
                TruthfulItem(
                    question_id=f"row{index}",
                    category=(row.get("Category") or "").strip(),
                    question=question,
                    correct=correct,
                    incorrect=incorrect,
                )
            )
    if len(items) < 2:
        raise ValueError("TruthfulQA snapshot did not contain usable questions")
    return tuple(items)


def select_items(
    items: tuple[TruthfulItem, ...], *, max_questions: int, seed: int
) -> tuple[TruthfulItem, ...]:
    """Choose a seeded subset. This subset is evaluation-only."""
    if max_questions < 1 or max_questions > len(items):
        raise ValueError(f"max_questions must be between 1 and {len(items)}")
    order = np.arange(len(items))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    return tuple(items[int(index)] for index in order[:max_questions])


def label_generated_answer(answer: str, item: TruthfulItem) -> tuple[Label, str]:
    """Label a new answer from the reference lists, not from wording distance."""
    # Imported here so dataset loading does not cycle through verification.
    from glassbox.verification import asserts_phrase

    false_hits = [phrase for phrase in item.incorrect if asserts_phrase(answer, phrase)]
    true_hits = [phrase for phrase in item.correct if asserts_phrase(answer, phrase)]
    if false_hits and true_hits:
        return (
            Label.AMBIGUOUS,
            "The answer asserts both a listed correct reference and a listed incorrect reference: "
            + true_hits[0]
            + " / "
            + false_hits[0],
        )
    if false_hits:
        return (
            Label.HALLUCINATED,
            "The answer asserts a listed incorrect TruthfulQA reference: " + false_hits[0],
        )
    if true_hits:
        return (
            Label.FAITHFUL,
            "The answer asserts a listed correct TruthfulQA reference and no listed incorrect reference: "
            + true_hits[0],
        )
    return (
        Label.UNVALIDATED,
        "The answer did not assert a listed correct or incorrect reference. "
        "A different wording is not labeled as a hallucination.",
    )


def download_snapshot(path: Path, *, allow_download: bool) -> Path:
    """Fetch the pinned CSV. Tests leave ``allow_download`` false."""
    import urllib.request

    reject_download(allow_download, resource="TruthfulQA CSV")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SOURCE_URL, destination)  # noqa: S310
    read_pinned_snapshot(destination)
    return destination


def _phrases(*fields: str) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for field in fields:
        for part in field.split(";"):
            phrase = part.strip()
            key = " ".join(phrase.lower().split())
            if len(phrase) < 8 or key in seen:
                continue
            seen.add(key)
            found.append(phrase)
    return tuple(found)
