"""Target records for audited short-answer examples.

These types describe audited short-answer records. ``load_examples`` fills them
from a local pinned file. Constructing one does not make its label a fact.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

from glassbox.constants import DATASET_IDS

_SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}")


class Label(enum.Enum):
    """Provisional label set.

    ``UNVALIDATED`` is the honest value until the audit in ``docs/DATASETS.md``
    accepts a label source for a specific dataset version.
    """

    UNVALIDATED = -1
    FAITHFUL = 0
    HALLUCINATED = 1
    AMBIGUOUS = 2


@dataclass(frozen=True)
class QAExample:
    """One short factual question and the fields needed to score it later."""

    example_id: str
    dataset: str
    question: str
    reference_answer: str | None
    model_answer: str | None
    label: Label
    label_source: str
    split: str

    def __post_init__(self) -> None:
        if not self.example_id or not isinstance(self.example_id, str):
            raise ValueError("example_id must be a non-empty string")
        if self.dataset not in DATASET_IDS:
            raise ValueError(f"dataset must be one of {', '.join(DATASET_IDS)}")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        _optional_text(self.reference_answer, "reference_answer")
        _optional_text(self.model_answer, "model_answer")
        if not isinstance(self.label, Label):
            raise ValueError("label must be a Label value")
        if not isinstance(self.label_source, str) or not self.label_source.strip():
            raise ValueError("label_source must name where the label came from")
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("split must be a non-empty string")


def make_example_id(dataset_id: str, source_id: str) -> str:
    """Build a stable id that cannot be used as a filesystem path.

    ``source_id`` is the id from the upstream release, not a hash of the model
    answer. Split membership should follow this id so regenerating an answer
    does not move an example between splits.
    """
    if dataset_id not in DATASET_IDS:
        raise ValueError(f"dataset_id must be one of {', '.join(DATASET_IDS)}")
    if not isinstance(source_id, str) or _SOURCE_ID.fullmatch(source_id) is None:
        raise ValueError("source_id must be a short stable id without whitespace or slashes")
    return f"{dataset_id}:{source_id}"


def _optional_text(value: str | None, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None")
