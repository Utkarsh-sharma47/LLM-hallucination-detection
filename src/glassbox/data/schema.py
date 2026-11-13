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
_SHA256_HEX = re.compile(r"^[a-f0-9]{64}$")
_ISO_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")


class Label(enum.Enum):
    """Provisional label set.

    ``UNVALIDATED`` is the honest value until the audit in ``docs/DATASETS.md``
    accepts a label source for a specific dataset version.

    Retrieval-aware labels (SUPPORTED, CONTRADICTED, CONFLICTING, UNVERIFIED)
    are assigned only when independent evidence retrieval is configured and a
    claim has no curated-catalog match. They are never derived from probe scores.
    """

    UNVALIDATED = -1
    FAITHFUL = 0
    HALLUCINATED = 1
    AMBIGUOUS = 2
    SUPPORTED = 3
    CONTRADICTED = 4
    CONFLICTING = 5
    UNVERIFIED = 6


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


@dataclass(frozen=True)
class EvidenceProvenance:
    """Provenance record for one piece of retrieved evidence.

    Every field must be populated. A missing field means the evidence is
    incomplete and the claim stays UNVERIFIED.
    """

    source_id: str
    source_url: str | None
    retrieved_at: str
    source_date: str | None
    content_hash: str
    query: str
    relevant_quote: str
    relation: str  # "supports", "contradicts", "ambiguous"

    def __post_init__(self) -> None:
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("source_id must be a non-empty string")
        if not self.retrieved_at or not isinstance(self.retrieved_at, str):
            raise ValueError("retrieved_at must be an ISO timestamp")
        if _ISO_TIMESTAMP.fullmatch(self.retrieved_at) is None:
            raise ValueError(
                f"retrieved_at must be an ISO 8601 timestamp, got {self.retrieved_at!r}"
            )
        if not self.content_hash or not isinstance(self.content_hash, str):
            raise ValueError("content_hash must be a non-empty string")
        if _SHA256_HEX.fullmatch(self.content_hash) is None:
            raise ValueError(
                f"content_hash must be a 64-character SHA-256 hex digest, got {self.content_hash!r}"
            )
        if not self.query or not isinstance(self.query, str):
            raise ValueError("query must be a non-empty string")
        if not self.relevant_quote or not isinstance(self.relevant_quote, str):
            raise ValueError("relevant_quote must be a non-empty string")
        if self.relation not in ("supports", "contradicts", "ambiguous"):
            raise ValueError("relation must be supports, contradicts, or ambiguous")


@dataclass(frozen=True)
class ClaimVerdict:
    """The result of verifying one atomic claim.

    ``evidence_source`` is ``"curated_catalog"`` or ``"independent_retrieval"``.
    ``provenance`` is populated only for retrieval-backed verdicts.
    """

    claim_id: str
    claim_text: str
    status: str  # Label name string
    evidence_source: str
    provenance: tuple[EvidenceProvenance, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim_id or not isinstance(self.claim_id, str):
            raise ValueError("claim_id must be a non-empty string")
        if not self.claim_text or not isinstance(self.claim_text, str):
            raise ValueError("claim_text must be a non-empty string")
        if self.evidence_source not in ("curated_catalog", "independent_retrieval", "none"):
            raise ValueError(
                "evidence_source must be curated_catalog, independent_retrieval, or none"
            )
