"""Records for evidence, claims, and verdicts."""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass

KIND_CURATED = "curated_catalog"
KIND_SNAPSHOT = "local_snapshot"


class Verdict(enum.StrEnum):
    """Claim-level outcome. Missing evidence is never CONTRADICTED."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    CONFLICTING = "CONFLICTING"
    UNVERIFIED = "UNVERIFIED"


class RetrievalError(RuntimeError):
    """A retriever could not run. The verdict reports it instead of guessing."""


@dataclass(frozen=True)
class EvidenceItem:
    """One quoted span from one source, with enough to audit it later.

    ``content_sha256`` hashes the full source text. ``text[span_start:span_end] == quote``
    for a local snapshot, so a reader can re-check the quote against the hashed source.
    A curated catalog item is labelled ``curated_catalog``. It is not retrieval.
    """

    evidence_id: str
    source_id: str
    locator: str
    evidence_kind: str
    query: str
    retrieved_at: str
    content_sha256: str
    quote: str
    span_start: int
    span_end: int
    source_date: str | None = None
    curated_stance: str | None = None
    matched_phrase: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Claim:
    """One atomic claim.

    ``values`` are the things the claim asserts (names, numbers). ``anchors`` say what the
    claim is about. ``rivals`` are values known to conflict with it. Without anchors or values
    a claim cannot be checked and stays UNVERIFIED.
    """

    claim_id: str
    text: str
    values: tuple[str, ...] = ()
    anchors: tuple[str, ...] = ()
    rivals: tuple[str, ...] = ()
    time_sensitive: bool = False


@dataclass(frozen=True)
class RetrievalResult:
    """What a retriever returned. ``failed`` is different from ``no_results``."""

    status: str
    items: tuple[EvidenceItem, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class ClaimVerdict:
    """The decision for one claim and the evidence behind it."""

    claim_id: str
    text: str
    verdict: Verdict
    reason: str
    rule_ids: tuple[str, ...]
    evidence: tuple[EvidenceItem, ...]
    considered: int
    retrieval_status: str
    retriever: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "rule_ids": list(self.rule_ids),
            "evidence": [item.to_dict() for item in self.evidence],
            "considered": self.considered,
            "retrieval_status": self.retrieval_status,
            "retriever": self.retriever,
            "error": self.error,
        }
