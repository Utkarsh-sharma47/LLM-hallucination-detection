"""Pluggable evidence retrieval for claim verification.

Retrieval is opt-in and never replaces curated-catalog checks. Every evidence
record carries full provenance: source identity, retrieval timestamp, content
hash, query, and the relevant quote span. A fixture-backed client supports
deterministic testing without network access.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from glassbox.data.schema import ClaimVerdict, EvidenceProvenance, Label


@dataclass(frozen=True)
class RetrievalSource:
    """One configured evidence source with a quality tier."""

    source_id: str
    source_type: str
    quality_tier: int
    config: dict[str, object]


@dataclass(frozen=True)
class RetrievalConfig:
    """Loaded retrieval configuration."""

    enabled: bool
    sources: tuple[RetrievalSource, ...]


class RetrievalClient(ABC):
    """Pluggable interface for evidence retrieval.

    Each implementation must return EvidenceProvenance records with every field
    populated. A client that cannot retrieve evidence must return an empty list,
    not raise an error — the caller treats an empty result as UNVERIFIED.
    """

    @abstractmethod
    def retrieve(self, claim: str, context: str | None = None) -> list[EvidenceProvenance]:
        """Return evidence records for a claim. Empty list means no evidence found."""
        ...

    @property
    @abstractmethod
    def source_label(self) -> str:
        """Human-readable label for the evidence source (e.g. 'wikipedia_api')."""
        ...


class FixtureRetrievalClient(RetrievalClient):
    """Deterministic client backed by a YAML fixture file. No network access."""

    def __init__(self, fixtures: dict[str, list[dict[str, object]]]) -> None:
        self._fixtures = fixtures
        self._source_label = "fixture"

    @property
    def source_label(self) -> str:
        return self._source_label

    def retrieve(self, claim: str, context: str | None = None) -> list[EvidenceProvenance]:
        normalized = " ".join(claim.lower().split())
        for key, records in self._fixtures.items():
            if " ".join(key.lower().split()) == normalized:
                return [_parse_evidence_record(r) for r in records]
        return []


def load_retrieval_config(path: Path) -> RetrievalConfig:
    """Load retrieval configuration from a YAML file."""
    if not path.is_file():
        return RetrievalConfig(enabled=False, sources=())
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("retrieval config must be a mapping")
    enabled = bool(payload.get("retrieval_enabled", False))
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list):
        return RetrievalConfig(enabled=enabled, sources=())
    sources: list[RetrievalSource] = []
    for raw in raw_sources:
        if not isinstance(raw, dict):
            continue
        sources.append(
            RetrievalSource(
                source_id=str(raw.get("id", "")),
                source_type=str(raw.get("type", "")),
                quality_tier=int(raw.get("quality_tier", 3)),
                config={k: v for k, v in raw.items() if k not in ("id", "type", "quality_tier")},
            )
        )
    return RetrievalConfig(enabled=enabled, sources=tuple(sources))


def load_fixture_client(fixture_path: Path) -> FixtureRetrievalClient:
    """Load a fixture-backed client from a YAML file."""
    payload = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), dict):
        raise ValueError("fixture file must have a 'claims' mapping")
    fixtures: dict[str, list[dict[str, object]]] = {}
    for claim_text, records in payload["claims"].items():
        if not isinstance(records, list):
            raise ValueError(f"claim '{claim_text}' must map to a list of evidence records")
        fixtures[str(claim_text)] = [dict(r) for r in records if isinstance(r, dict)]
    return FixtureRetrievalClient(fixtures)


def verify_claim_with_evidence(
    claim_text: str,
    claim_id: str,
    catalog_match_status: str | None,
    catalog_source: str,
    retrieval_client: RetrievalClient | None,
) -> ClaimVerdict:
    """Verify one atomic claim, preferring curated catalog over retrieval.

    Args:
        claim_text: The atomic claim to verify.
        claim_id: Stable identifier for this claim.
        catalog_match_status: If the curated catalog matched, the Label name.
        catalog_source: The source string from the catalog entry.
        retrieval_client: Optional retrieval client for UNVALIDATED claims.

    Returns:
        ClaimVerdict with evidence_source and optional provenance.
    """
    # Curated catalog always wins
    if catalog_match_status is not None:
        return ClaimVerdict(
            claim_id=claim_id,
            claim_text=claim_text,
            status=catalog_match_status,
            evidence_source="curated_catalog",
        )

    # No catalog match and no retrieval configured → UNVALIDATED
    if retrieval_client is None:
        return ClaimVerdict(
            claim_id=claim_id,
            claim_text=claim_text,
            status=Label.UNVALIDATED.name,
            evidence_source="none",
        )

    # Attempt retrieval
    try:
        evidence = retrieval_client.retrieve(claim_text)
    except Exception:
        # Failed fetch is UNVERIFIED, not CONTRADICTED
        return ClaimVerdict(
            claim_id=claim_id,
            claim_text=claim_text,
            status=Label.UNVERIFIED.name,
            evidence_source="independent_retrieval",
        )

    if not evidence:
        return ClaimVerdict(
            claim_id=claim_id,
            claim_text=claim_text,
            status=Label.UNVERIFIED.name,
            evidence_source="independent_retrieval",
        )

    # Classify based on evidence relations
    relations = {e.relation for e in evidence}
    if relations == {"supports"}:
        status = Label.SUPPORTED.name
    elif relations == {"contradicts"}:
        status = Label.CONTRADICTED.name
    elif "supports" in relations and "contradicts" in relations:
        status = Label.CONFLICTING.name
    else:
        status = Label.UNVERIFIED.name

    return ClaimVerdict(
        claim_id=claim_id,
        claim_text=claim_text,
        status=status,
        evidence_source="independent_retrieval",
        provenance=tuple(evidence),
    )


def verify_answer_with_retrieval(
    answer: str,
    catalog: tuple[Any, ...],
    retrieval_client: RetrievalClient | None = None,
) -> dict[str, object]:
    """Verify an answer using curated catalog first, then optional retrieval.

    The curated catalog path is identical to ``verify_answer``. Claims that
    match the catalog are labeled with ``evidence_source: curated_catalog``.
    Claims with no catalog match fall through to retrieval when a client is
    provided. Probe scores are never inputs to this function.
    """
    from glassbox.analysis.claims import _match, split_claims

    if not answer.strip():
        return {
            "overall": "UNVALIDATED",
            "note": "The model produced no text.",
            "claim_count": 0,
            "counts": {
                "FAITHFUL": 0,
                "HALLUCINATED": 0,
                "AMBIGUOUS": 0,
                "UNVALIDATED": 0,
                "SUPPORTED": 0,
                "CONTRADICTED": 0,
                "CONFLICTING": 0,
                "UNVERIFIED": 0,
            },
            "evidence_source": "none",
            "claims": [],
        }

    pieces = split_claims(answer)
    verdicts: list[ClaimVerdict] = []

    for idx, piece in enumerate(pieces):
        claim_id = f"claim_{idx}"
        # Check curated catalog first
        catalog_hits = [_match(piece, item) for item in catalog] if catalog else []
        catalog_hits = [h for h in catalog_hits if h is not None]

        if catalog_hits:
            # Use the first catalog hit's status
            hit = catalog_hits[0]
            verdicts.append(
                ClaimVerdict(
                    claim_id=claim_id,
                    claim_text=piece,
                    status=hit.status,
                    evidence_source="curated_catalog",
                )
            )
        else:
            # No catalog match — try retrieval
            verdict = verify_claim_with_evidence(
                claim_text=piece,
                claim_id=claim_id,
                catalog_match_status=None,
                catalog_source="",
                retrieval_client=retrieval_client,
            )
            verdicts.append(verdict)

    return _summarise_verdicts(tuple(verdicts))


def _summarise_verdicts(verdicts: tuple[ClaimVerdict, ...]) -> dict[str, object]:
    counts: dict[str, int] = {
        "FAITHFUL": 0,
        "HALLUCINATED": 0,
        "AMBIGUOUS": 0,
        "UNVALIDATED": 0,
        "SUPPORTED": 0,
        "CONTRADICTED": 0,
        "CONFLICTING": 0,
        "UNVERIFIED": 0,
    }
    for v in verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1

    statuses = {v.status for v in verdicts}
    if "HALLUCINATED" in statuses and "FAITHFUL" in statuses or "AMBIGUOUS" in statuses:
        overall = "AMBIGUOUS"
    elif "HALLUCINATED" in statuses:
        overall = "HALLUCINATED"
    elif "CONTRADICTED" in statuses and "SUPPORTED" in statuses or "CONFLICTING" in statuses:
        overall = "CONFLICTING"
    elif "CONTRADICTED" in statuses:
        overall = "CONTRADICTED"
    elif "SUPPORTED" in statuses and "UNVERIFIED" in statuses:
        overall = "UNVERIFIED"
    elif "SUPPORTED" in statuses and "UNVALIDATED" in statuses:
        overall = "UNVALIDATED"
    elif statuses == {"SUPPORTED"}:
        overall = "SUPPORTED"
    elif statuses == {"FAITHFUL"}:
        overall = "FAITHFUL"
    elif "UNVERIFIED" in statuses:
        overall = "UNVERIFIED"
    else:
        overall = "UNVALIDATED"

    evidence_sources = {v.evidence_source for v in verdicts}
    if "independent_retrieval" in evidence_sources and "curated_catalog" in evidence_sources:
        primary_source = "mixed"
    elif "independent_retrieval" in evidence_sources:
        primary_source = "independent_retrieval"
    elif "curated_catalog" in evidence_sources:
        primary_source = "curated_catalog"
    else:
        primary_source = "none"

    return {
        "overall": overall,
        "note": _verdict_note(overall, primary_source),
        "claim_count": len(verdicts),
        "counts": counts,
        "evidence_source": primary_source,
        "claims": [
            {
                "claim_id": v.claim_id,
                "text": v.claim_text,
                "status": v.status,
                "evidence_source": v.evidence_source,
                "provenance": [
                    {
                        "source_id": p.source_id,
                        "source_url": p.source_url,
                        "retrieved_at": p.retrieved_at,
                        "source_date": p.source_date,
                        "content_hash": p.content_hash,
                        "query": p.query,
                        "relevant_quote": p.relevant_quote,
                        "relation": p.relation,
                    }
                    for p in v.provenance
                ],
            }
            for v in verdicts
        ],
    }


def _verdict_note(overall: str, source: str) -> str:
    if source == "curated_catalog":
        if overall == "FAITHFUL":
            return "Every matched claim is supported by the curated catalog."
        if overall == "HALLUCINATED":
            return "A matched claim contradicts the curated catalog."
        if overall == "AMBIGUOUS":
            return "The answer contains both a supported and a contradicted claim per the curated catalog."
        return (
            "At least one claim has no catalog evidence. Missing evidence is not a hallucination."
        )
    if source == "independent_retrieval":
        if overall == "SUPPORTED":
            return "All claims are supported by independently retrieved evidence."
        if overall == "CONTRADICTED":
            return "At least one claim is contradicted by retrieved evidence."
        if overall == "CONFLICTING":
            return "Retrieved sources conflict: some support and some contradict the claims."
        return "Retrieved evidence is insufficient to verify one or more claims."
    if source == "mixed":
        return "Some claims were verified against the curated catalog; others used independent retrieval."
    return "No evidence source is available. Claims remain unvalidated."


def _parse_evidence_record(raw: dict[str, object]) -> EvidenceProvenance:
    return EvidenceProvenance(
        source_id=str(raw.get("source_id", "")),
        source_url=raw.get("source_url") if raw.get("source_url") is not None else None,
        retrieved_at=str(raw.get("retrieved_at", "")),
        source_date=raw.get("source_date") if raw.get("source_date") is not None else None,
        content_hash=str(raw.get("content_hash", "")),
        query=str(raw.get("query", "")),
        relevant_quote=str(raw.get("relevant_quote", "")),
        relation=str(raw.get("relation", "")),
    )


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def make_evidence_record(
    source_id: str,
    query: str,
    relevant_quote: str,
    relation: str,
    *,
    source_url: str | None = None,
    source_date: str | None = None,
    content: str,
) -> EvidenceProvenance:
    """Factory for EvidenceProvenance with auto-generated timestamp and hash.

    ``content`` is the full retrievable source text that ``relevant_quote`` was
    drawn from. It is hashed into ``content_hash`` so provenance can be audited
    against the original source. It must be provided explicitly — the factory
    never falls back to hashing ``relevant_quote`` because the field is named
    ``content_hash``, not ``quote_hash``.
    """
    return EvidenceProvenance(
        source_id=source_id,
        source_url=source_url,
        retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_date=source_date,
        content_hash=_hash_content(content),
        query=query,
        relevant_quote=relevant_quote,
        relation=relation,
    )
