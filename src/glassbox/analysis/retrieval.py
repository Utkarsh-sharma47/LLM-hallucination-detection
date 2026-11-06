"""Provenance-aware evidence retrieval (issue #40, library-only foundation).

Curated catalog always wins; retrieval only fills UNVALIDATED claims.
Opt-in, no network in this module: the only built-in client is a
fixture-backed deterministic client for tests. Failed fetch and empty
results both yield UNVERIFIED, never CONTRADICTED. Probe scores,
token probabilities, and entropy are never inputs here.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_HASH = re.compile(r"[0-9a-f]{64}")
_RELATIONS = frozenset({"supports", "contradicts", "neutral"})


@dataclass(frozen=True)
class EvidenceProvenance:
    """One retrieved evidence item with auditable provenance."""

    source_id: str
    query: str
    relevant_quote: str
    relation: str
    retrieved_at: str
    content_hash: str
    source_url: str | None = None
    source_date: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if not self.query.strip():
            raise ValueError("query must be a non-empty string")
        if not self.relevant_quote.strip():
            raise ValueError("relevant_quote must be a non-empty string")
        if self.relation not in _RELATIONS:
            raise ValueError(f"relation must be one of {sorted(_RELATIONS)}")
        try:
            datetime.fromisoformat(self.retrieved_at)
        except (TypeError, ValueError) as exc:
            raise ValueError("retrieved_at must be an ISO-8601 timestamp") from exc
        if not _HASH.fullmatch(self.content_hash or ""):
            raise ValueError("content_hash must be a 64-char lowercase SHA-256 hex digest")


def make_evidence_record(
    source_id: str,
    query: str,
    relevant_quote: str,
    relation: str,
    *,
    content: str,
    retrieved_at: str,
    source_url: str | None = None,
    source_date: str | None = None,
) -> EvidenceProvenance:
    """Create a record whose hash is SHA-256 of the explicit ``content``.

    ``content`` is required so the hash is auditable — it is never silently
    derived from the quote while labeled ``content_hash``.
    """
    if not content:
        raise ValueError("content is required for an auditable content_hash")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return EvidenceProvenance(
        source_id=source_id,
        query=query,
        relevant_quote=relevant_quote,
        relation=relation,
        retrieved_at=retrieved_at,
        content_hash=digest,
        source_url=source_url,
        source_date=source_date,
    )


class RetrievalClient(ABC):
    """Pluggable evidence source. Empty list means no evidence found."""

    @abstractmethod
    def retrieve(self, claim: str, context: str | None = None) -> list[EvidenceProvenance]: ...

    @property
    @abstractmethod
    def source_label(self) -> str: ...


class FixtureRetrievalClient(RetrievalClient):
    """Deterministic exact-match client backed by a dict. No network."""

    def __init__(self, fixtures: dict[str, list[EvidenceProvenance]]) -> None:
        self._fixtures = fixtures

    @property
    def source_label(self) -> str:
        return "fixture"

    def retrieve(self, claim: str, context: str | None = None) -> list[EvidenceProvenance]:
        normalized = " ".join(claim.lower().split())
        for key, records in self._fixtures.items():
            if " ".join(key.lower().split()) == normalized:
                return list(records)
        return []


def load_fixture_client(fixture_path: Path) -> FixtureRetrievalClient:
    """Load a fixture client from YAML: claims -> list of evidence mappings."""
    payload = yaml.safe_load(Path(fixture_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), dict):
        raise ValueError("fixture file must have a 'claims' mapping")
    fixtures: dict[str, list[EvidenceProvenance]] = {}
    for claim_text, records in payload["claims"].items():
        if not isinstance(records, list):
            raise ValueError(f"claim '{claim_text}' must map to a list")
        parsed: list[EvidenceProvenance] = []
        for raw in records:
            if not isinstance(raw, dict):
                raise ValueError(f"claim '{claim_text}' evidence must be mappings")
            parsed.append(
                EvidenceProvenance(
                    source_id=str(raw.get("source_id", "")),
                    query=str(raw.get("query", "")),
                    relevant_quote=str(raw.get("relevant_quote", "")),
                    relation=str(raw.get("relation", "")),
                    retrieved_at=str(raw.get("retrieved_at", "")),
                    content_hash=str(raw.get("content_hash", "")),
                    source_url=raw.get("source_url"),
                    source_date=raw.get("source_date"),
                )
            )
        fixtures[str(claim_text)] = parsed
    return FixtureRetrievalClient(fixtures)


def verify_claim_with_evidence(
    claim_text: str,
    *,
    catalog_status: str | None,
    retrieval_client: RetrievalClient | None = None,
) -> dict[str, Any]:
    """Verify one claim. Catalog wins; retrieval only for UNVALIDATED claims.

    Never reads a probe score. Returns a dict with status one of the
    catalog status (unchanged), SUPPORTED, CONTRADICTED, CONFLICTING,
    UNVERIFIED, or UNVALIDATED.
    """
    if catalog_status is not None:
        return {
            "claim_text": claim_text,
            "status": catalog_status,
            "evidence_source": "curated_catalog",
            "provenance": [],
        }
    if retrieval_client is None:
        return {
            "claim_text": claim_text,
            "status": "UNVALIDATED",
            "evidence_source": "none",
            "provenance": [],
        }
    try:
        evidence = retrieval_client.retrieve(claim_text)
    except Exception:
        return {
            "claim_text": claim_text,
            "status": "UNVERIFIED",
            "evidence_source": "independent_retrieval",
            "provenance": [],
            "note": "Retrieval failed; failure is not a contradiction.",
        }
    if not evidence:
        return {
            "claim_text": claim_text,
            "status": "UNVERIFIED",
            "evidence_source": "independent_retrieval",
            "provenance": [],
        }
    relations = {item.relation for item in evidence}
    if relations == {"supports"}:
        status = "SUPPORTED"
    elif relations == {"contradicts"}:
        status = "CONTRADICTED"
    elif "supports" in relations and "contradicts" in relations:
        status = "CONFLICTING"
    else:
        status = "UNVERIFIED"
    return {
        "claim_text": claim_text,
        "status": status,
        "evidence_source": "independent_retrieval",
        "provenance": [
            {
                "source_id": item.source_id,
                "source_url": item.source_url,
                "retrieved_at": item.retrieved_at,
                "source_date": item.source_date,
                "content_hash": item.content_hash,
                "query": item.query,
                "relevant_quote": item.relevant_quote,
                "relation": item.relation,
            }
            for item in evidence
        ],
    }
