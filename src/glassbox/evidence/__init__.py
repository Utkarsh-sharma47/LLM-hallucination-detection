"""Provenance-aware claim verification (issue #40, first stage, no network).

Every verdict names the evidence it used: source, locator, retrieval time, content hash,
query, and the quoted span. A probe score, token probability, or entropy is not an input to
anything in this package.
"""

from __future__ import annotations

from glassbox.evidence.models import (
    Claim,
    ClaimVerdict,
    EvidenceItem,
    RetrievalError,
    RetrievalResult,
    Verdict,
)
from glassbox.evidence.retrievers import CatalogRetriever, LocalSnapshotRetriever, Retriever
from glassbox.evidence.verify import (
    decompose,
    derive_claim,
    make_claim_id,
    verify_answer_claims,
    verify_claim,
)

__all__ = [
    "CatalogRetriever",
    "Claim",
    "ClaimVerdict",
    "EvidenceItem",
    "LocalSnapshotRetriever",
    "RetrievalError",
    "RetrievalResult",
    "Retriever",
    "Verdict",
    "decompose",
    "derive_claim",
    "make_claim_id",
    "verify_answer_claims",
    "verify_claim",
]
