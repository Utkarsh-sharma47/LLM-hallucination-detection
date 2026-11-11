"""Claim decomposition and verdicts from evidence.

Verdict rules are explicit and named. Each verdict records the rule and the quoted span that
triggered it. A passage that only mentions the topic gives no verdict. Nothing here reads a
probe score, a token probability, or an entropy: those are not parameters of any function.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime

from glassbox.analysis.claims import split_claims
from glassbox.evidence.models import (
    KIND_CURATED,
    Claim,
    ClaimVerdict,
    EvidenceItem,
    RetrievalError,
    Verdict,
)
from glassbox.evidence.retrievers import Retriever
from glassbox.evidence.textmatch import (
    anchors_needed,
    anchors_present,
    derive_terms,
    find_phrase,
    is_negated,
    stems,
    tokens,
)

RULE_CURATED = "curated_catalog_v1"
RULE_ASSERTED = "asserted_value_v1"
RULE_NEGATED = "negated_value_v1"
RULE_RIVAL = "rival_value_v1"


def make_claim_id(text: str) -> str:
    """Stable id from the normalized text. The same sentence always gets the same id."""
    normalized = " ".join(text.lower().split()).rstrip(".!? ")
    return "c-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def derive_claim(
    text: str,
    *,
    values: tuple[str, ...] | None = None,
    anchors: tuple[str, ...] | None = None,
    rivals: tuple[str, ...] = (),
    time_sensitive: bool = False,
) -> Claim:
    """Build a claim. Values and anchors are derived from the text unless given."""
    derived_values, derived_anchors = derive_terms(text)
    return Claim(
        claim_id=make_claim_id(text),
        text=text.strip(),
        values=derived_values if values is None else values,
        anchors=derived_anchors if anchors is None else anchors,
        rivals=rivals,
        time_sensitive=time_sensitive,
    )


def decompose(answer: str, *, time_sensitive: bool = False) -> tuple[Claim, ...]:
    """Split an answer into claims with stable ids. A repeated sentence is kept once."""
    claims: list[Claim] = []
    seen: set[str] = set()
    for piece in split_claims(answer):
        claim = derive_claim(piece, time_sensitive=time_sensitive)
        if claim.claim_id in seen:
            continue
        seen.add(claim.claim_id)
        claims.append(claim)
    return tuple(claims)


def verify_claim(
    claim: Claim,
    retriever: Retriever,
    *,
    query: str | None = None,
    now: datetime | None = None,
    max_age_days: int = 365,
) -> ClaimVerdict:
    """Retrieve evidence for one claim and decide. Failures and gaps stay UNVERIFIED."""
    moment = now or datetime.now(UTC)
    question = query or claim.text
    try:
        result = retriever.retrieve(question, claim, now=moment)
    except RetrievalError as exc:
        return _verdict(
            claim,
            retriever,
            Verdict.UNVERIFIED,
            "retrieval_failed",
            status="failed",
            error=str(exc),
        )
    if result.status == "failed":
        return _verdict(
            claim,
            retriever,
            Verdict.UNVERIFIED,
            "retrieval_failed",
            status="failed",
            error=result.error,
        )
    if not result.items:
        return _verdict(claim, retriever, Verdict.UNVERIFIED, "no_evidence", status=result.status)

    usable = [item for item in result.items if _fresh(claim, item, moment, max_age_days)]
    if not usable:
        return _verdict(
            claim,
            retriever,
            Verdict.UNVERIFIED,
            "stale_or_undated_evidence",
            status=result.status,
            considered=len(result.items),
        )
    supports: list[tuple[EvidenceItem, str]] = []
    contradicts: list[tuple[EvidenceItem, str]] = []
    for item in usable:
        for stance, rule in _stances(claim, item):
            (supports if stance == "support" else contradicts).append((item, rule))
    considered = len(result.items)
    if supports and contradicts:
        return _decided(
            claim,
            retriever,
            Verdict.CONFLICTING,
            "sources_disagree",
            supports + contradicts,
            result.status,
            considered,
        )
    if supports:
        return _decided(
            claim,
            retriever,
            Verdict.SUPPORTED,
            "evidence_asserts_claim",
            supports,
            result.status,
            considered,
        )
    if contradicts:
        return _decided(
            claim,
            retriever,
            Verdict.CONTRADICTED,
            "evidence_contradicts_claim",
            contradicts,
            result.status,
            considered,
        )
    return _verdict(
        claim,
        retriever,
        Verdict.UNVERIFIED,
        "no_rule_applied",
        status=result.status,
        considered=considered,
    )


def verify_answer_claims(
    answer: str,
    retriever: Retriever,
    *,
    query: str | None = None,
    now: datetime | None = None,
    max_age_days: int = 365,
    time_sensitive: bool = False,
) -> dict[str, object]:
    """Verify each claim of an answer separately. A mixed answer is never a blanket verdict."""
    claims = decompose(answer, time_sensitive=time_sensitive)
    moment = now or datetime.now(UTC)
    verdicts = [
        verify_claim(claim, retriever, query=query, now=moment, max_age_days=max_age_days)
        for claim in claims
    ]
    counts = {verdict.value: 0 for verdict in Verdict}
    for item in verdicts:
        counts[item.verdict.value] += 1
    return {
        "overall": _overall(counts, len(verdicts)),
        "claim_count": len(verdicts),
        "counts": counts,
        "claims": [item.to_dict() for item in verdicts],
        "retriever": retriever.name,
        "note": (
            "Evidence labels come from recorded source spans and named rules. "
            "Probe scores, token probabilities and entropy are not inputs."
        ),
    }


def _overall(counts: dict[str, int], total: int) -> str:
    if total == 0:
        return "UNVERIFIED"
    if counts["SUPPORTED"] == total:
        return "SUPPORTED"
    bad = counts["CONTRADICTED"] + counts["CONFLICTING"]
    if counts["SUPPORTED"] and bad:
        return "MIXED"
    if counts["CONTRADICTED"]:
        return "CONTRADICTED"
    if counts["CONFLICTING"]:
        return "CONFLICTING"
    if counts["SUPPORTED"]:
        return "PARTIALLY_VERIFIED"
    return "UNVERIFIED"


def _fresh(claim: Claim, item: EvidenceItem, now: datetime, max_age_days: int) -> bool:
    if not claim.time_sensitive or item.evidence_kind == KIND_CURATED:
        return True
    if item.source_date is None:
        return False
    try:
        published = date.fromisoformat(item.source_date)
    except ValueError:
        return False
    return (now.date() - published).days <= max_age_days


def _stances(claim: Claim, item: EvidenceItem) -> list[tuple[str, str]]:
    if item.evidence_kind == KIND_CURATED:
        if item.curated_stance == "accepted":
            return [("support", RULE_CURATED)]
        if item.curated_stance == "false_claim":
            return [("contradict", RULE_CURATED)]
        return []
    if not claim.anchors or not (claim.values or claim.rivals):
        return []
    words = tokens(item.quote)
    if anchors_present(set(stems(item.quote)), claim.anchors) < anchors_needed(len(claim.anchors)):
        return []
    found = {value: find_phrase(words, value) for value in claim.values}
    everything = bool(claim.values) and all(index >= 0 for index in found.values())
    if everything:
        negated = any(
            is_negated(words, index, max(1, len(tokens(value)))) for value, index in found.items()
        )
        return [("contradict", RULE_NEGATED)] if negated else [("support", RULE_ASSERTED)]
    for rival in claim.rivals:
        index = find_phrase(words, rival)
        if index >= 0 and not is_negated(words, index, max(1, len(tokens(rival)))):
            return [("contradict", RULE_RIVAL)]
    return []


def _verdict(
    claim: Claim,
    retriever: Retriever,
    verdict: Verdict,
    reason: str,
    *,
    status: str,
    considered: int = 0,
    error: str | None = None,
) -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=claim.claim_id,
        text=claim.text,
        verdict=verdict,
        reason=reason,
        rule_ids=(),
        evidence=(),
        considered=considered,
        retrieval_status=status,
        retriever=retriever.name,
        error=error,
    )


def _decided(
    claim: Claim,
    retriever: Retriever,
    verdict: Verdict,
    reason: str,
    used: list[tuple[EvidenceItem, str]],
    status: str,
    considered: int,
) -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=claim.claim_id,
        text=claim.text,
        verdict=verdict,
        reason=reason,
        rule_ids=tuple(dict.fromkeys(rule for _, rule in used)),
        evidence=tuple(dict.fromkeys(item for item, _ in used)),
        considered=considered,
        retrieval_status=status,
        retriever=retriever.name,
    )
