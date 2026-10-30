"""Claim checks against the local curated catalog.

A sentence with no catalog hit is UNVALIDATED. One supported claim does not
make a second false claim faithful. Probe scores are not read here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from glassbox.config import repo_root
from glassbox.data.schema import Label
from glassbox.verification import asserts_phrase

_SPLIT = re.compile(r"(?<=[.!?])\s+|\s+\band\b\s+", re.IGNORECASE)


@dataclass(frozen=True)
class CatalogClaim:
    """One curated claim and the phrases that support or contradict it."""

    claim_id: str
    claim: str
    accepted: tuple[str, ...]
    false_claims: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class ClaimFinding:
    """One slice of an answer after the catalog check."""

    text: str
    status: str
    claim_id: str | None
    source: str
    support: str
    contradiction: str


def default_catalog_path() -> Path:
    return repo_root() / "configs" / "claim_references.yaml"


def load_catalog(path: Path | None = None) -> tuple[CatalogClaim, ...]:
    """Load the curated catalog. A missing file is an empty catalog."""
    catalog_path = default_catalog_path() if path is None else Path(path)
    if not catalog_path.is_file():
        return ()
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("claim catalog must be a mapping with an items list")
    claims: list[CatalogClaim] = []
    for raw in payload["items"]:
        claims.append(_parse_claim(raw))
    return tuple(claims)


def split_claims(answer: str) -> tuple[str, ...]:
    """Split an answer into sentences and coordinated clauses."""
    pieces = [piece.strip(" ,;") for piece in _SPLIT.split(answer.strip())]
    return tuple(piece for piece in pieces if piece)


def verify_answer(answer: str, catalog: tuple[CatalogClaim, ...]) -> dict[str, object]:
    """Verify each slice. The probe is not an input."""
    if not answer.strip():
        return _summary((), "UNVALIDATED", "The model produced no text.")
    if not catalog:
        finding = ClaimFinding(
            text=answer.strip(),
            status="UNVALIDATED",
            claim_id=None,
            source="none",
            support="",
            contradiction="",
        )
        return _summary(
            (finding,),
            "UNVALIDATED",
            "Verification is unavailable. No claim catalog is configured.",
        )
    findings: list[ClaimFinding] = []
    for piece in split_claims(answer):
        hits = [_match(piece, item) for item in catalog]
        hits = [hit for hit in hits if hit is not None]
        if not hits:
            findings.append(
                ClaimFinding(
                    text=piece,
                    status="UNVALIDATED",
                    claim_id=None,
                    source="none",
                    support="",
                    contradiction="",
                )
            )
            continue
        findings.extend(hits)
    status = _overall(findings)
    note = _overall_note(status)
    return _summary(tuple(findings), status, note)


def _parse_claim(raw: object) -> CatalogClaim:
    if not isinstance(raw, dict):
        raise ValueError("each claim must be a mapping")
    claim_id = raw.get("id")
    claim = raw.get("claim")
    accepted = raw.get("accepted")
    false_claims = raw.get("false_claims")
    source = raw.get("source")
    if not isinstance(claim_id, str) or not claim_id:
        raise ValueError("claim id must be a non-empty string")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError(f"{claim_id}: claim text is required")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{claim_id}: source is required")
    return CatalogClaim(
        claim_id=claim_id,
        claim=claim.strip(),
        accepted=_phrases(accepted),
        false_claims=_phrases(false_claims),
        source=source.strip(),
    )


def _phrases(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError("accepted and false_claims must be lists of strings")
    return tuple(item.strip() for item in value)


def _match(piece: str, item: CatalogClaim) -> ClaimFinding | None:
    false_hit = next(
        (phrase for phrase in item.false_claims if asserts_phrase(piece, phrase)), None
    )
    true_hit = next((phrase for phrase in item.accepted if asserts_phrase(piece, phrase)), None)
    if false_hit and true_hit:
        status = Label.AMBIGUOUS.name
    elif false_hit:
        status = Label.HALLUCINATED.name
    elif true_hit:
        status = Label.FAITHFUL.name
    else:
        return None
    return ClaimFinding(
        text=piece,
        status=status,
        claim_id=item.claim_id,
        source=item.source,
        support=true_hit or "",
        contradiction=false_hit or "",
    )


def _overall(findings: tuple[ClaimFinding, ...] | list[ClaimFinding]) -> str:
    statuses = {item.status for item in findings}
    if Label.HALLUCINATED.name in statuses and Label.FAITHFUL.name in statuses:
        return Label.AMBIGUOUS.name
    if Label.AMBIGUOUS.name in statuses:
        return Label.AMBIGUOUS.name
    if Label.HALLUCINATED.name in statuses:
        return Label.HALLUCINATED.name
    if Label.FAITHFUL.name in statuses and Label.UNVALIDATED.name in statuses:
        return Label.UNVALIDATED.name
    if statuses == {Label.FAITHFUL.name}:
        return Label.FAITHFUL.name
    return Label.UNVALIDATED.name


def _overall_note(status: str) -> str:
    if status == Label.FAITHFUL.name:
        return "Every matched claim is supported by the curated catalog. Unlisted wording was not present."
    if status == Label.HALLUCINATED.name:
        return "A matched claim contradicts the curated catalog."
    if status == Label.AMBIGUOUS.name:
        return "The answer contains both a supported claim and a contradicted or conflicting claim."
    return "At least one claim has no catalog evidence, or no claim matched. Missing evidence is not a hallucination."


def _summary(findings: tuple[ClaimFinding, ...], status: str, note: str) -> dict[str, object]:
    counts = {"FAITHFUL": 0, "HALLUCINATED": 0, "AMBIGUOUS": 0, "UNVALIDATED": 0}
    for finding in findings:
        counts[finding.status] = counts.get(finding.status, 0) + 1
    return {
        "overall": status,
        "note": note,
        "claim_count": len(findings),
        "counts": counts,
        "claims": [
            {
                "text": item.text,
                "status": item.status,
                "claim_id": item.claim_id,
                "source": item.source,
                "support": item.support,
                "contradiction": item.contradiction,
            }
            for item in findings
        ],
    }
