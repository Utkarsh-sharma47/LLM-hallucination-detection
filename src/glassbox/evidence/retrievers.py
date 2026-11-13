"""Retriever interface with two offline implementations. Neither one touches the network."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from glassbox.analysis.claims import CatalogClaim, applies_to_question
from glassbox.evidence.models import (
    KIND_CURATED,
    KIND_SNAPSHOT,
    Claim,
    EvidenceItem,
    RetrievalResult,
)
from glassbox.evidence.textmatch import (
    anchors_needed,
    anchors_present,
    find_phrase,
    sentence_spans,
    stem,
    stems,
    tokens,
)
from glassbox.verification import asserts_phrase


class Retriever(Protocol):
    """Anything that can return evidence for a claim. A network retriever must be opt-in."""

    name: str

    def retrieve(self, query: str, claim: Claim, *, now: datetime) -> RetrievalResult: ...


def stamp(now: datetime) -> str:
    return now.astimezone(UTC).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CatalogRetriever:
    """The hand-written catalog behind the same interface.

    Evidence is labelled ``curated_catalog``. It is a curated phrase match, not retrieval.
    """

    name = "curated_catalog"

    def __init__(self, catalog: tuple[CatalogClaim, ...], question: str | None = None):
        self._catalog = catalog
        self._question = question

    def retrieve(self, query: str, claim: Claim, *, now: datetime) -> RetrievalResult:
        items: list[EvidenceItem] = []
        for entry in self._catalog:
            if not applies_to_question(entry, self._question):
                continue
            for stance, phrases in (
                ("false_claim", entry.false_claims),
                ("accepted", entry.accepted),
            ):
                hit = next((p for p in phrases if asserts_phrase(claim.text, p)), None)
                if hit is None:
                    continue
                items.append(
                    EvidenceItem(
                        evidence_id=f"{entry.claim_id}:{stance}",
                        source_id=entry.claim_id,
                        locator=entry.source,
                        evidence_kind=KIND_CURATED,
                        query=query,
                        retrieved_at=stamp(now),
                        content_sha256=sha256_text(entry.claim),
                        quote=entry.claim,
                        span_start=0,
                        span_end=len(entry.claim),
                        curated_stance=stance,
                        matched_phrase=hit,
                    )
                )
        return RetrievalResult("ok" if items else "no_results", tuple(items))


class LocalSnapshotRetriever:
    """Sentences from a frozen local JSONL file of sources.

    Each line needs ``id``, ``locator`` and ``text``; ``source_date`` (YYYY-MM-DD) is optional.
    The content hash covers the whole source text, and every item records the character span
    of its quote inside that text.
    """

    name = "local_snapshot"

    def __init__(self, path: Path, *, top_k: int = 5):
        self._sources = _read_sources(Path(path))
        self._top_k = top_k

    def retrieve(self, query: str, claim: Claim, *, now: datetime) -> RetrievalResult:
        anchors = claim.anchors or tuple(
            word.lower() for word in tokens(query) if word.lower() not in {"the", "a", "an"}
        )
        need = anchors_needed(len(anchors)) if anchors else 1
        scored: list[tuple[int, int, EvidenceItem]] = []
        for order, source in enumerate(self._sources):
            text = str(source["text"])
            digest = sha256_text(text)
            for start, end in sentence_spans(text):
                quote = text[start:end]
                sentence_stems = set(stems(quote))
                matched = anchors_present(sentence_stems, anchors)
                if matched < need:
                    continue
                value_hits = sum(
                    1 for value in claim.values if stem(value) in sentence_stems
                ) + sum(1 for rival in claim.rivals if find_phrase(tokens(quote), rival) >= 0)
                if (claim.values or claim.rivals) and value_hits == 0:
                    continue
                scored.append(
                    (
                        matched + value_hits,
                        -order,
                        EvidenceItem(
                            evidence_id=f"{source['id']}#{start}-{end}",
                            source_id=str(source["id"]),
                            locator=str(source["locator"]),
                            evidence_kind=KIND_SNAPSHOT,
                            query=query,
                            retrieved_at=stamp(now),
                            content_sha256=digest,
                            quote=quote,
                            span_start=start,
                            span_end=end,
                            source_date=source.get("source_date"),
                        ),
                    )
                )
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        items = tuple(row[2] for row in scored[: self._top_k])
        return RetrievalResult("ok" if items else "no_results", items)


def _read_sources(path: Path) -> list[dict[str, str | None]]:
    if not path.is_file():
        raise FileNotFoundError(f"Evidence snapshot is not at {path}")
    sources: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "locator", "text"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{path} line {number} needs a non-empty {key}")
        if row["id"] in seen:
            raise ValueError(f"{path} line {number} repeats source id {row['id']}")
        seen.add(row["id"])
        date = row.get("source_date")
        if date is not None and not isinstance(date, str):
            raise ValueError(f"{path} line {number} has a non-text source_date")
        sources.append(
            {"id": row["id"], "locator": row["locator"], "text": row["text"], "source_date": date}
        )
    if not sources:
        raise ValueError(f"{path} has no sources")
    return sources
