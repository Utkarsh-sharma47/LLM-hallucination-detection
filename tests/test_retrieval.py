"""Provenance retrieval library tests. No network, no Qwen."""

from __future__ import annotations

import pytest

from glassbox.analysis.retrieval import (
    EvidenceProvenance,
    FixtureRetrievalClient,
    make_evidence_record,
    verify_claim_with_evidence,
)


def _record(relation: str, quote: str = "the sky is blue") -> EvidenceProvenance:
    return make_evidence_record(
        "src-1",
        "q",
        quote,
        relation,
        content=quote,
        retrieved_at="2026-11-06T00:00:00+00:00",
    )


def test_provenance_requires_auditable_hash() -> None:
    record = _record("supports")
    assert len(record.content_hash) == 64
    with pytest.raises(ValueError, match="content_hash"):
        EvidenceProvenance(
            source_id="s",
            query="q",
            relevant_quote="x",
            relation="supports",
            retrieved_at="2026-11-06T00:00:00+00:00",
            content_hash="short-placeholder",
        )
    with pytest.raises(ValueError, match="content is required"):
        make_evidence_record(
            "s", "q", "x", "supports", content="", retrieved_at="2026-11-06T00:00:00+00:00"
        )


def test_catalog_wins_over_retrieval() -> None:
    client = FixtureRetrievalClient({"the sky is blue": [_record("contradicts")]})
    result = verify_claim_with_evidence(
        "the sky is blue", catalog_status="FAITHFUL", retrieval_client=client
    )
    assert result["status"] == "FAITHFUL"
    assert result["evidence_source"] == "curated_catalog"


def test_supported_contradicted_conflicting_unverified() -> None:
    client = FixtureRetrievalClient(
        {
            "a": [_record("supports")],
            "b": [_record("contradicts")],
            "c": [_record("supports"), _record("contradicts")],
            "d": [_record("neutral")],
        }
    )
    assert (
        verify_claim_with_evidence("a", catalog_status=None, retrieval_client=client)["status"]
        == "SUPPORTED"
    )
    assert (
        verify_claim_with_evidence("b", catalog_status=None, retrieval_client=client)["status"]
        == "CONTRADICTED"
    )
    assert (
        verify_claim_with_evidence("c", catalog_status=None, retrieval_client=client)["status"]
        == "CONFLICTING"
    )
    assert (
        verify_claim_with_evidence("d", catalog_status=None, retrieval_client=client)["status"]
        == "UNVERIFIED"
    )
    assert (
        verify_claim_with_evidence("missing", catalog_status=None, retrieval_client=client)[
            "status"
        ]
        == "UNVERIFIED"
    )
    assert verify_claim_with_evidence("a", catalog_status=None)["status"] == "UNVALIDATED"


def test_failed_retrieval_is_unverified_not_contradicted() -> None:
    class Boom(FixtureRetrievalClient):
        def retrieve(self, claim, context=None):  # noqa: ANN001, ANN202
            raise RuntimeError("network down")

    result = verify_claim_with_evidence("a", catalog_status=None, retrieval_client=Boom({}))
    assert result["status"] == "UNVERIFIED"
