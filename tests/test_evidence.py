"""Offline fixture tests for provenance-aware claim verification. No network, no model."""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from glassbox.analysis.claims import load_catalog
from glassbox.evidence import (
    CatalogRetriever,
    Claim,
    LocalSnapshotRetriever,
    RetrievalError,
    RetrievalResult,
    Verdict,
    decompose,
    derive_claim,
    make_claim_id,
    verify_answer_claims,
    verify_claim,
)

FIXTURE = Path(__file__).parent / "fixtures" / "evidence_snapshot.jsonl"
NOW = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def snapshot() -> LocalSnapshotRetriever:
    return LocalSnapshotRetriever(FIXTURE)


def _texts() -> dict[str, str]:
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line]
    return {row["id"]: row["text"] for row in rows}


def test_supported_claim_records_a_checkable_span(snapshot) -> None:
    claim = derive_claim("Canberra is the capital of Australia.")
    result = verify_claim(claim, snapshot, now=NOW)
    assert result.verdict is Verdict.SUPPORTED
    assert result.rule_ids == ("asserted_value_v1",)
    item = result.evidence[0]
    source = _texts()[item.source_id]
    assert source[item.span_start : item.span_end] == item.quote
    assert item.content_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert item.query == claim.text
    assert item.retrieved_at == "2026-11-07T12:00:00+00:00"
    assert item.locator.startswith("fixture://")
    assert item.evidence_kind == "local_snapshot"


def test_negating_evidence_contradicts(snapshot) -> None:
    claim = derive_claim("Sydney is the capital of Australia.")
    result = verify_claim(claim, snapshot, now=NOW)
    assert result.verdict is Verdict.CONTRADICTED
    assert result.rule_ids == ("negated_value_v1",)
    assert result.evidence[0].source_id == "atlas-sydney"


def test_paraphrased_evidence_still_supports(snapshot) -> None:
    claim = derive_claim("Canberra is the capital of Australia.")
    result = verify_claim(claim, snapshot, now=NOW)
    ids = {item.source_id for item in result.evidence}
    assert "atlas-paraphrase" in ids


def test_missing_evidence_is_unverified_not_contradicted(snapshot) -> None:
    claim = derive_claim("Reykjavik is the capital of Iceland.")
    result = verify_claim(claim, snapshot, now=NOW)
    assert result.verdict is Verdict.UNVERIFIED
    assert result.reason == "no_evidence"
    assert result.retrieval_status == "no_results"
    assert result.evidence == ()


def test_topic_mention_without_an_asserted_value_gives_no_verdict(snapshot) -> None:
    claim = derive_claim("Perth is the capital of Australia.")
    result = verify_claim(claim, snapshot, now=NOW)
    assert result.verdict is Verdict.UNVERIFIED
    assert result.reason == "no_rule_applied"
    assert result.considered > 0


def test_rival_values_need_an_explicit_claim_field(snapshot) -> None:
    plain = derive_claim("Perth is the capital of Australia.")
    ruled = derive_claim("Perth is the capital of Australia.", rivals=("Canberra",))
    assert verify_claim(plain, snapshot, now=NOW).verdict is Verdict.UNVERIFIED
    outcome = verify_claim(ruled, snapshot, now=NOW)
    assert outcome.verdict is Verdict.CONTRADICTED
    assert outcome.rule_ids == ("rival_value_v1",)


def test_disagreeing_sources_are_conflicting(snapshot) -> None:
    claim = derive_claim("The tower is 300 metres tall.")
    result = verify_claim(claim, snapshot, now=NOW)
    assert result.verdict is Verdict.CONFLICTING
    assert {item.source_id for item in result.evidence} == {"tower-a", "tower-b"}
    assert set(result.rule_ids) == {"asserted_value_v1", "negated_value_v1"}


def test_stale_evidence_is_ignored_for_time_sensitive_claims(snapshot) -> None:
    claim = derive_claim("Alice Smith is the president of Examplestan.", time_sensitive=True)
    assert verify_claim(claim, snapshot, now=NOW).verdict is Verdict.UNVERIFIED
    assert verify_claim(claim, snapshot, now=NOW).reason == "stale_or_undated_evidence"
    permanent = derive_claim("Alice Smith is the president of Examplestan.")
    assert verify_claim(permanent, snapshot, now=NOW).verdict is Verdict.SUPPORTED


def test_undated_evidence_cannot_support_a_time_sensitive_claim(snapshot) -> None:
    claim = derive_claim("Bob Jones is the treasurer of Examplestan.", time_sensitive=True)
    result = verify_claim(claim, snapshot, now=NOW)
    assert result.verdict is Verdict.UNVERIFIED
    assert result.reason == "stale_or_undated_evidence"


class _Failing:
    name = "failing"

    def retrieve(self, query, claim, *, now):
        raise RetrievalError("snapshot offline")


class _FailedStatus:
    name = "failed_status"

    def retrieve(self, query, claim, *, now):
        return RetrievalResult("failed", error="timeout")


@pytest.mark.parametrize(
    ("retriever", "error"), [(_Failing(), "snapshot offline"), (_FailedStatus(), "timeout")]
)
def test_retrieval_failure_is_visible_and_not_a_contradiction(retriever, error) -> None:
    result = verify_claim(derive_claim("Canberra is the capital of Australia."), retriever, now=NOW)
    assert result.verdict is Verdict.UNVERIFIED
    assert result.reason == "retrieval_failed"
    assert result.retrieval_status == "failed"
    assert result.error == error


def test_mixed_answer_is_evaluated_per_claim_and_never_blanket_supported(snapshot) -> None:
    report = verify_answer_claims(
        "Canberra is the capital of Australia and Sydney is the capital of Australia.",
        snapshot,
        now=NOW,
    )
    verdicts = [item["verdict"] for item in report["claims"]]
    assert verdicts == ["SUPPORTED", "CONTRADICTED"]
    assert report["overall"] == "MIXED"
    assert report["counts"]["SUPPORTED"] == 1


def test_claim_ids_are_stable_and_distinct() -> None:
    first = make_claim_id("Canberra is the capital of Australia.")
    assert first == make_claim_id("  canberra   is the capital of australia ")
    assert first != make_claim_id("Sydney is the capital of Australia.")
    ids = [claim.claim_id for claim in decompose("A is big. A is big. B is small.")]
    assert len(ids) == len(set(ids)) == 2


def test_claim_without_a_checkable_value_stays_unverified(snapshot) -> None:
    claim = Claim(claim_id="c-x", text="It is the capital.", values=(), anchors=("capital",))
    assert verify_claim(claim, snapshot, now=NOW).verdict is Verdict.UNVERIFIED


def test_catalog_checks_are_labelled_curated_not_retrieval() -> None:
    catalog = load_catalog()
    question = "What is the capital of Australia?"
    retriever = CatalogRetriever(catalog, question)
    wrong = verify_claim(derive_claim("Sydney is the capital of Australia."), retriever, now=NOW)
    right = verify_claim(derive_claim("The capital of Australia is Canberra."), retriever, now=NOW)
    assert wrong.verdict is Verdict.CONTRADICTED
    assert right.verdict is Verdict.SUPPORTED
    for result in (wrong, right):
        assert result.rule_ids == ("curated_catalog_v1",)
        assert {item.evidence_kind for item in result.evidence} == {"curated_catalog"}
        assert "Not retrieved at runtime" in result.evidence[0].locator
    unscoped = verify_claim(
        derive_claim("Sydney is the capital of Australia."), CatalogRetriever(catalog), now=NOW
    )
    assert unscoped.verdict is Verdict.UNVERIFIED


def test_probe_signals_are_not_parameters_anywhere() -> None:
    banned = ("probe", "score", "entropy", "probab", "logit")
    for function in (verify_claim, verify_answer_claims, derive_claim, decompose):
        for name in inspect.signature(function).parameters:
            assert not any(word in name.lower() for word in banned), (function.__name__, name)


def test_snapshot_loader_rejects_bad_files(tmp_path) -> None:
    missing = tmp_path / "none.jsonl"
    with pytest.raises(FileNotFoundError):
        LocalSnapshotRetriever(missing)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "locator": "x", "text": ""}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty text"):
        LocalSnapshotRetriever(bad)
    dup = tmp_path / "dup.jsonl"
    row = json.dumps({"id": "a", "locator": "x", "text": "Some text here."})
    dup.write_text(f"{row}\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="repeats source id"):
        LocalSnapshotRetriever(dup)


def test_content_hash_changes_when_the_source_changes(tmp_path) -> None:
    claim = derive_claim("Canberra is the capital of Australia.")
    hashes = []
    for suffix in ("", " Extra sentence."):
        path = tmp_path / f"s{len(suffix)}.jsonl"
        text = "Canberra is the capital of Australia." + suffix
        path.write_text(
            json.dumps({"id": "s", "locator": "f://s", "text": text}) + "\n", encoding="utf-8"
        )
        item = verify_claim(claim, LocalSnapshotRetriever(path), now=NOW).evidence[0]
        hashes.append(item.content_sha256)
    assert hashes[0] != hashes[1]
