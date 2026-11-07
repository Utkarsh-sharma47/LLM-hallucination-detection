"""Tests for provenance-aware evidence retrieval.

All tests use fixture-backed clients. No network access is required.
The curated catalog path is verified to be unchanged.
"""

from __future__ import annotations

from pathlib import Path

from glassbox.analysis.claims import load_catalog, verify_answer
from glassbox.analysis.retrieval import (
    FixtureRetrievalClient,
    load_fixture_client,
    verify_answer_with_retrieval,
    verify_claim_with_evidence,
)
from glassbox.config import repo_root
from glassbox.data.schema import Label

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "retrieval_fixtures.yaml"


def _make_client() -> FixtureRetrievalClient:
    return load_fixture_client(_FIXTURE_PATH)


def _catalog():
    return load_catalog(repo_root() / "configs" / "claim_references.yaml")


# ---------------------------------------------------------------------------
# Curated catalog behavior is preserved
# ---------------------------------------------------------------------------


def test_curated_catalog_unchanged_for_known_claim() -> None:
    """A claim in the curated catalog still gets a catalog verdict."""
    catalog = _catalog()
    result = verify_answer("New Delhi is the capital of India.", catalog)
    assert result["overall"] == "FAITHFUL"


def test_curated_catalog_unchanged_for_false_claim() -> None:
    """A false claim in the curated catalog still gets HALLUCINATED."""
    catalog = _catalog()
    result = verify_answer("Mumbai is the capital of India.", catalog)
    assert result["overall"] == "HALLUCINATED"


def test_curated_catalog_unchanged_for_unknown_claim() -> None:
    """A claim outside the catalog stays UNVALIDATED without retrieval."""
    catalog = _catalog()
    result = verify_answer("Paris is the capital of France.", catalog)
    assert result["overall"] == "UNVALIDATED"


# ---------------------------------------------------------------------------
# Retrieval path: supported claims
# ---------------------------------------------------------------------------


def test_retrieval_supports_known_fact() -> None:
    """A true claim outside the catalog is SUPPORTED by fixture evidence."""
    client = _make_client()
    result = verify_answer_with_retrieval(
        "Paris is the capital of France.", _catalog(), retrieval_client=client
    )
    assert result["overall"] == "SUPPORTED"
    assert result["evidence_source"] == "independent_retrieval"
    claims = result["claims"]
    assert len(claims) == 1
    assert claims[0]["status"] == "SUPPORTED"
    assert claims[0]["evidence_source"] == "independent_retrieval"
    assert len(claims[0]["provenance"]) == 1
    assert claims[0]["provenance"][0]["relation"] == "supports"


def test_retrieval_supports_python_creator() -> None:
    """Another supported claim verified through fixtures."""
    client = _make_client()
    result = verify_answer_with_retrieval(
        "Python was created by Guido van Rossum.", _catalog(), retrieval_client=client
    )
    assert result["overall"] == "SUPPORTED"


# ---------------------------------------------------------------------------
# Retrieval path: contradicted claims
# ---------------------------------------------------------------------------


def test_retrieval_contradicts_false_fact() -> None:
    """A false claim is CONTRADICTED by fixture evidence."""
    client = _make_client()
    result = verify_answer_with_retrieval("The Earth is flat.", _catalog(), retrieval_client=client)
    assert result["overall"] == "CONTRADICTED"
    claims = result["claims"]
    assert claims[0]["status"] == "CONTRADICTED"
    assert claims[0]["provenance"][0]["relation"] == "contradicts"


def test_retrieval_contradicts_wrong_boiling_point() -> None:
    """A numerically wrong claim is contradicted."""
    client = _make_client()
    result = verify_answer_with_retrieval(
        "Water boils at 50 degrees Celsius.", _catalog(), retrieval_client=client
    )
    assert result["overall"] == "CONTRADICTED"


def test_retrieval_contradicts_wrong_sunrise_direction() -> None:
    """A directionally wrong claim is contradicted."""
    client = _make_client()
    result = verify_answer_with_retrieval(
        "The sun rises in the west.", _catalog(), retrieval_client=client
    )
    assert result["overall"] == "CONTRADICTED"


# ---------------------------------------------------------------------------
# Retrieval path: conflicting sources
# ---------------------------------------------------------------------------


def test_conflicting_sources_produce_conflicting_label() -> None:
    """When sources disagree, the verdict is CONFLICTING."""
    client = _make_client()
    result = verify_answer_with_retrieval(
        "Vitamin C prevents the common cold.",
        _catalog(),
        retrieval_client=client,
    )
    assert result["overall"] == "CONFLICTING"
    claims = result["claims"]
    relations = [p["relation"] for p in claims[0]["provenance"]]
    assert "supports" in relations
    assert "contradicts" in relations


# ---------------------------------------------------------------------------
# Retrieval path: unverified (no evidence found)
# ---------------------------------------------------------------------------


def test_no_evidence_produces_unverified() -> None:
    """A claim with no matching fixture evidence is UNVERIFIED."""
    client = _make_client()
    result = verify_answer_with_retrieval(
        "The moon is made of green cheese.", _catalog(), retrieval_client=client
    )
    assert result["overall"] == "UNVERIFIED"


# ---------------------------------------------------------------------------
# Retrieval path: absent client
# ---------------------------------------------------------------------------


def test_no_retrieval_client_produces_unvalidated() -> None:
    """Without a retrieval client, unknown claims stay UNVALIDATED."""
    result = verify_answer_with_retrieval(
        "Paris is the capital of France.", _catalog(), retrieval_client=None
    )
    assert result["overall"] == "UNVALIDATED"
    assert result["evidence_source"] == "none"


# ---------------------------------------------------------------------------
# Provenance completeness
# ---------------------------------------------------------------------------


def test_provenance_fields_are_populated() -> None:
    """Every evidence record must have all provenance fields."""
    client = _make_client()
    result = verify_answer_with_retrieval(
        "Paris is the capital of France.", _catalog(), retrieval_client=client
    )
    prov = result["claims"][0]["provenance"][0]
    assert prov["source_id"]
    assert prov["retrieved_at"]
    assert prov["content_hash"]
    assert prov["query"]
    assert prov["relevant_quote"]
    assert prov["relation"] in ("supports", "contradicts", "ambiguous")


# ---------------------------------------------------------------------------
# Claim decomposition
# ---------------------------------------------------------------------------


def test_compound_answer_decomposes_into_separate_claims() -> None:
    """A multi-claim answer is split and each claim is independently verified."""
    client = _make_client()
    result = verify_answer_with_retrieval(
        "Python was created by Guido van Rossum. The Earth is flat.",
        _catalog(),
        retrieval_client=client,
    )
    assert result["claim_count"] == 2
    statuses = {c["status"] for c in result["claims"]}
    assert "SUPPORTED" in statuses
    assert "CONTRADICTED" in statuses
    # Mixed support/contradiction → CONFLICTING overall
    assert result["overall"] == "CONFLICTING"


# ---------------------------------------------------------------------------
# Failed fetch is not contradiction
# ---------------------------------------------------------------------------


def test_failed_retrieval_is_unverified_not_contradicted() -> None:
    """When retrieval raises, the claim is UNVERIFIED, not CONTRADICTED."""

    class FailingClient(FixtureRetrievalClient):
        def retrieve(self, claim, context=None):
            raise RuntimeError("simulated network failure")

    client = FailingClient({})
    verdict = verify_claim_with_evidence(
        claim_text="Some claim",
        claim_id="test_1",
        catalog_match_status=None,
        catalog_source="",
        retrieval_client=client,
    )
    assert verdict.status == Label.UNVERIFIED.name
    assert verdict.evidence_source == "independent_retrieval"


# ---------------------------------------------------------------------------
# Curated catalog takes precedence over retrieval
# ---------------------------------------------------------------------------


def test_catalog_match_bypasses_retrieval() -> None:
    """Even with a retrieval client, a catalog match uses the catalog verdict."""
    client = _make_client()
    result = verify_answer_with_retrieval(
        "New Delhi is the capital of India.", _catalog(), retrieval_client=client
    )
    assert result["overall"] == "FAITHFUL"
    assert result["evidence_source"] == "curated_catalog"


# ---------------------------------------------------------------------------
# Probe scores are never inputs
# ---------------------------------------------------------------------------


def test_verify_claim_with_evidence_has_no_probe_parameter() -> None:
    """The function signature must not accept probe scores."""
    import inspect

    sig = inspect.signature(verify_claim_with_evidence)
    param_names = list(sig.parameters.keys())
    for forbidden in ("probe_score", "probability", "entropy", "token_prob"):
        assert forbidden not in param_names, (
            f"verify_claim_with_evidence must not accept {forbidden}"
        )


# ---------------------------------------------------------------------------
# Empty answer
# ---------------------------------------------------------------------------


def test_empty_answer_with_retrieval() -> None:
    """Empty answer produces zero claims."""
    client = _make_client()
    result = verify_answer_with_retrieval("", _catalog(), retrieval_client=client)
    assert result["claim_count"] == 0
    assert result["overall"] == "UNVALIDATED"


# ---------------------------------------------------------------------------
# Paraphrased evidence
# ---------------------------------------------------------------------------


def test_paraphrased_claim_with_evidence() -> None:
    """A claim that is a paraphrase of a known fact is verified through fixtures."""
    client = _make_client()
    result = verify_answer_with_retrieval(
        "The Great Wall of China is visible from space with the naked eye.",
        _catalog(),
        retrieval_client=client,
    )
    assert result["overall"] == "CONTRADICTED"
