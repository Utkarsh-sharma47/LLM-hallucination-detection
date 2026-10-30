"""Tests for Glass Box report generation."""

import json

import numpy as np
import pytest

from glassbox.glassbox_report import build_report
from glassbox.models.backend import GeneratedToken, GenerationResult
from glassbox.models.distributions import TokenAlternative
from glassbox.probes.logistic import PROBE_NOT_TRAINED
from glassbox.reporting import TurnRecord
from glassbox.verification import ReferenceResult


@pytest.fixture
def sample_generation():
    """Create a sample GenerationResult for testing."""
    tokens = (
        GeneratedToken(
            token_id=1,
            text="New",
            probability=0.72,
            top_alternatives=(
                TokenAlternative(token_id=2, text="Delhi", probability=0.12),
                TokenAlternative(token_id=3, text="The", probability=0.08),
            ),
            is_argmax=True,
            entropy_nats=0.8,
            normalized_entropy=0.3,
            margin=0.60,
            rank=1,
            step=1,
            conditioned_on="prompt",
            decoding="greedy",
            temperature=0.0,
        ),
        GeneratedToken(
            token_id=4,
            text="Delhi",
            probability=0.94,
            top_alternatives=(
                TokenAlternative(token_id=5, text="York", probability=0.02),
                TokenAlternative(token_id=6, text="Delhi's", probability=0.01),
            ),
            is_argmax=True,
            entropy_nats=0.2,
            normalized_entropy=0.1,
            margin=0.92,
            rank=1,
            step=2,
            conditioned_on="prompt_plus_generated",
            decoding="greedy",
            temperature=0.0,
        ),
    )
    return GenerationResult(
        question="What is capital of India?",
        prompt_text="Question: What is capital of India?\nAnswer:",
        prompt_mode="plain",
        answer="New Delhi",
        model_name="Qwen2.5-0.5B-Instruct",
        model_revision="main",
        device="cpu",
        dtype="float32",
        latency_seconds=1.5,
        prompt_tokens=10,
        generated_tokens=2,
        tokens=tokens,
        hidden_states_available=True,
        logits_available=True,
        feature=np.array([0.1, 0.2, 0.3] * 100),
        feature_layer=24,
        hidden_state_count=25,
        position="last_generated_token",
        layer_vectors={24: np.array([0.1, 0.2, 0.3] * 100)},
        load_count=1,
        note="",
        decoding="greedy",
        temperature=0.0,
        answer_mean_feature=np.array([0.1, 0.2, 0.3] * 100),
        token_spans={24: np.array([[0.1, 0.2, 0.3] * 100, [0.2, 0.3, 0.4] * 100])},
        capture_mode="generated",
        capture_bytes=2400,
        capture_note="Capture completed.",
    )


@pytest.fixture
def sample_turn(sample_generation):
    """Create a sample TurnRecord for testing."""
    return TurnRecord(
        generation=sample_generation,
        probe_status="experimental probe loaded",
        experimental_score=1.0,
        score_note="Experimental probe score from logistic regression.",
        factual_risk_score=None,
        reference=ReferenceResult(
            configured=True,
            status="faithful",
            detail="Curated item in-capital: the answer asserts an accepted alias.",
        ),
        probe_prediction="hallucinated",
        probe_threshold=0.5,
        calibration_status="not validated",
        probe_task="Task: paired-answer discrimination on supplied HaluEval answers.",
    )


def test_report_to_text(sample_generation, sample_turn, tmp_path):
    """Test that report generates valid text output."""
    catalog_path = tmp_path / "claim_references.yaml"
    catalog_path.write_text(
        "schema: claim_references_v1\nitems:\n  - id: test\n    claim: Test claim\n    accepted: ['test']\n    false_claims: []\n    source: test\n"
    )

    report = build_report(
        question="What is capital of India?",
        generation=sample_generation,
        turn=sample_turn,
        probe=None,
        catalog_path=catalog_path,
    )

    text = report.to_text()

    # Check for required sections
    assert "GLASS BOX | LLM GENERATION ANALYSIS" in text
    assert "[01] QUESTION" in text
    assert "[02] GENERATED ANSWER" in text
    assert "[03] TOKEN GENERATION PROBABILITIES" in text
    assert "[04] FINAL TOKEN ALTERNATIVES" in text
    assert "[05] FACTUALITY VERIFICATION" in text
    assert "[06] INTERNAL MODEL ANALYSIS" in text
    assert "[07] PROBE ANALYSIS" in text
    assert "[08] LAYER-WISE ANALYSIS" in text
    assert "[09] EARLY-WARNING ANALYSIS" in text
    assert "[10] FINAL CONCLUSION" in text

    # Check content
    assert "What is capital of India?" in text
    assert "New Delhi" in text
    assert "Generated tokens: 2" in text


def test_report_to_json(sample_generation, sample_turn, tmp_path):
    """Test that report generates valid JSON output."""
    catalog_path = tmp_path / "claim_references.yaml"
    catalog_path.write_text(
        "schema: claim_references_v1\nitems:\n  - id: test\n    claim: Test claim\n    accepted: ['test']\n    false_claims: []\n    source: test\n"
    )

    report = build_report(
        question="What is capital of India?",
        generation=sample_generation,
        turn=sample_turn,
        probe=None,
        catalog_path=catalog_path,
    )

    json_data = report.to_json()

    # Check required top-level keys
    assert "question" in json_data
    assert "answer" in json_data
    assert "token_probabilities" in json_data
    assert "token_alternatives" in json_data
    assert "generation_config" in json_data
    assert "hidden_state_metadata" in json_data
    assert "probe" in json_data
    assert "factuality" in json_data
    assert "calibration" in json_data
    assert "conclusion" in json_data

    # Check token probabilities structure
    assert json_data["token_probabilities"]["available"] is True
    assert json_data["token_probabilities"]["token_count"] == 2
    assert "average_selected_probability" in json_data["token_probabilities"]
    assert "geometric_mean_probability" in json_data["token_probabilities"]

    # Check that JSON is serializable
    json_str = json.dumps(json_data)
    assert json_str


def test_token_probabilities_calculation(sample_generation, sample_turn, tmp_path):
    """Test token probability statistics are calculated correctly."""
    catalog_path = tmp_path / "claim_references.yaml"
    catalog_path.write_text(
        "schema: claim_references_v1\nitems:\n  - id: test\n    claim: Test claim\n    accepted: ['test']\n    false_claims: []\n    source: test\n"
    )

    report = build_report(
        question="Test",
        generation=sample_generation,
        turn=sample_turn,
        probe=None,
        catalog_path=catalog_path,
    )

    json_data = report.to_json()
    probs = json_data["token_probabilities"]

    # Check mean probability
    expected_mean = (0.72 + 0.94) / 2
    assert abs(probs["average_selected_probability"] - expected_mean) < 0.01

    # Check geometric mean
    expected_geo = np.sqrt(0.72 * 0.94)
    assert abs(probs["geometric_mean_probability"] - expected_geo) < 0.01


def test_no_tokens_case(tmp_path):
    """Test report handles empty token list."""
    generation = GenerationResult(
        question="Test",
        prompt_text="Test",
        prompt_mode="plain",
        answer="",
        model_name="test",
        model_revision="main",
        device="cpu",
        dtype="float32",
        latency_seconds=0.1,
        prompt_tokens=5,
        generated_tokens=0,
        tokens=(),
        hidden_states_available=False,
        logits_available=False,
        feature=None,
        feature_layer=None,
        hidden_state_count=None,
        position="last_generated_token",
        layer_vectors={},
        load_count=1,
        note="",
        decoding="greedy",
        temperature=0.0,
        answer_mean_feature=None,
        token_spans={},
        capture_mode="final",
        capture_bytes=0,
        capture_note="",
    )

    turn = TurnRecord(
        generation=generation,
        probe_status=PROBE_NOT_TRAINED,
        experimental_score=None,
        score_note="No probe trained.",
        factual_risk_score=None,
        reference=ReferenceResult(
            configured=False,
            status="not_configured",
            detail="No reference available.",
        ),
    )

    catalog_path = tmp_path / "claim_references.yaml"
    catalog_path.write_text("schema: claim_references_v1\nitems: []\n")

    report = build_report(
        question="Test",
        generation=generation,
        turn=turn,
        probe=None,
        catalog_path=catalog_path,
    )

    text = report.to_text()
    assert "No tokens were generated" in text

    json_data = report.to_json()
    assert json_data["token_probabilities"]["available"] is False


def test_probe_compatibility_messaging(sample_generation, sample_turn, tmp_path):
    """Test that pairwise probe is marked as incompatible."""
    catalog_path = tmp_path / "claim_references.yaml"
    catalog_path.write_text(
        "schema: claim_references_v1\nitems:\n  - id: test\n    claim: Test claim\n    accepted: ['test']\n    false_claims: []\n    source: test\n"
    )

    # Simulate pairwise probe by setting probe_task appropriately
    sample_turn.probe_task = "Task: paired-answer discrimination on supplied HaluEval answers."

    report = build_report(
        question="Test",
        generation=sample_generation,
        turn=sample_turn,
        probe=None,
        catalog_path=catalog_path,
    )

    text = report.to_text()
    # The probe section should show the experimental status
    assert "[07] PROBE ANALYSIS" in text
    assert "HALLUCINATION PROBABILITY: NOT AVAILABLE" in text


def test_json_serialization_with_numpy(tmp_path):
    """Test that numpy arrays are properly serialized in JSON."""
    generation = GenerationResult(
        question="Test",
        prompt_text="Test",
        prompt_mode="plain",
        answer="Test",
        model_name="test",
        model_revision="main",
        device="cpu",
        dtype="float32",
        latency_seconds=0.1,
        prompt_tokens=5,
        generated_tokens=1,
        tokens=(
            GeneratedToken(
                token_id=1,
                text="Test",
                probability=0.9,
                top_alternatives=(),
                is_argmax=True,
                entropy_nats=0.1,
                normalized_entropy=0.1,
                margin=0.8,
                rank=1,
                step=1,
                conditioned_on="prompt",
                decoding="greedy",
                temperature=0.0,
            ),
        ),
        hidden_states_available=True,
        logits_available=True,
        feature=np.array([0.1, 0.2, 0.3]),
        feature_layer=24,
        hidden_state_count=25,
        position="last_generated_token",
        layer_vectors={24: np.array([0.1, 0.2, 0.3])},
        load_count=1,
        note="",
        decoding="greedy",
        temperature=0.0,
        answer_mean_feature=np.array([0.1, 0.2, 0.3]),
        token_spans={},
        capture_mode="final",
        capture_bytes=0,
        capture_note="",
    )

    turn = TurnRecord(
        generation=generation,
        probe_status=PROBE_NOT_TRAINED,
        experimental_score=None,
        score_note="No probe trained.",
        factual_risk_score=None,
        reference=ReferenceResult(
            configured=False,
            status="not_configured",
            detail="No reference available.",
        ),
    )

    catalog_path = tmp_path / "claim_references.yaml"
    catalog_path.write_text("schema: claim_references_v1\nitems: []\n")

    report = build_report(
        question="Test",
        generation=generation,
        turn=turn,
        probe=None,
        catalog_path=catalog_path,
    )

    # Should not raise an error
    json_str = json.dumps(report.to_json(), default=str)
    assert json_str
