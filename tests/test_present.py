"""The risk report must not relabel a token probability as hallucination risk."""

from glassbox.models.backend import GeneratedToken, GenerationResult
from glassbox.models.distributions import TokenAlternative
from glassbox.present import render_turn_text, risk_level
from glassbox.probes.logistic import PROBE_NOT_TRAINED
from glassbox.reporting import TurnRecord
from glassbox.verification import ReferenceResult


def _turn(probability: float) -> TurnRecord:
    token = GeneratedToken(
        token_id=3,
        text="Canberra",
        probability=probability,
        top_alternatives=(
            TokenAlternative(token_id=3, text="Canberra", probability=probability),
            TokenAlternative(token_id=4, text="Sydney", probability=0.2),
        ),
        is_argmax=True,
        normalized_entropy=0.4,
        margin=probability - 0.2,
    )
    generation = GenerationResult(
        question="What is the capital of Australia?",
        prompt_text="Question: What is the capital of Australia?\nAnswer:",
        prompt_mode="plain",
        answer="Canberra",
        model_name="example-model",
        model_revision="test",
        device="cpu",
        dtype="float32",
        latency_seconds=0.5,
        prompt_tokens=6,
        generated_tokens=1,
        tokens=(token,),
        hidden_states_available=False,
        logits_available=True,
        feature=None,
        feature_layer=None,
        hidden_state_count=None,
        position="last_prompt_token",
        layer_vectors={},
        load_count=1,
    )
    return TurnRecord(
        generation=generation,
        probe_status=PROBE_NOT_TRAINED,
        experimental_score=None,
        score_note="No trained probe is loaded.",
        factual_risk_score=None,
        reference=ReferenceResult(
            configured=True,
            status="faithful",
            detail="Curated item au-capital: the answer asserts an accepted alias. This reference check is not a probe score.",
        ),
    )


def test_token_probability_is_shown_and_is_not_the_risk_score():
    text = render_turn_text(_turn(0.7))
    risk = text.split("FACTUAL RISK", 1)[1].split("MODEL UNCERTAINTY", 1)[0]
    assert "70.0%" in text
    assert "70.0%" not in risk
    assert "PROBE NOT TRAINED" in text
    assert "unavailable" in risk
    assert "NOT ASSESSED" in text
    assert "CORRECT" in text
    assert "can be correct" in text
    assert "can be false" in text
    for invented in ("82.4%", "11.2%", "96.1%", "98.0%"):
        assert invented not in text


def test_missing_probe_is_not_a_risk_level():
    assert risk_level(_turn(0.7)) == "RISK NOT ASSESSED"
    rendered = render_turn_text(_turn(0.7))
    token_section = rendered.split("OUTPUT TOKEN", 1)[1].split("MODEL UNCERTAINTY", 1)[0]
    assert "Probe" in token_section
    assert "PROBE NOT TRAINED" in token_section
