"""Decoding probabilities match the distribution that selects the token."""

import csv

import numpy as np

from glassbox.models.backend import GeneratedToken, GenerationResult, _choose_id, _token_from_logits
from glassbox.models.distributions import (
    probabilities_from_logits,
    scale_logits,
    write_distribution_csv,
)
from glassbox.present import render_turn_text
from glassbox.probes.logistic import PROBE_NOT_TRAINED
from glassbox.reporting import TurnRecord
from glassbox.verification import ReferenceResult


class _Tokenizer:
    def decode(self, ids, **_kwargs):
        return {0: "No", 1: "Yes", 2: "Maybe", 3: "Tail"}[int(ids[0])]


def test_greedy_probability_is_the_full_softmax_of_the_selected_id():
    logits = np.array([0.0, 1.0, 4.0])
    chosen = _choose_id(logits, 0.0)
    token = _token_from_logits(_Tokenizer(), chosen, logits, top_k=2, temperature=0.0, step=1)
    expected = probabilities_from_logits(logits)[chosen]
    assert chosen == 2
    assert token.decoding == "greedy"
    assert token.conditioned_on == "prompt"
    assert np.isclose(token.probability, expected)
    assert all(item.token_id != chosen for item in token.top_alternatives)
    assert len(token.top_alternatives) == 2
    assert token.vocabulary_probabilities is None


def test_temperature_reports_the_distribution_that_was_sampled(monkeypatch):
    logits = np.array([0.0, 1.0, 4.0])
    monkeypatch.setattr(np.random, "choice", lambda _size, p: 0)
    chosen = _choose_id(logits, 2.0)
    token = _token_from_logits(
        _Tokenizer(),
        chosen,
        logits,
        top_k=2,
        temperature=2.0,
        step=2,
        retain_full=True,
    )
    scaled = probabilities_from_logits(scale_logits(logits, 2.0))
    raw = probabilities_from_logits(logits)
    assert chosen == 0
    assert token.conditioned_on == "prompt_plus_generated"
    assert token.decoding == "temperature"
    assert np.isclose(token.probability, scaled[0])
    assert not np.isclose(token.probability, raw[0])
    assert np.isclose(float(token.vocabulary_probabilities.sum()), 1.0)


def test_full_vocabulary_export_is_not_printed(tmp_path):
    probabilities = np.array([0.5, 0.2, 0.2, 0.125], dtype=np.float32)
    token = GeneratedToken(
        token_id=0,
        text="Yes",
        probability=0.5,
        top_alternatives=(),
        is_argmax=True,
        step=1,
        conditioned_on="prompt",
        decoding="greedy",
        vocabulary_probabilities=probabilities,
    )
    generation = GenerationResult(
        question="Who wrote The Odyssey?",
        prompt_text="prompt",
        prompt_mode="chat_template",
        answer="Yes",
        model_name="example",
        model_revision="test",
        device="cpu",
        dtype="float32",
        latency_seconds=0.1,
        prompt_tokens=4,
        generated_tokens=1,
        tokens=(token,),
        hidden_states_available=False,
        logits_available=True,
        feature=None,
        feature_layer=None,
        hidden_state_count=None,
        position="last_generated_token",
        layer_vectors={},
        load_count=1,
    )
    turn = TurnRecord(
        generation=generation,
        probe_status=PROBE_NOT_TRAINED,
        experimental_score=None,
        score_note="No trained probe is loaded.",
        factual_risk_score=None,
        reference=ReferenceResult(False, "not_configured", "NO REFERENCE AVAILABLE."),
    )
    rendered = render_turn_text(turn, width=100)
    collapsed = " ".join(rendered.split())
    assert "12.5%" not in rendered
    assert "not input-token probabilities" in collapsed
    assert "Step 1 is conditioned on the prompt" in collapsed
    path = write_distribution_csv(tmp_path / "full.csv", [token], mode="full", text_for_id=str)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 4
    assert rows[3]["piece_id"] == "3"
    assert float(rows[3]["probability"]) == 0.125
    assert rows[0]["selected"] == "1"
    assert rows[0]["conditioned_on"] == "prompt"
