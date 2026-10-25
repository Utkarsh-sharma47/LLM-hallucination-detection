"""Shortcut baselines and display rules. These tests do not load Qwen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from glassbox.data.schema import Label
from glassbox.data.truthfulqa import TruthfulItem, label_generated_answer
from glassbox.evaluation.metrics import bootstrap_interval
from glassbox.present import render_turn_text
from glassbox.probes.preprocess import apply_standard_scaler, fit_standard_scaler
from glassbox.probes.text_baseline import score_text_rows
from glassbox.reporting import TurnRecord, format_probe_view
from glassbox.verification import CuratedFact, label_answer


def test_scaler_matches_population_mean_and_scale() -> None:
    matrix = np.asarray([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]], dtype=np.float64)
    mean, scale = fit_standard_scaler(matrix)
    assert mean.tolist() == pytest.approx([3.0, 2.0])
    assert scale[1] == pytest.approx(1.0)
    transformed = apply_standard_scaler(matrix, mean, scale)
    assert transformed[:, 0].mean() == pytest.approx(0.0)
    assert transformed[:, 1].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_text_classifier_uses_answer_words_and_reports_the_leak() -> None:
    rows = []
    for index in range(12):
        for split, label, answer in (
            ("train", 0, f"alpha paris fact {index}"),
            ("train", 1, f"beta lyon fiction {index}"),
            ("validation", 0, f"alpha paris held {index}"),
            ("validation", 1, f"beta lyon held {index}"),
            ("test", 0, "alpha paris final"),
            ("test", 1, "beta lyon final"),
        ):
            rows.append(
                {
                    "question_id": f"q{index}-{split}-{label}",
                    "split": split,
                    "label": label,
                    "answer": answer,
                    "reference": "alpha paris fact",
                }
            )
    report = score_text_rows(rows, seed=0)
    assert report["answer_text_tfidf"]["metrics"]["auroc"] == pytest.approx(1.0)
    assert report["lexical_overlap"]["fair_baseline"] is False
    assert "None of these scores are labels of a Qwen generation." in str(report["note"])


def test_bootstrap_refuses_a_tiny_class() -> None:
    labels = np.asarray([0, 0, 1])
    scores = np.asarray([0.1, 0.2, 0.8])
    interval = bootstrap_interval(labels, scores, min_per_class=20)
    assert interval["low"] is None
    assert "not computed" in str(interval["note"])


def test_both_reference_hits_are_ambiguous() -> None:
    item = TruthfulItem(
        question_id="row0",
        category="example",
        question="Where is the capital?",
        correct=("Paris is the capital",),
        incorrect=("Lyon is the capital",),
    )
    label, _detail = label_generated_answer(
        "Paris is the capital, but Lyon is the capital too.",
        item,
    )
    assert label is Label.AMBIGUOUS
    fact = CuratedFact(
        item_id="example",
        split="test",
        question="Where?",
        accepted=("Paris",),
        false_claims=("Lyon",),
    )
    assert label_answer(fact, "Paris and Lyon") is Label.AMBIGUOUS


def test_saturated_experimental_score_is_not_a_risk() -> None:
    from glassbox.models.backend import GeneratedToken, GenerationResult
    from glassbox.models.distributions import TokenAlternative
    from glassbox.verification import ReferenceResult

    token = GeneratedToken(
        token_id=3,
        text="Paris",
        probability=0.99,
        top_alternatives=(TokenAlternative(token_id=3, text="Paris", probability=0.99),),
        is_argmax=True,
        normalized_entropy=0.01,
        margin=0.9,
    )
    generation = GenerationResult(
        question="What is the capital of France?",
        prompt_text="prompt",
        prompt_mode="chat_template",
        answer="The capital of France is Paris.",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        model_revision="7ae557604adf67be50417f59c2c2f167def9a775",
        device="cpu",
        dtype="bfloat16",
        latency_seconds=1.0,
        prompt_tokens=38,
        generated_tokens=1,
        tokens=(token,),
        hidden_states_available=True,
        logits_available=True,
        feature=None,
        feature_layer=24,
        hidden_state_count=25,
        position="last_generated_token",
        layer_vectors={},
        load_count=1,
    )
    turn = TurnRecord(
        generation=generation,
        probe_status="experimental probe loaded",
        experimental_score=1.0,
        score_note="not a calibrated hallucination probability",
        factual_risk_score=None,
        reference=ReferenceResult(
            configured=True, status="faithful", detail="reference is separate"
        ),
        probe_prediction="hallucinated",
        probe_threshold=0.65,
        calibration_status="not validated",
        probe_task=(
            "Task: paired-answer discrimination on supplied HaluEval answers. "
            "This is not a measurement of answers generated by Qwen."
        ),
    )
    text = format_probe_view(turn)
    assert "EXPERIMENTAL PROBE SCORE" in text
    assert "RISK NOT ASSESSED" in text
    assert "paired-answer discrimination" in text
    assert "1.000" in text
    rendered = render_turn_text(turn)
    assert "FAITHFUL" in rendered
    assert "HALLUCINATED" not in rendered
    assert "Predicted class: hallucinated at threshold 0.65." in rendered
    assert "Raw score: 1.000" in rendered
    assert "RISK NOT ASSESSED" in rendered
    assert "not a measurement of" in rendered
    assert "answers generated by Qwen" in rendered


def test_saved_probe_matches_demo_config() -> None:
    probe = Path("artifacts/probes/final-hidden-state/metadata.json")
    if not probe.is_file():
        pytest.skip("saved probe is local and gitignored")
    metadata = json.loads(probe.read_text(encoding="utf-8"))
    demo = yaml.safe_load(Path("configs/demo.yaml").read_text(encoding="utf-8"))
    assert metadata["model_name"] == demo["model"]["name"]
    assert metadata["model_revision"] == demo["model"]["revision"]
    assert metadata["dtype"] == demo["model"]["dtype"]
    assert metadata["position"] == demo["generation"]["position"]
    assert metadata["calibration_validated"] is False
    assert metadata["label_protocol"] == "halueval_qa_pair_v1"
    assert metadata["experimental"] is True
