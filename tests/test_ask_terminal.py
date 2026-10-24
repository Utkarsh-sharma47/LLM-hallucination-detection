"""Terminal evidence and probe checks. These tests do not load Qwen."""

from pathlib import Path

import numpy as np
import pytest

from glassbox.activations.extract import feature_index
from glassbox.cli import main
from glassbox.data.schema import Label
from glassbox.models.backend import GenerationResult
from glassbox.present import reference_label, render_turn_text
from glassbox.probes.logistic import LogisticProbe
from glassbox.probes.store import load_probe, save_probe
from glassbox.reporting import TurnRecord
from glassbox.session import _apply_probe
from glassbox.verification import CuratedFact, ReferenceResult, assess_answer, label_answer


def test_last_generated_token_is_the_answer_index() -> None:
    assert feature_index(4, 3, "last_generated_token") == 6
    assert feature_index(4, 3, "last_prompt_token") == 3
    with pytest.raises(ValueError, match="answer token"):
        feature_index(4, 0, "last_generated_token")


def test_incompatible_probe_does_not_invent_a_score(tmp_path: Path) -> None:
    loaded = _probe(tmp_path / "probe", width=2)
    turn = _turn(feature=np.zeros(3), feature_dim_layer=24)
    _apply_probe(turn, loaded)
    assert turn.probe_status == "PROBE INCOMPATIBLE"
    assert turn.experimental_score is None
    assert turn.probe_prediction is None
    assert turn.factual_risk_score is None
    text = render_turn_text(turn)
    assert "PROBE INCOMPATIBLE" in text
    assert "Raw score" not in text
    assert "RISK NOT ASSESSED" in text


def test_compatible_probe_shows_the_raw_score(tmp_path: Path) -> None:
    loaded = _probe(tmp_path / "probe", width=2)
    turn = _turn(feature=np.array([0.0, 0.0]), feature_dim_layer=24)
    _apply_probe(turn, loaded)
    assert turn.experimental_score is not None
    assert 0.0 <= turn.experimental_score <= 1.0
    assert turn.probe_prediction in {"faithful", "hallucinated"}
    assert turn.factual_risk_score is None
    text = render_turn_text(turn)
    assert "EXPERIMENTAL PROBE SCORE" in text
    assert f"Raw score: {turn.experimental_score:.3f}" in text
    assert "RISK NOT ASSESSED" in text
    assert "FAITHFUL" in text


def test_missing_reference_is_not_a_hallucination() -> None:
    reference = ReferenceResult(
        configured=False,
        status="not_configured",
        detail="NO REFERENCE AVAILABLE. This question is outside the curated list.",
        label=None,
    )
    turn = _turn(feature=None, feature_dim_layer=None, reference=reference)
    text = render_turn_text(turn)
    assert reference_label("not_configured") == "NO REFERENCE AVAILABLE"
    assert "NO REFERENCE AVAILABLE" in text
    assert "HALLUCINATED" not in text


def test_ambiguous_answer_stays_ambiguous() -> None:
    fact = CuratedFact(
        item_id="both",
        question="Where is the example?",
        accepted=("alpha bravo",),
        false_claims=("echo foxtrot",),
        split="demo",
    )
    answer = "Alpha bravo. Echo foxtrot."
    assert label_answer(fact, answer) is Label.AMBIGUOUS
    result = assess_answer(fact, answer)
    turn = _turn(
        feature=None,
        feature_dim_layer=None,
        reference=result,
    )
    text = render_turn_text(turn)
    assert "AMBIGUOUS" in text
    assert "FAITHFUL" not in text
    assert "HALLUCINATED" not in text


def test_incomplete_probe_directory_fails_before_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("model backend was constructed")

    monkeypatch.setattr("glassbox.cli.HuggingFaceBackend", boom)
    probe = tmp_path / "probe"
    probe.mkdir()
    code = main(
        ["--offline", "--probe", str(probe), "--once", "--question", "What is two plus two?"]
    )
    assert code == 1
    error = capsys.readouterr().err
    assert "incomplete" in error.lower()
    assert "No prediction was invented." in error


def _probe(directory: Path, *, width: int):
    probe = LogisticProbe()
    probe.fit(
        np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.1], [0.9, 0.8]]),
        np.array([0, 1, 0, 1]),
    )
    save_probe(
        directory,
        probe,
        metadata={
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_revision": "b" * 40,
            "layer_index": 24,
            "hidden_state_count": 25,
            "position": "last_generated_token",
            "prompt_mode": "chat_template",
            "dtype": "bfloat16",
            "threshold": 0.65,
            "calibration_validated": False,
            "label_protocol": "halueval_qa_pair_v1",
        },
        mean=np.zeros(width),
        scale=np.ones(width),
    )
    return load_probe(directory)


def _turn(
    *,
    feature: np.ndarray | None,
    feature_dim_layer: int | None,
    reference: ReferenceResult | None = None,
) -> TurnRecord:
    if reference is None:
        reference = ReferenceResult(
            configured=True,
            status="faithful",
            detail="Curated item example: the answer asserts an accepted alias.",
            label=Label.FAITHFUL,
        )
    return TurnRecord(
        generation=GenerationResult(
            question="What is the capital of Australia?",
            prompt_text="question",
            prompt_mode="chat_template",
            answer="Canberra",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            model_revision="b" * 40,
            device="cpu",
            dtype="bfloat16",
            latency_seconds=0.1,
            prompt_tokens=4,
            generated_tokens=1,
            tokens=(),
            hidden_states_available=feature is not None,
            logits_available=True,
            feature=feature,
            feature_layer=feature_dim_layer,
            hidden_state_count=25 if feature is not None else None,
            position="last_generated_token",
            layer_vectors={},
            load_count=1,
        ),
        probe_status="experimental probe loaded",
        experimental_score=None,
        score_note="",
        factual_risk_score=None,
        reference=reference,
    )
