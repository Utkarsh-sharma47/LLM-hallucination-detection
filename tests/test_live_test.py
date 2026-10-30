"""Live-test reporting without loading Qwen."""

from pathlib import Path

import numpy as np
import pytest

from glassbox.cli import main
from glassbox.data.schema import Label
from glassbox.probes.live_test import (
    EARLY_WARNING_NOTE,
    PARIS_REGRESSION,
    RISK_BANNER,
    SCORE_BANNER,
    SUITE_RUBRIC,
    ProbeRead,
    assemble_case,
    assert_live_directory,
    factuality_status,
    format_case,
    refuse_intermediate_score,
    rubric_label,
    score_vector,
    summarize_cases,
)
from glassbox.probes.logistic import LogisticProbe
from glassbox.probes.store import LoadedProbe


def test_suite_covers_the_requested_categories():
    categories = {str(item["category"]) for item in SUITE_RUBRIC}
    assert {
        "established_fact",
        "misconception",
        "confident_false",
        "mixed_claim",
        "no_reference",
        "paraphrase",
        "length_variation",
    } <= categories
    assert PARIS_REGRESSION["supplied_answer"] == "Paris."
    assert "fr-capital" in {str(item["case_id"]) for item in SUITE_RUBRIC}


def test_paris_period_is_faithful_and_independent_of_the_probe_score():
    label, _detail, configured = rubric_label(PARIS_REGRESSION, "Paris.")
    assert label is Label.FAITHFUL
    assert factuality_status(label, configured=configured) == "FAITHFUL"
    case = assemble_case(
        case_id="fr-capital-paris-period",
        category="recorded_false_positive",
        question="What is the capital of France?",
        answer="Paris.",
        source="supplied_completion",
        label=label,
        configured=True,
        evidence_detail="asserts Paris",
        evidence_source="live_suite_rubric",
        probe_read=ProbeRead(
            True,
            0.99,
            "hallucinated",
            0.65,
            "experimental",
            "halueval_qa_pair_v1",
            "artifacts/probes/final-hidden-state",
        ),
        hidden={"position": "last_generated_token", "layer": 24, "dimension": 4},
        include_in_aggregate=False,
    )
    assert case["factuality_status"] == "FAITHFUL"
    assert case["probe_prediction"] == "hallucinated"
    assert case["reference_check"] == "supported"
    text = format_case(case)
    assert "PROBE PREDICTION hallucinated" in text
    assert "PROBE SCORE 0.990" in text
    assert "FINAL FACTUALITY STATUS FAITHFUL" in text
    assert SCORE_BANNER in text
    assert RISK_BANNER in text
    assert "Early warning was not measured" in text


def test_missing_reference_stays_unvalidated():
    spec = next(item for item in SUITE_RUBRIC if item["case_id"] == "no-reference-kemper")
    label, detail, configured = rubric_label(spec, "Kemper")
    assert configured is False
    assert label is Label.UNVALIDATED
    assert "NO REFERENCE AVAILABLE" in detail
    assert factuality_status(label, configured=configured) == "UNVALIDATED"


def test_wording_mismatch_is_not_a_hallucination():
    spec = next(item for item in SUITE_RUBRIC if item["case_id"] == "fr-capital")
    label, _detail, configured = rubric_label(spec, "The city on the Seine.")
    assert configured is True
    assert label is Label.UNVALIDATED


def test_intermediate_positions_are_not_scored():
    assert "last_generated_token" in refuse_intermediate_score()
    assert "lead time" in EARLY_WARNING_NOTE
    read = score_vector(
        _probe(),
        np.zeros(4),
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        model_revision="a" * 40,
        layer_index=24,
        hidden_state_count=25,
        position="last_prompt_token",
        prompt_mode="chat_template",
        dtype="bfloat16",
    )
    assert read.applied is False
    assert read.score is None
    assert "last_generated_token" in read.note


def test_small_suite_is_inconclusive_and_does_not_fit_a_threshold():
    cases = [
        _binary_case("a", "FAITHFUL", 0.2),
        _binary_case("b", "HALLUCINATED", 0.9),
    ]
    summary = summarize_cases(cases, threshold=0.65)
    assert summary["aggregate_performance"] == "inconclusive"
    assert summary["successful_detector"] is False
    assert summary["threshold_was_fit_on_this_suite"] is False
    assert summary["threshold_used"] == 0.65
    assert summary["tfidf_baseline"]["computed"] is False
    assert summary["calibration"] == "rejected_or_unavailable"
    assert summary["early_warning"] == "not established"
    assert summary["confusion"] == {"tn": 1, "fp": 0, "fn": 0, "tp": 1}


def test_one_class_does_not_invent_metrics():
    summary = summarize_cases([_binary_case("a", "FAITHFUL", 0.1)], threshold=0.65)
    assert "metrics" not in summary
    assert summary["aggregate_performance"] == "inconclusive"


def test_existing_and_protected_directories_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "glassbox.probes.live_test.generation_experiment_dir",
        lambda run_id: tmp_path / run_id,
    )
    target = tmp_path / "live-terminal-001"
    target.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        assert_live_directory("live-terminal-001")
    with pytest.raises(ValueError, match="protected"):
        assert_live_directory("final-hidden-state")


def test_missing_probe_does_not_load_a_model(tmp_path, monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("model backend was constructed")

    monkeypatch.setattr("glassbox.probes.live_test.HuggingFaceBackend", forbidden)
    code = main(
        [
            "live-test",
            "--probe",
            str(tmp_path / "missing"),
            "--suite",
            "--run-id",
            "live-test-missing",
        ]
    )
    assert code == 1
    error = capsys.readouterr().err
    assert "missing or incomplete" in error
    assert "does not train" in error


def test_incomplete_probe_directory_is_rejected(tmp_path, monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("model backend was constructed")

    monkeypatch.setattr("glassbox.probes.live_test.HuggingFaceBackend", forbidden)
    folder = tmp_path / "empty-probe"
    folder.mkdir()
    code = main(["live-test", "--probe", str(folder), "--suite"])
    assert code == 1
    assert "incomplete" in capsys.readouterr().err


def _binary_case(case_id: str, status: str, score: float) -> dict[str, object]:
    return {
        "case_id": case_id,
        "source": "generated",
        "include_in_aggregate": True,
        "factuality_status": status,
        "probe_applied": True,
        "probe_score": score,
        "probe_prediction": "hallucinated" if score >= 0.65 else "faithful",
        "answer": "x" * 10,
        "question": case_id,
        "reference_check": "supported" if status == "FAITHFUL" else "contradicted",
        "evidence_detail": "test",
    }


def _probe() -> LoadedProbe:
    return LoadedProbe(
        probe=LogisticProbe(),
        metadata={
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_revision": "a" * 40,
            "layer_index": 24,
            "hidden_state_count": 25,
            "position": "last_generated_token",
            "prompt_mode": "chat_template",
            "dtype": "bfloat16",
            "feature_dim": 4,
            "label_protocol": "halueval_qa_pair_v1",
            "threshold": 0.65,
            "calibration": "none",
            "calibration_validated": False,
            "preprocessing": {
                "type": "standard_scaler",
                "mean": [0.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0, 1.0],
            },
        },
        directory=Path("memory-probe"),
    )
