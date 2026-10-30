"""Length-shortcut diagnostics. These tests do not load Qwen or rewrite probes."""

from __future__ import annotations

import json
from pathlib import Path

from glassbox.cli import main
from glassbox.evaluation.generation_calibration import PAIR_THRESHOLD
from glassbox.evaluation.length_shortcut import (
    controlled_variants,
    length_matched_subset,
    summarize_length_effects,
)
from glassbox.verification import CuratedFact


def test_controlled_variants_are_length_matched_by_question() -> None:
    fact = CuratedFact(
        item_id="fr-capital",
        question="What is the capital of France?",
        accepted=("Paris",),
        false_claims=("Lyon",),
        split="train",
    )
    variants = controlled_variants(fact)
    formats = {str(item["format"]): item for item in variants}
    assert formats["short_faithful"]["answer"] == "Paris"
    assert formats["short_faithful"]["label"] == 0
    assert formats["sentence_faithful"]["label"] == 0
    assert "Paris" in str(formats["sentence_faithful"]["answer"])
    assert formats["short_hallucinated"]["label"] == 1
    assert formats["sentence_hallucinated"]["label"] == 1
    assert len(str(formats["sentence_faithful"]["answer"])) > len(
        str(formats["short_faithful"]["answer"])
    )


def test_sentence_faithful_false_positives_are_reported_as_format_effects() -> None:
    rows = [
        _row("q1", "short_faithful", "Paris", 0, 0.02),
        _row("q1", "sentence_faithful", "The answer is Paris.", 0, 0.99),
        _row("q1", "short_hallucinated", "Lyon", 1, 0.10),
        _row("q1", "sentence_hallucinated", "The answer is Lyon.", 1, 0.99),
        _row("q2", "short_faithful", "Ottawa", 0, 0.01),
        _row("q2", "sentence_faithful", "The answer is Ottawa.", 0, 0.97),
        _row("q2", "short_hallucinated", "Toronto", 1, 0.05),
        _row("q2", "sentence_hallucinated", "The answer is Toronto.", 1, 0.96),
    ]
    report = summarize_length_effects(rows, threshold=PAIR_THRESHOLD)
    assert report["threshold"] == 0.65
    assert report["threshold_was_fit_on_these_rows"] is False
    assert report["pair_probe_rewritten"] is False
    assert report["calibration_validated"] is False
    assert report["successful_detector"] is False
    assert report["formats"]["sentence_faithful"]["false_positive_rate"] == 1.0
    assert report["formats"]["short_faithful"]["false_positive_rate"] == 0.0
    assert report["interpretation"]["learns"] == "superficial_answer_patterns"
    matched = length_matched_subset(rows, tolerance=15)
    assert matched
    assert report["overall"]["inconclusive"] is True


def test_length_eval_cli_does_not_rewrite_pair_probe_metadata(tmp_path: Path, monkeypatch) -> None:
    pair = tmp_path / "probes" / "final-hidden-state"
    pair.mkdir(parents=True)
    metadata = pair / "metadata.json"
    metadata.write_text('{"threshold": 0.65, "calibration_validated": false}\n', encoding="utf-8")
    before = metadata.read_bytes()
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            [
                _row("q1", "short_faithful", "Paris", 0, 0.1),
                _row("q1", "sentence_faithful", "The answer is Paris.", 0, 0.9),
                _row("q1", "short_hallucinated", "Lyon", 1, 0.2),
                _row("q1", "sentence_hallucinated", "The answer is Lyon.", 1, 0.9),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("glassbox.evaluation.length_eval.repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        "glassbox.evaluation.length_eval.generation_experiment_dir",
        lambda run_id: tmp_path / "artifacts" / "experiments" / run_id,
    )
    code = main(
        [
            "eval-length-shortcut",
            "--scores",
            str(scores),
            "--probe",
            str(pair),
            "--run-id",
            "length-unit-001",
        ]
    )
    assert code == 0
    assert metadata.read_bytes() == before
    assert (tmp_path / "artifacts" / "experiments" / "length-unit-001" / "report.json").is_file()
    assert not (tmp_path / "probes" / "final-hidden-state" / "probe.joblib").exists()


def _row(question_id: str, fmt: str, answer: str, label: int, score: float) -> dict[str, object]:
    return {
        "question_id": question_id,
        "format": fmt,
        "answer": answer,
        "label": label,
        "probe_score": score,
    }
