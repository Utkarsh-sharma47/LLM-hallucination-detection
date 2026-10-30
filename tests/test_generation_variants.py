"""Generation variant comparison. These tests do not load Qwen or replace probes."""

from __future__ import annotations

import json
from pathlib import Path

from glassbox.cli import main
from glassbox.evaluation.generation_calibration import PAIR_THRESHOLD
from glassbox.probes.generation_variants import evaluate_generation_variants


def test_variants_report_inconclusive_below_twenty_and_keep_pair_probe(
    tmp_path: Path,
) -> None:
    pair = tmp_path / "final-hidden-state"
    pair.mkdir()
    metadata = pair / "metadata.json"
    metadata.write_text(
        json.dumps({"threshold": 0.65, "calibration_validated": False}) + "\n",
        encoding="utf-8",
    )
    before = metadata.read_bytes()
    rows = _population(train=8, validation=4, test=4)
    report = evaluate_generation_variants(
        rows,
        seed=0,
        frozen_threshold=0.25,
        pair_probe_path=pair,
    )
    assert report["live_probe_untouched"] is True
    assert report["target_met"] is False
    assert report["successful_detector"] is False
    assert report["pair_threshold_unchanged"] == PAIR_THRESHOLD
    assert report["generation_calibration_gate"]["status"] == "rejected"
    assert report["generation_calibration_gate"]["factual_risk_available"] is False
    assert report["variants"]["mean_pool"]["available"] is False
    assert report["variants"]["frozen_final_piece"]["threshold_was_refit"] is False
    assert metadata.read_bytes() == before


def test_length_can_explain_scores_when_labels_follow_answer_length() -> None:
    rows = []
    cursor = 0
    for split, count in (("train", 40), ("validation", 40), ("test", 40)):
        for _ in range(count // 2):
            rows.append(_row(f"q{cursor}", split, 0, "yes", 0.1, 4))
            cursor += 1
            rows.append(
                _row(
                    f"q{cursor}",
                    split,
                    1,
                    "this hallucinated answer is much longer than the faithful one",
                    0.9,
                    40,
                )
            )
            cursor += 1
    report = evaluate_generation_variants(rows, seed=0, frozen_threshold=0.5)
    length = report["variants"]["answer_length"]
    tfidf = report["variants"]["answer_tfidf"]
    assert isinstance(length, dict)
    assert isinstance(tfidf, dict)
    assert report["detector_conclusion"]["successful_detector"] is False
    assert report["target_met"] is True


def test_variant_cli_refuses_historical_and_pair_probe_names(capsys) -> None:
    code = main(
        [
            "compare-generation-variants",
            "--predictions",
            "missing.json",
            "--run-id",
            "halueval-generation-001",
        ]
    )
    assert code == 1
    assert "historical generation run" in capsys.readouterr().err
    code = main(
        [
            "compare-generation-variants",
            "--predictions",
            "missing.json",
            "--run-id",
            "final-hidden-state",
        ]
    )
    assert code == 1
    assert "protected" in capsys.readouterr().err


def test_variant_cli_writes_a_new_report_without_a_live_probe(tmp_path: Path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.json"
    rows = _population(train=8, validation=4, test=4)
    predictions.write_text(json.dumps({"scored_rows": rows}), encoding="utf-8")
    monkeypatch.setattr("glassbox.probes.generation_variants.repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        "glassbox.probes.generation_variants.generation_experiment_dir",
        lambda run_id: tmp_path / "artifacts" / "experiments" / run_id,
    )
    monkeypatch.setattr(
        "glassbox.probes.generation_variants.generation_probe_dir",
        lambda run_id: tmp_path / "artifacts" / "probes" / "generation-runs" / run_id,
    )
    code = main(
        [
            "compare-generation-variants",
            "--predictions",
            str(predictions),
            "--run-id",
            "unit-variants",
            "--threshold",
            "0.25",
        ]
    )
    assert code == 0
    assert (tmp_path / "artifacts" / "experiments" / "unit-variants" / "report.json").is_file()
    assert not (tmp_path / "artifacts" / "probes" / "generation-runs" / "unit-variants").exists()
    assert not (tmp_path / "artifacts" / "probes" / "final-hidden-state").exists()


def _population(*, train: int, validation: int, test: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cursor = 0
    for split, count in (("train", train), ("validation", validation), ("test", test)):
        half = count // 2
        for _ in range(half):
            rows.append(_row(f"q{cursor}", split, 0, "alpha", 0.1, 4))
            cursor += 1
        for _ in range(count - half):
            rows.append(_row(f"q{cursor}", split, 1, "beta contradicted answer", 0.9, 4))
            cursor += 1
    return rows


def _row(
    question_id: str,
    split: str,
    label: int,
    answer: str,
    score: float,
    dim: int = 4,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "group_id": question_id,
        "split": split,
        "label": label,
        "label_name": "HALLUCINATED" if label == 1 else "FAITHFUL",
        "answer": answer,
        "right_answer": "alpha",
        "false_spans": ["beta"],
        "probe_score": score,
    }
