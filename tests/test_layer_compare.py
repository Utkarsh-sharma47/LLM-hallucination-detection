"""Issue #13 layer comparison. These tests do not load Qwen or replace probes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from glassbox.cli import main
from glassbox.evaluation.generation_calibration import PAIR_THRESHOLD
from glassbox.probes.layer_compare import REGISTERED_REPRESENTATIONS, evaluate_layer_comparison


def test_representations_are_named_before_scoring_and_missing_layers_stay_unavailable() -> None:
    names = [str(item["name"]) for item in REGISTERED_REPRESENTATIONS]
    assert "saved_probe_representation" in names
    assert "mean_pool_final_layer" in names
    rows = _population()
    report = evaluate_layer_comparison(rows, seed=0)
    assert report["early_warning_lead"] is None
    assert report["pair_threshold_unchanged"] == PAIR_THRESHOLD
    assert report["pair_probe_rewritten"] is False
    assert report["successful_detector"] is False
    assert report["detector_conclusion"]["ranked_by_pair_protocol_auroc"] is False
    saved = report["representations"]["saved_probe_representation"]
    assert saved["available"] is True
    assert saved["saved_as_live_probe"] is False
    assert report["representations"]["middle_layer_last_piece"]["available"] is False
    assert report["representations"]["embedding_last_piece"]["available"] is False
    mean = report["representations"]["mean_pool_final_layer"]
    assert mean["available"] is True
    interval = saved.get("auroc_interval")
    if isinstance(interval, dict):
        assert interval.get("point") is None


def test_layer_cli_refuses_pair_probe_name(capsys) -> None:
    code = main(
        [
            "compare-generation-layers",
            "--predictions",
            "missing.json",
            "--run-id",
            "final-hidden-state",
        ]
    )
    assert code == 1
    assert "protected" in capsys.readouterr().err


def test_layer_cli_does_not_rewrite_pair_metadata(tmp_path: Path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.json"
    serializable = []
    for row in _population():
        item = dict(row)
        item["vector"] = np.asarray(row["vector"]).tolist()
        item["answer_mean"] = np.asarray(row["answer_mean"]).tolist()
        serializable.append(item)
    predictions.write_text(json.dumps({"rows": serializable}), encoding="utf-8")
    pair = tmp_path / "final-hidden-state"
    pair.mkdir()
    metadata = pair / "metadata.json"
    metadata.write_text('{"threshold": 0.65, "calibration_validated": false}\n', encoding="utf-8")
    before = metadata.read_bytes()
    monkeypatch.setattr("glassbox.probes.layer_compare.repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        "glassbox.probes.layer_compare.generation_experiment_dir",
        lambda run_id: tmp_path / "artifacts" / "experiments" / run_id,
    )
    code = main(
        [
            "compare-generation-layers",
            "--predictions",
            str(predictions),
            "--pair-probe",
            str(pair),
            "--run-id",
            "layer-unit",
        ]
    )
    assert code == 0
    assert metadata.read_bytes() == before
    report = json.loads(
        (tmp_path / "artifacts" / "experiments" / "layer-unit" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["early_warning_lead"] is None
    assert report["live_probe_untouched"] is True


def _population() -> list[dict[str, object]]:
    rows = []
    cursor = 0
    for split, count in (("train", 8), ("validation", 4), ("test", 4)):
        for _ in range(count // 2):
            rows.append(_row(f"q{cursor}", split, 0, 0.1))
            cursor += 1
            rows.append(_row(f"q{cursor}", split, 1, 0.9))
            cursor += 1
    return rows


def _row(question_id: str, split: str, label: int, score: float) -> dict[str, object]:
    vector = np.full(4, score, dtype=np.float64)
    mean = np.full(4, min(score + 0.05, 1.0), dtype=np.float64)
    return {
        "question_id": question_id,
        "split": split,
        "label": label,
        "label_name": "HALLUCINATED" if label == 1 else "FAITHFUL",
        "answer": "long answer text" if label == 1 else "short",
        "right_answer": "short",
        "false_spans": ["other"],
        "probe_score": score,
        "vector": vector,
        "answer_mean": mean,
        "group_id": question_id,
    }
