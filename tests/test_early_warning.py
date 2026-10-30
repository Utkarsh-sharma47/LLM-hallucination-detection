"""Issue #7 early-warning bookkeeping. These tests do not load Qwen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from glassbox.cli import main
from glassbox.evaluation.early_warning import (
    EVENT_DEFINITION,
    EVENT_MEASUREMENT_ENABLED,
    event_index_from_prefix_labels,
    record_without_event,
    score_generated_positions,
)
from glassbox.evaluation.generation_calibration import PAIR_THRESHOLD
from glassbox.evaluation.metrics import early_warning_lead


def test_two_positions_are_scored_without_publishing_a_lead() -> None:
    pieces = np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float64)
    result = score_generated_positions(
        pieces,
        lambda vector: float(np.clip(vector.mean(), 0.0, 1.0)),
        event_index=1,
        threshold=PAIR_THRESHOLD,
    )
    assert result["n_positions"] == 2
    assert result["early_warning_lead"] is None
    assert result["event_index"] is None
    assert EVENT_MEASUREMENT_ENABLED is False
    assert "not a calibrated" in EVENT_DEFINITION
    assert "pair-protocol AUROC" in EVENT_DEFINITION


def test_missing_event_and_missing_flag_keep_lead_null() -> None:
    assert record_without_event()["early_warning_lead"] is None
    assert event_index_from_prefix_labels(["UNVALIDATED", "AMBIGUOUS", "FAITHFUL"]) is None
    assert event_index_from_prefix_labels(["FAITHFUL", "HALLUCINATED"]) == 1
    assert early_warning_lead([False, False], 1) is None


def test_early_warning_cli_refuses_historical_run(capsys) -> None:
    code = main(
        ["eval-early-warning", "--scores", "missing.json", "--run-id", "halueval-generation-001"]
    )
    assert code == 1
    assert "historical generation run" in capsys.readouterr().err


def test_early_warning_cli_writes_null_lead(tmp_path: Path, monkeypatch) -> None:
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps({"piece_vectors": [[0.1, 0.2], [0.8, 0.9]], "event_index": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr("glassbox.evaluation.early_warning.repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        "glassbox.evaluation.early_warning.generation_experiment_dir",
        lambda run_id: tmp_path / "artifacts" / "experiments" / run_id,
    )
    code = main(["eval-early-warning", "--scores", str(scores), "--run-id", "early-warning-unit"])
    assert code == 0
    report = json.loads(
        (tmp_path / "artifacts" / "experiments" / "early-warning-unit" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["early_warning_lead"] is None
    assert report["event_measurement_enabled"] is False
