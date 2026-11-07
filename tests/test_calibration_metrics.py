"""Calibration metric unit tests. No Qwen, no network."""

from __future__ import annotations

import pytest

from glassbox.evaluation.calibration_metrics import brier_score, calibration_report


def test_brier_perfect_and_uninformative() -> None:
    assert brier_score([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert brier_score([0, 1], [0.5, 0.5]) == pytest.approx(0.25)
    assert brier_score([0, 1], [1.0, 0.0]) == pytest.approx(1.0)


def test_brier_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        brier_score([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        brier_score([0, 1], [0.1, 1.5])
    with pytest.raises(ValueError):
        brier_score([], [])


def test_small_sample_interval_is_null_with_reason() -> None:
    report = calibration_report(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["a", "b", "c", "d"], n_resamples=20
    )
    assert report["support_status"] == "insufficient"
    assert report["brier_interval"]["low"] is None
    assert report["ece_interval"]["low"] is None
    assert "below" in report["brier_interval"]["note"]
    assert report["brier"] == pytest.approx(brier_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]))


def test_grouped_interval_is_deterministic() -> None:
    labels = [0] * 20 + [1] * 20
    scores = [0.1] * 20 + [0.9] * 20
    groups = [f"g{i}" for i in range(40)]
    first = calibration_report(labels, scores, groups, n_resamples=100)
    second = calibration_report(labels, scores, groups, n_resamples=100)
    assert first["brier_interval"]["low"] == second["brier_interval"]["low"]
    assert first["ece_interval"]["low"] is not None
    assert first["resampling_unit"] == "question_group"
