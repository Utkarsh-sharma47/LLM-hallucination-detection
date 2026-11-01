"""Generation calibration metric helpers do not load a model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glassbox.evaluation.calibration_metrics import brier_score, calibration_report
from glassbox.probes.store import LoadedProbe, experimental_score, raw_score


def test_brier_score_perfect_uninformative_and_miscalibrated() -> None:
    labels = [0, 0, 1, 1]
    assert brier_score(labels, [0.0, 0.0, 1.0, 1.0]) == 0.0
    assert brier_score(labels, [0.5, 0.5, 0.5, 0.5]) == 0.25
    assert brier_score(labels, [0.9, 0.9, 0.1, 0.1]) == pytest.approx(0.81)


def test_calibration_report_uses_fixed_bins_and_grouped_intervals() -> None:
    labels = [0] * 20 + [1] * 20
    scores = [0.0] * 20 + [1.0] * 20
    groups = [f"q{index}" for index in range(40)]
    report = calibration_report(labels, scores, groups, n_resamples=200)

    assert report["brier"] == 0.0
    assert report["ece"] == 0.0
    assert report["bins"] == 10
    assert report["support_status"] == "sufficient"
    assert report["resampling_unit"] == "question_group"
    assert report["brier_interval"]["low"] == 0.0
    assert report["brier_interval"]["high"] == 0.0
    assert report["ece_interval"]["low"] == 0.0
    assert report["ece_interval"]["high"] == 0.0


def test_calibration_intervals_are_null_below_class_support() -> None:
    report = calibration_report(
        [0] * 19 + [1] * 20,
        [0.1] * 19 + [0.9] * 20,
        [f"q{index}" for index in range(39)],
    )

    assert report["support_status"] == "insufficient"
    assert report["brier_interval"]["low"] is None
    assert report["ece_interval"]["high"] is None
    assert "below 20" in report["brier_interval"]["note"]


def test_calibration_report_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        calibration_report([0, 1], [0.2], ["q1", "q2"])
    with pytest.raises(ValueError, match="probabilities"):
        brier_score([0, 1], [0.2, float("nan")])
    with pytest.raises(ValueError, match="one non-empty id"):
        calibration_report([0, 1], [0.2, 0.8], ["q1", ""])


def test_generation_gate_score_uses_classifier_probability_without_platt() -> None:
    class FixedProbability:
        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            assert features.shape == (1,)
            return np.asarray([0.72])

    loaded = LoadedProbe(
        probe=FixedProbability(),
        metadata={
            "calibration": "generation_validation_ece_gate",
            "calibration_validated": True,
            "preprocessing": {"mean": [0.0], "scale": [1.0]},
        },
        directory=Path("."),
    )

    assert experimental_score(loaded, np.asarray([1.0])) == 0.72


def test_raw_score_bypasses_a_saved_platt_transform() -> None:
    class FixedProbability:
        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            return np.asarray([0.72])

    loaded = LoadedProbe(
        probe=FixedProbability(),
        metadata={
            "calibration": "platt",
            "calibration_validated": True,
            "platt": {"coef": 0.0, "intercept": 0.0},
            "preprocessing": {"mean": [0.0], "scale": [1.0]},
        },
        directory=Path("."),
    )

    assert raw_score(loaded, np.asarray([1.0])) == 0.72
    assert experimental_score(loaded, np.asarray([1.0])) == 0.5
