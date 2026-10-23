"""Tests for metric helpers."""

from __future__ import annotations

import numpy as np
import pytest

from glassbox.constants import CLASSIFICATION_METRICS
from glassbox.evaluation.metrics import compute_binary_metrics, early_warning_lead


def test_perfect_separation_scores_all_classification_metrics() -> None:
    scores = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert tuple(scores) == CLASSIFICATION_METRICS
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0
    assert scores["auroc"] == 1.0
    assert scores["auprc"] == 1.0
    assert scores["false_positive_rate"] == 0.0


def test_false_positive_rate_uses_negative_labels() -> None:
    scores = compute_binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.9, 0.1, 0.8, 0.2]),
    )
    assert scores["false_positive_rate"] == 0.5
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5


def test_metric_inputs_are_checked() -> None:
    with pytest.raises(ValueError, match="both class"):
        compute_binary_metrics([1, 1], [0.2, 0.8])
    with pytest.raises(ValueError, match="same length"):
        compute_binary_metrics([0, 1], [0.2])
    with pytest.raises(ValueError, match="probabilities"):
        compute_binary_metrics([0, 1], [0.2, float("nan")])
    with pytest.raises(ValueError, match="probabilities"):
        compute_binary_metrics([0, 1], [0.2, 1.4])
    with pytest.raises(ValueError, match="integer"):
        compute_binary_metrics([False, True], [0.1, 0.9])


def test_early_warning_lead_counts_only_timely_flags() -> None:
    assert early_warning_lead([False, False, True, True], 3) == 1
    assert early_warning_lead([True, False], 1) == 1
    assert early_warning_lead([False, False], 1) is None
    assert early_warning_lead([False, True], 0) is None
    with pytest.raises(ValueError, match="outside"):
        early_warning_lead([False, True], -1)
    with pytest.raises(ValueError, match="outside"):
        early_warning_lead([False], 1)
    with pytest.raises(ValueError, match="booleans"):
        early_warning_lead([False, 1], 1)  # type: ignore[list-item]
