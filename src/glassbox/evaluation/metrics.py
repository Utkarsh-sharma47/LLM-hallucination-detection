"""Scoring helpers for binary hallucination probes.

``compute_binary_metrics`` scores a fixed prediction vector. The default
threshold of 0.5 is for tests and debugging. The evaluation protocol chooses
thresholds on a validation split and freezes them before the test split is read.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from glassbox.constants import CLASSIFICATION_METRICS

# Smaller labeled classes do not receive a confidence interval. The generation
# dataset records this same minimum and does not lower it.
BOOTSTRAP_MIN_PER_CLASS = 20


def compute_binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Score binary labels against probabilities in ``[0, 1]``.

    The positive class is ``1``. Both classes must be present, so AUROC and
    AUPRC are defined. Scores are probabilities, which makes the threshold and
    the false-positive rate interpretable.
    """
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    decision = _as_unit_interval(threshold, "threshold")
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("y_true and y_score must be 1-D arrays of the same length")
    if labels.size == 0:
        raise ValueError("y_true must be non-empty")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("y_true must be integer 0/1 labels")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true values must be 0 or 1")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("y_true must contain both class 0 and class 1")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("y_score values must be finite probabilities in [0, 1]")

    predicted = (scores >= decision).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predicted,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    true_negative, false_positive, _, _ = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    negatives = true_negative + false_positive
    if negatives == 0:
        raise ValueError("false_positive_rate requires at least one negative label")
    values = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "false_positive_rate": float(false_positive / negatives),
    }
    return {name: values[name] for name in CLASSIFICATION_METRICS}


def early_warning_lead(flags: Sequence[bool], event_index: int) -> int | None:
    """Return how many steps before ``event_index`` the detector first fired.

    ``flags[t]`` is whether the detector fired at generation step ``t``.
    ``event_index`` is the first step counted as the hallucination event.
    The lead is ``event_index - first_flag_index`` when that flag is at or
    before the event. Zero means the first timely flag lands on the event step.
    The result is ``None`` when no flag occurs at or before the event.

    This is bookkeeping for a label definition that the project has not settled.
    It does not decide whether a flag is a true positive.
    """
    if isinstance(event_index, bool) or not isinstance(event_index, int):
        raise ValueError("event_index must be an integer")
    if event_index < 0 or event_index >= len(flags):
        raise ValueError("event_index is outside the flag sequence")
    for flagged in flags:
        if not isinstance(flagged, bool):
            raise ValueError("flags must be booleans")
    for index, flagged in enumerate(flags[: event_index + 1]):
        if flagged:
            return event_index - index
    return None


def bootstrap_interval(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    metric: str = "auroc",
    threshold: float = 0.5,
    n_resamples: int = 1000,
    seed: int = 0,
    min_per_class: int = BOOTSTRAP_MIN_PER_CLASS,
) -> dict[str, object]:
    """Percentile interval from resampling rows that were already scored.

    The interval is not computed when the smaller class has fewer than
    ``min_per_class`` rows. A tiny labeled set does not get a confidence
    interval that looks like a generalization result.
    """
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if metric not in CLASSIFICATION_METRICS:
        raise ValueError(f"Unknown metric {metric}")
    faithful = int(np.sum(labels == 0))
    hallucinated = int(np.sum(labels == 1))
    if min(faithful, hallucinated) < min_per_class:
        return {
            "metric": metric,
            "point": None,
            "low": None,
            "high": None,
            "n": int(labels.size),
            "note": (
                "Confidence interval was not computed. "
                f"The smaller class has {min(faithful, hallucinated)} rows, below {min_per_class}."
            ),
        }
    point = compute_binary_metrics(labels, scores, threshold=threshold)[metric]
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    count = int(labels.size)
    for _ in range(n_resamples):
        chosen = rng.integers(0, count, count)
        if set(labels[chosen].tolist()) != {0, 1}:
            continue
        try:
            samples.append(
                compute_binary_metrics(labels[chosen], scores[chosen], threshold=threshold)[metric]
            )
        except ValueError:
            continue
    if len(samples) < 100:
        return {
            "metric": metric,
            "point": point,
            "low": None,
            "high": None,
            "n": count,
            "resamples_used": len(samples),
            "note": "Too many resamples lacked both classes, so the interval was not reported.",
        }
    low, high = np.percentile(np.asarray(samples, dtype=np.float64), [2.5, 97.5])
    return {
        "metric": metric,
        "point": point,
        "low": float(low),
        "high": float(high),
        "n": count,
        "resamples_used": len(samples),
        "note": (
            "Percentile interval from resampling these scored rows with replacement. "
            "It does not transfer to another dataset."
        ),
    }


def bootstrap_grouped_interval(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    groups: Sequence[str],
    *,
    metric: str = "auroc",
    threshold: float = 0.5,
    n_resamples: int = 1000,
    seed: int = 0,
    min_per_class: int = 20,
) -> dict[str, object]:
    """Percentile interval from resampling question groups, not answer rows.

    Every row that shares a group id is copied together. An interval is not
    reported when the smaller class has fewer than ``min_per_class`` rows.
    """
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    group_ids = [str(item) for item in groups]
    if len(group_ids) != int(labels.shape[0]):
        raise ValueError("groups must have one id for every scored row")
    faithful = int(np.sum(labels == 0))
    hallucinated = int(np.sum(labels == 1))
    if min(faithful, hallucinated) < min_per_class:
        return {
            "metric": metric,
            "point": None,
            "low": None,
            "high": None,
            "n": int(labels.size),
            "groups": len(set(group_ids)),
            "unit": "question_group",
            "note": (
                "Confidence interval was not computed. "
                f"The smaller class has {min(faithful, hallucinated)} rows, below {min_per_class}."
            ),
        }
    point = compute_binary_metrics(labels, scores, threshold=threshold)[metric]
    members: dict[str, list[int]] = {}
    for index, group in enumerate(group_ids):
        members.setdefault(group, []).append(index)
    unique = list(members)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        drawn = rng.integers(0, len(unique), len(unique))
        chosen: list[int] = []
        for slot in drawn:
            chosen.extend(members[unique[int(slot)]])
        picked = np.asarray(chosen, dtype=int)
        if set(labels[picked].tolist()) != {0, 1}:
            continue
        try:
            samples.append(
                compute_binary_metrics(labels[picked], scores[picked], threshold=threshold)[metric]
            )
        except ValueError:
            continue
    if len(samples) < 100:
        return {
            "metric": metric,
            "point": point,
            "low": None,
            "high": None,
            "n": int(labels.size),
            "groups": len(unique),
            "resamples_used": len(samples),
            "unit": "question_group",
            "note": "Too many resamples lacked both classes, so the interval was not reported.",
        }
    low, high = np.percentile(np.asarray(samples, dtype=np.float64), [2.5, 97.5])
    return {
        "metric": metric,
        "point": point,
        "low": float(low),
        "high": float(high),
        "n": int(labels.size),
        "groups": len(unique),
        "resamples_used": len(samples),
        "unit": "question_group",
        "note": (
            "Percentile interval from resampling question groups with replacement. "
            "Rows from the same group stay together. It does not transfer to another dataset."
        ),
    }


def _as_unit_interval(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number
