"""Fixed-bin calibration metrics and question-group bootstrap intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from glassbox.evaluation.metrics import BOOTSTRAP_MIN_PER_CLASS
from glassbox.probes.scoring import ECE_BINS, expected_calibration_error

CALIBRATION_RESAMPLES = 1000
CALIBRATION_MIN_VALID_RESAMPLES = 100


def brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Mean squared error of binary probabilities."""
    truth, scores = _validated_arrays(labels, probabilities)
    return float(np.mean(np.square(scores - truth)))


def calibration_report(
    labels: Sequence[int],
    probabilities: Sequence[float],
    groups: Sequence[str],
    *,
    bins: int = ECE_BINS,
    n_resamples: int = CALIBRATION_RESAMPLES,
    seed: int = 0,
    min_per_class: int = BOOTSTRAP_MIN_PER_CLASS,
) -> dict[str, object]:
    """Report fixed-bin ECE/Brier and percentile intervals resampled by group."""
    truth, scores = _validated_arrays(labels, probabilities)
    group_ids = [str(group) for group in groups]
    if len(group_ids) != len(truth) or any(not group.strip() for group in group_ids):
        raise ValueError("groups must contain one non-empty id for every scored row")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if isinstance(min_per_class, bool) or not isinstance(min_per_class, int) or min_per_class < 1:
        raise ValueError("min_per_class must be a positive integer")

    point = {
        "brier": brier_score(truth, scores),
        "ece": expected_calibration_error(truth, scores, bins=bins),
    }
    faithful = int(np.sum(truth == 0))
    hallucinated = int(np.sum(truth == 1))
    unique_groups = list(dict.fromkeys(group_ids))
    enough_class_support = min(faithful, hallucinated) >= min_per_class
    report: dict[str, object] = {
        "n": int(truth.size),
        "class_counts": {"faithful": faithful, "hallucinated": hallucinated},
        "groups": len(unique_groups),
        "resampling_unit": "question_group",
        "bins": bins,
        "binning": "equal-width bins on [0, 1]; left-closed, right-open except final bin",
        "brier": point["brier"],
        "ece": point["ece"],
        "interval_method": "95% percentile bootstrap over question groups",
        "resamples_requested": n_resamples,
        "support_status": "sufficient" if enough_class_support else "insufficient",
    }
    if not enough_class_support:
        reason = (
            "Intervals not computed: the smaller class has "
            f"{min(faithful, hallucinated)} rows, below {min_per_class}."
        )
        report["brier_interval"] = _null_interval(reason)
        report["ece_interval"] = _null_interval(reason)
        return report

    members: dict[str, list[int]] = {}
    for index, group in enumerate(group_ids):
        members.setdefault(group, []).append(index)
    if len(members) < min_per_class:
        report["support_status"] = "insufficient"
        reason = (
            f"Intervals not computed: only {len(members)} independent groups, "
            f"below {min_per_class}."
        )
        report["brier_interval"] = _null_interval(reason)
        report["ece_interval"] = _null_interval(reason)
        return report

    rng = np.random.default_rng(seed)
    brier_samples: list[float] = []
    ece_samples: list[float] = []
    for _ in range(n_resamples):
        draw = rng.integers(0, len(unique_groups), len(unique_groups))
        chosen = [index for slot in draw for index in members[unique_groups[int(slot)]]]
        picked = np.asarray(chosen, dtype=int)
        sampled_labels = truth[picked]
        if set(sampled_labels.tolist()) != {0, 1}:
            continue
        sampled_scores = scores[picked]
        brier_samples.append(brier_score(sampled_labels, sampled_scores))
        ece_samples.append(expected_calibration_error(sampled_labels, sampled_scores, bins=bins))

    if len(brier_samples) < CALIBRATION_MIN_VALID_RESAMPLES:
        reason = (
            "Intervals not computed: only "
            f"{len(brier_samples)} valid group resamples; "
            f"at least {CALIBRATION_MIN_VALID_RESAMPLES} are required."
        )
        report["brier_interval"] = _null_interval(reason, len(brier_samples))
        report["ece_interval"] = _null_interval(reason, len(ece_samples))
        return report

    report["resamples_used"] = len(brier_samples)
    report["brier_interval"] = _percentile_interval(brier_samples)
    report["ece_interval"] = _percentile_interval(ece_samples)
    return report


def _validated_arrays(
    labels: Sequence[int], probabilities: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 1 or scores.ndim != 1 or truth.shape != scores.shape or truth.size == 0:
        raise ValueError("labels and probabilities must be non-empty aligned 1-D arrays")
    if not np.issubdtype(truth.dtype, np.integer) or not np.isin(truth, [0, 1]).all():
        raise ValueError("labels must be integer 0/1 values")
    if not np.isfinite(scores).all() or (scores < 0.0).any() or (scores > 1.0).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    return truth.astype(np.int8, copy=False), scores


def _null_interval(reason: str, resamples_used: int = 0) -> dict[str, object]:
    return {
        "low": None,
        "high": None,
        "resamples_used": resamples_used,
        "note": reason,
    }


def _percentile_interval(samples: Sequence[float]) -> dict[str, object]:
    low, high = np.percentile(np.asarray(samples, dtype=np.float64), [2.5, 97.5])
    return {
        "low": float(low),
        "high": float(high),
        "resamples_used": len(samples),
        "note": "Percentile interval from question-group bootstrap resampling.",
    }
