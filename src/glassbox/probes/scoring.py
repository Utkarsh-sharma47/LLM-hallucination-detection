"""Operating points, calibration, and baselines for one probe run.

Thresholds and the Platt map are fit without the test split. A constant
baseline does not get an invented AUROC.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from glassbox.evaluation.metrics import compute_binary_metrics

# Chosen before the HaluEval run. Do not retune these after seeing test scores.
CALIBRATION_ECE_LIMIT = 0.10
CALIBRATION_MIN_PER_CLASS = 15
ECE_BINS = 10


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Pick the validation threshold with the highest F1, then the lower FPR."""
    best_threshold = 0.5
    best_key = (-1.0, -1.0)
    for step in range(1, 20):
        threshold = step / 20
        metrics = compute_binary_metrics(labels, scores, threshold=threshold)
        key = (metrics["f1"], -metrics["false_positive_rate"])
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def confusion_counts(labels: np.ndarray, scores: np.ndarray, *, threshold: float) -> dict[str, int]:
    """Return tn, fp, fn, tp at a frozen threshold."""
    predicted = (np.asarray(scores) >= threshold).astype(int)
    truth = np.asarray(labels)
    return {
        "tn": int(np.sum((truth == 0) & (predicted == 0))),
        "fp": int(np.sum((truth == 0) & (predicted == 1))),
        "fn": int(np.sum((truth == 1) & (predicted == 0))),
        "tp": int(np.sum((truth == 1) & (predicted == 1))),
    }


def expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, *, bins: int = ECE_BINS
) -> float:
    """Weighted absolute gap between confidence and accuracy across bins."""
    truth = np.asarray(labels)
    values = np.asarray(scores, dtype=np.float64)
    if truth.shape != values.shape or truth.size == 0:
        raise ValueError("labels and scores must be non-empty and aligned")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(truth.size)
    error = 0.0
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if index == bins - 1:
            mask = (values >= start) & (values <= end)
        else:
            mask = (values >= start) & (values < end)
        count = int(np.sum(mask))
        if count == 0:
            continue
        confidence = float(values[mask].mean())
        accuracy = float(truth[mask].mean())
        error += (count / total) * abs(accuracy - confidence)
    return float(error)


def fit_platt(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float] | None:
    """Fit a one-dimensional logistic map on validation scores."""
    if set(np.asarray(labels).tolist()) != {0, 1}:
        return None
    model = LogisticRegression(random_state=seed, max_iter=1000)
    model.fit(np.asarray(scores, dtype=np.float64).reshape(-1, 1), np.asarray(labels))
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def apply_platt(scores: np.ndarray, coef: float, intercept: float) -> np.ndarray:
    logits = coef * np.asarray(scores, dtype=np.float64) + intercept
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def class_counts(labels: np.ndarray) -> dict[str, int]:
    values = np.asarray(labels)
    return {"faithful": int(np.sum(values == 0)), "hallucinated": int(np.sum(values == 1))}


def majority_baseline(train_labels: np.ndarray, test_labels: np.ndarray) -> dict[str, object]:
    """Hard predictions of the training majority class. AUROC is not defined."""
    train = np.asarray(train_labels)
    test = np.asarray(test_labels)
    faithful = int(np.sum(train == 0))
    hallucinated = int(np.sum(train == 1))
    predicted_label = 1 if hallucinated >= faithful else 0
    predicted = np.full(test.shape, predicted_label, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test,
        predicted,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    negatives = int(np.sum(test == 0))
    false_positives = int(np.sum((test == 0) & (predicted == 1)))
    fpr = None if negatives == 0 else float(false_positives / negatives)
    return {
        "predicted_class": int(predicted_label),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": fpr,
        "auroc": None,
        "auprc": None,
        "note": "A constant prediction does not define AUROC or AUPRC. Those values were not filled in.",
    }


def overlap_score(answer: str, reference: str) -> float:
    """High when the answer shares few tokens with the reference answer.

    Under the pair protocol the faithful text is the reference, so this baseline
    can recover the label from the strings. It is not a detector of new generations.
    """
    left = set(_tokens(answer))
    right = set(_tokens(reference))
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    jaccard = len(left & right) / len(union)
    return float(1.0 - jaccard)


def _tokens(text: str) -> list[str]:
    return [
        part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if part
    ]
