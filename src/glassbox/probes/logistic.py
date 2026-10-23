"""Logistic regression on frozen hidden states.

``predict_proba`` is the classifier's positive-class output. It is not a
calibrated probability of hallucination unless a saved artifact says so.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from glassbox.config import ProbeConfig

PROBE_NOT_TRAINED = "PROBE NOT TRAINED"


class ProbeNotTrainedError(RuntimeError):
    """A score was requested from a probe that has not been fit."""

    def __str__(self) -> str:
        return PROBE_NOT_TRAINED


def sklearn_class_weight(setting: str) -> str | None:
    """Map the config string to the scikit-learn ``class_weight`` argument.

    ``"none"`` becomes ``None``. This mapping is an untested default, not a
    result of class-balance tuning.
    """
    if setting == "balanced":
        return "balanced"
    if setting == "none":
        return None
    raise ValueError("class_weight setting must be 'balanced' or 'none'")


class LogisticProbe:
    """Layer-wise logistic regression on frozen hidden states.

    The positive class is ``1`` (hallucinated under the label protocol that
    produced the training labels).
    """

    def __init__(self, config: ProbeConfig | None = None) -> None:
        self.config = config or ProbeConfig()
        if self.config.kind != "logistic_regression":
            raise ValueError("LogisticProbe requires probe.kind='logistic_regression'")
        self._classifier: LogisticRegression | None = None
        self.feature_dim: int | None = None

    @property
    def classifier(self) -> LogisticRegression | None:
        """The fitted scikit-learn classifier, if ``fit`` has run."""
        return self._classifier

    @property
    def trained(self) -> bool:
        """Whether ``fit`` has produced a classifier."""
        return self._classifier is not None

    def fit(self, features: Any, labels: Any) -> LogisticProbe:
        """Fit logistic regression. Both classes must be present."""
        if features is None or labels is None:
            raise ValueError("features and labels are required")
        matrix = np.asarray(features, dtype=np.float64)
        targets = np.asarray(labels)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("features must be a non-empty 2-D array")
        if targets.shape != (matrix.shape[0],):
            raise ValueError("labels must have one value per feature row")
        if not np.isfinite(matrix).all():
            raise ValueError("features must be finite")
        if targets.dtype == np.bool_ or not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("labels must be integer 0/1 values")
        if not np.isin(targets, [0, 1]).all() or set(targets.tolist()) != {0, 1}:
            raise ValueError("labels must contain both class 0 and class 1")
        classifier = LogisticRegression(
            class_weight=sklearn_class_weight(self.config.class_weight),
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        classifier.fit(matrix, targets)
        self._classifier = classifier
        self.feature_dim = int(matrix.shape[1])
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        """Return the positive-class output for each row.

        This is the logistic regression score, not a calibrated hallucination
        probability.
        """
        if self._classifier is None or self.feature_dim is None:
            raise ProbeNotTrainedError()
        if features is None:
            raise ValueError("features are required")
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have dimension {self.feature_dim}, got shape {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("features must be finite")
        probabilities = self._classifier.predict_proba(matrix)
        classes = [int(label) for label in self._classifier.classes_]
        positive = classes.index(1)
        return np.asarray(probabilities[:, positive], dtype=np.float64)

    def attach(self, classifier: LogisticRegression, feature_dim: int) -> LogisticProbe:
        """Attach a classifier that was loaded from disk."""
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self._classifier = classifier
        self.feature_dim = int(feature_dim)
        return self


def build_probe(config: ProbeConfig) -> LogisticProbe:
    """Construct the probe named by ``config.kind``."""
    if config.kind != "logistic_regression":
        raise ValueError(f"Unsupported probe kind: {config.kind}")
    return LogisticProbe(config)
