"""One scaler for probe training and for later scoring.

The arithmetic is the population mean and standard deviation, with a zero
scale replaced by 1 so a constant feature does not divide by zero. Training
and inference both call these functions. Do not fit a second copy on the
test split or on a single demo answer.
"""

from __future__ import annotations

import numpy as np


def fit_standard_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit mean and scale on rows that are already known to be training rows."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("scaler fit expects a non-empty 2-D matrix")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    return mean, np.where(scale == 0, 1.0, scale)


def apply_standard_scaler(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Apply a scaler that was fit earlier. This function does not refit."""
    matrix = np.asarray(features, dtype=np.float64)
    center = np.asarray(mean, dtype=np.float64)
    safe_scale = np.where(
        np.asarray(scale, dtype=np.float64) == 0, 1.0, np.asarray(scale, dtype=np.float64)
    )
    return (matrix - center) / safe_scale
