"""Save and load a probe with the metadata required to apply it honestly."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from glassbox.probes.logistic import LogisticProbe
from glassbox.probes.preprocess import apply_standard_scaler
from glassbox.probes.scoring import apply_platt

PROBE_FILENAME = "probe.joblib"
METADATA_FILENAME = "metadata.json"


class ProbeCompatibilityError(ValueError):
    """A saved probe does not match the model representation being scored."""


@dataclass
class LoadedProbe:
    """A classifier plus the preprocessing and evaluation record."""

    probe: LogisticProbe
    metadata: dict[str, Any]
    directory: Path

    @property
    def calibrated(self) -> bool:
        """True only when a calibrator was fit and the held-out check passed.

        A method name alone is not enough. ``calibration_validated`` has to be
        the boolean true written by the evaluation, not a string.
        """
        method = str(self.metadata.get("calibration", "none"))
        return method not in {"", "none"} and self.metadata.get("calibration_validated") is True


def save_probe(
    directory: Path,
    probe: LogisticProbe,
    *,
    metadata: dict[str, Any],
    mean: np.ndarray,
    scale: np.ndarray,
) -> Path:
    """Write the classifier and metadata under ``directory``."""
    if not probe.trained:
        raise ValueError("Cannot save a probe that has not been fit")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": probe.classifier,
        "mean": np.asarray(mean, dtype=np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
        "feature_dim": probe.feature_dim,
    }
    joblib.dump(payload, destination / PROBE_FILENAME)
    record = dict(metadata)
    record["preprocessing"] = {
        "type": "standard_scaler",
        "mean": payload["mean"].tolist(),
        "scale": payload["scale"].tolist(),
    }
    record["feature_dim"] = probe.feature_dim
    (destination / METADATA_FILENAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_probe(directory: Path) -> LoadedProbe:
    """Load a probe directory created by ``save_probe``."""
    folder = Path(directory)
    model_path = folder / PROBE_FILENAME
    metadata_path = folder / METADATA_FILENAME
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Probe directory is incomplete: {folder}")
    payload = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_dim = int(payload["feature_dim"])
    if int(metadata.get("feature_dim", -1)) != feature_dim:
        raise ProbeCompatibilityError("metadata feature_dim does not match the saved classifier")
    probe = LogisticProbe()
    probe.attach(payload["classifier"], feature_dim)
    return LoadedProbe(probe=probe, metadata=metadata, directory=folder)


def check_compatibility(
    loaded: LoadedProbe,
    *,
    model_name: str,
    model_revision: str,
    layer_index: int,
    hidden_state_count: int,
    position: str,
    prompt_mode: str,
    feature_dim: int,
    dtype: str | None = None,
) -> None:
    """Raise when the live representation does not match the training record."""
    metadata = loaded.metadata
    expected = {
        "model_name": model_name,
        "model_revision": model_revision,
        "layer_index": layer_index,
        "hidden_state_count": hidden_state_count,
        "position": position,
        "prompt_mode": prompt_mode,
        "feature_dim": feature_dim,
    }
    mismatches: list[str] = []
    for key, value in expected.items():
        if metadata.get(key) != value:
            mismatches.append(f"{key}: saved {metadata.get(key)!r} != live {value!r}")
    saved_dtype = metadata.get("dtype")
    if dtype is not None and saved_dtype is not None and saved_dtype != dtype:
        mismatches.append(f"dtype: saved {saved_dtype!r} != live {dtype!r}")
    preprocessing = metadata.get("preprocessing")
    if not isinstance(preprocessing, dict) or preprocessing.get("type") != "standard_scaler":
        mismatches.append("preprocessing is missing a standard_scaler record")
    else:
        mean = preprocessing.get("mean")
        scale = preprocessing.get("scale")
        if not isinstance(mean, list) or not isinstance(scale, list):
            mismatches.append("preprocessing mean and scale must be lists")
        elif len(mean) != feature_dim or len(scale) != feature_dim:
            mismatches.append("preprocessing length does not match the feature dimension")
    if mismatches:
        raise ProbeCompatibilityError("; ".join(mismatches))


def experimental_score(loaded: LoadedProbe, feature: np.ndarray) -> float:
    """Apply the saved scaler and return the uncalibrated positive-class score."""
    preprocessing = loaded.metadata["preprocessing"]
    mean = np.asarray(preprocessing["mean"], dtype=np.float64)
    scale = np.asarray(preprocessing["scale"], dtype=np.float64)
    vector = np.asarray(feature, dtype=np.float64)
    if vector.shape != mean.shape:
        raise ProbeCompatibilityError(
            f"feature dimension {vector.shape} does not match scaler {mean.shape}"
        )
    transformed = apply_standard_scaler(vector, mean, scale)
    raw = float(loaded.probe.predict_proba(transformed)[0])
    if not loaded.calibrated:
        return raw
    platt = loaded.metadata.get("platt")
    if not isinstance(platt, dict) or "coef" not in platt or "intercept" not in platt:
        raise ProbeCompatibilityError("calibrated probe is missing Platt coefficients")
    calibrated = apply_platt(np.asarray([raw]), float(platt["coef"]), float(platt["intercept"]))
    return float(calibrated[0])
