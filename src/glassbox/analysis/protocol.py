"""Decide whether a saved probe may be described as a generation detector.

A pairwise HaluEval probe stays a pairwise experiment. An uncalibrated score
is never renamed into a hallucination probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from glassbox.probes.store import LoadedProbe

PAIR_PROTOCOL = "halueval_qa_pair_v1"
GENERATION_PROTOCOL = "halueval_generation_evidence_v1"
PROBABILITY_UNAVAILABLE = "NOT AVAILABLE"
GENERATION_UNAVAILABLE = "GENERATION PROBE UNAVAILABLE"

_GENERATION_FIELDS = (
    "label_protocol",
    "model_name",
    "model_revision",
    "layer_index",
    "position",
    "feature_dim",
    "threshold",
    "calibration_validated",
)


@dataclass(frozen=True)
class ProbeAccount:
    """How one artifact may be shown. Missing fields stay explicit."""

    role: str
    protocol: str
    compatible_generation_detector: bool
    calibration_validated: bool
    probability_status: str
    generation_status: str
    missing_fields: tuple[str, ...]
    threshold: float | None
    note: str


def account_probe(probe: LoadedProbe | None) -> ProbeAccount:
    """Classify a probe. A pair protocol cannot fill in as a generation probe."""
    if probe is None:
        return ProbeAccount(
            role="missing",
            protocol="",
            compatible_generation_detector=False,
            calibration_validated=False,
            probability_status=PROBABILITY_UNAVAILABLE,
            generation_status=GENERATION_UNAVAILABLE,
            missing_fields=_GENERATION_FIELDS,
            threshold=None,
            note="No generation-specific probe is loaded.",
        )
    metadata = probe.metadata
    protocol = str(metadata.get("label_protocol") or "")
    missing = tuple(name for name in _GENERATION_FIELDS if name not in metadata)
    threshold = _threshold(metadata.get("threshold"))
    calibrated = metadata.get("calibration_validated") is True and probe.calibrated
    if protocol == PAIR_PROTOCOL:
        return ProbeAccount(
            role="pairwise_experimental",
            protocol=protocol,
            compatible_generation_detector=False,
            calibration_validated=False,
            probability_status=PROBABILITY_UNAVAILABLE,
            generation_status=GENERATION_UNAVAILABLE,
            missing_fields=missing,
            threshold=threshold,
            note=(
                "Pairwise experimental probe trained to rank supplied HaluEval answers. "
                "It is not a generation-specific factuality predictor."
            ),
        )
    if (
        protocol != GENERATION_PROTOCOL
        or missing
        or metadata.get("position") != "last_generated_token"
    ):
        return ProbeAccount(
            role="incompatible",
            protocol=protocol,
            compatible_generation_detector=False,
            calibration_validated=False,
            probability_status=PROBABILITY_UNAVAILABLE,
            generation_status=GENERATION_UNAVAILABLE,
            missing_fields=missing,
            threshold=threshold,
            note="The artifact does not meet the generation-probe contract.",
        )
    if not calibrated:
        return ProbeAccount(
            role="generation_experimental",
            protocol=protocol,
            compatible_generation_detector=False,
            calibration_validated=False,
            probability_status=PROBABILITY_UNAVAILABLE,
            generation_status="GENERATION PROBE EXPERIMENTAL",
            missing_fields=(),
            threshold=threshold,
            note=(
                "Generation-protocol probe. Calibration was not validated, so the "
                "logistic output stays a raw score."
            ),
        )
    return ProbeAccount(
        role="generation_calibrated",
        protocol=protocol,
        compatible_generation_detector=True,
        calibration_validated=True,
        probability_status="calibrated",
        generation_status="GENERATION PROBE CALIBRATED",
        missing_fields=(),
        threshold=threshold,
        note="Calibrated generation probe. The held-out record must still be cited.",
    )


def probability_lines(account: ProbeAccount, metadata: dict[str, Any] | None) -> dict[str, object]:
    """Fill probability fields only for a validated generation probe."""
    if account.probability_status != "calibrated" or metadata is None:
        return {
            "hallucination_probability": PROBABILITY_UNAVAILABLE,
            "factuality_probability": PROBABILITY_UNAVAILABLE,
            "calibration_status": "not validated",
            "calibration_method": None,
            "ece": None,
            "brier": None,
            "threshold": account.threshold,
            "validation_rows": None,
        }
    ece = metadata.get("test_ece")
    return {
        "hallucination_probability": "see calibrated score",
        "factuality_probability": "see calibrated score",
        "calibration_status": "validated",
        "calibration_method": metadata.get("calibration"),
        "ece": ece if isinstance(ece, (int, float)) else None,
        "brier": metadata.get("brier"),
        "threshold": account.threshold,
        "validation_rows": metadata.get("calibration_rows"),
    }


def _threshold(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
