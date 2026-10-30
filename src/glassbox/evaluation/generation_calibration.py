"""Pre-registered generation-level calibration gate.

This is issue #18. It decides whether a pair-probe score on Qwen generations
may be described as a calibrated factual-risk probability. It does not train,
does not choose a threshold, and does not rewrite the live pair probe.

The operating threshold stays the frozen HaluEval value 0.65. The ECE limit
stays 0.10. The five-label TruthfulQA pilot is ineligible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from glassbox.data.schema import Label
from glassbox.evaluation.metrics import BOOTSTRAP_MIN_PER_CLASS
from glassbox.probes.scoring import CALIBRATION_ECE_LIMIT, expected_calibration_error

# Frozen on halueval-pair-002. Do not replace after seeing generation scores.
PAIR_THRESHOLD = 0.65
GENERATION_CALIBRATION_MIN_PER_CLASS = BOOTSTRAP_MIN_PER_CLASS
GENERATION_CALIBRATION_ECE_LIMIT = CALIBRATION_ECE_LIMIT
PILOT_FAITHFUL = 3
PILOT_HALLUCINATED = 2
SCORE_NAME = "EXPERIMENTAL PROBE SCORE"
RISK_STATUS = "RISK NOT ASSESSED"
GATE_NOTE = (
    "Pre-registered generation calibration gate. "
    f"Threshold stays {PAIR_THRESHOLD}. "
    f"Each of faithful and hallucinated needs at least {GENERATION_CALIBRATION_MIN_PER_CLASS} "
    "validation rows whose labels were assigned without the probe score. "
    f"Validation ECE must be at most {GENERATION_CALIBRATION_ECE_LIMIT}. "
    "The test split is not used to choose the threshold or the ECE limit. "
    "UNVALIDATED and AMBIGUOUS stay out of the ECE denominator. "
    "A passing gate is the only path that may describe the score as a calibrated probability. "
    "Pair-protocol AUROC is not evidence for that claim."
)


def evaluate_generation_calibration_gate(
    rows: Sequence[Mapping[str, object]],
    *,
    threshold: float = PAIR_THRESHOLD,
    pair_probe_metadata: Mapping[str, object] | None = None,
    pair_probe_path: Path | None = None,
    is_five_label_pilot: bool = False,
) -> dict[str, object]:
    """Return the gate decision. Never writes probe metadata."""
    if is_five_label_pilot:
        return _rejected(
            "The five-label TruthfulQA pilot is ineligible for this gate.",
            counts=_counts(rows),
            eligible=(),
            pair_probe_path=pair_probe_path,
            pair_probe_metadata=pair_probe_metadata,
        )
    if not math_is_threshold(threshold):
        return _rejected(
            f"Refusing a generation calibration gate at threshold {threshold}. "
            f"The frozen HaluEval threshold is {PAIR_THRESHOLD}.",
            counts=_counts(rows),
            eligible=(),
            pair_probe_path=pair_probe_path,
            pair_probe_metadata=pair_probe_metadata,
        )
    validation_rows = [row for row in rows if row.get("split") == "validation"]
    if not validation_rows:
        return _rejected(
            "The generation calibration gate requires a dedicated validation split. "
            "Held-out, live-suite, or unlabeled rows are not used to accept calibration.",
            counts=_counts(rows),
            eligible=(),
            pair_probe_path=pair_probe_path,
            pair_probe_metadata=pair_probe_metadata,
        )
    scored_rows = validation_rows
    eligible = [
        row
        for row in scored_rows
        if row.get("label") in {Label.FAITHFUL.name, Label.HALLUCINATED.name}
        and row.get("probe_score") is not None
        and not row.get("split_overlap")
    ]
    counts = _counts(scored_rows)
    counts["eligible_faithful"] = sum(row["label"] == Label.FAITHFUL.name for row in eligible)
    counts["eligible_hallucinated"] = sum(
        row["label"] == Label.HALLUCINATED.name for row in eligible
    )
    smaller = min(counts["eligible_faithful"], counts["eligible_hallucinated"])
    if smaller < GENERATION_CALIBRATION_MIN_PER_CLASS:
        return _rejected(
            (
                "Calibration was not validated. The smaller eligible class has "
                f"{smaller} rows, below {GENERATION_CALIBRATION_MIN_PER_CLASS}."
            ),
            counts=counts,
            eligible=eligible,
            pair_probe_path=pair_probe_path,
            pair_probe_metadata=pair_probe_metadata,
        )
    labels = np.asarray(
        [1 if row["label"] == Label.HALLUCINATED.name else 0 for row in eligible], dtype=int
    )
    scores = np.asarray([float(row["probe_score"]) for row in eligible], dtype=np.float64)
    ece = expected_calibration_error(labels, scores)
    if ece > GENERATION_CALIBRATION_ECE_LIMIT:
        return _rejected(
            (
                f"Calibration was not validated. Validation ECE {ece:.3f} is above "
                f"{GENERATION_CALIBRATION_ECE_LIMIT}."
            ),
            counts=counts,
            eligible=eligible,
            pair_probe_path=pair_probe_path,
            pair_probe_metadata=pair_probe_metadata,
            ece=ece,
        )
    return {
        "status": "accepted",
        "calibration_validated": True,
        "factual_risk_available": True,
        "score_name": "calibrated probability of the hallucinated class",
        "risk_status": "calibrated generation probability",
        "threshold": PAIR_THRESHOLD,
        "ece": float(ece),
        "ece_limit": GENERATION_CALIBRATION_ECE_LIMIT,
        "min_per_class": GENERATION_CALIBRATION_MIN_PER_CLASS,
        "sample_counts": counts,
        "pair_probe_rewritten": False,
        "pair_probe_path": None if pair_probe_path is None else str(pair_probe_path),
        "pair_probe_calibration_validated": _pair_flag(pair_probe_metadata),
        "note": GATE_NOTE,
    }


def math_is_threshold(threshold: object) -> bool:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return False
    return abs(float(threshold) - PAIR_THRESHOLD) <= 1e-12


def _rejected(
    reason: str,
    *,
    counts: dict[str, int],
    eligible: Sequence[Mapping[str, object]],
    pair_probe_path: Path | None,
    pair_probe_metadata: Mapping[str, object] | None,
    ece: float | None = None,
) -> dict[str, object]:
    return {
        "status": "rejected",
        "calibration_validated": False,
        "factual_risk_available": False,
        "score_name": SCORE_NAME,
        "risk_status": RISK_STATUS,
        "threshold": PAIR_THRESHOLD,
        "ece": ece,
        "ece_limit": GENERATION_CALIBRATION_ECE_LIMIT,
        "min_per_class": GENERATION_CALIBRATION_MIN_PER_CLASS,
        "sample_counts": counts,
        "eligible_n": len(eligible),
        "pair_probe_rewritten": False,
        "pair_probe_path": None if pair_probe_path is None else str(pair_probe_path),
        "pair_probe_calibration_validated": _pair_flag(pair_probe_metadata),
        "reason": reason,
        "note": GATE_NOTE,
    }


def _counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {
        "generated": len(rows),
        "faithful": sum(row.get("label") == Label.FAITHFUL.name for row in rows),
        "hallucinated": sum(row.get("label") == Label.HALLUCINATED.name for row in rows),
        "ambiguous": sum(row.get("label") == Label.AMBIGUOUS.name for row in rows),
        "unvalidated": sum(row.get("label") == Label.UNVALIDATED.name for row in rows),
    }


def _pair_flag(metadata: Mapping[str, object] | None) -> bool | None:
    if metadata is None:
        return None
    flag = metadata.get("calibration_validated")
    return flag if isinstance(flag, bool) else None
