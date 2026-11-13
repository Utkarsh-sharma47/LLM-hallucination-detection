"""Score generated prefixes when features and evidence labels are available.

Lead time is emitted only when each scored position has a stored evidence label
from the registered protocol. This is not a calibrated factuality risk.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from glassbox.config import repo_root
from glassbox.evaluation.generation_calibration import PAIR_THRESHOLD
from glassbox.evaluation.metrics import early_warning_lead
from glassbox.probes.generation_eval import PROTECTED_PROBE_NAMES, generation_experiment_dir

EVENT_PROTOCOL_ID = "generation_prefix_evidence_v1"
# Measurement is supported, but remains null on individual records without labels.
EVENT_MEASUREMENT_ENABLED = True
EVENT_DEFINITION = (
    "The hallucination event index is the smallest generated-piece index i such that "
    "the answer prefix through piece i is labeled HALLUCINATED by the generation "
    "evidence protocol (halueval_generation_evidence_v1 or truthfulqa_generation_v1) "
    "or by a documented human review that cites passage evidence. Prefixes that remain "
    "UNVALIDATED or AMBIGUOUS are not events. The empty prefix is not an event. "
    "early_warning_lead uses the existing helper and stays null when this event is not "
    "stored on the run, when no prefix is an event, or when no probe flag occurs at or "
    "before the event. Thresholds are chosen on validation only. This measurement is "
    "not a calibrated risk, is not an activation edit, and is not evidence for "
    "pair-protocol AUROC."
)


def event_index_from_prefix_labels(labels: Sequence[str]) -> int | None:
    """Return the first HALLUCINATED prefix, or None.

    UNVALIDATED and AMBIGUOUS prefixes are not events.
    """
    for index, name in enumerate(labels):
        if name == "HALLUCINATED":
            return index
    return None


def score_generated_positions(
    piece_vectors: np.ndarray,
    scorer: Callable[[np.ndarray], float],
    *,
    event_index: int | None = None,
    threshold: float = PAIR_THRESHOLD,
    prefix_labels: Sequence[str] | None = None,
) -> dict[str, object]:
    """Score every generated piece of one sequence. Does not load Qwen.

    ``piece_vectors`` has shape [generated_pieces, dim]. Two or more positions
    are required so earlier pieces can be compared with the final piece.
    """
    matrix = np.asarray(piece_vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("piece_vectors must contain at least two generated pieces")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite probability in [0, 1]")
    if prefix_labels is not None and len(prefix_labels) != matrix.shape[0]:
        raise ValueError("prefix_labels must have one evidence label per generated piece")
    derived_event = (
        event_index_from_prefix_labels(prefix_labels) if prefix_labels is not None else None
    )
    if event_index is not None and event_index != derived_event:
        raise ValueError("event_index must match the first HALLUCINATED prefix label")
    scores = [float(scorer(matrix[index])) for index in range(matrix.shape[0])]
    if any(not np.isfinite(score) or score < 0.0 or score > 1.0 for score in scores):
        raise ValueError("position scores must be finite probabilities in [0, 1]")
    flags = [score >= threshold for score in scores]
    stored_event = derived_event if prefix_labels is not None else None
    lead = (
        early_warning_lead(flags, stored_event)
        if prefix_labels is not None and stored_event is not None
        else None
    )
    return {
        "protocol": EVENT_PROTOCOL_ID,
        "event_definition": EVENT_DEFINITION,
        "event_measurement_enabled": EVENT_MEASUREMENT_ENABLED and prefix_labels is not None,
        "event_index": stored_event,
        "prefix_labels": list(prefix_labels) if prefix_labels is not None else None,
        "position_scores": scores,
        "flags": flags,
        "threshold": threshold,
        "early_warning_lead": lead,
        "n_positions": int(matrix.shape[0]),
        "note": "Lead time requires one stored evidence label per scored position and a validation-selected prefix-probe threshold. This is not a calibrated factual-risk score.",
    }


def early_warning_command(args: object) -> int:
    """Apply the saved prefix probe to stored prefix features and evidence labels."""
    import json
    import sys

    import joblib

    from glassbox.evaluation.records import write_json

    try:
        run_id = str(args.run_id)
        if run_id in PROTECTED_PROBE_NAMES or run_id.startswith("final-hidden-state"):
            raise ValueError(f"Refusing protected run id {run_id}")
        if run_id == "halueval-generation-001":
            raise ValueError("Refusing to overwrite the historical generation run")
        artifacts = repo_root() / "artifacts"
        directory = generation_experiment_dir(run_id)
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite {directory}")
        payload = json.loads(Path(args.scores).read_text(encoding="utf-8"))
        vectors = np.asarray(payload["piece_vectors"], dtype=np.float64)
        labels = payload.get("prefix_labels")
        valid_labels = {"FAITHFUL", "HALLUCINATED", "UNVALIDATED", "AMBIGUOUS"}
        if not isinstance(labels, list) or not all(
            isinstance(label, str) and label in valid_labels for label in labels
        ):
            raise ValueError(
                "scores JSON must include prefix_labels from the registered evidence protocol"
            )
        probe_directory = Path(
            getattr(args, "probe", None) or (artifacts / "probes" / "early-warning-task7-ai-v1-001")
        )
        bundle_path = (
            probe_directory
            if probe_directory.is_file()
            else probe_directory / "prefix_probe.joblib"
        )
        bundle = joblib.load(bundle_path)
        if not isinstance(bundle, dict) or "probe" not in bundle or "threshold" not in bundle:
            raise ValueError(f"Invalid prefix-probe bundle: {bundle_path}")
        probe = bundle["probe"]
        threshold = float(bundle["threshold"])
        if vectors.ndim != 2 or vectors.shape[1] != int(getattr(probe, "n_features_in_", -1)):
            raise ValueError("piece_vectors do not match the saved prefix probe feature dimension")

        result = score_generated_positions(
            vectors,
            lambda vector: float(probe.predict_proba(vector.reshape(1, -1))[0, 1]),
            threshold=threshold,
            prefix_labels=labels,
        )
        result["probe_applied"] = True
        result["score_source"] = str(bundle_path)
        result["threshold_selected_on"] = "validation prefix labels only"
        result["question_id"] = payload.get("question_id")
        result["probe_layer"] = bundle.get("layer_index")
        write_json(directory / "report.json", result, root=artifacts)
        np.savez_compressed(
            directory / "prefix_features.npz",
            piece_vectors=vectors.astype(np.float32),
            prefix_labels=np.asarray(labels),
        )
        print(f"Early-warning lead: {result['early_warning_lead']}")
        print(f"Wrote {directory / 'report.json'}")
        return 0
    except (OSError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Early-warning evaluation stopped: {exc}", file=sys.stderr)
        return 1


def record_without_event(metrics: dict[str, object] | None = None) -> dict[str, object]:
    """Fields a run record must keep when the event is not stored."""
    payload = {
        "early_warning_lead": None,
        "event_definition": EVENT_DEFINITION,
        "event_measurement_enabled": False,
        "event_protocol": EVENT_PROTOCOL_ID,
    }
    if metrics is not None:
        payload["metrics_early_warning_lead"] = metrics.get("early_warning_lead")
    return payload
