"""Issue #7: score earlier generated pieces without publishing a lead time.

The event definition is recorded here and in docs/ROADMAP.md. No run stores
per-prefix evidence labels yet, so ``early_warning_lead`` stays null.
This measurement is not a calibrated risk and is not evidence for the
pair-protocol AUROC.
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
# Per-prefix labels are not stored on any run. Do not publish a lead time.
EVENT_MEASUREMENT_ENABLED = False
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
) -> dict[str, object]:
    """Score every generated piece of one sequence. Does not load Qwen.

    ``piece_vectors`` has shape [generated_pieces, dim]. Two or more positions
    are required so earlier pieces can be compared with the final piece.
    """
    matrix = np.asarray(piece_vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("piece_vectors must contain at least two generated pieces")
    scores = [float(scorer(matrix[index])) for index in range(matrix.shape[0])]
    if any(not np.isfinite(score) or score < 0.0 or score > 1.0 for score in scores):
        raise ValueError("position scores must be finite probabilities in [0, 1]")
    flags = [score >= threshold for score in scores]
    lead = None
    stored_event = None
    if EVENT_MEASUREMENT_ENABLED and event_index is not None:
        stored_event = event_index
        lead = early_warning_lead(flags, event_index)
    elif event_index is not None and not EVENT_MEASUREMENT_ENABLED:
        # The helper is still exercised so a missing flag stays null, but the
        # published field remains unset until prefix labels exist on a run.
        _ = early_warning_lead(flags, event_index)
    return {
        "protocol": EVENT_PROTOCOL_ID,
        "event_definition": EVENT_DEFINITION,
        "event_measurement_enabled": EVENT_MEASUREMENT_ENABLED,
        "event_index": stored_event,
        "position_scores": scores,
        "flags": flags,
        "threshold": threshold,
        "early_warning_lead": lead,
        "n_positions": int(matrix.shape[0]),
        "note": (
            "early_warning_lead stays null until a run stores per-prefix evidence labels. "
            "This is not a calibrated factual-risk score."
        ),
    }


def early_warning_command(args: object) -> int:
    """Score stored piece vectors. Does not load Qwen or publish a lead time."""
    import json
    import sys

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
        event_index = payload.get("event_index")
        threshold = float(payload.get("threshold", PAIR_THRESHOLD))

        def _score(vector: np.ndarray) -> float:
            values = np.asarray(vector, dtype=np.float64)
            return float(np.clip(values.mean(), 0.0, 1.0))

        result = score_generated_positions(
            vectors,
            _score,
            event_index=None if event_index is None else int(event_index),
            threshold=threshold,
        )
        result["probe_applied"] = False
        result["score_source"] = "clipped_mean_of_stored_vector"
        result["note"] = (
            "position_scores are the clipped mean of each stored vector, not a saved probe. "
            "early_warning_lead stays null until a run stores per-prefix evidence labels. "
            "This is not a calibrated factual-risk score."
        )
        result.update(record_without_event())
        write_json(directory / "report.json", result, root=artifacts)
        print("early_warning_lead stays null; event measurement is not enabled")
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
        "event_measurement_enabled": EVENT_MEASUREMENT_ENABLED,
        "event_protocol": EVENT_PROTOCOL_ID,
    }
    if metrics is not None:
        payload["metrics_early_warning_lead"] = metrics.get("early_warning_lead")
    return payload
