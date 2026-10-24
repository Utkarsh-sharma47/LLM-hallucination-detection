"""Fail-closed record for one generation evaluation.

A finished record names the model, dataset split, and probe, and keeps one
row per generated answer. ``UNVALIDATED`` and ``AMBIGUOUS`` stay out of both
binary classes. Pair-protocol metrics are not copied here, and this writer
does not set ``calibration_validated``.

The writer does not load Qwen and does not read
``artifacts/probes/final-hidden-state/``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

from glassbox.data.generation_dataset import assert_dataset_destination
from glassbox.data.schema import Label
from glassbox.evaluation.records import write_json
from glassbox.reproducibility import require_commit_hash

REQUIRED_PROVENANCE = (
    "model_name",
    "model_revision",
    "dtype",
    "dataset_id",
    "dataset_sha256",
    "split_id",
    "seed",
    "probe_directory",
    "probe_model_revision",
    "layer",
    "position",
    "prompt_mode",
    "calibration_validated",
    "config_path",
    "git_commit",
    "worktree_clean",
)
_BINARY = {Label.FAITHFUL.name, Label.HALLUCINATED.name}
_LABELS = {label.name for label in Label}
_METRIC_KEYS = frozenset(
    {
        "auroc",
        "auprc",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "ece",
        "metrics",
        "confusion",
        "threshold",
    }
)
_NOTE = "Pair-protocol metrics are not this result. This record does not validate calibration."


def build_generation_evaluation_record(
    provenance: Mapping[str, object],
    rows: list[Mapping[str, object]],
    *,
    score_requested: bool,
    probe_available: bool,
) -> dict[str, object]:
    """Assemble a finished record or raise before any file is written."""
    if not isinstance(score_requested, bool) or not isinstance(probe_available, bool):
        raise ValueError("score_requested and probe_available must be booleans")
    if score_requested and not probe_available:
        raise ValueError("A score was requested and the probe is missing")
    checked = _provenance(provenance)
    if not rows:
        raise ValueError("A finished record needs one row per generated example")
    stored = [
        _row(row, index=index, score_requested=score_requested) for index, row in enumerate(rows)
    ]
    record: dict[str, object] = {
        "schema_version": 1,
        "protocol": "generation_evaluation",
        "finished_reproduction": True,
        "pair_protocol_metrics_included": False,
        "calibration_validated": False,
        "provenance": checked,
        "rows": stored,
        "excluded_from_binary_metrics": {
            "ambiguous": sum(row["label"] == Label.AMBIGUOUS.name for row in stored),
            "unvalidated": sum(row["label"] == Label.UNVALIDATED.name for row in stored),
        },
        "note": _NOTE,
    }
    _reject_metric_keys(record)
    return record


def write_generation_evaluation(
    provenance: Mapping[str, object],
    rows: list[Mapping[str, object]],
    destination: Path,
    *,
    artifacts_root: Path,
    score_requested: bool,
    probe_available: bool,
) -> Path:
    """Write the record only after it passes. A failure leaves no file."""
    record = build_generation_evaluation_record(
        provenance,
        rows,
        score_requested=score_requested,
        probe_available=probe_available,
    )
    path = Path(destination)
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}")
    assert_dataset_destination(path, artifacts_root)
    return write_json(path, record, root=artifacts_root)


def write_command(args: argparse.Namespace) -> int:
    """Write a record from a JSON payload. Does not load Qwen."""
    destination = Path(args.output)
    try:
        payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object")
        provenance = payload.get("provenance")
        rows = payload.get("rows")
        if not isinstance(provenance, dict) or not isinstance(rows, list):
            raise ValueError("Payload needs provenance and rows")
        write_generation_evaluation(
            provenance,
            rows,
            destination,
            artifacts_root=Path(args.artifacts_root),
            score_requested=bool(payload.get("score_requested")),
            probe_available=bool(payload.get("probe_available")),
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Generation evaluation was not recorded: {exc}")
        return 1
    print(f"Wrote {destination}")
    print(_NOTE)
    return 0


def _provenance(provenance: Mapping[str, object]) -> dict[str, object]:
    missing = [name for name in REQUIRED_PROVENANCE if name not in provenance]
    if missing:
        raise ValueError(f"Missing provenance: {missing}")
    if provenance.get("worktree_clean") is not True:
        raise ValueError("A dirty worktree cannot be published as a finished reproduction")
    if provenance.get("calibration_validated") is not False:
        raise ValueError("This generation record does not set calibration_validated")
    seed = provenance.get("seed")
    layer = provenance.get("layer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise ValueError("layer must be an integer")
    text_fields = (
        "model_name",
        "dtype",
        "dataset_id",
        "dataset_sha256",
        "split_id",
        "probe_directory",
        "position",
        "prompt_mode",
        "config_path",
    )
    checked: dict[str, object] = {
        "model_revision": require_commit_hash(
            _text(provenance, "model_revision"), field="model_revision"
        ),
        "probe_model_revision": require_commit_hash(
            _text(provenance, "probe_model_revision"), field="probe_model_revision"
        ),
        "git_commit": require_commit_hash(_text(provenance, "git_commit"), field="git_commit"),
        "seed": seed,
        "layer": layer,
        "worktree_clean": True,
        "calibration_validated": False,
    }
    for name in text_fields:
        checked[name] = _text(provenance, name)
    return checked


def _text(provenance: Mapping[str, object], name: str) -> str:
    value = provenance.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _row(row: Mapping[str, object], *, index: int, score_requested: bool) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise ValueError(f"Row {index} must be an object")
    question_id = row.get("question_id")
    answer = row.get("answer")
    label = row.get("label")
    phrases = row.get("reference_phrases")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError(f"Row {index} needs a question_id")
    if not isinstance(answer, str):
        raise ValueError(f"{question_id} needs an answer string")
    if not isinstance(label, str) or label not in _LABELS:
        raise ValueError(f"{question_id} has an unknown label")
    if not isinstance(phrases, list) or not all(isinstance(phrase, str) for phrase in phrases):
        raise ValueError(f"{question_id} needs a list of reference phrases")
    if label == Label.UNVALIDATED.name and phrases:
        raise ValueError(f"{question_id} is unvalidated and cannot carry a justifying phrase")
    if (
        label in {Label.FAITHFUL.name, Label.HALLUCINATED.name, Label.AMBIGUOUS.name}
        and not phrases
    ):
        raise ValueError(f"{question_id} needs the reference phrase that justified the label")
    binary = label in _BINARY
    if "in_binary_metric_set" in row and row["in_binary_metric_set"] is not binary:
        raise ValueError(f"{question_id} cannot move an excluded label into the binary metric set")
    score, reason = _score(row, question_id=question_id, score_requested=score_requested)
    return {
        "question_id": question_id,
        "answer": answer,
        "label": label,
        "reference_phrases": list(phrases),
        "probe_score": score,
        "missing_score_reason": reason,
        "in_binary_metric_set": binary,
    }


def _score(
    row: Mapping[str, object], *, question_id: str, score_requested: bool
) -> tuple[float | None, str | None]:
    score = row.get("probe_score", None)
    reason = row.get("missing_score_reason")
    if score is None:
        if score_requested and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError(f"{question_id} is missing a probe score")
        if reason is None:
            return None, None
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{question_id} needs a missing-score reason")
        return None, reason.strip()
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise ValueError(f"{question_id} probe_score must be a finite number")
    if reason not in (None, ""):
        raise ValueError(f"{question_id} cannot have both a probe score and a missing-score reason")
    return float(score), None


def _reject_metric_keys(payload: object, *, prefix: str = "") -> None:
    if isinstance(payload, Mapping):
        for key, nested in payload.items():
            if str(key).lower() in _METRIC_KEYS:
                raise ValueError(f"Generation record must not store {prefix}{key}")
            _reject_metric_keys(nested, prefix=f"{prefix}{key}.")
        return
    if isinstance(payload, list):
        for index, nested in enumerate(payload):
            _reject_metric_keys(nested, prefix=f"{prefix}{index}.")
