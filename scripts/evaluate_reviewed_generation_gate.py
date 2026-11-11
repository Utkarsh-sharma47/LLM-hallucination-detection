"""Evaluate issue #18 on the frozen task-7 split and reviewed decisions.

This is an offline analysis: it reuses cached Qwen vectors and never loads the
model. The fixed pair threshold is applied, question IDs used by the pair probe
are excluded, and the saved pair probe is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from glassbox.data.schema import Label
from glassbox.evaluation.generation_calibration import (
    PAIR_THRESHOLD,
    evaluate_generation_calibration_gate,
)
from glassbox.probes.scoring import expected_calibration_error
from glassbox.probes.store import check_compatibility, experimental_score, load_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("artifacts/reviews/task7-halueval-generation-2400-001-ai-v1"),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("artifacts/experiments/task7-halueval-generation-2400-001"),
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=Path("artifacts/probes/final-hidden-state"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/experiments/task7-halueval-generation-2400-001/"
            "reviewed-calibration-gate.json"
        ),
    )
    args = parser.parse_args()
    try:
        report = evaluate(args.review_dir, args.experiment_dir, args.probe_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Calibration gate evaluation failed closed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "accepted" else 1


def evaluate(review_dir: Path, experiment_dir: Path, probe_dir: Path) -> dict[str, object]:
    manifest = _read_json(review_dir / "source-manifest.json")
    _verify_hashes(review_dir, experiment_dir, manifest)
    attestation = _read_attestation(review_dir, manifest)

    annotations = _read_annotations(review_dir / "annotations.jsonl")
    decisions = _read_decisions(review_dir, annotations)
    annotation_ids = {str(row["question_id"]) for row in annotations.values()}
    if annotation_ids != set(decisions) or len(annotations) != 2400:
        raise ValueError("Expected one reviewed decision for every one of the 2,400 frozen rows")

    source = _read_json(experiment_dir / "predictions.json")
    source_rows = {str(row["question_id"]): row for row in source["rows"]}
    if len(source_rows) != 2400:
        raise ValueError("The frozen prediction file must contain 2,400 unique questions")

    with np.load(experiment_dir / "features.npz", allow_pickle=False) as bundle:
        feature_ids = bundle["question_id"].astype(str)
        feature_splits = bundle["split"].astype(str)
        vectors = bundle["vectors"]
    if len(feature_ids) != len(set(feature_ids.tolist())) or vectors.shape[0] != 2400:
        raise ValueError("Cached features must contain 2,400 unique question IDs")
    features = {
        question_id: (split, vectors[index])
        for index, (question_id, split) in enumerate(zip(feature_ids, feature_splits, strict=True))
    }

    loaded = load_probe(probe_dir)
    metadata = loaded.metadata
    if metadata.get("label_protocol") != "halueval_qa_pair_v1":
        raise ValueError("The selected artifact is not the HaluEval pair probe")
    used = metadata.get("dataset", {}).get("used_question_ids")
    if not isinstance(used, dict) or not all(
        isinstance(used.get(k), list) for k in ("train", "validation", "test")
    ):
        raise ValueError("The pair probe lacks complete train/validation/test question IDs")
    pair_question_ids = {str(item) for values in used.values() for item in values}
    if not pair_question_ids:
        raise ValueError("The pair probe records no used question IDs")

    gate_rows: list[dict[str, object]] = []
    diagnostic_labels: list[int] = []
    diagnostic_scores: list[float] = []
    validation_counts = {label.name: 0 for label in Label}
    overlap_counts = {label.name: 0 for label in Label}
    for question_id, row in source_rows.items():
        annotation = annotations[decisions[question_id]["source_index"]]
        label = decisions[question_id]["label"]
        if annotation["question_id"] != question_id or annotation["split"] != row["split"]:
            raise ValueError(f"Review/source identity mismatch for {question_id}")
        if label not in {item.name for item in Label}:
            raise ValueError(f"Unknown reviewed label {label!r} for {question_id}")
        if row["split"] != "validation":
            continue
        validation_counts[label] += 1
        is_overlap = question_id in pair_question_ids
        if is_overlap:
            overlap_counts[label] += 1
        vector_record = features.get(question_id)
        score = None
        if vector_record is not None:
            feature_split, vector = vector_record
            if feature_split != row["split"]:
                raise ValueError(f"Feature split mismatch for {question_id}")
            check_compatibility(
                loaded,
                model_name=str(row["model_name"]),
                model_revision=str(row["model_revision"]),
                layer_index=int(row["feature_layer"]),
                hidden_state_count=int(row["hidden_state_count"]),
                position=str(row["position"]),
                prompt_mode=str(row["prompt_mode"]),
                feature_dim=int(np.asarray(vector).size),
                dtype=str(row["dtype"]),
            )
            score = experimental_score(loaded, np.asarray(vector, dtype=np.float64))
        gate_rows.append(
            {
                "question_id": question_id,
                "group_id": annotation["group_id"],
                "split": "validation",
                "label": label,
                "probe_score": score,
                "split_overlap": is_overlap,
            }
        )
        if (
            not is_overlap
            and score is not None
            and label
            in {
                Label.FAITHFUL.name,
                Label.HALLUCINATED.name,
            }
        ):
            diagnostic_labels.append(int(label == Label.HALLUCINATED.name))
            diagnostic_scores.append(float(score))

    decision = evaluate_generation_calibration_gate(
        gate_rows,
        threshold=PAIR_THRESHOLD,
        pair_probe_metadata=metadata,
        pair_probe_path=probe_dir,
        score_source="pair_probe",
    )
    decision["pair_probe_artifact_threshold"] = metadata.get("threshold")
    decision["fixed_gate_threshold"] = PAIR_THRESHOLD
    decision["human_review_attestation"] = {
        "reviewer_id": attestation["reviewer_id"],
        "recorded_at": attestation["recorded_at"],
        "scope_count": attestation["decision_count"],
        "annotation_sha256": attestation["annotation_sha256"],
        "decision_files_sha256": attestation["decision_files_sha256"],
    }
    decision["validation_label_counts_before_overlap_filter"] = validation_counts
    decision["validation_label_counts_excluded_for_pair_probe_overlap"] = overlap_counts
    decision["validation_label_counts_after_overlap_filter"] = {
        label: validation_counts[label] - overlap_counts[label] for label in validation_counts
    }
    decision["calibration_source"] = "saved pair-probe scores on user-reviewed Qwen generations"
    decision["review_policy"] = manifest["policy_version"]
    decision["diagnostic_validation_ece"] = (
        expected_calibration_error(np.asarray(diagnostic_labels), np.asarray(diagnostic_scores))
        if diagnostic_labels and set(diagnostic_labels) == {0, 1}
        else None
    )
    decision["diagnostic_ece_is_gate_eligible"] = (
        min(
            decision["validation_label_counts_after_overlap_filter"][Label.FAITHFUL.name],
            decision["validation_label_counts_after_overlap_filter"][Label.HALLUCINATED.name],
        )
        >= 20
    )
    decision["diagnostic_ece_note"] = (
        "For information only when the 20-per-class gate fails; it cannot change the rejected status."
    )
    return decision


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_annotations(path: Path) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        index = item.get("source_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in rows:
            raise ValueError(f"Invalid or duplicate annotation index at line {line_number}")
        rows[index] = item
    return rows


def _read_decisions(
    review_dir: Path, annotations: dict[int, dict[str, object]]
) -> dict[str, dict[str, object]]:
    decisions: dict[str, dict[str, object]] = {}
    for path in sorted(review_dir.glob("decisions-*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            if not isinstance(row, list) or len(row) < 4:
                raise ValueError(f"Malformed decision row in {path.name}")
            index, label = row[0], row[1]
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(f"Malformed source index in {path.name}")
            annotation = annotations.get(index)
            if annotation is None:
                raise ValueError(f"Decision index {index} has no annotation")
            question_id = str(annotation["question_id"])
            if question_id in decisions:
                raise ValueError(f"Duplicate decision for {question_id}")
            decisions[question_id] = {"source_index": index, "label": str(label)}
    return decisions


def _verify_hashes(review_dir: Path, experiment_dir: Path, manifest: dict[str, object]) -> None:
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict):
        raise ValueError("Source manifest is missing source file hashes")
    for name, expected in source_files.items():
        path = experiment_dir / str(name)
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen source is missing: {path}")
        actual = _sha256(path)
        if not isinstance(expected, dict) or actual != expected.get("sha256"):
            raise ValueError(f"Frozen source hash mismatch for {name}")
    annotation_hash = _sha256(review_dir / "annotations.jsonl")
    if annotation_hash != manifest.get("annotation_sha256"):
        raise ValueError("Annotation hash mismatch")
    files = sorted(review_dir.glob("decisions-*.json"))
    joined = "".join(f"{path.name} {_sha256(path)}\n" for path in files)
    decisions_hash = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    attestation_path = review_dir / "human-review-attestations.jsonl"
    if not attestation_path.is_file():
        raise ValueError("Missing human review attestation")
    attestation_lines = [
        line for line in attestation_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    attestations = [json.loads(line) for line in attestation_lines]
    attestation = next(
        (
            item
            for item in reversed(attestations)
            if item.get("event_type") == "human_review_attestation"
        ),
        None,
    )
    if not isinstance(attestation, dict):
        raise ValueError("No human review attestation was found")
    if (
        attestation.get("decision_files_sha256") != decisions_hash
        or attestation.get("annotation_sha256") != annotation_hash
        or attestation.get("decision_count") != 2400
    ):
        raise ValueError("Human review attestation does not match this decision snapshot")


def _read_attestation(review_dir: Path, manifest: dict[str, object]) -> dict[str, object]:
    path = review_dir / "human-review-attestations.jsonl"
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("event_type") == "human_review_attestation":
            return item
    raise ValueError("Human review attestation was not found")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
