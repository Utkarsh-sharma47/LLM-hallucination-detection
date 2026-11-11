"""Validate the immutable AI review and rebuild summaries without loading Qwen."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ("FAITHFUL", "HALLUCINATED", "AMBIGUOUS", "UNVALIDATED")
SPLITS = ("train", "validation", "test")
BINARY = {"FAITHFUL": 0, "HALLUCINATED": 1}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(source: Path | None = None, features: Path | None = None) -> tuple[list, dict]:
    manifest = read_json(ROOT / "source-manifest.json")
    annotation_bytes = (ROOT / "annotations.jsonl").read_bytes()
    require(
        sha256(annotation_bytes) == manifest["annotation_sha256"],
        "Annotation snapshot changed; preserve this version and publish revisions separately.",
    )
    rows = [json.loads(line) for line in annotation_bytes.decode("utf-8").splitlines()]
    require([row["source_index"] for row in rows] == list(range(2400)),
            "Expected exactly 2,400 ordered, unique source indices.")
    require(len({row["question_id"] for row in rows}) == len(rows), "Duplicate question IDs.")
    decisions = []
    for path in sorted(ROOT.glob("decisions-*.json")):
        decisions.extend(read_json(path))
    decisions.sort(key=lambda item: item[0])
    require([item[0] for item in decisions] == list(range(2400)),
            "Missing or duplicate batch decisions.")
    groups: dict[str, str] = {}
    passages: dict[str, str] = {}
    source_hash = manifest["source_files"]["predictions.json"]["sha256"]
    for row, decision in zip(rows, decisions, strict=True):
        require(row["label"] in LABELS and row["split"] in SPLITS, "Invalid label or split.")
        require(row["review_number"] == row["source_index"] + 1, "Review number mismatch.")
        require(row["source_file_sha256"] == source_hash, "Source hash mismatch.")
        require(row["policy_version"] == manifest["policy_version"], "Policy mismatch.")
        require(row["reviewer_type"] == "AI" and row["human_reviewed"] is False,
                "AI provenance must remain explicit.")
        require(row["annotation_method"] == "AI-assisted" and bool(row["reviewer"]),
                "Missing reviewer provenance.")
        require(bool(row["evidence_quote"]) and row["evidence_quote"] in row["knowledge"],
                f"Evidence does not match passage: {row['question_id']}")
        require(len(row["evidence_note"]) >= 20, "Missing rationale.")
        require(datetime.fromisoformat(row["timestamp"]).tzinfo is not None,
                "Review timestamp must have a timezone.")
        require(row["eligible_for_binary_metrics"] == (row["label"] in BINARY),
                "Binary candidate flag mismatch.")
        require(decision == [row["source_index"], row["label"], row["evidence_quote"],
                             row["evidence_note"], row["flags"]],
                f"Batch decision differs from annotation: {row['question_id']}")
        group = row["group_id"]
        passage = re.sub(r"[^a-z0-9]+", " ",
                         re.sub(r"\s+", " ", row["knowledge"].strip().lower())).strip()
        for key, locations in ((group, groups), (passage, passages)):
            require(key not in locations or locations[key] == row["split"],
                    "Knowledge group or normalized passage crosses splits.")
            locations[key] = row["split"]
    source_checked = False
    disagreements = None
    if source is not None:
        data = source.read_bytes()
        require(sha256(data) == source_hash, "Wrong frozen predictions.json.")
        original = json.loads(data)["rows"]
        require(len(original) == len(rows), "Source row count mismatch.")
        for row, old in zip(rows, original, strict=True):
            for field in ("question_id", "group_id", "split", "question", "answer", "knowledge"):
                require(row[field] == old[field], f"Source field changed: {field}")
            digest = sha256(json.dumps(old, sort_keys=True, ensure_ascii=False).encode())
            require(digest == row["source_row_sha256"], "Original row hash mismatch.")
        disagreements = sum(row["label"] != old["label_name"]
                            for row, old in zip(rows, original, strict=True))
        source_checked = True
    if features is not None:
        require(sha256(features.read_bytes()) ==
                manifest["source_files"]["features.npz"]["sha256"],
                "Wrong cached features.npz.")
    counts = {split: {label: sum(row["split"] == split and row["label"] == label
                                for row in rows) for label in LABELS} for split in SPLITS}
    summary = {
        "run_id": manifest["run_id"],
        "policy_version": manifest["policy_version"],
        "reviewed": len(rows),
        "remaining": 0,
        "label_counts": dict(collections.Counter(row["label"] for row in rows)),
        "split_label_counts": counts,
        "binary_candidate_counts": {
            split: counts[split]["FAITHFUL"] + counts[split]["HALLUCINATED"]
            for split in SPLITS
        },
        "excluded_from_binary": sum(row["label"] not in BINARY for row in rows),
        "reviewer_type": "AI",
        "independent_human_reviews": 0,
        "inter_rater_agreement": None,
        "adjudicated": False,
        "consistency_audit_completed": False,
        "probe_retrained": False,
        "source_sha256": source_hash,
        "annotation_sha256": manifest["annotation_sha256"],
        "group_split_isolation": True,
        "normalized_passage_split_isolation": True,
    }
    checks = {
        "annotations": len(rows),
        "source_content_checked": source_checked,
        "feature_file_hash_checked": features is not None,
        "disagreements_with_automatic_labels": disagreements,
        "summary": summary,
    }
    return rows, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Original frozen predictions.json")
    parser.add_argument("--features", type=Path, help="Original cached features.npz")
    parser.add_argument("--rebuild", action="store_true", help="Write derived files")
    args = parser.parse_args()
    rows, checks = validate(args.source, args.features)
    if args.rebuild:
        output = ROOT / "generated"
        output.mkdir(exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(checks["summary"], indent=2) + "\n", encoding="utf-8", newline="\n")
        # This index is a data handoff, not a human-review import or a trained model.
        with (output / "training-index.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                if row["label"] not in BINARY:
                    continue
                item = {key: row[key] for key in (
                    "source_index", "question_id", "group_id", "split", "label",
                    "source_row_sha256", "source_file_sha256", "policy_version",
                    "reviewer_type", "human_reviewed",
                )}
                item["target"] = BINARY[row["label"]]
                item["annotation_sha256"] = checks["summary"]["annotation_sha256"]
                item["adjudicated"] = False
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
