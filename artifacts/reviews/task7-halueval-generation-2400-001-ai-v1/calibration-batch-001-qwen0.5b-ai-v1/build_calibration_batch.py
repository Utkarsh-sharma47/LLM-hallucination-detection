"""Validate and assemble calibration batch into merged annotations and index."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BINARY_LABELS = {"FAITHFUL": 0, "HALLUCINATED": 1}
ALL_LABELS = ("FAITHFUL", "HALLUCINATED", "AMBIGUOUS", "UNVALIDATED")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    queue_file = ROOT / "review-queue.jsonl"
    meta_file = ROOT / "review-meta.json"
    manifest_file = ROOT / "queue-manifest.json"

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    queue_rows = [json.loads(line) for line in queue_file.read_text(encoding="utf-8").splitlines()]
    if len(queue_rows) != manifest["queue_rows"]:
        raise ValueError(f"Queue count {len(queue_rows)} != manifest {manifest['queue_rows']}")

    decisions = []
    for path in sorted(ROOT.glob("decisions-*.json")):
        decisions.extend(json.loads(path.read_text(encoding="utf-8")))

    if len(decisions) != len(queue_rows):
        raise ValueError(f"Decisions count {len(decisions)} != queue count {len(queue_rows)}")

    decisions.sort(key=lambda d: d[0])
    if [d[0] for d in decisions] != list(range(len(queue_rows))):
        raise ValueError("Decisions have missing or duplicate indices")

    output_dir = ROOT / "generated"
    output_dir.mkdir(exist_ok=True)

    annotations = []
    binary_index = []

    for q, d in zip(queue_rows, decisions, strict=True):
        idx, label, quote, note, flags = d
        if label not in ALL_LABELS:
            raise ValueError(f"Unknown label {label}")
        if quote and quote not in q["knowledge"]:
            raise ValueError(f"Quote not found in knowledge for row {idx}")

        row = {
            "source_index": idx,
            "question_id": q["question_id"],
            "group_id": q["group_id"],
            "question": q["question"],
            "answer": q["answer"],
            "knowledge": q["knowledge"],
            "label": label,
            "evidence_quote": quote,
            "evidence_note": note,
            "flags": flags,
            "reviewer": meta["reviewer"],
            "reviewer_type": meta["reviewer_type"],
            "human_reviewed": meta["human_reviewed"],
            "policy_version": meta["policy_version"],
            "eligible_for_binary_metrics": label in BINARY_LABELS,
        }
        annotations.append(row)

        if label in BINARY_LABELS:
            binary_index.append({
                "source_index": idx,
                "question_id": q["question_id"],
                "group_id": q["group_id"],
                "label": label,
                "target": BINARY_LABELS[label],
            })

    # Write merged annotations.jsonl
    annotations_path = output_dir / "annotations.jsonl"
    with annotations_path.open("w", encoding="utf-8", newline="\n") as f:
        for item in annotations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write binary index
    index_path = output_dir / "calibration-index.jsonl"
    with index_path.open("w", encoding="utf-8", newline="\n") as f:
        for item in binary_index:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    counts = dict(collections.Counter(a["label"] for a in annotations))
    summary = {
        "batch_id": ROOT.name,
        "policy_version": meta["policy_version"],
        "reviewer": meta["reviewer"],
        "total_rows": len(annotations),
        "label_counts": counts,
        "binary_candidates": {
            "total": len(binary_index),
            "faithful": counts.get("FAITHFUL", 0),
            "hallucinated": counts.get("HALLUCINATED", 0),
        },
        "annotations_sha256": sha256_bytes(annotations_path.read_bytes()),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

