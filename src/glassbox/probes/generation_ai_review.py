"""Train a generation probe from the AI-reviewed Task 7 labels (issue #41 handoff).

Run from the repository root::

    python -m glassbox.probes.generation_ai_review --dry-run
    python -m glassbox.probes.generation_ai_review --run-id task7-ai-review-v1-001

This does not call Qwen. It reuses the frozen answers and cached final-token vectors of the
original Task 7 run, replaces only the label with the AI review label, and hands the rows to
the same ``finalize_generation_run`` used by ``train-generation-probe``. It does not use
``read_human_reviews``: that path would mark these labels ``human_reviewed=True``. Here every
row, the probe metadata and the report keep ``reviewer_type=AI``, ``human_reviewed=False`` and
``adjudicated=False``. The original probe and run are never modified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from glassbox.config import repo_root
from glassbox.data.schema import Label
from glassbox.demo_config import default_probe_dir
from glassbox.probes.generation_eval import (
    assert_output_allowed,
    generation_experiment_dir,
    generation_probe_dir,
)
from glassbox.probes.generation_train import _git_state, finalize_generation_run
from glassbox.reproducibility import package_versions

SOURCE_RUN = "task7-halueval-generation-2400-001"
REVIEW_DIR = "task7-halueval-generation-2400-001-ai-v1"
DEFAULT_RUN_ID = "task7-ai-review-v1-001"
BINARY = {"FAITHFUL", "HALLUCINATED"}
AI_NOTE = (
    "Labels are one AI-assisted, passage-grounded pass (ai_passage_review_v1). They are not "
    "independent human reviews and were not adjudicated. Treat results as provisional."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m glassbox.probes.generation_ai_review")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--review-dir", type=Path, default=None)
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument("--pair-probe", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="verify and join, write nothing")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"PROBE NOT TRAINED: {exc}", file=sys.stderr)
        return 1


def run(args: argparse.Namespace) -> int:
    artifacts = repo_root() / "artifacts"
    review_dir = Path(args.review_dir) if args.review_dir else artifacts / "reviews" / REVIEW_DIR
    source_dir = (
        Path(args.source_run_dir) if args.source_run_dir else artifacts / "experiments" / SOURCE_RUN
    )
    predictions_path = source_dir / "predictions.json"
    features_path = source_dir / "features.npz"
    for path in (review_dir / "annotations.jsonl", predictions_path, features_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")

    _verify_bundle(review_dir, predictions_path, features_path)
    annotations = _read_annotations(review_dir / "annotations.jsonl")
    manifest = json.loads((review_dir / "source-manifest.json").read_text(encoding="utf-8"))
    source_report = json.loads((source_dir / "report.json").read_text(encoding="utf-8"))
    rows = _join(predictions_path, features_path, annotations)
    counts = _split_counts(rows)
    for split, found in counts.items():
        print(f"{split:11s} {found}")
    if args.dry_run:
        print("Dry run: verification and join passed. Nothing was written.")
        return 0

    probe_dir = generation_probe_dir(args.run_id)
    experiment_dir = generation_experiment_dir(args.run_id)
    assert_output_allowed(artifacts, args.run_id, probe_dir)
    if experiment_dir.exists() or probe_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.run_id}")

    review_record = _review_record(manifest, annotations)
    dataset_record = dict(source_report["dataset"])
    dataset_record["label_review"] = review_record
    dataset_record["label_policy_note"] = (
        "The text in label_policy describes the original automatic labeler. The labels used "
        "here come from ai_passage_review_v1; see label_review."
    )
    provenance = _provenance(source_report, manifest)
    provenance["label_review"] = review_record

    code = finalize_generation_run(
        rows,
        artifacts=artifacts,
        probe_dir=probe_dir,
        experiment_dir=experiment_dir,
        run_id=args.run_id,
        seed=args.seed,
        dataset_record=dataset_record,
        provenance=provenance,
        pair_probe_dir=Path(args.pair_probe) if args.pair_probe else default_probe_dir(),
        external_rows=None,
        external_note="External evaluation was not run for the AI-review retrain.",
        settings=None,
        reviews_applied=0,
    )
    _stamp_ai_provenance(experiment_dir, probe_dir, review_record)
    print(AI_NOTE)
    return code


def _verify_bundle(review_dir: Path, predictions: Path, features: Path) -> None:
    script = review_dir / "manage_review.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source",
            str(predictions),
            "--features",
            str(features),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            "The review bundle failed its own validator:\n" + (result.stderr or result.stdout)
        )
    summary = json.loads(result.stdout)
    if not (summary.get("source_content_checked") and summary.get("feature_file_hash_checked")):
        raise ValueError("Validator did not confirm source and feature hashes")


def _read_annotations(path: Path) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        question_id = str(item["question_id"])
        if question_id in found:
            raise ValueError(f"annotations.jsonl line {number} repeats {question_id}")
        if item.get("reviewer_type") != "AI" or item.get("human_reviewed") is not False:
            raise ValueError(f"annotations.jsonl line {number} lost its AI provenance")
        Label[str(item["label"])]
        found[question_id] = item
    return found


def _join(
    predictions: Path, features: Path, annotations: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    original = json.loads(predictions.read_text(encoding="utf-8"))["rows"]
    data = np.load(features)
    ids = [str(item) for item in data["question_id"]]
    if len(set(ids)) != len(ids):
        raise ValueError("features.npz repeats a question_id")
    vectors = {
        question_id: (str(split), np.asarray(vector, dtype=np.float64))
        for question_id, split, vector in zip(ids, data["split"], data["vectors"], strict=True)
    }
    if len(annotations) != len(original):
        raise ValueError("annotations and predictions cover different numbers of rows")
    rows: list[dict[str, object]] = []
    for item in original:
        question_id = str(item["question_id"])
        note = annotations.get(question_id)
        if note is None:
            raise ValueError(f"No AI annotation for {question_id}")
        if question_id not in vectors:
            raise ValueError(f"No cached vector for {question_id}")
        split, vector = vectors[question_id]
        if not (split == str(item["split"]) == str(note["split"])):
            raise ValueError(f"Split mismatch for {question_id}")
        if str(note["answer"]) != str(item["answer"]):
            raise ValueError(f"Answer text changed for {question_id}")
        row = dict(item)
        label = Label[str(note["label"])]
        row["automatic_label_name"] = item["label_name"]
        row["label_name"] = label.name
        row["label"] = int(label.value)
        row["vector"] = vector
        row["review"] = {
            "reviewer_type": "AI",
            "human_reviewed": False,
            "adjudicated": False,
            "policy_version": note.get("policy_version"),
            "flags": note.get("flags", []),
        }
        rows.append(row)
    return rows


def _split_counts(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        split = counts.setdefault(str(row["split"]), {})
        name = str(row["label_name"])
        split[name] = split.get(name, 0) + 1
    return counts


def _review_record(manifest: dict[str, object], annotations: dict[str, dict[str, object]]):
    return {
        "policy_version": manifest.get("policy_version"),
        "reviewer_type": "AI",
        "human_reviewed": False,
        "adjudicated": False,
        "independent_reviews": 0,
        "inter_rater_agreement": None,
        "annotation_sha256": manifest.get("annotation_sha256"),
        "source_files": manifest.get("source_files"),
        "annotation_count": len(annotations),
        "note": AI_NOTE,
    }


def _provenance(source_report: dict[str, object], manifest: dict[str, object]):
    original = dict(source_report["provenance"])
    commit, dirty, blockers = _git_state()
    blockers = list(blockers)
    blockers.append(
        "answers and vectors come from a source run recorded as not reproducible "
        f"({SOURCE_RUN}); labels are one AI-assisted pass"
    )
    keep = (
        "model_name",
        "model_revision",
        "encoder_revision",
        "encoder_class",
        "model_dtype",
        "feature_precision",
        "dataset_sha256",
    )
    record = {key: original[key] for key in keep if key in original}
    record.update(
        {
            "git_commit": commit,
            "worktree_dirty": dirty,
            "publishable_as_reproducible": False,
            "reproducibility_blockers": blockers,
            "source_run": SOURCE_RUN,
            "source_git_commit": original.get("git_commit"),
            "seed": int(original.get("seed", 0)),
            "python": sys.version.split()[0],
            "packages": package_versions(),
        }
    )
    return record


def _stamp_ai_provenance(experiment_dir: Path, probe_dir: Path, review_record: dict) -> None:
    """Add the AI-review block to files finalize wrote. Nothing else is changed."""
    for path in (probe_dir / "metadata.json", experiment_dir / "report.json"):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["label_review"] = review_record
        payload["labels_are_ai_assisted"] = True
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
