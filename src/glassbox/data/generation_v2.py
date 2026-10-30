"""Assemble Generation Dataset v2 without rewriting generation-001.

Cached answers keep their original split. New questions stay unlabeled until
they are generated and passed through the evidence protocol, or until a human
review cites passage evidence. Unvalidated rows are not converted into binary
labels. The live pair probe is not replaced.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from glassbox.data.generation_expand import (
    HISTORICAL_GENERATION_RUN,
    TARGET_PER_CLASS,
    assert_no_split_leakage,
    cached_answers_for_split,
    load_generation_rows,
    plan_expansion,
)
from glassbox.data.halueval import SplitAssignment
from glassbox.data.schema import Label
from glassbox.evaluation.metrics import BOOTSTRAP_MIN_PER_CLASS
from glassbox.probes.generation_eval import PROTECTED_PROBE_NAMES

V2_DEFAULT_RUN = "halueval-generation-002"
PROTECTED_RUNS = frozenset({HISTORICAL_GENERATION_RUN, *PROTECTED_PROBE_NAMES})


def assemble_generation_v2(
    plan: Mapping[str, object],
    cached_rows: Sequence[Mapping[str, object]],
    *,
    generated_rows: Sequence[Mapping[str, object]] | None = None,
    cached_vectors: Mapping[str, np.ndarray] | None = None,
    cached_means: Mapping[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Join frozen 001 rows with any newly generated rows on the planned split."""
    selected = _assignment_from_plan(plan)
    reused, missing = cached_answers_for_split(selected, cached_rows)
    by_id = {str(row["question_id"]): dict(row) for row in reused}
    generated = list(generated_rows or [])
    generated_ids = {str(row["question_id"]) for row in generated}
    for row in generated:
        question_id = str(row["question_id"])
        if question_id in by_id:
            raise ValueError(
                f"Generated row {question_id} would replace a cached generation-001 answer"
            )
        expected = _split_of(plan, question_id)
        if expected is None:
            raise ValueError(f"Generated row {question_id} is not in the expansion plan")
        copied = dict(row)
        copied["split"] = expected
        copied["source"] = "generated_v2"
        by_id[question_id] = copied
    remaining = [qid for qid in missing if qid not in generated_ids]
    rows = []
    for question_id, row in by_id.items():
        copied = dict(row)
        copied.setdefault("source", "cached_generation_001")
        if cached_vectors and question_id in cached_vectors:
            copied["vector"] = np.asarray(cached_vectors[question_id], dtype=np.float64)
        if cached_means and question_id in cached_means:
            copied["answer_mean"] = np.asarray(cached_means[question_id], dtype=np.float64)
        rows.append(copied)
    train = [str(row["question_id"]) for row in rows if row["split"] == "train"]
    validation = [str(row["question_id"]) for row in rows if row["split"] == "validation"]
    test = [str(row["question_id"]) for row in rows if row["split"] == "test"]
    assert_no_split_leakage(train, validation, test)
    counts = _split_label_counts(rows)
    shortfall = {
        split: {
            "faithful": max(0, TARGET_PER_CLASS - counts[split]["FAITHFUL"]),
            "hallucinated": max(0, TARGET_PER_CLASS - counts[split]["HALLUCINATED"]),
        }
        for split in ("validation", "test")
    }
    mean_pool_n = sum(1 for row in rows if row.get("answer_mean") is not None)
    return {
        "run_id_default": V2_DEFAULT_RUN,
        "historical_run": HISTORICAL_GENERATION_RUN,
        "rewrites_historical_run": False,
        "pair_probe_rewritten": False,
        "rows": rows,
        "n_rows": len(rows),
        "missing_question_ids": remaining,
        "missing_question_count": len(remaining),
        "class_counts": counts,
        "shortfall_for_target": shortfall,
        "target_per_class": TARGET_PER_CLASS,
        "target_met": all(
            shortfall[split]["faithful"] == 0 and shortfall[split]["hallucinated"] == 0
            for split in ("validation", "test")
        ),
        "mean_pool_rows": mean_pool_n,
        "unvalidated_not_auto_labeled": True,
        "invented_human_labels": False,
        "note": (
            "Unvalidated cached rows stay unvalidated. Missing questions have no labels "
            "until they are generated or reviewed. This assembly does not train a live probe."
        ),
    }


def load_vector_map(path: Path, *, field: str = "vectors") -> dict[str, np.ndarray]:
    """Read a generation-001-style npz keyed by question id."""
    data = np.load(path)
    if "question_id" not in data.files or field not in data.files:
        raise ValueError(f"{path} needs question_id and {field}")
    return {
        str(question_id): np.asarray(vector, dtype=np.float64)
        for question_id, vector in zip(data["question_id"], data[field], strict=True)
    }


def jsonable_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Drop ndarray fields so the row list can be written as JSON."""
    cleaned = []
    for row in rows:
        item = {
            key: value
            for key, value in dict(row).items()
            if key not in {"vector", "answer_mean"} and not isinstance(value, np.ndarray)
        }
        item["has_final_piece_vector"] = row.get("vector") is not None
        item["has_mean_pool_vector"] = row.get("answer_mean") is not None
        cleaned.append(item)
    return cleaned


def v2_command(args: object) -> int:
    """Assemble v2 from cached 001 rows. Does not rewrite historical artifacts."""
    import sys

    from glassbox.config import repo_root
    from glassbox.data.halueval import (
        PINNED_SHA256,
        prepare_pairs,
        read_pinned_snapshot,
        snapshot_sha256,
    )
    from glassbox.evaluation.generation_calibration import PAIR_THRESHOLD
    from glassbox.evaluation.records import write_json
    from glassbox.probes.generation_eval import (
        generation_experiment_dir,
        generation_probe_dir,
        halueval_snapshot_path,
    )
    from glassbox.probes.generation_variants import evaluate_generation_variants

    try:
        run_id = str(args.run_id)
        if run_id in PROTECTED_RUNS or run_id.startswith("final-hidden-state"):
            raise ValueError(f"Refusing protected run id {run_id}")
        artifacts = repo_root() / "artifacts"
        directory = generation_experiment_dir(run_id)
        probe_dir = generation_probe_dir(run_id)
        if directory.exists() or probe_dir.exists():
            raise FileExistsError("Refusing to overwrite an existing v2 run")
        snapshot = Path(args.snapshot) if args.snapshot else halueval_snapshot_path()
        if not snapshot.is_file():
            raise FileNotFoundError(f"HaluEval snapshot is not at {snapshot}")
        digest = snapshot_sha256(snapshot)
        if digest != PINNED_SHA256:
            raise ValueError(
                f"HaluEval snapshot hash {digest} does not match pinned {PINNED_SHA256}"
            )
        prepared = prepare_pairs(read_pinned_snapshot(snapshot), seed=int(args.seed))
        predictions = (
            Path(args.predictions)
            if args.predictions
            else artifacts / "experiments" / HISTORICAL_GENERATION_RUN / "predictions.json"
        )
        cached = load_generation_rows(predictions) if predictions.is_file() else []
        generated = load_generation_rows(Path(args.generated)) if args.generated else []
        plan = plan_expansion(
            prepared.pairs.values(),
            cached,
            seed=int(args.seed),
            max_questions=int(args.max_questions),
        )
        vectors = None
        means = None
        if args.features:
            vectors = load_vector_map(Path(args.features))
        assembled = assemble_generation_v2(
            plan,
            cached,
            generated_rows=generated,
            cached_vectors=vectors,
            cached_means=means,
        )
        pair_before = None
        pair_path = Path(args.pair_probe) if args.pair_probe else None
        if pair_path is not None and (pair_path / "metadata.json").is_file():
            pair_before = (pair_path / "metadata.json").read_bytes()
        report = evaluate_generation_variants(
            assembled["rows"],
            seed=int(args.seed),
            frozen_threshold=None if args.threshold is None else float(args.threshold),
            pair_probe_path=pair_path,
        )
        payload = {
            "task": "generation_dataset_v2",
            "plan": {
                key: value
                for key, value in plan.items()
                if key not in {"question_ids", "groups", "new_question_ids"}
            },
            "assembly": {key: value for key, value in assembled.items() if key != "rows"},
            "variants": report,
            "pair_threshold_unchanged": PAIR_THRESHOLD,
            "bootstrap_min_per_class": BOOTSTRAP_MIN_PER_CLASS,
            "dataset_sha256": digest,
            "successful_detector": False,
        }
        write_json(directory / "report.json", payload, root=artifacts)
        write_json(
            directory / "predictions.json",
            {"rows": jsonable_rows(assembled["rows"])},
            root=artifacts,
        )
        if pair_before is not None and (pair_path / "metadata.json").read_bytes() != pair_before:
            raise RuntimeError("Generation v2 rewrote the pair-probe metadata")
        if probe_dir.exists():
            raise RuntimeError("Generation v2 wrote a live probe directory")
        print(
            "v2 assembled "
            f"{assembled['n_rows']} rows, missing {assembled['missing_question_count']}, "
            f"target_met={assembled['target_met']}"
        )
        print(f"Wrote {directory / 'report.json'}")
        return 0
    except (OSError, ValueError, FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Generation dataset v2 stopped: {exc}", file=sys.stderr)
        return 1


def _assignment_from_plan(plan: Mapping[str, object]) -> SplitAssignment:
    ids = plan.get("question_ids")
    if not isinstance(ids, Mapping):
        raise ValueError("expansion plan is missing question_ids")
    return SplitAssignment(
        train=tuple(str(item) for item in ids["train"]),
        validation=tuple(str(item) for item in ids["validation"]),
        test=tuple(str(item) for item in ids["test"]),
        seed=int(plan.get("seed", 0)),
    )


def _split_of(plan: Mapping[str, object], question_id: str) -> str | None:
    ids = plan.get("question_ids")
    if not isinstance(ids, Mapping):
        return None
    for split in ("train", "validation", "test"):
        if question_id in {str(item) for item in ids[split]}:
            return split
    return None


def _split_label_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[Mapping[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for row in rows:
        grouped[str(row["split"])].append(row)
    return {split: _label_counts(items) for split, items in grouped.items()}


def _label_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    names = Counter(str(row.get("label_name") or _label_name(row.get("label"))) for row in rows)
    return {
        "questions": len(rows),
        "FAITHFUL": int(names.get("FAITHFUL", 0)),
        "HALLUCINATED": int(names.get("HALLUCINATED", 0)),
        "AMBIGUOUS": int(names.get("AMBIGUOUS", 0)),
        "UNVALIDATED": int(names.get("UNVALIDATED", 0)),
        "EXCLUDED": int(names.get("EXCLUDED", 0)),
    }


def _label_name(value: object) -> str:
    if isinstance(value, Label):
        return value.name
    if value in (0, "0"):
        return "FAITHFUL"
    if value in (1, "1"):
        return "HALLUCINATED"
    if value in (2, "2"):
        return "AMBIGUOUS"
    return "UNVALIDATED"
