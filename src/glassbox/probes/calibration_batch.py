"""Preselected validation batch for the generation calibration gate (issue #18).

The gate asks whether the saved pair probe's score can be called a calibrated probability on
Qwen's own generated answers. The existing reviewed validation rows are too few once the
questions the pair probe saw are removed, so this module builds a new batch:

* ``plan``      choose HaluEval questions with a model-free, hash-ordered rule. Questions the pair
                probe, the task-7 run, or any recorded split used are excluded, and so is every
                question that shares a knowledge passage with one of them. The plan is written
                before any answer exists and carries its own hash.
* ``verify``    re-check a plan against the files in the repository.
* ``generate``  produce Qwen answers and final-token vectors for the plan, one shard at a time.
* ``queue``     write a blind review queue: question, answer and passage only. No probe score
                and no automatic label.
* ``gate``      score the reviewed answers with the saved pair probe at the fixed threshold and
                run the pre-registered gate. The pair probe is never modified, and nothing here
                chooses a threshold or refits anything on this batch.

If the gate fails, the batch is not used to tune anything. Recalibration would need a separate
fit set (the recorded reserve) and a fresh untouched evaluation set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from glassbox.config import repo_root
from glassbox.data.generation_splits import knowledge_group
from glassbox.data.halueval import (
    HaluPair,
    SplitAssignment,
    prepare_pairs,
    read_pinned_snapshot,
    snapshot_sha256,
)
from glassbox.evaluation.generation_calibration import (
    PAIR_THRESHOLD,
    evaluate_generation_calibration_gate,
)
from glassbox.evaluation.records import write_json
from glassbox.probes.generation_eval import halueval_snapshot_path
from glassbox.probes.scoring import expected_calibration_error
from glassbox.probes.store import check_compatibility, experimental_score, load_probe

DEFAULT_BATCH_ID = "pair-calibration-validation-001"
SELECTION_SEED = "calibration-batch-v1"
DEFAULT_SIZE = 800
SHARD_SIZE = 100
POLICY_VERSION = "ai_passage_review_v1"
RUBRIC_PATH = Path("artifacts/reviews/task7-halueval-generation-2400-001-ai-v1/RUBRIC.md")
PAIR_PROTOCOL = "halueval_qa_pair_v1"
_ROW_ID = re.compile(r"\brow\d+\b")
_LABELS = {"FAITHFUL", "HALLUCINATED", "AMBIGUOUS", "UNVALIDATED"}
_BINARY = {"FAITHFUL", "HALLUCINATED"}

DECISION_RULES = (
    "Questions are chosen by sha256 order of the group id, whole knowledge groups at a time, "
    "before any answer exists. The score of any probe is not an input.",
    "Shards are reviewed in order. The analysis set is the longest prefix of fully reviewed "
    "shards. Review out of order is refused.",
    "A row is dropped only when generation produced no answer or no hidden state, or when the "
    "review label is AMBIGUOUS or UNVALIDATED. Dropped rows are counted in the report.",
    f"The gate runs at the fixed threshold {PAIR_THRESHOLD} with the pre-registered ECE limit "
    "and the 20-per-class minimum. Nothing is tuned on this batch.",
    "If the gate fails, this batch is not used to refit the pair probe or to choose a threshold. "
    "Recalibration needs the reserve as a fit set and a fresh untouched evaluation set.",
)


# --------------------------------------------------------------------------- planning


def select_batch(
    pairs: Mapping[str, HaluPair],
    excluded_ids: set[str],
    *,
    size: int,
    seed: str = SELECTION_SEED,
) -> dict[str, object]:
    """Pick whole knowledge groups in hash order until at least ``size`` questions are chosen."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")
    group_of = {question_id: knowledge_group(pair.knowledge) for question_id, pair in pairs.items()}
    excluded_groups = {group_of[item] for item in excluded_ids if item in group_of}
    members: dict[str, list[str]] = {}
    for question_id in sorted(pairs):
        if question_id in excluded_ids or group_of[question_id] in excluded_groups:
            continue
        members.setdefault(group_of[question_id], []).append(question_id)
    eligible_questions = sum(len(items) for items in members.values())
    if eligible_questions < size:
        raise ValueError(f"Only {eligible_questions} eligible questions remain; need {size}")
    ordered_groups = sorted(members, key=lambda group: _digest(f"{seed}:group:{group}"))
    chosen: list[str] = []
    chosen_groups: set[str] = set()
    for group in ordered_groups:
        if len(chosen) >= size:
            break
        chosen.extend(members[group])
        chosen_groups.add(group)
    selected = sorted(chosen, key=lambda item: _digest(f"{seed}:order:{item}"))
    reserve = sorted(
        item for group, items in members.items() if group not in chosen_groups for item in items
    )
    return {
        "selected": selected,
        "reserve": reserve,
        "eligible_question_count": eligible_questions,
        "eligible_group_count": len(members),
        "excluded_question_count": len([item for item in excluded_ids if item in pairs]),
        "excluded_group_count": len(excluded_groups),
        "group_of": group_of,
    }


def scan_exclusions(
    artifacts: Path, *, skip_dir: Path | None = None
) -> tuple[list[dict[str, object]], set[str]]:
    """Every question id a recorded probe, split or earlier plan has touched."""
    root = Path(artifacts)
    files = sorted(
        set(root.glob("probes/**/metadata.json"))
        | set(root.glob("experiments/*/splits.json"))
        | set(root.glob("experiments/*/predictions.json"))
    )
    records: list[dict[str, object]] = []
    ids: set[str] = set()
    for path in files:
        text = path.read_text(encoding="utf-8")
        found = set(_ROW_ID.findall(text))
        ids |= found
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _digest(text.replace("\r\n", "\n")),
                "question_ids_found": len(found),
            }
        )
    for plan_path in sorted(root.glob("calibration/*/plan.json")):
        if skip_dir is not None and plan_path.parent.resolve() == Path(skip_dir).resolve():
            continue
        other = json.loads(plan_path.read_text(encoding="utf-8"))
        selected = {str(item) for item in other.get("selected_question_ids", [])}
        ids |= selected
        records.append(
            {
                "path": plan_path.relative_to(root).as_posix(),
                "sha256": str(other.get("plan_sha256")),
                "question_ids_found": len(selected),
            }
        )
    return records, ids


def build_plan(
    pairs: Mapping[str, HaluPair],
    *,
    batch_id: str,
    size: int,
    dataset_sha256: str,
    artifacts: Path,
    shard_size: int = SHARD_SIZE,
) -> dict[str, object]:
    batch_dir = Path(artifacts) / "calibration" / batch_id
    sources, excluded = scan_exclusions(artifacts, skip_dir=batch_dir)
    chosen = select_batch(pairs, excluded, size=size)
    selected = list(chosen["selected"])  # type: ignore[arg-type]
    reserve = list(chosen["reserve"])  # type: ignore[arg-type]
    plan: dict[str, object] = {
        "schema": "calibration_batch_plan_v1",
        "batch_id": batch_id,
        "purpose": "Issue #18 generation calibration gate for the saved pair probes.",
        "selection_seed": SELECTION_SEED,
        "requested_size": size,
        "selected_count": len(selected),
        "shard_size": shard_size,
        "shard_count": -(-len(selected) // shard_size),
        "dataset_sha256": dataset_sha256,
        "kept_pairs": len(pairs),
        "exclusion_sources": sources,
        "excluded_question_count": chosen["excluded_question_count"],
        "excluded_group_count": chosen["excluded_group_count"],
        "eligible_question_count": chosen["eligible_question_count"],
        "eligible_group_count": chosen["eligible_group_count"],
        "reserve_count": len(reserve),
        "selected_question_ids": selected,
        "reserve_question_ids": reserve,
        "decision_rules": list(DECISION_RULES),
        "created_before_generation": True,
        "score_of_any_probe_used_in_selection": False,
    }
    plan["plan_sha256"] = plan_hash(plan)
    return plan


def plan_hash(plan: Mapping[str, object]) -> str:
    core = {
        key: plan[key]
        for key in (
            "schema",
            "batch_id",
            "selection_seed",
            "requested_size",
            "shard_size",
            "dataset_sha256",
            "selected_question_ids",
            "reserve_question_ids",
        )
    }
    return _digest(json.dumps(core, sort_keys=True, separators=(",", ":")))


def verify_plan(
    plan: Mapping[str, object],
    pairs: Mapping[str, HaluPair],
    artifacts: Path,
) -> list[str]:
    """Return a list of problems. An empty list means the plan still holds."""
    problems: list[str] = []
    if plan.get("plan_sha256") != plan_hash(plan):
        problems.append("plan_sha256 does not match the plan content")
    selected = [str(item) for item in plan.get("selected_question_ids", [])]  # type: ignore[union-attr]
    reserve = [str(item) for item in plan.get("reserve_question_ids", [])]  # type: ignore[union-attr]
    if len(selected) != len(set(selected)) or len(reserve) != len(set(reserve)):
        problems.append("a question id is repeated")
    if set(selected) & set(reserve):
        problems.append("selected and reserve share a question")
    missing = [item for item in selected + reserve if item not in pairs]
    if missing:
        problems.append(f"{len(missing)} planned questions are not in the filtered snapshot")
        return problems
    batch_dir = Path(artifacts) / "calibration" / str(plan.get("batch_id"))
    _, excluded = scan_exclusions(artifacts, skip_dir=batch_dir)
    touched = sorted(set(selected) & excluded)
    if touched:
        problems.append(f"{len(touched)} planned questions appear in recorded probe or run files")
    groups = {item: knowledge_group(pairs[item].knowledge) for item in selected + reserve}
    excluded_groups = {knowledge_group(pairs[item].knowledge) for item in excluded if item in pairs}
    shared = {groups[item] for item in selected} & excluded_groups
    if shared:
        problems.append(f"{len(shared)} planned passages are shared with excluded questions")
    if {groups[item] for item in selected} & {groups[item] for item in reserve}:
        problems.append("a passage is shared between the batch and the reserve")
    shard_size = int(plan.get("shard_size", SHARD_SIZE))  # type: ignore[call-overload]
    if plan.get("shard_count") != -(-len(selected) // shard_size):
        problems.append("shard_count does not match selected_count and shard_size")
    return problems


# ------------------------------------------------------------------------- generation


def shard_ids(plan: Mapping[str, object], shard: int) -> list[str]:
    size = int(plan["shard_size"])  # type: ignore[call-overload]
    selected = [str(item) for item in plan["selected_question_ids"]]  # type: ignore[union-attr]
    if shard < 0 or shard * size >= len(selected):
        raise ValueError(f"Shard {shard} is outside the plan")
    return selected[shard * size : (shard + 1) * size]


def shard_paths(run_dir: Path, shard: int) -> tuple[Path, Path]:
    return run_dir / f"shard-{shard:03d}.json", run_dir / f"shard-{shard:03d}.npz"


RowsFn = Callable[[object, object, dict[str, str], object, dict, object], list[dict[str, object]]]


def generate_shards(
    plan: Mapping[str, object],
    pairs: Mapping[str, HaluPair],
    *,
    run_dir: Path,
    shards: Sequence[int],
    backend: object,
    loaded: object,
    settings: object,
    rows_fn: RowsFn,
    run_meta: Mapping[str, object],
) -> list[int]:
    """Generate the named shards. A shard whose json file exists is skipped, so a stopped run
    resumes. The json is written last: its existence means the shard is complete."""
    from glassbox.probes.generation_train import _public_row

    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run.json"
    meta = dict(run_meta)
    meta["plan_sha256"] = plan["plan_sha256"]
    if meta_path.is_file():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("plan_sha256", "model_name", "model_revision", "dtype"):
            if previous.get(key) != meta.get(key):
                raise ValueError(f"Refusing to mix runs: {key} differs from {meta_path}")
    else:
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    group_of = {item: knowledge_group(pair.knowledge) for item, pair in pairs.items()}
    done: list[int] = []
    for shard in shards:
        json_path, npz_path = shard_paths(run_dir, shard)
        if json_path.is_file():
            continue
        ids = shard_ids(plan, shard)
        prepared = SimpleNamespace(pairs=pairs, assignment=SplitAssignment((), tuple(ids), (), 0))
        rows = rows_fn(backend, prepared, group_of, settings, {}, loaded)
        if [str(row["question_id"]) for row in rows] != ids:
            raise RuntimeError(f"Shard {shard} came back in a different order or size")
        base = int(plan["shard_size"]) * shard  # type: ignore[call-overload]
        kept = [row for row in rows if row.get("vector") is not None]
        if kept:
            np.savez_compressed(
                npz_path,
                question_id=np.asarray([str(row["question_id"]) for row in kept]),
                vectors=np.vstack([np.asarray(row["vector"], dtype=np.float32) for row in kept]),
            )
        public = []
        for offset, row in enumerate(rows):
            record = _public_row(row)
            record["plan_index"] = base + offset
            record["shard"] = shard
            record["automatic_label_name"] = record.get("label_name")
            record["split"] = "validation"
            public.append(record)
        temporary = json_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"shard": shard, "rows": public}, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(json_path)
        done.append(shard)
    return done


def read_shard(run_dir: Path, shard: int) -> list[dict[str, object]]:
    json_path, _ = shard_paths(run_dir, shard)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return [dict(row) for row in payload["rows"]]


def read_vectors(run_dir: Path, shard: int) -> dict[str, np.ndarray]:
    _, npz_path = shard_paths(run_dir, shard)
    if not npz_path.is_file():
        return {}
    with np.load(npz_path, allow_pickle=False) as bundle:
        return {
            str(question_id): np.asarray(vector, dtype=np.float64)
            for question_id, vector in zip(bundle["question_id"], bundle["vectors"], strict=True)
        }


def completed_shards(run_dir: Path, plan: Mapping[str, object]) -> list[int]:
    return [
        shard
        for shard in range(int(plan["shard_count"]))  # type: ignore[call-overload]
        if shard_paths(run_dir, shard)[0].is_file()
    ]


# ------------------------------------------------------------------------ review queue


def write_queue(
    plan: Mapping[str, object], run_dir: Path, review_dir: Path
) -> tuple[Path, int, list[str]]:
    """Blind queue for the complete shards. Only question, answer and passage are included."""
    lines: list[str] = []
    dropped: list[str] = []
    for shard in completed_shards(run_dir, plan):
        for row in read_shard(run_dir, shard):
            if row.get("label_name") == "EXCLUDED" or not str(row.get("answer", "")).strip():
                dropped.append(str(row["question_id"]))
                continue
            lines.append(
                json.dumps(
                    {
                        "source_index": row["plan_index"],
                        "shard": shard,
                        "question_id": row["question_id"],
                        "group_id": row["group_id"],
                        "question": row["question"],
                        "answer": row["answer"],
                        "knowledge": row["knowledge"],
                    },
                    sort_keys=True,
                )
            )
    review_dir.mkdir(parents=True, exist_ok=True)
    queue_path = review_dir / "review-queue.jsonl"
    text = "\n".join(lines) + ("\n" if lines else "")
    if queue_path.is_file() and queue_path.read_text(encoding="utf-8") != text:
        existing = queue_path.read_text(encoding="utf-8").splitlines()
        if existing != lines[: len(existing)]:
            raise ValueError("Refusing to change queue rows that may already be reviewed")
    queue_path.write_text(text, encoding="utf-8")
    (review_dir / "queue-manifest.json").write_text(
        json.dumps(
            {
                "plan_sha256": plan["plan_sha256"],
                "queue_rows": len(lines),
                "dropped_before_review": dropped,
                "fields_withheld": ["probe score", "automatic label", "right answer"],
                "decision_format": "decisions-SSS.json: [[source_index, label, quote, note, flags], ...]",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    meta = review_dir / "review-meta.json"
    if not meta.is_file():
        meta.write_text(
            json.dumps(
                {
                    "reviewer": None,
                    "reviewer_type": None,
                    "human_reviewed": None,
                    "policy_version": POLICY_VERSION,
                    "rubric": RUBRIC_PATH.as_posix(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return queue_path, len(lines), dropped


# ------------------------------------------------------------------------------- gate


def read_queue(review_dir: Path) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    path = review_dir / "review-queue.jsonl"
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        index = item["source_index"]
        if index in rows:
            raise ValueError(f"review-queue.jsonl line {number} repeats index {index}")
        rows[index] = item
    return rows


def read_decisions(review_dir: Path, queue: Mapping[int, Mapping[str, object]]):
    decisions: dict[int, dict[str, object]] = {}
    for path in sorted(review_dir.glob("decisions-*.json")):
        match = re.fullmatch(r"decisions-(\d{3})\.json", path.name)
        if match is None:
            raise ValueError(f"Unexpected decision file name {path.name}")
        file_shard = int(match.group(1))
        for row in json.loads(path.read_text(encoding="utf-8")):
            if not isinstance(row, list) or len(row) < 4:
                raise ValueError(f"Malformed decision row in {path.name}")
            index, label, quote, note = row[0], str(row[1]), str(row[2]), str(row[3])
            flags = row[4] if len(row) > 4 else []
            item = queue.get(index)
            if item is None:
                raise ValueError(f"Decision index {index} in {path.name} is not in the queue")
            if item["shard"] != file_shard:
                raise ValueError(f"Decision {index} is filed under the wrong shard")
            if index in decisions:
                raise ValueError(f"Duplicate decision for index {index}")
            if label not in _LABELS:
                raise ValueError(f"Unknown label {label!r} for index {index}")
            passage = _squash(str(item["knowledge"]))
            if label in _BINARY and not quote.strip():
                raise ValueError(f"Index {index}: {label} needs a quote from the passage")
            if quote.strip() and _squash(quote) not in passage:
                raise ValueError(f"Index {index}: the quote is not in the passage")
            if len(note.strip()) < 20:
                raise ValueError(f"Index {index}: the note needs at least 20 characters")
            if not isinstance(flags, list):
                raise ValueError(f"Index {index}: flags must be a list")
            decisions[index] = {"label": label, "quote": quote, "note": note, "flags": flags}
    return decisions


def analysis_shards(
    queue: Mapping[int, Mapping[str, object]],
    decisions: Mapping[int, object],
    shard_count: int,
) -> list[int]:
    """Longest prefix of fully reviewed shards. Reviewing past a gap is refused."""
    by_shard: dict[int, list[int]] = {}
    for index, item in queue.items():
        by_shard.setdefault(int(item["shard"]), []).append(index)  # type: ignore[call-overload]
    complete: list[int] = []
    gap = False
    for shard in range(shard_count):
        indexes = by_shard.get(shard, [])
        reviewed = [index for index in indexes if index in decisions]
        if gap and reviewed:
            raise ValueError(f"Shard {shard} was reviewed after an unreviewed shard")
        if indexes and len(reviewed) == len(indexes) and not gap:
            complete.append(shard)
        else:
            gap = True
    return complete


def evaluate_batch(
    *,
    plan: Mapping[str, object],
    run_dir: Path,
    review_dir: Path,
    probe_dir: Path,
    artifacts: Path,
) -> dict[str, object]:
    if plan.get("plan_sha256") != plan_hash(plan):
        raise ValueError("The plan does not match its recorded hash")
    run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run_meta.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("The run was generated from a different plan")
    review_meta = json.loads((review_dir / "review-meta.json").read_text(encoding="utf-8"))
    for key in ("reviewer", "reviewer_type", "human_reviewed"):
        if review_meta.get(key) in (None, ""):
            raise ValueError(f"review-meta.json must record {key} before the gate runs")
    if review_meta.get("policy_version") != POLICY_VERSION:
        raise ValueError("The review used a different policy version")
    rubric = Path(artifacts).parent / RUBRIC_PATH
    queue = read_queue(review_dir)
    decisions = read_decisions(review_dir, queue)
    shards = analysis_shards(queue, decisions, int(plan["shard_count"]))  # type: ignore[call-overload]
    if not shards:
        raise ValueError("No shard is fully reviewed yet")
    batch_dir = Path(artifacts) / "calibration" / str(plan["batch_id"])
    _, excluded = scan_exclusions(artifacts, skip_dir=batch_dir)
    probe_files = [Path(probe_dir) / "metadata.json", Path(probe_dir) / "probe.joblib"]
    before = [_file_hash(path) for path in probe_files]
    loaded = load_probe(probe_dir)
    metadata = loaded.metadata
    if metadata.get("label_protocol") != PAIR_PROTOCOL:
        raise ValueError("The probe is not the HaluEval pair probe")

    counts = {name: 0 for name in sorted(_LABELS)}
    dropped = {"generation_excluded": 0, "no_vector": 0}
    gate_rows: list[dict[str, object]] = []
    labels: list[int] = []
    scores: list[float] = []
    selected_in_scope = 0
    for shard in shards:
        vectors = read_vectors(run_dir, shard)
        for row in read_shard(run_dir, shard):
            selected_in_scope += 1
            question_id = str(row["question_id"])
            if question_id in excluded:
                raise ValueError(f"{question_id} appears in a recorded probe or run file")
            if row.get("label_name") == "EXCLUDED":
                dropped["generation_excluded"] += 1
                continue
            decision = decisions[int(row["plan_index"])]
            label = str(decision["label"])
            counts[label] += 1
            vector = vectors.get(question_id)
            if vector is None:
                dropped["no_vector"] += 1
                continue
            check_compatibility(
                loaded,
                model_name=str(row["model_name"]),
                model_revision=str(row["model_revision"]),
                layer_index=int(row["feature_layer"]),
                hidden_state_count=int(row["hidden_state_count"]),
                position=str(row["position"]),
                prompt_mode=str(row["prompt_mode"]),
                feature_dim=int(vector.size),
                dtype=str(row["dtype"]),
            )
            score = experimental_score(loaded, vector)
            gate_rows.append(
                {
                    "question_id": question_id,
                    "group_id": row["group_id"],
                    "split": "validation",
                    "label": label,
                    "probe_score": score,
                    "split_overlap": False,
                }
            )
            if label in _BINARY:
                labels.append(int(label == "HALLUCINATED"))
                scores.append(float(score))

    decision_report = evaluate_generation_calibration_gate(
        gate_rows,
        threshold=PAIR_THRESHOLD,
        pair_probe_metadata=metadata,
        pair_probe_path=probe_dir,
        score_source="pair_probe",
    )
    after = [_file_hash(path) for path in probe_files]
    if before != after:
        raise RuntimeError("The pair probe files changed during the gate run")
    both = bool(labels) and set(labels) == {0, 1}
    decision_report.update(
        {
            "batch_id": plan["batch_id"],
            "plan_sha256": plan["plan_sha256"],
            "run_id": run_dir.name,
            "calibration_source": (
                "saved pair-probe scores on a preselected, separately reviewed batch of Qwen "
                "generations whose questions the probe never saw"
            ),
            "shards_analyzed": shards,
            "shard_count": plan["shard_count"],
            "questions_in_scope": selected_in_scope,
            "review_label_counts": counts,
            "dropped_rows": dropped,
            "reviewer": review_meta["reviewer"],
            "reviewer_type": review_meta["reviewer_type"],
            "human_reviewed": review_meta["human_reviewed"],
            "review_policy": review_meta["policy_version"],
            "rubric_sha256": _file_hash(rubric) if rubric.is_file() else None,
            "review_queue_withheld": ["probe score", "automatic label", "right answer"],
            "fixed_gate_threshold": PAIR_THRESHOLD,
            "pair_probe_artifact_threshold": metadata.get("threshold"),
            "pair_probe_unchanged": True,
            "pair_probe_files_sha256": dict(
                zip(("metadata.json", "probe.joblib"), after, strict=True)
            ),
            "tuned_on_this_batch": False,
            "diagnostic_ece": (
                float(expected_calibration_error(np.asarray(labels), np.asarray(scores)))
                if both
                else None
            ),
            "diagnostic_ece_is_gate_eligible": decision_report.get("status") == "accepted",
            "diagnostic_ece_note": (
                "Reported for information. It cannot change the gate status, and this batch is "
                "not used to choose a threshold or refit the probe."
            ),
            "reliability_bins": _reliability(labels, scores),
            "mean_score_by_class": {
                "faithful": _mean(
                    score for label, score in zip(labels, scores, strict=True) if label == 0
                ),
                "hallucinated": _mean(
                    score for label, score in zip(labels, scores, strict=True) if label == 1
                ),
            },
        }
    )
    return decision_report


# --------------------------------------------------------------------------- helpers


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _squash(text: str) -> str:
    return " ".join(text.split())


def _mean(values) -> float | None:
    items = list(values)
    return float(sum(items) / len(items)) if items else None


def _reliability(labels: Sequence[int], scores: Sequence[float], bins: int = 10):
    table = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        picked = [
            (label, score)
            for label, score in zip(labels, scores, strict=True)
            if low <= score < high or (index == bins - 1 and score == 1.0)
        ]
        table.append(
            {
                "range": [low, high],
                "n": len(picked),
                "mean_score": _mean(score for _, score in picked),
                "observed_hallucinated_rate": _mean(label for label, _ in picked),
            }
        )
    return table


# ----------------------------------------------------------------------------- commands


def _paths(artifacts: Path, batch_id: str, run_id: str | None):
    batch_dir = artifacts / "calibration" / batch_id
    run_dir = artifacts / "experiments" / run_id if run_id else None
    review_dir = artifacts / "reviews" / f"{run_id}-ai-v1" if run_id else None
    return batch_dir, run_dir, review_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m glassbox.probes.calibration_batch")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "verify", "generate", "queue", "gate"):
        item = sub.add_parser(name)
        item.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
        if name == "plan":
            item.add_argument("--size", type=int, default=DEFAULT_SIZE)
        if name in {"generate", "queue", "gate"}:
            item.add_argument("--run-id", required=True)
        if name == "generate":
            item.add_argument("--config", type=Path, default=None)
            item.add_argument("--model", default=None)
            item.add_argument("--offline", action="store_true")
            item.add_argument("--allow-model-download", action="store_true")
            item.add_argument("--shards", default=None, help="e.g. 0-3 or 0,2 (default all)")
        if name == "gate":
            item.add_argument("--probe-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"Calibration batch stopped: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    artifacts = repo_root() / "artifacts"
    batch_dir, run_dir, review_dir = _paths(artifacts, args.batch_id, getattr(args, "run_id", None))
    plan_path = batch_dir / "plan.json"
    if args.command == "plan":
        if plan_path.exists():
            raise FileExistsError(f"Refusing to replace {plan_path}. A plan is written once.")
        snapshot = halueval_snapshot_path()
        digest = snapshot_sha256(snapshot)
        prepared = prepare_pairs(read_pinned_snapshot(snapshot), seed=0)
        plan = build_plan(
            prepared.pairs,
            batch_id=args.batch_id,
            size=args.size,
            dataset_sha256=digest,
            artifacts=artifacts,
        )
        write_json(plan_path, plan, root=artifacts)
        print(f"Wrote {plan_path}")
        print(
            f"selected {plan['selected_count']} questions in {plan['shard_count']} shards; "
            f"reserve {plan['reserve_count']}; excluded {plan['excluded_question_count']}"
        )
        print(f"plan_sha256 {plan['plan_sha256']}")
        print("Commit this file before generating any answer.")
        return 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if args.command == "verify":
        snapshot = halueval_snapshot_path()
        read_pinned_snapshot(snapshot)
        pairs = prepare_pairs(read_pinned_snapshot(snapshot), seed=0).pairs
        problems = verify_plan(plan, pairs, artifacts)
        for problem in problems:
            print(f"PROBLEM: {problem}", file=sys.stderr)
        print("plan holds" if not problems else "plan does not hold")
        return 1 if problems else 0
    assert run_dir is not None and review_dir is not None
    if args.command == "queue":
        path, rows, dropped = write_queue(plan, run_dir, review_dir)
        print(f"Wrote {path} with {rows} rows; {len(dropped)} dropped before review")
        print(
            f"Edit {review_dir / 'review-meta.json'} to record the reviewer, then add decisions-SSS.json files."
        )
        return 0
    if args.command == "gate":
        report = evaluate_batch(
            plan=plan,
            run_dir=run_dir,
            review_dir=review_dir,
            probe_dir=Path(args.probe_dir),
            artifacts=artifacts,
        )
        out = batch_dir / f"gate-{args.run_id}-shards-{len(report['shards_analyzed'])}.json"
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite {out}")
        write_json(out, report, root=artifacts)
        print(
            json.dumps(
                {k: report[k] for k in ("status", "reason", "ece", "eligible_n", "diagnostic_ece")},
                indent=2,
            )
        )
        print(f"Wrote {out}")
        return 0 if report["status"] == "accepted" else 1
    return _generate_command(args, plan, run_dir)


def _generate_command(args: argparse.Namespace, plan: dict[str, object], run_dir: Path) -> int:
    from glassbox.demo_config import load_demo_config
    from glassbox.models.backend import GenerationSettings, HuggingFaceBackend
    from glassbox.probes.generation_train import _generate_rows, _git_state
    from glassbox.probes.training import REGISTERED_POSITION
    from glassbox.reproducibility import require_commit_hash

    if args.allow_model_download and args.offline:
        raise ValueError("--offline refuses --allow-model-download")
    snapshot = halueval_snapshot_path()
    pairs = prepare_pairs(read_pinned_snapshot(snapshot), seed=0).pairs
    artifacts = repo_root() / "artifacts"
    problems = verify_plan(plan, pairs, artifacts)
    if problems:
        raise ValueError("The plan does not hold: " + "; ".join(problems))
    config = load_demo_config(
        args.config,
        offline=(not args.allow_model_download) or args.offline,
        model_name=args.model,
    )
    if config.generation.position != REGISTERED_POSITION:
        raise ValueError(f"Demo position is {config.generation.position}")
    settings = GenerationSettings(
        max_new_tokens=config.generation.max_new_tokens,
        temperature=0.0,
        top_k=config.generation.top_k,
        layer_index=config.generation.layer_index,
        position=REGISTERED_POSITION,
        capture_all_layers=False,
    )
    backend = HuggingFaceBackend(config.spec)
    loaded = backend.ensure_loaded()
    require_commit_hash(loaded.resolved_revision, field="model.revision")
    commit, dirty, _ = _git_state()
    total = int(plan["shard_count"])  # type: ignore[call-overload]
    shards = _parse_shards(args.shards, total)
    done = generate_shards(
        plan,
        pairs,
        run_dir=run_dir,
        shards=shards,
        backend=backend,
        loaded=loaded,
        settings=settings,
        rows_fn=_generate_rows,
        run_meta={
            "model_name": loaded.spec.name,
            "model_revision": str(loaded.resolved_revision),
            "dtype": loaded.dtype,
            "git_commit": commit,
            "worktree_dirty": dirty,
            "feature_position": REGISTERED_POSITION,
        },
    )
    print(
        f"Generated shards {done or 'none (all already present)'}; {len(completed_shards(run_dir, plan))}/{total} complete"
    )
    return 0


def _parse_shards(text: str | None, total: int) -> list[int]:
    if not text:
        return list(range(total))
    shards: list[int] = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-", 1)
            shards.extend(range(int(low), int(high) + 1))
        else:
            shards.append(int(part))
    if any(item < 0 or item >= total for item in shards):
        raise ValueError(f"Shards must be between 0 and {total - 1}")
    return shards


if __name__ == "__main__":
    raise SystemExit(main())
