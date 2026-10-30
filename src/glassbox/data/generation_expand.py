"""Plan a larger generation set without rewriting historical runs.

The 720-answer ``halueval-generation-001`` record stays frozen. New questions
are assigned with the same knowledge-group split seed so existing rows keep
their split. Unvalidated automatic labels are not promoted to binary classes.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from glassbox.data.generation_splits import (
    assert_group_isolation,
    assign_generation_splits,
    question_ids,
    subset_preserving_groups,
)
from glassbox.data.halueval import HaluPair, SplitAssignment
from glassbox.data.schema import Label

TARGET_PER_CLASS = 20
HISTORICAL_GENERATION_RUN = "halueval-generation-001"


def plan_expansion(
    pairs: Iterable[HaluPair],
    labeled_rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    max_questions: int,
    target_per_class: int = TARGET_PER_CLASS,
) -> dict[str, object]:
    """Count how a larger prefix split would change labeled coverage.

    This does not generate answers and does not invent labels for new questions.
    """
    if max_questions < 3:
        raise ValueError("max_questions must cover train, validation, and test")
    assignment, group_of = assign_generation_splits(pairs, seed=seed)
    selected = subset_preserving_groups(assignment, group_of, max_questions)
    assert_group_isolation(selected, group_of)
    known = {str(row["question_id"]): row for row in labeled_rows}
    overlap = _split_overlap(selected, known)
    new_ids = {
        split: [qid for qid in selected.ids(split) if qid not in known]
        for split in ("train", "validation", "test")
    }
    known_counts = {
        split: _label_counts(overlap[split]) for split in ("train", "validation", "test")
    }
    missing = {
        split: _missing_for_target(known_counts[split], target_per_class)
        for split in ("validation", "test")
    }
    return {
        "seed": seed,
        "max_questions": max_questions,
        "split_algorithm": "knowledge-group-v1",
        "historical_run": HISTORICAL_GENERATION_RUN,
        "rewrites_historical_run": False,
        "target_per_class": target_per_class,
        "selected_questions": {name: len(ids) for name, ids in question_ids(selected).items()},
        "known_labeled_in_selected": known_counts,
        "new_question_ids": new_ids,
        "new_question_counts": {split: len(ids) for split, ids in new_ids.items()},
        "shortfall_for_target": missing,
        "target_met_with_known_labels": all(
            missing[split]["faithful"] == 0 and missing[split]["hallucinated"] == 0
            for split in ("validation", "test")
        ),
        "note": (
            "New questions have no labels until they are generated and passed through "
            "the evidence protocol, or until a human review cites passage evidence. "
            "Unvalidated rows are not converted into FAITHFUL or HALLUCINATED here."
        ),
        "question_ids": {name: list(ids) for name, ids in question_ids(selected).items()},
        "groups": dict(group_of),
    }


def cached_answers_for_split(
    selected: SplitAssignment, cached_rows: Sequence[Mapping[str, object]]
) -> tuple[list[dict[str, object]], list[str]]:
    """Reuse historical generations whose question ids remain in the new split."""
    by_id = {str(row["question_id"]): dict(row) for row in cached_rows}
    reused: list[dict[str, object]] = []
    missing: list[str] = []
    located = {
        qid: split for split in ("train", "validation", "test") for qid in selected.ids(split)
    }
    for question_id, split in located.items():
        row = by_id.get(question_id)
        if row is None:
            missing.append(question_id)
            continue
        if row.get("split") not in {None, split} and row.get("split") != split:
            raise ValueError(
                f"Cached row {question_id} was in {row.get('split')} and would move to {split}. "
                "Refusing a split change that would leak historical evaluation rows."
            )
        copied = dict(row)
        copied["split"] = split
        reused.append(copied)
    return reused, missing


def assert_no_split_leakage(
    train: Sequence[str], validation: Sequence[str], test: Sequence[str]
) -> None:
    """Refuse overlapping question ids."""
    sets = {"train": set(train), "validation": set(validation), "test": set(test)}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise ValueError(f"Split leakage between {left} and {right}: {sorted(overlap)[:8]}")


def load_generation_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a row list")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _split_overlap(
    selected: SplitAssignment, known: Mapping[str, Mapping[str, object]]
) -> dict[str, list[Mapping[str, object]]]:
    grouped = {"train": [], "validation": [], "test": []}
    for split in grouped:
        for question_id in selected.ids(split):
            row = known.get(question_id)
            if row is None:
                continue
            if row.get("split") not in {None, split} and row.get("split") != split:
                raise ValueError(
                    f"Known row {question_id} would move from {row.get('split')} to {split}"
                )
            grouped[split].append(row)
    return grouped


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


def _missing_for_target(counts: Mapping[str, int], target: int) -> dict[str, int]:
    return {
        "faithful": max(0, target - int(counts.get("FAITHFUL", 0))),
        "hallucinated": max(0, target - int(counts.get("HALLUCINATED", 0))),
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


def expand_command(args: object) -> int:
    """Write an expansion plan. This does not rewrite historical artifacts."""
    from glassbox.config import repo_root
    from glassbox.data.halueval import (
        PINNED_SHA256,
        prepare_pairs,
        read_pinned_snapshot,
        snapshot_sha256,
    )
    from glassbox.evaluation.records import write_json
    from glassbox.probes.generation_eval import (
        generation_experiment_dir,
        halueval_snapshot_path,
    )

    try:
        run_id = str(args.run_id)
        if run_id == HISTORICAL_GENERATION_RUN or run_id.startswith("final-hidden-state"):
            raise ValueError(f"Refusing to reuse historical run id {run_id}")
        artifacts = repo_root() / "artifacts"
        directory = generation_experiment_dir(run_id)
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite {directory}")
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
            else (
                repo_root()
                / "artifacts"
                / "experiments"
                / HISTORICAL_GENERATION_RUN
                / "predictions.json"
            )
        )
        labeled = load_generation_rows(predictions) if predictions.is_file() else []
        plan = plan_expansion(
            prepared.pairs.values(),
            labeled,
            seed=int(args.seed),
            max_questions=int(args.max_questions),
        )
        plan["dataset_sha256"] = digest
        plan["predictions"] = str(predictions) if predictions.is_file() else None
        write_json(directory / "expansion_plan.json", plan, root=artifacts)
        print(
            "Known validation labels "
            f"{plan['known_labeled_in_selected']['validation']}; "
            f"shortfall {plan['shortfall_for_target']['validation']}"
        )
        print(f"Wrote {directory / 'expansion_plan.json'}")
        return 0
    except (OSError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Expansion plan stopped: {exc}", file=sys.stderr)
        return 1
