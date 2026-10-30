"""Question splits for generated-answer probes.

Membership is chosen from knowledge-passage groups before any model call.
Two questions that share a passage stay in one split, and a later subset
does not cut a group in half.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

from glassbox.data.halueval import HaluPair, SplitAssignment
from glassbox.data.splits import assert_disjoint_ids

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def knowledge_group(knowledge: str) -> str:
    """Stable id for one passage. Whitespace and case do not create a new group."""
    normalized = _NON_ALNUM.sub(" ", _WHITESPACE.sub(" ", knowledge.strip().lower())).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def assign_generation_splits(
    pairs: Iterable[HaluPair], *, seed: int
) -> tuple[SplitAssignment, dict[str, str]]:
    """Assign whole knowledge groups with the same proportions as the pair split."""
    from glassbox.data.halueval import assign_splits

    grouped: dict[str, list[HaluPair]] = {}
    group_of: dict[str, str] = {}
    for pair in pairs:
        group = knowledge_group(pair.knowledge)
        grouped.setdefault(group, []).append(pair)
        group_of[pair.question_id] = group
    if len(grouped) < 5:
        raise ValueError("Need at least five knowledge groups to fill three splits")
    proxies = [
        HaluPair(
            question_id=group,
            source_row=members[0].source_row,
            question=members[0].question,
            knowledge=members[0].knowledge,
            right_answer=members[0].right_answer,
            hallucinated_answer=members[0].hallucinated_answer,
        )
        for group, members in grouped.items()
    ]
    group_assignment = assign_splits(proxies, seed=seed)
    return _flatten(group_assignment, grouped), group_of


def subset_preserving_groups(
    assignment: SplitAssignment,
    group_of: dict[str, str],
    max_questions: int,
) -> SplitAssignment:
    """Take a prefix of each split without separating a knowledge group."""
    if isinstance(max_questions, bool) or not isinstance(max_questions, int) or max_questions < 3:
        raise ValueError("max_questions must be an integer covering train, validation, and test")
    n_test = max(1, int(round(max_questions * 0.2)))
    n_val = max(1, int(round(max_questions * 0.2)))
    n_train = max_questions - n_test - n_val
    if n_train < 1:
        raise ValueError("max_questions leaves no training questions")
    return SplitAssignment(
        train=_take(assignment.train, group_of, n_train),
        validation=_take(assignment.validation, group_of, n_val),
        test=_take(assignment.test, group_of, n_test),
        seed=assignment.seed,
    )


def assert_group_isolation(assignment: SplitAssignment, group_of: dict[str, str]) -> None:
    """Raise when a question or a knowledge group crosses splits."""
    assert_disjoint_ids(assignment.train, assignment.validation, assignment.test)
    located: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for question_id in assignment.ids(split):
            group = group_of[question_id]
            previous = located.get(group)
            if previous is not None and previous != split:
                raise ValueError(f"Knowledge group {group} is in {previous} and {split}")
            located[group] = split


def question_ids(assignment: SplitAssignment) -> dict[str, tuple[str, ...]]:
    return {
        "train": assignment.train,
        "validation": assignment.validation,
        "test": assignment.test,
    }


def _flatten(
    group_assignment: SplitAssignment, grouped: dict[str, list[HaluPair]]
) -> SplitAssignment:
    def questions(group_ids: Sequence[str]) -> tuple[str, ...]:
        ordered: list[str] = []
        for group in group_ids:
            members = sorted(grouped[group], key=lambda pair: pair.question_id)
            ordered.extend(pair.question_id for pair in members)
        return tuple(ordered)

    assignment = SplitAssignment(
        train=questions(group_assignment.train),
        validation=questions(group_assignment.validation),
        test=questions(group_assignment.test),
        seed=group_assignment.seed,
    )
    assert_disjoint_ids(assignment.train, assignment.validation, assignment.test)
    return assignment


def _take(ids: tuple[str, ...], group_of: dict[str, str], limit: int) -> tuple[str, ...]:
    chosen: list[str] = []
    index = 0
    while index < len(ids):
        group = group_of[ids[index]]
        block: list[str] = []
        while index < len(ids) and group_of[ids[index]] == group:
            block.append(ids[index])
            index += 1
        if chosen and len(chosen) + len(block) > limit:
            break
        if not chosen and len(block) > limit:
            raise ValueError(
                "A knowledge group is larger than the requested split prefix. "
                "Refusing to divide it."
            )
        chosen.extend(block)
        if len(chosen) >= limit:
            break
    if not chosen:
        raise ValueError("The split prefix is empty")
    return tuple(chosen)
