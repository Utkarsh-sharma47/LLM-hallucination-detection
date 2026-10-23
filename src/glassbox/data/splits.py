"""Split checks used to keep train and evaluation examples apart."""

from __future__ import annotations

from collections.abc import Iterable


def assert_disjoint_ids(*splits: Iterable[str]) -> None:
    """Raise when an example id repeats within a split or across splits.

    Call this with ids from ``make_example_id`` after the train, validation,
    and test membership has been chosen and before any probe is fit.
    """
    if len(splits) < 2:
        raise ValueError("Provide at least two splits")
    seen: dict[str, int] = {}
    for split_index, split in enumerate(splits):
        split_ids = list(split)
        if len(split_ids) != len(set(split_ids)):
            raise ValueError(f"Split {split_index} contains duplicate ids")
        for example_id in split_ids:
            if not isinstance(example_id, str) or not example_id:
                raise ValueError("Example ids must be non-empty strings")
            previous = seen.get(example_id)
            if previous is not None:
                raise ValueError(
                    f"Example id {example_id!r} is in split {previous} and split {split_index}"
                )
            seen[example_id] = split_index
