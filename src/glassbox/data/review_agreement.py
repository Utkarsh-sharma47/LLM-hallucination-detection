"""Inter-rater agreement for adjudicated review (issue #41 helpers).

Computes observed agreement and Cohen's kappa from *already collected*
independent judgments. This invents no labels: with fewer than two
independent judgments on an overlap subset, agreement is unavailable.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from glassbox.data.review_log import ReviewEvent


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float | None:
    """Cohen's kappa for two aligned label sequences, or None if undefined."""
    if len(first) != len(second) or not first:
        return None
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / n
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        (first_counts[label] / n) * (second_counts[label] / n)
        for label in set(first_counts) | set(second_counts)
    )
    if expected >= 1.0:
        # Single unanimous label: kappa undefined (0/0), report as None.
        return None
    return (observed - expected) / (1.0 - expected)


def agreement_report(first: Sequence[str], second: Sequence[str]) -> dict[str, object]:
    """Agreement summary for a pre-registered overlap subset."""
    if len(first) != len(second):
        raise ValueError("aligned label sequences must have the same length")
    n = len(first)
    if n == 0:
        return {
            "n_overlap": 0,
            "observed_agreement": None,
            "cohen_kappa": None,
            "disagreements": 0,
            "note": "No overlap subset; agreement unavailable.",
        }
    disagreements = sum(a != b for a, b in zip(first, second, strict=True))
    observed = (n - disagreements) / n
    return {
        "n_overlap": n,
        "observed_agreement": observed,
        "cohen_kappa": cohen_kappa(first, second),
        "disagreements": disagreements,
        "per_class_counts_first": dict(Counter(first)),
        "per_class_counts_second": dict(Counter(second)),
        "note": (
            "Agreement describes reviewer consistency, not truth. "
            "Adjudicated labels must still cite passage evidence."
        ),
    }


def agreement_from_events(
    events: Sequence[ReviewEvent],
) -> dict[str, object]:
    """Overlap agreement from review events (first two distinct reviewers).

    Groups by question_id, keeps the first judgment of each distinct
    reviewer, and compares the first two reviewers on shared questions.
    Original events are never modified.
    """
    by_question: dict[str, dict[str, str]] = {}
    for event in events:
        reviewers = by_question.setdefault(event.question_id, {})
        reviewers.setdefault(event.reviewer, event.label.name)
    first_labels: list[str] = []
    second_labels: list[str] = []
    reviewer_pairs: Counter[str] = Counter()
    for reviewers in by_question.values():
        if len(reviewers) < 2:
            continue
        items = list(reviewers.items())
        (rev_a, lab_a), (rev_b, lab_b) = items[0], items[1]
        first_labels.append(lab_a)
        second_labels.append(lab_b)
        reviewer_pairs[f"{rev_a} vs {rev_b}"] += 1
    report = agreement_report(first_labels, second_labels)
    report["reviewer_pairs"] = dict(reviewer_pairs)
    report["n_questions_with_two_reviewers"] = len(first_labels)
    report["n_questions_total"] = len(by_question)
    return report


def unresolved_questions(events: Sequence[ReviewEvent]) -> list[str]:
    """Questions with 2+ distinct reviewers who disagree (need adjudication)."""
    by_question: dict[str, set[str]] = {}
    for event in events:
        by_question.setdefault(event.question_id, set()).add(event.label.name)
    return sorted(q for q, labels in by_question.items() if len(labels) > 1)


def label_counts(events: Sequence[ReviewEvent]) -> dict[str, int]:
    """Current-label counts (latest event per question)."""
    latest: dict[str, str] = {}
    for event in events:
        latest[event.question_id] = event.label.name
    return dict(Counter(latest.values()))


def summary_from_events(events: Sequence[ReviewEvent]) -> dict[str, object]:
    """Full annotation summary: counts, agreement, unresolved (no invention)."""
    agreement = agreement_from_events(events)
    return {
        "n_events": len(events),
        "n_questions": len({e.question_id for e in events}),
        "current_label_counts": label_counts(events),
        "agreement": agreement,
        "unresolved_disagreements": unresolved_questions(events),
        "adjudicated": False,
        "note": (
            "This summary never invents binary labels. Questions without "
            "independent review stay UNVALIDATED until reviewed with evidence."
        ),
    }
