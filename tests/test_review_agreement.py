"""Agreement helpers for issue #41. Synthetic data only; no invented labels."""

from __future__ import annotations

from glassbox.data.review_agreement import (
    agreement_from_events,
    agreement_report,
    cohen_kappa,
    label_counts,
    summary_from_events,
    unresolved_questions,
)
from glassbox.data.review_log import ReviewEvent
from glassbox.data.schema import Label


def _event(qid: str, label: Label, reviewer: str, revision: int = 1) -> ReviewEvent:
    return ReviewEvent(
        question_id=qid,
        label=label,
        reviewer=reviewer,
        evidence_note="passage quote supports this label for testing purposes",
        timestamp="2026-11-06T00:00:00+00:00",
        dataset_id="test",
        source_example_id=qid,
        revision=revision,
    )


def test_kappa_perfect_and_chance() -> None:
    assert cohen_kappa(["A", "B"], ["A", "B"]) == 1.0
    assert cohen_kappa(["A", "A"], ["A", "A"]) is None  # unanimous: undefined
    assert cohen_kappa([], []) is None


def test_agreement_report_counts() -> None:
    report = agreement_report(["F", "H", "F"], ["F", "F", "F"])
    assert report["n_overlap"] == 3
    assert report["disagreements"] == 1
    assert report["observed_agreement"] == 2 / 3


def test_agreement_from_events_uses_first_two_reviewers() -> None:
    events = [
        _event("q1", Label.FAITHFUL, "alice"),
        _event("q1", Label.FAITHFUL, "bob"),
        _event("q2", Label.HALLUCINATED, "alice"),
        _event("q2", Label.FAITHFUL, "bob"),
        _event("q3", Label.UNVALIDATED, "alice"),  # single reviewer: not overlap
    ]
    report = agreement_from_events(events)
    assert report["n_overlap"] == 2
    assert report["disagreements"] == 1
    assert unresolved_questions(events) == ["q2"]


def test_summary_never_adjudicates() -> None:
    events = [_event("q1", Label.FAITHFUL, "alice")]
    summary = summary_from_events(events)
    assert summary["adjudicated"] is False
    assert label_counts(events) == {"FAITHFUL": 1}
    assert summary["agreement"]["n_overlap"] == 0
