"""Append-only human review history for generated-answer labels.

Reviews do not invent binary labels. An unvalidated automatic label stays
unvalidated until a reviewer assigns a different label and cites evidence.
Hallucination here is a factual contradiction or an unsupported factual
assertion under the documented protocol, not a wording mismatch.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from glassbox.data.evidence import HUMAN_REVIEW_SOURCE, LABEL_POLICY
from glassbox.data.schema import Label

REVIEW_PROTOCOL = (
    "Hallucinated means the generated answer asserts a fact that contradicts the "
    "trusted passage or asserts an unsupported factual claim under the generation "
    "evidence protocol. A wording difference from a reference string is not a "
    "hallucination. Unvalidated rows are not converted into FAITHFUL or "
    "HALLUCINATED without an explicit review and evidence note."
)
MIN_EVIDENCE_CHARS = 20
ALLOWED_LABELS = frozenset(item.name for item in Label)
HISTORICAL_QUEUE_MARKERS = (
    "halueval-generation-001/review_queue.json",
    "halueval-generation-001/report.json",
    "halueval-generation-001/predictions.json",
)


@dataclass(frozen=True)
class ReviewEvent:
    """One review action. Later events for the same question are revisions."""

    question_id: str
    label: Label
    reviewer: str
    evidence_note: str
    timestamp: str
    dataset_id: str
    source_example_id: str
    previous_label: str | None = None
    revision: int = 1

    def as_dict(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "label": self.label.name,
            "reviewer": self.reviewer,
            "evidence_note": self.evidence_note,
            "timestamp": self.timestamp,
            "dataset_id": self.dataset_id,
            "source_example_id": self.source_example_id,
            "previous_label": self.previous_label,
            "revision": self.revision,
            "label_source": HUMAN_REVIEW_SOURCE,
        }


def parse_review_event(payload: Mapping[str, object], *, line_number: int = 1) -> ReviewEvent:
    """Validate one review object. Missing required fields are refused."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"Review line {line_number} is not an object")
    question_id = _required_text(payload.get("question_id"), "question_id", line_number)
    reviewer = _required_text(payload.get("reviewer"), "reviewer", line_number)
    note = payload.get("evidence_note")
    if not isinstance(note, str) or len(note.strip()) < MIN_EVIDENCE_CHARS:
        raise ValueError(
            f"Review line {line_number} needs an evidence_note of at least "
            f"{MIN_EVIDENCE_CHARS} characters. Uncertain rows stay unvalidated."
        )
    label_name = payload.get("label")
    if not isinstance(label_name, str) or label_name not in ALLOWED_LABELS:
        raise ValueError(
            f"Review line {line_number} has an invalid label {label_name!r}. "
            f"Allowed labels: {', '.join(sorted(ALLOWED_LABELS))}."
        )
    dataset_id = _required_text(payload.get("dataset_id"), "dataset_id", line_number)
    source = _required_text(payload.get("source_example_id"), "source_example_id", line_number)
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp.strip():
        timestamp = utc_now()
    previous = payload.get("previous_label")
    if previous is not None and (not isinstance(previous, str) or previous not in ALLOWED_LABELS):
        raise ValueError(f"Review line {line_number} has an invalid previous_label")
    revision = payload.get("revision", 1)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError(f"Review line {line_number} has an invalid revision")
    return ReviewEvent(
        question_id=question_id,
        label=Label[label_name],
        reviewer=reviewer,
        evidence_note=note.strip(),
        timestamp=timestamp.strip(),
        dataset_id=dataset_id,
        source_example_id=source,
        previous_label=previous,
        revision=revision,
    )


def load_history(path: Path) -> list[ReviewEvent]:
    """Read an append-only JSONL history. Empty files are allowed."""
    location = Path(path)
    if not location.is_file():
        return []
    events: list[ReviewEvent] = []
    for line_number, line in enumerate(location.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        events.append(parse_review_event(payload, line_number=line_number))
    return events


def current_reviews(events: Sequence[ReviewEvent]) -> dict[str, ReviewEvent]:
    """Latest review per question. Earlier events remain in the history."""
    current: dict[str, ReviewEvent] = {}
    for event in events:
        current[event.question_id] = event
    return current


def append_reviews(
    path: Path,
    payloads: Sequence[Mapping[str, object]],
    *,
    existing: Sequence[ReviewEvent] | None = None,
) -> list[ReviewEvent]:
    """Append validated events. Automatic unvalidated labels are not rewritten."""
    location = Path(path)
    _refuse_historical_artifact(location)
    history = list(existing) if existing is not None else load_history(location)
    current = current_reviews(history)
    added: list[ReviewEvent] = []
    for index, payload in enumerate(payloads, start=1):
        event = parse_review_event(payload, line_number=index)
        previous = current.get(event.question_id)
        if previous is not None:
            expected_previous = previous.label.name
            if event.previous_label is None:
                event = ReviewEvent(
                    question_id=event.question_id,
                    label=event.label,
                    reviewer=event.reviewer,
                    evidence_note=event.evidence_note,
                    timestamp=event.timestamp,
                    dataset_id=event.dataset_id,
                    source_example_id=event.source_example_id,
                    previous_label=expected_previous,
                    revision=previous.revision + 1,
                )
            elif event.previous_label != expected_previous:
                raise ValueError(
                    f"Review of {event.question_id} expected previous_label "
                    f"{expected_previous}, got {event.previous_label}."
                )
            elif event.revision <= previous.revision:
                event = ReviewEvent(
                    question_id=event.question_id,
                    label=event.label,
                    reviewer=event.reviewer,
                    evidence_note=event.evidence_note,
                    timestamp=event.timestamp,
                    dataset_id=event.dataset_id,
                    source_example_id=event.source_example_id,
                    previous_label=event.previous_label,
                    revision=previous.revision + 1,
                )
        history.append(event)
        current[event.question_id] = event
        added.append(event)
    if added:
        location.parent.mkdir(parents=True, exist_ok=True)
        with location.open("a", encoding="utf-8") as handle:
            for event in added:
                handle.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")
    return added


def export_snapshot(events: Sequence[ReviewEvent]) -> list[dict[str, object]]:
    """One current review per question for ``--reviews`` on training."""
    snapshot = []
    for event in current_reviews(events).values():
        snapshot.append(
            {
                "question_id": event.question_id,
                "label": event.label.name,
                "reviewer": event.reviewer,
                "evidence_note": event.evidence_note,
            }
        )
    return snapshot


def display_queue_item(row: Mapping[str, object]) -> str:
    """Show the evidence an annotator needs. This does not assign a label."""
    decision = row.get("decision") if isinstance(row.get("decision"), Mapping) else {}
    automatic = row.get("label_name") or decision.get("automatic_label") or "UNVALIDATED"
    lines = [
        f"QUESTION ID {row.get('question_id')}",
        f"SPLIT {row.get('split')}",
        f"QUESTION {row.get('question')}",
        f"PASSAGE {row.get('knowledge')}",
        f"GENERATED ANSWER {row.get('answer')}",
        f"RIGHT ANSWER {row.get('right_answer')}",
        f"BENCHMARK FALSE ANSWER {row.get('hallucinated_answer')}",
        f"AUTOMATIC LABEL {automatic}",
        f"AUTOMATIC DETAIL {decision.get('detail') or row.get('exclusion')}",
        f"PROTOCOL {REVIEW_PROTOCOL}",
        f"LABEL POLICY {LABEL_POLICY}",
        "Assign FAITHFUL, HALLUCINATED, AMBIGUOUS, or UNVALIDATED with an evidence note.",
        "Do not convert UNVALIDATED to a binary label without citing passage evidence.",
    ]
    return "\n".join(lines)


def unreviewed_rows(
    rows: Sequence[Mapping[str, object]], events: Sequence[ReviewEvent]
) -> list[Mapping[str, object]]:
    """Queue items that still have only an automatic label."""
    reviewed = set(current_reviews(events))
    return [row for row in rows if str(row.get("question_id")) not in reviewed]


def apply_current_labels(
    rows: Sequence[Mapping[str, object]], events: Sequence[ReviewEvent]
) -> list[dict[str, object]]:
    """Attach current human labels without inventing missing reviews."""
    current = current_reviews(events)
    result = []
    for row in rows:
        payload = dict(row)
        automatic = str(row.get("label_name") or "UNVALIDATED")
        payload["automatic_label"] = automatic
        event = current.get(str(row.get("question_id")))
        if event is None:
            payload["human_reviewed"] = False
            payload["current_label"] = automatic
        else:
            payload["human_reviewed"] = True
            payload["current_label"] = event.label.name
            payload["reviewer"] = event.reviewer
            payload["review_revision"] = event.revision
            payload["label_source"] = HUMAN_REVIEW_SOURCE
        result.append(payload)
    return result


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _required_text(value: object, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Review line {line_number} is missing {field}")
    return value.strip()


def _refuse_historical_artifact(path: Path) -> None:
    resolved = str(Path(path).resolve())
    for marker in HISTORICAL_QUEUE_MARKERS:
        if resolved.endswith(marker):
            raise ValueError(
                f"Refusing to write review history into historical artifact {path}. "
                "Keep the generation-001 queue unchanged and store a new JSONL history."
            )
