"""HaluEval QA pair records for the hidden-state probe.

The labels are the benchmark fields ``right_answer`` and ``hallucinated_answer``.
They are not a new judgment of a Qwen generation. Filtering below removes pairs
this audit could not treat as a faithful/hallucinated contrast. Rows that fail
a filter stay out of every split. They are not moved into training.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from glassbox.data.schema import Label, make_example_id
from glassbox.data.splits import assert_disjoint_ids
from glassbox.guards import reject_download

PINNED_COMMIT = "207f479c555a68b8d2431631a95083b84b2accbe"
PINNED_SHA256 = "89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/RUCAIBox/HaluEval/{PINNED_COMMIT}/data/qa_data.json"
)
DATASET_ID = "halueval_qa"
LABEL_SOURCE = "halueval_qa_pair_v1"
_FIELDS = ("knowledge", "question", "right_answer", "hallucinated_answer")
_YES_NO = {"yes", "no"}
_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class HaluPair:
    """One question whose two supplied answers stay in the same split."""

    question_id: str
    source_row: int
    question: str
    knowledge: str
    right_answer: str
    hallucinated_answer: str

    @property
    def example_ids(self) -> tuple[str, str]:
        return (
            make_example_id(DATASET_ID, f"{self.question_id}.faithful"),
            make_example_id(DATASET_ID, f"{self.question_id}.hallucinated"),
        )


@dataclass(frozen=True)
class FilterReport:
    """Counts for every row the snapshot did not keep."""

    source_rows: int
    kept: int
    dropped: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_rows": self.source_rows,
            "kept": self.kept,
            "dropped": dict(self.dropped),
        }


@dataclass(frozen=True)
class SplitAssignment:
    """Question ids assigned before any hidden state is computed."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int

    def ids(self, split: str) -> tuple[str, ...]:
        return {"train": self.train, "validation": self.validation, "test": self.test}[split]

    def subset(self, max_questions: int) -> SplitAssignment:
        """Take a prefix of each split. Prefixes do not change split membership."""
        if isinstance(max_questions, bool) or not isinstance(max_questions, int):
            raise ValueError("max_questions must be an integer")
        if max_questions < 3:
            raise ValueError("max_questions must cover train, validation, and test")
        n_test = max(1, int(round(max_questions * 0.2)))
        n_val = max(1, int(round(max_questions * 0.2)))
        n_train = max_questions - n_test - n_val
        if n_train < 1:
            raise ValueError("max_questions leaves no training questions")
        if n_train > len(self.train) or n_val > len(self.validation) or n_test > len(self.test):
            raise ValueError(
                "max_questions asks for more questions than the filtered split contains: "
                f"train={len(self.train)}, validation={len(self.validation)}, test={len(self.test)}"
            )
        return SplitAssignment(
            train=self.train[:n_train],
            validation=self.validation[:n_val],
            test=self.test[:n_test],
            seed=self.seed,
        )


@dataclass
class PreparedPairs:
    """Filtered pairs plus the split that was fixed before extraction."""

    pairs: dict[str, HaluPair]
    assignment: SplitAssignment
    report: FilterReport
    sha256: str
    source_commit: str = PINNED_COMMIT
    audit_notes: list[str] = field(default_factory=list)


def snapshot_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pinned_snapshot(path: Path) -> list[dict[str, str]]:
    """Read the pinned JSONL file and refuse a different hash."""
    digest = snapshot_sha256(path)
    if digest != PINNED_SHA256:
        raise ValueError(
            f"HaluEval snapshot hash {digest} does not match pinned {PINNED_SHA256}. "
            "Refusing to train on an unaudited file."
        )
    return read_jsonl(path)


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Line {line_number} is not a JSON object")
        row = {key: payload.get(key) for key in _FIELDS}
        if any(not isinstance(row[key], str) for key in _FIELDS):
            raise ValueError(f"Line {line_number} is missing a string field")
        rows.append({key: str(row[key]) for key in _FIELDS})
    return rows


def download_snapshot(path: Path, *, allow_download: bool) -> Path:
    """Fetch the pinned QA file. Tests leave ``allow_download`` false."""
    reject_download(allow_download, resource="HaluEval QA snapshot")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SOURCE_URL, destination)  # noqa: S310
    read_pinned_snapshot(destination)
    return destination


def filter_pairs(rows: Iterable[Mapping[str, str]]) -> tuple[list[HaluPair], FilterReport]:
    """Drop pairs that this audit cannot label, without reassigning them."""
    dropped: dict[str, int] = {}
    seen_questions: set[str] = set()
    kept: list[HaluPair] = []
    source_rows = 0
    for source_row, row in enumerate(rows):
        source_rows += 1
        reason = _drop_reason(row, seen_questions)
        if reason is not None:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        question_key = _normalize(str(row["question"]))
        seen_questions.add(question_key)
        kept.append(
            HaluPair(
                question_id=f"row{source_row}",
                source_row=source_row,
                question=str(row["question"]).strip(),
                knowledge=str(row["knowledge"]).strip(),
                right_answer=str(row["right_answer"]).strip(),
                hallucinated_answer=str(row["hallucinated_answer"]).strip(),
            )
        )
    report = FilterReport(source_rows=source_rows, kept=len(kept), dropped=dropped)
    return kept, report


def assign_splits(pairs: Iterable[HaluPair], *, seed: int) -> SplitAssignment:
    """Assign question ids with a fixed seed before any model call."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    ids = tuple(pair.question_id for pair in pairs)
    if len(ids) != len(set(ids)):
        raise ValueError("question ids must be unique before the split")
    if len(ids) < 5:
        raise ValueError("Need at least five filtered questions to fill three splits")
    order = list(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    n_test = max(1, int(round(len(order) * 0.2)))
    n_val = max(1, int(round(len(order) * 0.2)))
    test = tuple(order[:n_test])
    validation = tuple(order[n_test : n_test + n_val])
    train = tuple(order[n_test + n_val :])
    if not train:
        raise ValueError("The split left the training set empty")
    assert_disjoint_ids(train, validation, test)
    return SplitAssignment(train=train, validation=validation, test=test, seed=seed)


def prepare_pairs(rows: Iterable[Mapping[str, str]], *, seed: int) -> PreparedPairs:
    pairs, report = filter_pairs(rows)
    assignment = assign_splits(pairs, seed=seed)
    return PreparedPairs(
        pairs={pair.question_id: pair for pair in pairs},
        assignment=assignment,
        report=report,
        sha256=PINNED_SHA256,
        audit_notes=list(AUDIT_NOTES),
    )


def pair_label(which: str) -> Label:
    if which == "faithful":
        return Label.FAITHFUL
    if which == "hallucinated":
        return Label.HALLUCINATED
    raise ValueError("which must be 'faithful' or 'hallucinated'")


def _drop_reason(row: Mapping[str, str], seen_questions: set[str]) -> str | None:
    if any(key not in row or not isinstance(row[key], str) for key in _FIELDS):
        return "missing_field"
    question = _normalize(str(row["question"]))
    right = _normalize(str(row["right_answer"]))
    hallucinated = _normalize(str(row["hallucinated_answer"]))
    knowledge = str(row["knowledge"]).strip()
    if not question or not right or not hallucinated or not knowledge.strip():
        return "empty_text"
    if len(question) > 400 or len(right) > 300 or len(hallucinated) > 300:
        return "text_too_long"
    if right == hallucinated:
        return "identical_answers"
    if right in _YES_NO:
        return "yes_no_answer"
    if right in hallucinated or hallucinated in right:
        return "answer_contains_the_other"
    if _same_numbers(right, hallucinated):
        return "numeric_restatement"
    if question in seen_questions:
        return "duplicate_question"
    return None


def _same_numbers(right: str, hallucinated: str) -> bool:
    """Drop a pair whose only numbers are the same.

    A manual read found gold ``9 January 1982`` paired with ``born in 1982``.
    That hallucinated field restates the year instead of contradicting it.
    """
    right_numbers = set(re.findall(r"\d+", right))
    hall_numbers = set(re.findall(r"\d+", hallucinated))
    return bool(hall_numbers) and hall_numbers <= right_numbers


def _normalize(text: str) -> str:
    lowered = text.strip().lower()
    return _NON_ALNUM.sub(" ", _WHITESPACE.sub(" ", lowered)).strip()


AUDIT_NOTES = [
    "Source: HaluEval QA (Li et al., EMNLP 2023), file data/qa_data.json at commit "
    f"{PINNED_COMMIT}, sha256 {PINNED_SHA256}. 10,000 JSONL rows with keys "
    "knowledge, question, right_answer, hallucinated_answer.",
    "right_answer is the HotpotQA answer. hallucinated_answer is the benchmark's generated answer. "
    "A manual read of 12 unfiltered rows (seed 0) found gold facts repeated inside the hallucinated "
    "field and a yes/no answer paraphrased, so those patterns are filtered. "
    "A later read of filtered rows found a gold date restated as the same year, so identical number "
    "sets are filtered too. Remaining rows are still benchmark labels, not a full re-annotation.",
    "The model prompt is the demo question prompt plus the supplied answer. The knowledge "
    "passage is stored for audit and is not fed to the model, because the demo does not have a passage.",
    "This labels supplied answer strings. It does not label a new Qwen generation.",
]
