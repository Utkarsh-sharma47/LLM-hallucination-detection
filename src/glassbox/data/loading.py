"""Local dataset entry points.

``load_examples`` reads a pinned file that is already on disk. It does not
download, and it does not import the ``datasets`` package. A missing file with
downloads disabled raises ``DownloadDisabledError``. A missing file with
downloads requested still raises ``NotImplementedError`` because this function
is not a download client.
"""

from __future__ import annotations

from pathlib import Path

from glassbox.config import repo_root
from glassbox.constants import DATASET_IDS
from glassbox.data.halueval import LABEL_SOURCE, prepare_pairs, read_pinned_snapshot
from glassbox.data.schema import Label, QAExample, make_example_id
from glassbox.data.truthfulqa import read_pinned_snapshot as read_truthfulqa
from glassbox.guards import reject_download


def default_snapshot(dataset_id: str) -> Path:
    """Return the gitignored path used for one pinned dataset."""
    root = repo_root() / "artifacts" / "datasets"
    if dataset_id == "halueval_qa":
        return root / "halueval" / "qa_data.json"
    if dataset_id == "truthfulqa":
        return root / "truthfulqa" / "TruthfulQA.csv"
    known = ", ".join(DATASET_IDS)
    raise ValueError(f"Unknown dataset {dataset_id!r}. Expected one of {known}.")


def load_examples(
    dataset_id: str,
    *,
    allow_download: bool = False,
    snapshot: Path | None = None,
) -> list[QAExample]:
    """Load audited examples from a local pinned snapshot.

    HaluEval rows use the pair-protocol labels. Those labels discriminate two
    supplied answers. They are not labels of a new Qwen generation. TruthfulQA
    rows are reference questions with ``UNVALIDATED`` labels. The CSV does not
    contain a model answer, so this loader does not invent one.
    """
    if dataset_id not in DATASET_IDS:
        known = ", ".join(DATASET_IDS)
        raise ValueError(f"Unknown dataset {dataset_id!r}. Expected one of {known}.")
    path = Path(snapshot) if snapshot is not None else default_snapshot(dataset_id)
    if not path.is_file():
        reject_download(allow_download, resource=f"dataset {dataset_id}")
        raise NotImplementedError(
            f"Downloading {dataset_id} is not implemented in load_examples. "
            "Place the pinned file locally. This function does not import the datasets package."
        )
    if dataset_id == "halueval_qa":
        return _halueval_examples(path)
    return _truthfulqa_examples(path)


def _halueval_examples(path: Path) -> list[QAExample]:
    prepared = prepare_pairs(read_pinned_snapshot(path), seed=0)
    examples: list[QAExample] = []
    for question_id, pair in prepared.pairs.items():
        split = _split_for(question_id, prepared.assignment)
        faithful_id, hallucinated_id = pair.example_ids
        examples.append(
            QAExample(
                example_id=faithful_id,
                dataset="halueval_qa",
                question=pair.question,
                reference_answer=pair.right_answer,
                model_answer=pair.right_answer,
                label=Label.FAITHFUL,
                label_source=LABEL_SOURCE,
                split=split,
            )
        )
        examples.append(
            QAExample(
                example_id=hallucinated_id,
                dataset="halueval_qa",
                question=pair.question,
                reference_answer=pair.right_answer,
                model_answer=pair.hallucinated_answer,
                label=Label.HALLUCINATED,
                label_source=LABEL_SOURCE,
                split=split,
            )
        )
    return examples


def _truthfulqa_examples(path: Path) -> list[QAExample]:
    items = read_truthfulqa(path)
    return [
        QAExample(
            example_id=make_example_id("truthfulqa", item.question_id),
            dataset="truthfulqa",
            question=item.question,
            reference_answer=item.correct[0],
            model_answer=None,
            label=Label.UNVALIDATED,
            label_source="truthfulqa_reference_only_v1",
            split="reference",
        )
        for item in items
    ]


def _split_for(question_id: str, assignment) -> str:
    if question_id in assignment.train:
        return "train"
    if question_id in assignment.validation:
        return "validation"
    if question_id in assignment.test:
        return "test"
    raise ValueError(f"Question {question_id} is missing from the split assignment")
