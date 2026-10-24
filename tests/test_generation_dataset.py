"""Generation dataset records stay unlabeled when the reference is not asserted.

These tests do not load Qwen, do not download TruthfulQA, and do not read
``artifacts/probes/final-hidden-state/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glassbox.cli import main
from glassbox.data.generation_dataset import (
    PILOT_AMBIGUOUS,
    PILOT_FAITHFUL,
    PILOT_GENERATED,
    PILOT_HALLUCINATED,
    PILOT_LABELED_BINARY,
    PILOT_UNVALIDATED,
    RUBRIC_TEXT,
    FrozenSplit,
    build_generation_dataset,
    freeze_question_split,
    load_answer_file,
    write_dataset_file,
)
from glassbox.data.schema import Label
from glassbox.data.truthfulqa import PINNED_SHA256, TruthfulItem
from glassbox.evaluation.metrics import BOOTSTRAP_MIN_PER_CLASS

REVISION = "a" * 40
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = "bfloat16"


def _item(index: int) -> TruthfulItem:
    return TruthfulItem(
        question_id=f"row{index}",
        category="Misconceptions",
        question=f"What is example question {index}?",
        correct=(f"correct reference phrase {index}",),
        incorrect=(f"incorrect reference phrase {index}",),
    )


def _catalog(*indexes: int) -> dict[str, TruthfulItem]:
    return {f"row{index}": _item(index) for index in indexes}


def _answers(indexes: tuple[int, ...], texts: dict[int, str]) -> dict[str, str]:
    return {f"row{index}": texts[index] for index in indexes}


def _build(indexes: tuple[int, ...], texts: dict[int, str]) -> dict[str, object]:
    chosen = tuple(_item(index) for index in indexes)
    split = freeze_question_split(
        chosen,
        max_questions=len(chosen),
        seed=0,
        dataset_sha256=PINNED_SHA256,
    )
    return build_generation_dataset(
        split=split,
        items={item.question_id: item for item in chosen},
        answers=_answers(indexes, texts),
        model_name=MODEL,
        model_revision=REVISION,
        dtype=DTYPE,
    )


def test_rubric_names_the_four_labels() -> None:
    for word in ("FAITHFUL", "HALLUCINATED", "AMBIGUOUS", "UNVALIDATED"):
        assert word in RUBRIC_TEXT
    assert "wording difference is not a hallucination label" in RUBRIC_TEXT
    assert "evaluation-only" in RUBRIC_TEXT


def test_split_is_stable_and_frozen_before_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    items = tuple(_item(index) for index in range(6))
    first = freeze_question_split(items, max_questions=4, seed=0, dataset_sha256=PINNED_SHA256)
    second = freeze_question_split(items, max_questions=4, seed=0, dataset_sha256=PINNED_SHA256)
    other = freeze_question_split(items, max_questions=4, seed=1, dataset_sha256=PINNED_SHA256)
    assert first.question_ids == second.question_ids
    assert first.frozen_before_generation is True
    assert first.question_ids != other.question_ids

    def forbidden(*_args, **_kwargs):
        raise AssertionError("select_items ran while labeling")

    monkeypatch.setattr("glassbox.data.generation_dataset.select_items", forbidden)
    texts = {
        question_id: f"correct reference phrase {question_id.removeprefix('row')}"
        for question_id in first.question_ids
    }
    record = build_generation_dataset(
        split=first,
        items={item.question_id: item for item in items},
        answers=texts,
        model_name=MODEL,
        model_revision=REVISION,
        dtype=DTYPE,
    )
    assert [row["question_id"] for row in record["rows"]] == list(first.question_ids)


def test_unfrozen_split_is_refused() -> None:
    split = FrozenSplit(
        seed=0,
        max_questions=1,
        dataset_sha256=PINNED_SHA256,
        question_ids=("row0",),
        frozen_before_generation=False,
    )
    with pytest.raises(ValueError, match="frozen before generation"):
        build_generation_dataset(
            split=split,
            items=_catalog(0),
            answers={"row0": "correct reference phrase 0"},
            model_name=MODEL,
            model_revision=REVISION,
            dtype=DTYPE,
        )


def test_labels_keep_unvalidated_and_ambiguous_out_of_both_classes() -> None:
    indexes = (0, 1, 2, 3)
    texts = {
        0: "correct reference phrase 0",
        1: "incorrect reference phrase 1",
        2: "correct reference phrase 2 and incorrect reference phrase 2",
        3: "The seeds leave the body without growing.",
    }
    record = _build(indexes, texts)
    by_label = {row["label"]: row for row in record["rows"]}
    assert by_label[Label.FAITHFUL.name]["reference_phrases"] == ["correct reference phrase 0"]
    assert by_label[Label.HALLUCINATED.name]["reference_phrases"] == [
        "incorrect reference phrase 1"
    ]
    assert by_label[Label.AMBIGUOUS.name]["reference_phrases"] == [
        "correct reference phrase 2",
        "incorrect reference phrase 2",
    ]
    assert by_label[Label.UNVALIDATED.name]["reference_phrases"] == []
    binary = [row for row in record["rows"] if row["in_binary_metric_set"]]
    assert {row["label"] for row in binary} == {Label.FAITHFUL.name, Label.HALLUCINATED.name}
    assert len(record["rows"]) == 4
    counts = record["class_counts"]
    assert counts["faithful"] == 1
    assert counts["hallucinated"] == 1
    assert counts["ambiguous"] == 1
    assert counts["unvalidated"] == 1
    assert counts["interval_minimum"] == 20
    assert counts["faithful_meets_interval_minimum"] is False
    assert counts["hallucinated_meets_interval_minimum"] is False
    for row in record["rows"]:
        assert row["seed"] == 0
        assert row["dataset_sha256"] == PINNED_SHA256
        assert row["model_name"] == MODEL
        assert row["model_revision"] == REVISION
        assert row["dtype"] == DTYPE


def test_pilot_stays_below_the_interval_minimum_and_pair_metrics_stay_out() -> None:
    record = _build((0,), {0: "correct reference phrase 0"})
    pilot = record["pilot"]
    assert pilot["generated"] == PILOT_GENERATED == 40
    assert pilot["faithful"] == PILOT_FAITHFUL == 3
    assert pilot["hallucinated"] == PILOT_HALLUCINATED == 2
    assert pilot["ambiguous"] == PILOT_AMBIGUOUS == 0
    assert pilot["unvalidated"] == PILOT_UNVALIDATED == 35
    assert pilot["labeled_binary"] == PILOT_LABELED_BINARY == 5
    assert pilot["interval_minimum"] == BOOTSTRAP_MIN_PER_CLASS == 20
    assert pilot["below_interval_minimum"] is True
    assert pilot["faithful_meets_interval_minimum"] is False
    assert pilot["hallucinated_meets_interval_minimum"] is False
    assert record["trains_probe"] is False
    assert record["pair_protocol_metrics_included"] is False
    encoded = json.dumps(record)
    assert "halueval-pair-002" not in encoded
    assert "0.996" not in encoded
    assert "auroc" not in encoded
    assert "probe_score" not in encoded


def test_meeting_the_count_minimum_does_not_add_detector_metrics() -> None:
    indexes = tuple(range(40))
    texts = {index: f"correct reference phrase {index}" for index in range(20)}
    texts.update({index: f"incorrect reference phrase {index}" for index in range(20, 40)})
    record = _build(indexes, texts)
    counts = record["class_counts"]
    assert counts["faithful"] == 20
    assert counts["hallucinated"] == 20
    assert counts["faithful_meets_interval_minimum"] is True
    assert counts["hallucinated_meets_interval_minimum"] is True
    assert record["pilot"]["below_interval_minimum"] is True
    assert "metrics" not in record
    assert record["trains_probe"] is False


def test_answer_loader_drops_probe_scores_and_metrics(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "metrics": {"auroc": 0.5},
                "rows": [
                    {
                        "question_id": "row0",
                        "answer": "correct reference phrase 0",
                        "probe_score": 0.99,
                        "label": "HALLUCINATED",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_answer_file(path) == {"row0": "correct reference phrase 0"}


def test_write_refuses_the_saved_probe_directory(tmp_path: Path) -> None:
    record = _build((0,), {0: "correct reference phrase 0"})
    probe = tmp_path / "probes" / "final-hidden-state"
    probe.mkdir(parents=True)
    sentinel = probe / "metadata.json"
    sentinel.write_text("keep", encoding="utf-8")
    before = sentinel.read_bytes()
    with pytest.raises(ValueError, match="pair-probe"):
        write_dataset_file(record, probe / "dataset.json", artifacts_root=tmp_path)
    previous = tmp_path / "probes" / "final-hidden-state.previous" / "dataset.json"
    with pytest.raises(ValueError, match="pair-probe"):
        write_dataset_file(record, previous, artifacts_root=tmp_path)
    assert sentinel.read_bytes() == before
    assert not previous.exists()
    destination = tmp_path / "experiments" / "truthfulqa-generation-dataset" / "dataset.json"
    write_dataset_file(record, destination, artifacts_root=tmp_path)
    assert json.loads(destination.read_text(encoding="utf-8"))["trains_probe"] is False
    with pytest.raises(ValueError, match="overwrite"):
        write_dataset_file(record, destination, artifacts_root=tmp_path)
    assert sentinel.read_bytes() == before


def test_cli_builds_a_dataset_without_loading_qwen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("model backend was constructed")

    monkeypatch.setattr("glassbox.cli.HuggingFaceBackend", forbidden)
    items = tuple(_item(index) for index in range(4))
    split = freeze_question_split(items, max_questions=4, seed=2, dataset_sha256=PINNED_SHA256)
    split_path = tmp_path / "split.json"
    write_dataset_file(split.as_dict(), split_path, artifacts_root=tmp_path)
    answers = {
        "rows": [
            {
                "question_id": question_id,
                "answer": f"correct reference phrase {question_id.removeprefix('row')}",
                "probe_score": 1.0,
            }
            for question_id in split.question_ids
        ],
        "metrics": {"auroc": 0.996},
    }
    answer_path = tmp_path / "answers.json"
    answer_path.write_text(json.dumps(answers), encoding="utf-8")
    output = tmp_path / "dataset.json"
    monkeypatch.setattr(
        "glassbox.data.generation_dataset.read_pinned_snapshot",
        lambda _path: items,
    )
    code = main(
        [
            "build-generation-dataset",
            "--split",
            str(split_path),
            "--answers",
            str(answer_path),
            "--snapshot",
            str(tmp_path / "missing.csv"),
            "--model-name",
            MODEL,
            "--model-revision",
            REVISION,
            "--dtype",
            DTYPE,
            "--output",
            str(output),
            "--artifacts-root",
            str(tmp_path),
        ]
    )
    assert code == 0
    captured = capsys.readouterr().out
    assert "No probe was trained" in captured
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["rows"][0]["label"] == Label.FAITHFUL.name
    assert "auroc" not in json.dumps(written)
    assert "probe_score" not in json.dumps(written)
