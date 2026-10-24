"""Generation transfer of the saved pair probe. These tests do not load Qwen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from glassbox.data.generation_dataset import FrozenSplit
from glassbox.data.schema import Label
from glassbox.data.truthfulqa import PINNED_SHA256, TruthfulItem
from glassbox.evaluation.probe_transfer import (
    BASELINE_THRESHOLD,
    PLANNED_MAX_QUESTIONS,
    RISK_STATUS,
    SAVED_PAIR_THRESHOLD,
    SCORE_NAME,
    build_transfer,
    length_score,
    publish_transfer,
    question_overlaps,
    reference_overlap_score,
    reference_tfidf_scores,
    require_pair_probe,
    score_feature,
    training_question_texts,
)
from glassbox.probes.logistic import LogisticProbe
from glassbox.probes.scoring import CALIBRATION_ECE_LIMIT
from glassbox.probes.store import ProbeCompatibilityError, load_probe, save_probe

_COMMIT = "a" * 40
_REVISION = "b" * 40
_PROBE_REVISION = "c" * 40


def test_saved_threshold_is_the_halueval_value_and_a_refit_is_refused() -> None:
    assert SAVED_PAIR_THRESHOLD == 0.65
    assert BASELINE_THRESHOLD == 0.5
    assert PLANNED_MAX_QUESTIONS == 200
    require_pair_probe(_metadata())
    with pytest.raises(ValueError, match="0.65"):
        require_pair_probe(_metadata(threshold=0.25))


def test_length_and_reference_baselines_are_not_fit_on_labels() -> None:
    correct = ("alpha bravo charlie delta",)
    incorrect = ("echo foxtrot golf hotel",)
    faithful = "Alpha bravo charlie delta."
    hallucinated = "Echo foxtrot golf hotel."
    assert length_score("one") < length_score("one two three four five six seven eight nine")
    assert reference_overlap_score(faithful, correct, incorrect) < reference_overlap_score(
        hallucinated, correct, incorrect
    )
    scores = reference_tfidf_scores(
        (
            ("faithful", faithful, correct, incorrect),
            ("hallucinated", hallucinated, correct, incorrect),
        )
    )
    assert scores["faithful"] < scores["hallucinated"]


def test_question_text_overlap_is_excluded_and_ids_are_not_treated_as_a_match() -> None:
    leaked = _item("leaked", "Which river is the longest?", "alpha bravo charlie delta")
    same_id = _item("row1868", "What color is a zebra?", "alpha bravo charlie delta")
    overlaps = question_overlaps(
        (leaked, same_id),
        {"train": ("Which river is the longest?",), "validation": (), "test": ()},
    )
    assert overlaps["leaked"] == ("train",)
    assert overlaps["row1868"] == ()


def test_missing_training_question_fails_closed() -> None:
    pair = type("Pair", (), {"question_id": "row1", "question": "Known question"})()
    with pytest.raises(ValueError, match="row9"):
        training_question_texts(
            [pair],
            {"train": ["row9"], "validation": ["row1"], "test": ["row1"]},
        )


def test_incompatible_feature_raises_and_does_not_invent_a_score(tmp_path: Path) -> None:
    directory = tmp_path / "probe"
    _save(directory)
    before = hashlib.sha256((directory / "metadata.json").read_bytes()).hexdigest()
    loaded = load_probe(directory)
    live = _live(feature_dim=3)
    with pytest.raises(ProbeCompatibilityError):
        score_feature(loaded, np.zeros(3), live=live)
    assert hashlib.sha256((directory / "metadata.json").read_bytes()).hexdigest() == before
    assert not (tmp_path / "experiments").exists()


def test_publish_keeps_every_label_and_drops_ineligible_rows_from_binary_metrics(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifacts" / "experiments" / "transfer-unit"
    metrics = publish_transfer(**_publish_kwargs(destination, tmp_path / "artifacts"))
    record = json.loads((destination / "record.json").read_text(encoding="utf-8"))
    labels = {row["question_id"]: row["label"] for row in record["rows"]}
    assert labels == {
        "faithful-low": Label.FAITHFUL.name,
        "faithful-high": Label.FAITHFUL.name,
        "hallucinated-high": Label.HALLUCINATED.name,
        "hallucinated-low": Label.HALLUCINATED.name,
        "paraphrase": Label.UNVALIDATED.name,
        "both": Label.AMBIGUOUS.name,
        "leaked": Label.FAITHFUL.name,
        "missing": Label.FAITHFUL.name,
    }
    assert len(record["rows"]) == 8
    assert record["calibration_validated"] is False
    assert _keys(record).isdisjoint({"auroc", "precision", "recall", "f1", "ece", "threshold"})
    counts = metrics["sample_counts"]
    assert counts["generated"] == 8
    assert counts["faithful"] == 4
    assert counts["hallucinated"] == 2
    assert counts["ambiguous"] == 1
    assert counts["unvalidated"] == 1
    assert counts["binary_eligible"] == 4
    assert counts["eligible_faithful"] == 2
    assert counts["eligible_hallucinated"] == 2
    by_id = {row["question_id"]: row for row in metrics["rows"]}
    assert by_id["paraphrase"]["in_binary_evaluation"] is False
    assert by_id["both"]["in_binary_evaluation"] is False
    assert by_id["leaked"]["in_binary_evaluation"] is False
    assert by_id["leaked"]["split_overlap"] == ["train"]
    assert by_id["missing"]["in_binary_evaluation"] is False
    assert by_id["paraphrase"]["label"] == Label.UNVALIDATED.name
    assert by_id["paraphrase"]["probe_prediction"] == Label.HALLUCINATED.name
    assert {row["question_id"] for row in metrics["false_positives"]} == {"faithful-high"}
    assert {row["question_id"] for row in metrics["false_negatives"]} == {"hallucinated-low"}
    assert metrics["calibration_validated"] is False
    assert metrics["generation_ece_set_calibration_validated"] is False
    assert metrics["detector_claim"] is False
    assert metrics["score_name"] == SCORE_NAME
    assert metrics["risk_status"] == RISK_STATUS
    assert metrics["probe"]["auroc_interval"]["low"] is None
    assert metrics["probe"]["auroc_interval"]["high"] is None
    assert "has 2 rows" in str(metrics["probe"]["auroc_interval"]["note"])
    assert metrics["probe"]["threshold"] == 0.65
    assert metrics["length_baseline"]["threshold"] == 0.5
    assert metrics["outperformed_text_baselines"] is None
    assert metrics["generation_settings"]["answer_piece_limit"] == 32
    markers = ("secret", "password", "token", "credential", "api_key", "private_key")
    assert all(all(marker not in key for marker in markers) for key in _keys(metrics))


def test_interval_is_computed_only_at_the_predeclared_minimum_and_ece_does_not_validate(
    tmp_path: Path,
) -> None:
    items = []
    answers = {}
    scores = {}
    for index in range(20):
        faithful = _item(f"f{index}", f"Faithful question {index}?", "alpha bravo charlie delta")
        hallucinated = _item(
            f"h{index}",
            f"Hallucinated question {index}?",
            "alpha bravo charlie delta",
            answer_kind="hallucinated",
        )
        items.extend((faithful, hallucinated))
        answers[faithful.question_id] = "Alpha bravo charlie delta."
        answers[hallucinated.question_id] = "Echo foxtrot golf hotel."
        scores[faithful.question_id] = 0.0
        scores[hallucinated.question_id] = 1.0
    split = FrozenSplit(
        seed=0,
        max_questions=len(items),
        dataset_sha256=PINNED_SHA256,
        question_ids=tuple(item.question_id for item in items),
        frozen_before_generation=True,
    )
    _dataset, _rows, metrics = build_transfer(
        split=split,
        items={item.question_id: item for item in items},
        answers=answers,
        probe_scores=scores,
        score_notes={},
        training_questions={
            "train": ("unrelated training question",),
            "validation": (),
            "test": (),
        },
        provenance=_provenance(),
        probe_metadata_calibration_validated=True,
        generation_settings={"temperature": 0.0},
        probe_metadata_sha256="d" * 64,
    )
    assert metrics["sample_counts"]["binary_eligible"] == 40
    assert metrics["probe"]["auroc"] == 1.0
    assert metrics["probe"]["auroc_interval"]["low"] is not None
    assert metrics["probe"]["auroc_interval"]["high"] is not None
    assert metrics["generation_ece"] == 0.0
    assert metrics["calibration_validated"] is False
    assert metrics["probe_metadata_calibration_validated"] is True
    assert metrics["generation_ece_set_calibration_validated"] is False
    assert metrics["detector_claim"] is False
    assert not (tmp_path / "record.json").exists()


def test_missing_provenance_and_a_failed_write_leave_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    destination = root / "experiments" / "broken"
    kwargs = _publish_kwargs(destination, root)
    broken = dict(kwargs["provenance"])
    del broken["git_commit"]
    kwargs["provenance"] = broken
    with pytest.raises(ValueError, match="git_commit"):
        publish_transfer(**kwargs)
    assert not destination.exists()

    def fail_record(*_args: object, **_kwargs: object) -> Path:
        raise ValueError("record failed")

    monkeypatch.setattr(
        "glassbox.evaluation.probe_transfer.write_generation_evaluation", fail_record
    )
    with pytest.raises(ValueError, match="record failed"):
        publish_transfer(**_publish_kwargs(destination, root))
    assert not (destination / "split.json").exists()
    assert not (destination / "dataset.json").exists()
    assert not (destination / "record.json").exists()
    assert not (destination / "metrics.json").exists()


def test_existing_run_and_probe_directory_are_refused(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    destination = root / "experiments" / "kept"
    destination.mkdir(parents=True)
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        publish_transfer(**_publish_kwargs(destination, root))
    assert sentinel.read_text(encoding="utf-8") == "keep"
    probe_destination = root / "final-hidden-state" / "record.json"
    with pytest.raises(ValueError, match="pair-probe"):
        publish_transfer(**_publish_kwargs(probe_destination.parent, root))


def test_probe_metadata_hash_is_unchanged_by_publish(tmp_path: Path) -> None:
    probe = tmp_path / "artifacts" / "probes" / "final-hidden-state"
    _save(probe)
    before = (probe / "metadata.json").read_bytes()
    root = tmp_path / "artifacts"
    publish_transfer(**_publish_kwargs(root / "experiments" / "beside-probe", root))
    assert (probe / "metadata.json").read_bytes() == before


def test_transfer_source_does_not_select_a_threshold_or_validate_calibration() -> None:
    import glassbox.evaluation.probe_transfer as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "select_threshold(" not in source
    assert "calibration_is_validated(" not in source
    assert CALIBRATION_ECE_LIMIT == 0.10


def test_missing_snapshot_does_not_construct_a_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("backend constructed")

    monkeypatch.setattr("glassbox.models.backend.HuggingFaceBackend", boom)
    from glassbox.cli import main

    code = main(
        [
            "--offline",
            "eval-saved-probe",
            "--snapshot",
            str(tmp_path / "missing.csv"),
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--run-id",
            "not-a-run",
        ]
    )
    assert code == 1
    assert not (tmp_path / "artifacts").exists()
    assert "not recorded" in capsys.readouterr().err


def test_bad_probe_metadata_stops_before_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("backend constructed")

    monkeypatch.setattr("glassbox.models.backend.HuggingFaceBackend", boom)
    probe = tmp_path / "probe"
    _save(probe, threshold=0.25)
    snapshot = tmp_path / "TruthfulQA.csv"
    halueval = tmp_path / "qa_data.json"
    snapshot.write_text("not a real snapshot", encoding="utf-8")
    halueval.write_text("{}\n", encoding="utf-8")
    from glassbox.cli import main

    code = main(
        [
            "--offline",
            "eval-saved-probe",
            "--snapshot",
            str(snapshot),
            "--halueval",
            str(halueval),
            "--probe",
            str(probe),
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--run-id",
            "bad-threshold",
        ]
    )
    assert code == 1
    assert not (tmp_path / "artifacts" / "experiments" / "bad-threshold").exists()


def _publish_kwargs(destination: Path, root: Path) -> dict[str, object]:
    items = _population()
    split = FrozenSplit(
        seed=0,
        max_questions=len(items),
        dataset_sha256=PINNED_SHA256,
        question_ids=tuple(item.question_id for item in items),
        frozen_before_generation=True,
    )
    answers = {item.question_id: _answer(item.question_id) for item in items}
    scores: dict[str, float | None] = {
        "faithful-low": 0.1,
        "faithful-high": 0.8,
        "hallucinated-high": 0.9,
        "hallucinated-low": 0.2,
        "paraphrase": 0.99,
        "both": 0.2,
        "leaked": 0.95,
        "missing": None,
    }
    notes = {item.question_id: "unit score" for item in items}
    notes["missing"] = "hidden state unavailable"
    return {
        "split": split,
        "items": {item.question_id: item for item in items},
        "answers": answers,
        "probe_scores": scores,
        "score_notes": notes,
        "training_questions": {
            "train": ("Which river is the longest?",),
            "validation": ("A validation question that is not reused",),
            "test": ("A test question that is not reused",),
        },
        "provenance": _provenance(),
        "destination": destination,
        "artifacts_root": root,
        "probe_metadata_calibration_validated": False,
        "generation_settings": {
            "answer_piece_limit": 32,
            "temperature": 0.0,
            "top_k": 5,
            "layer_index": -1,
            "position": "last_generated_token",
            "seed": 0,
            "max_questions": 8,
            "threshold": SAVED_PAIR_THRESHOLD,
            "baseline_threshold": BASELINE_THRESHOLD,
            "length_smoothing": 8,
        },
        "probe_metadata_sha256": "d" * 64,
    }


def _population() -> tuple[TruthfulItem, ...]:
    return (
        _item("faithful-low", "Where is the first dam?", "alpha bravo charlie delta"),
        _item("faithful-high", "Where is the second dam?", "alpha bravo charlie delta"),
        _item(
            "hallucinated-high",
            "Where is the third dam?",
            "alpha bravo charlie delta",
            answer_kind="hallucinated",
        ),
        _item(
            "hallucinated-low",
            "Where is the fourth dam?",
            "alpha bravo charlie delta",
            answer_kind="hallucinated",
        ),
        _item("paraphrase", "Where is the fifth dam?", "alpha bravo charlie delta"),
        _item("both", "Where is the sixth dam?", "alpha bravo charlie delta", answer_kind="both"),
        _item("leaked", "Which river is the longest?", "alpha bravo charlie delta"),
        _item("missing", "Where is the eighth dam?", "alpha bravo charlie delta"),
    )


def _answer(question_id: str) -> str:
    if question_id.startswith("hallucinated"):
        return "Echo foxtrot golf hotel."
    if question_id == "paraphrase":
        return "The structure is a large concrete barrier in a canyon."
    if question_id == "both":
        return "Alpha bravo charlie delta. Echo foxtrot golf hotel."
    return "Alpha bravo charlie delta."


def _item(question_id: str, question: str, correct: str, **_unused: str) -> TruthfulItem:
    incorrect = "echo foxtrot golf hotel"
    return TruthfulItem(
        question_id=question_id,
        category="unit",
        question=question,
        correct=(correct,),
        incorrect=(incorrect,),
    )


def _provenance() -> dict[str, object]:
    return {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": _REVISION,
        "dtype": "bfloat16",
        "dataset_id": "truthfulqa",
        "dataset_sha256": PINNED_SHA256,
        "split_id": "unit-split",
        "seed": 0,
        "probe_directory": "probe",
        "probe_model_revision": _PROBE_REVISION,
        "layer": 24,
        "position": "last_generated_token",
        "prompt_mode": "chat_template",
        "calibration_validated": False,
        "config_path": "configs/demo.yaml",
        "git_commit": _COMMIT,
        "worktree_clean": True,
    }


def _metadata(*, threshold: float = 0.65) -> dict[str, object]:
    return {
        "threshold": threshold,
        "position": "last_generated_token",
        "label_protocol": "halueval_qa_pair_v1",
        "positive_class": "hallucinated",
        "calibration_validated": False,
        "layer_index": 24,
        "feature_dim": 2,
        "model_revision": _PROBE_REVISION,
        "dataset": {
            "used_question_ids": {
                "train": ["row1"],
                "validation": ["row2"],
                "test": ["row3"],
            }
        },
    }


def _save(directory: Path, *, threshold: float = 0.65) -> None:
    probe = LogisticProbe()
    probe.fit(np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.0], [0.9, 1.0]]), np.array([0, 1, 0, 1]))
    metadata = _metadata(threshold=threshold)
    metadata.update(
        {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hidden_state_count": 25,
            "prompt_mode": "chat_template",
            "dtype": "bfloat16",
        }
    )
    save_probe(directory, probe, metadata=metadata, mean=np.zeros(2), scale=np.ones(2))


def _live(*, feature_dim: int) -> dict[str, object]:
    return {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": _PROBE_REVISION,
        "layer_index": 24,
        "hidden_state_count": 25,
        "position": "last_generated_token",
        "prompt_mode": "chat_template",
        "feature_dim": feature_dim,
        "dtype": "bfloat16",
    }


def _keys(payload: object) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            found.add(str(key).lower())
            found.update(_keys(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_keys(value))
    return found
