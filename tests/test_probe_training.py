"""The probe refuses a one-class training split and does not borrow test rows."""

import argparse
from pathlib import Path

import numpy as np
import pytest

from glassbox.data.schema import Label
from glassbox.evaluation.metrics import BOOTSTRAP_MIN_PER_CLASS
from glassbox.probes.logistic import PROBE_NOT_TRAINED
from glassbox.probes.scoring import CALIBRATION_ECE_LIMIT
from glassbox.probes.store import LoadedProbe, load_probe
from glassbox.probes.training import (
    PROTECTED_PAIR_RUN,
    FeatureExample,
    assert_new_experiment_dir,
    train_command,
    train_from_examples,
)


def _example(
    example_id: str, question_id: str, split: str, label: Label, value: float
) -> FeatureExample:
    vector = np.full(4, value, dtype=np.float64)
    return FeatureExample(
        example_id=example_id,
        question_id=question_id,
        split=split,
        label=label,
        answer="yes" if label is Label.FAITHFUL else "no",
        reference_answer="yes",
        vectors={0: vector, 1: vector},
        prompt_mode="chat_template",
        position="last_generated_token",
        hidden_state_count=2,
        model_name="example",
        model_revision="a" * 40,
    )


def _rows(split: str, start: int, count: int, label: Label, value: float) -> list[FeatureExample]:
    rows = []
    for offset in range(count):
        index = start + offset
        which = "faithful" if label is Label.FAITHFUL else "hallucinated"
        rows.append(_example(f"halueval_qa:q{index}.{which}", f"q{index}", split, label, value))
    return rows


def test_one_class_training_does_not_use_the_test_split(tmp_path: Path):
    examples = _rows("train", 0, 4, Label.FAITHFUL, 0.0)
    examples.extend(_rows("test", 100, 20, Label.HALLUCINATED, 1.0))
    examples.extend(_rows("test", 120, 20, Label.FAITHFUL, 0.0))
    examples.extend(_rows("validation", 200, 4, Label.HALLUCINATED, 1.0))
    outcome = train_from_examples(examples, layer_index=-1, output_dir=tmp_path, min_per_class=20)
    assert outcome.saved is False
    assert PROBE_NOT_TRAINED in outcome.reason
    assert "not moved" in outcome.reason
    assert not any(tmp_path.iterdir())


def test_separable_features_are_scored_on_held_out_ids(tmp_path: Path):
    examples: list[FeatureExample] = []
    examples.extend(_rows("train", 0, 20, Label.FAITHFUL, 0.0))
    examples.extend(_rows("train", 20, 20, Label.HALLUCINATED, 1.0))
    examples.extend(_rows("validation", 100, 15, Label.FAITHFUL, 0.0))
    examples.extend(_rows("validation", 115, 15, Label.HALLUCINATED, 1.0))
    examples.extend(_rows("test", 200, 15, Label.FAITHFUL, 0.0))
    examples.extend(_rows("test", 215, 15, Label.HALLUCINATED, 1.0))
    outcome = train_from_examples(examples, layer_index=-1, output_dir=tmp_path)
    assert outcome.saved is True
    assert outcome.metrics is not None
    assert outcome.metrics["auroc"] == 1.0
    assert outcome.report["test_confusion"] == {"tn": 15, "fp": 0, "fn": 0, "tp": 15}
    scores = outcome.report["test_scores"]
    assert len(scores) == 30
    assert {row["example_id"] for row in scores} == {
        item.example_id for item in examples if item.split == "test"
    }
    assert {row["label"] for row in scores} == {0, 1}
    assert all(isinstance(row["probe_score"], float) for row in scores)
    interval = outcome.report["test_auroc_interval"]
    assert interval["low"] is None
    assert interval["high"] is None
    assert "below 20" in str(interval["note"])
    assert "does not transfer" in str(outcome.report["test_auroc_interval_note"])
    assert outcome.report["baselines"]["probe_auroc_interval"] == interval
    loaded = load_probe(tmp_path)
    assert loaded.metadata["test_scores"] == scores
    assert loaded.metadata["test_auroc_interval"]["low"] is None
    assert isinstance(loaded, LoadedProbe)
    train_ids = set(loaded.metadata["split_ids"]["train"])
    test_ids = set(loaded.metadata["split_ids"]["test"])
    assert train_ids.isdisjoint(test_ids)
    assert loaded.metadata["position"] == "last_generated_token"
    assert loaded.metadata["calibration"] in {"none", "platt"}
    if loaded.calibrated:
        assert loaded.metadata["calibration_validated"] is True
        assert loaded.metadata["test_ece"] <= 0.10
    else:
        assert loaded.metadata["calibration_validated"] is False


def test_scaler_is_fit_on_training_rows_only(tmp_path: Path):
    examples: list[FeatureExample] = []
    examples.extend(_rows("train", 0, 20, Label.FAITHFUL, 0.0))
    examples.extend(_rows("train", 20, 20, Label.HALLUCINATED, 1.0))
    examples.extend(_rows("validation", 100, 15, Label.FAITHFUL, 0.0))
    examples.extend(_rows("validation", 115, 15, Label.HALLUCINATED, 1.0))
    examples.extend(_rows("test", 200, 15, Label.FAITHFUL, 5.0))
    examples.extend(_rows("test", 215, 15, Label.HALLUCINATED, 5.0))
    outcome = train_from_examples(examples, layer_index=-1, output_dir=tmp_path)
    assert outcome.saved is True
    loaded = load_probe(tmp_path)
    mean = np.asarray(loaded.metadata["preprocessing"]["mean"], dtype=np.float64)
    assert mean.shape == (4,)
    assert np.allclose(mean, 0.5)


def test_method_name_without_the_validated_flag_is_not_calibrated(tmp_path: Path):
    examples = []
    examples.extend(_rows("train", 0, 20, Label.FAITHFUL, 0.0))
    examples.extend(_rows("train", 20, 20, Label.HALLUCINATED, 1.0))
    examples.extend(_rows("validation", 100, 15, Label.FAITHFUL, 0.0))
    examples.extend(_rows("validation", 115, 15, Label.HALLUCINATED, 1.0))
    examples.extend(_rows("test", 200, 15, Label.FAITHFUL, 0.0))
    examples.extend(_rows("test", 215, 15, Label.HALLUCINATED, 1.0))
    train_from_examples(examples, layer_index=-1, output_dir=tmp_path)
    loaded = load_probe(tmp_path)
    loaded.metadata["calibration"] = "platt"
    loaded.metadata["calibration_validated"] = False
    assert loaded.calibrated is False


def test_small_test_split_writes_null_interval_and_keeps_calibration_unvalidated(
    tmp_path: Path,
) -> None:
    examples: list[FeatureExample] = []
    examples.extend(_rows("train", 0, 20, Label.FAITHFUL, 0.0))
    examples.extend(_rows("train", 20, 20, Label.HALLUCINATED, 1.0))
    examples.extend(_rows("validation", 100, 15, Label.FAITHFUL, 0.0))
    examples.extend(_rows("validation", 115, 15, Label.HALLUCINATED, 1.0))
    examples.extend(_rows("test", 200, 15, Label.FAITHFUL, 1.0))
    examples.extend(_rows("test", 215, 15, Label.HALLUCINATED, 0.0))
    outcome = train_from_examples(examples, layer_index=-1, output_dir=tmp_path)
    assert outcome.saved is True
    assert len(outcome.report["test_scores"]) == 30
    interval = outcome.report["test_auroc_interval"]
    assert interval["low"] is None
    assert interval["high"] is None
    assert interval["n"] == 30
    assert "below 20" in str(interval["note"])
    assert BOOTSTRAP_MIN_PER_CLASS == 20
    assert CALIBRATION_ECE_LIMIT == 0.10
    assert outcome.report["test_ece"] > CALIBRATION_ECE_LIMIT
    assert outcome.calibration_validated is False
    loaded = load_probe(tmp_path)
    assert loaded.metadata["calibration_validated"] is False
    assert loaded.calibrated is False
    assert loaded.metadata["test_auroc_interval"]["low"] is None


def test_held_out_interval_is_written_when_each_class_meets_the_minimum(tmp_path: Path) -> None:
    examples: list[FeatureExample] = []
    examples.extend(_rows("train", 0, 20, Label.FAITHFUL, 0.0))
    examples.extend(_rows("train", 20, 20, Label.HALLUCINATED, 1.0))
    examples.extend(_rows("validation", 100, 15, Label.FAITHFUL, 0.0))
    examples.extend(_rows("validation", 115, 15, Label.HALLUCINATED, 1.0))
    examples.extend(_rows("test", 200, 20, Label.FAITHFUL, 0.0))
    examples.extend(_rows("test", 220, 20, Label.HALLUCINATED, 1.0))
    outcome = train_from_examples(examples, layer_index=-1, output_dir=tmp_path)
    assert outcome.saved is True
    interval = outcome.report["test_auroc_interval"]
    assert interval["point"] == pytest.approx(1.0)
    assert interval["low"] is not None
    assert interval["high"] is not None
    assert interval["n"] == 40
    assert "does not transfer" in str(interval["note"])
    assert "does not transfer" in str(outcome.report["test_auroc_interval_note"])
    assert "does not validate calibration" in str(outcome.report["test_auroc_interval_note"])


def test_saved_pair_run_id_is_not_rewritten(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="halueval-pair-002"):
        assert_new_experiment_dir(tmp_path / PROTECTED_PAIR_RUN)
    kept = tmp_path / "existing-run"
    kept.mkdir()
    (kept / "record.json").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        assert_new_experiment_dir(kept)
    assert (kept / "record.json").read_text(encoding="utf-8") == "keep"


def test_train_command_refuses_protected_run_id_before_loading_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Qwen backend constructed")

    monkeypatch.setattr("glassbox.probes.training.HuggingFaceBackend", boom)
    monkeypatch.setattr("glassbox.probes.training.repo_root", lambda: tmp_path)
    code = train_command(argparse.Namespace(run_id=PROTECTED_PAIR_RUN))
    assert code == 1
    err = capsys.readouterr().err
    assert PROTECTED_PAIR_RUN in err
    assert not (tmp_path / "artifacts" / "experiments" / PROTECTED_PAIR_RUN).exists()
