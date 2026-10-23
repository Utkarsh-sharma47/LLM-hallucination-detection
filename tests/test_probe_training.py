"""The probe refuses a one-class training split and does not borrow test rows."""

from pathlib import Path

import numpy as np

from glassbox.data.schema import Label
from glassbox.probes.logistic import PROBE_NOT_TRAINED
from glassbox.probes.store import LoadedProbe, load_probe
from glassbox.probes.training import FeatureExample, train_from_examples


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
    loaded = load_probe(tmp_path)
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
