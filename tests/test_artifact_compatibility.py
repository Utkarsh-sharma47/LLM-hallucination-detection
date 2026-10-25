"""Split stability, artifact rejection, and a fresh-process reload. No Qwen weights."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from glassbox.data.schema import Label
from glassbox.data.truthfulqa import TruthfulItem, select_items
from glassbox.probes.store import (
    ProbeCompatibilityError,
    check_compatibility,
    experimental_score,
    load_probe,
)
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


def _items() -> tuple[TruthfulItem, ...]:
    return tuple(
        TruthfulItem(
            question_id=f"row{index}",
            category="example",
            question=f"Question {index}?",
            correct=(f"yes {index}",),
            incorrect=(f"no {index}",),
        )
        for index in range(6)
    )


def _fit(tmp_path: Path) -> None:
    examples: list[FeatureExample] = []
    examples.extend(_rows("train", 0, 20, Label.FAITHFUL, 0.0))
    examples.extend(_rows("train", 20, 20, Label.HALLUCINATED, 1.0))
    examples.extend(_rows("validation", 100, 15, Label.FAITHFUL, 0.0))
    examples.extend(_rows("validation", 115, 15, Label.HALLUCINATED, 1.0))
    examples.extend(_rows("test", 200, 15, Label.FAITHFUL, 0.0))
    examples.extend(_rows("test", 215, 15, Label.HALLUCINATED, 1.0))
    outcome = train_from_examples(examples, layer_index=-1, output_dir=tmp_path)
    assert outcome.saved is True


def test_same_seed_returns_the_same_question_ids() -> None:
    items = _items()
    first = select_items(items, max_questions=3, seed=0)
    again = select_items(items, max_questions=3, seed=0)
    other = select_items(items, max_questions=3, seed=1)
    assert [item.question_id for item in first] == [item.question_id for item in again]
    assert [item.question_id for item in first] != [item.question_id for item in other]


def test_matching_metadata_is_accepted_and_each_mismatch_is_rejected(tmp_path: Path) -> None:
    _fit(tmp_path)
    loaded = load_probe(tmp_path)
    loaded.metadata["dtype"] = "bfloat16"
    kwargs = {
        "model_name": loaded.metadata["model_name"],
        "model_revision": loaded.metadata["model_revision"],
        "layer_index": loaded.metadata["layer_index"],
        "hidden_state_count": loaded.metadata["hidden_state_count"],
        "position": loaded.metadata["position"],
        "prompt_mode": loaded.metadata["prompt_mode"],
        "feature_dim": loaded.metadata["feature_dim"],
        "dtype": "bfloat16",
    }
    check_compatibility(loaded, **kwargs)
    replacements = {
        "model_name": "other-model",
        "model_revision": "c" * 40,
        "layer_index": 0,
        "position": "last_prompt_token",
        "feature_dim": 3,
        "dtype": "float32",
    }
    for key, value in replacements.items():
        changed = dict(kwargs)
        changed[key] = value
        with pytest.raises(ProbeCompatibilityError, match=key):
            check_compatibility(loaded, **changed)


def test_missing_and_incompatible_artifacts_do_not_score(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_probe(tmp_path / "absent")
    _fit(tmp_path)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_dim"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ProbeCompatibilityError, match="feature_dim"):
        load_probe(tmp_path)


def test_fresh_process_reproduces_the_saved_score(tmp_path: Path) -> None:
    _fit(tmp_path)
    vector = np.full(4, 1.0, dtype=np.float64)
    parent = experimental_score(load_probe(tmp_path), vector)
    np.save(tmp_path / "vector.npy", vector)
    code = "\n".join(
        [
            "import numpy as np",
            "from pathlib import Path",
            "from glassbox.probes.store import experimental_score, load_probe",
            f"probe = load_probe(Path({str(tmp_path)!r}))",
            f"vector = np.load(Path({str(tmp_path / 'vector.npy')!r}))",
            "print(f'{experimental_score(probe, vector):.12f}')",
        ]
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.stdout.strip() == f"{parent:.12f}"
