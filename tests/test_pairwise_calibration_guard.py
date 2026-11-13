"""A calibrated pairwise probe must never become a generation risk score."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from glassbox.data.schema import Label
from glassbox.models.backend import GenerationResult
from glassbox.probes.logistic import LogisticProbe
from glassbox.probes.store import load_probe, save_probe
from glassbox.reporting import TurnRecord
from glassbox.session import _apply_probe
from glassbox.verification import ReferenceResult


def _calibrated_pairwise(directory: Path):
    probe = LogisticProbe()
    probe.fit(
        np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.1], [0.9, 0.8]]),
        np.array([0, 1, 0, 1]),
    )
    save_probe(
        directory,
        probe,
        metadata={
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_revision": "b" * 40,
            "layer_index": 24,
            "hidden_state_count": 25,
            "position": "last_generated_token",
            "prompt_mode": "chat_template",
            "dtype": "bfloat16",
            "threshold": 0.65,
            "calibration_validated": False,
            "label_protocol": "halueval_qa_pair_v1",
        },
        mean=np.zeros(2),
        scale=np.ones(2),
    )
    meta_path = directory / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["calibration"] = "platt"
    meta["calibration_validated"] = True
    meta["platt"] = {"coef": 1.0, "intercept": 0.0}
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return load_probe(directory)


def _turn() -> TurnRecord:
    reference = ReferenceResult(
        configured=True,
        status="faithful",
        detail="test",
        label=Label.FAITHFUL,
    )
    return TurnRecord(
        generation=GenerationResult(
            question="What is the capital of Australia?",
            prompt_text="question",
            prompt_mode="chat_template",
            answer="Canberra",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            model_revision="b" * 40,
            device="cpu",
            dtype="bfloat16",
            latency_seconds=0.1,
            prompt_tokens=4,
            generated_tokens=1,
            tokens=(),
            hidden_states_available=True,
            logits_available=True,
            feature=np.array([0.0, 0.0]),
            feature_layer=24,
            hidden_state_count=25,
            position="last_generated_token",
            layer_vectors={},
            load_count=1,
        ),
        probe_status="experimental probe loaded",
        experimental_score=None,
        score_note="test",
        factual_risk_score=None,
        reference=reference,
        probe_prediction=None,
        probe_threshold=None,
        calibration_status="",
    )


def test_calibrated_pairwise_probe_sets_no_factual_risk(tmp_path: Path) -> None:
    loaded = _calibrated_pairwise(tmp_path / "probe")
    assert loaded.calibrated is True
    turn = _turn()
    _apply_probe(turn, loaded)
    assert turn.experimental_score is not None
    assert turn.factual_risk_score is None
    assert turn.calibration_status == "not validated"
    assert turn.probe_status == "experimental probe loaded"
