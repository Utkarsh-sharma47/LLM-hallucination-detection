"""Tests for model, activation, and probe placeholders."""

from __future__ import annotations

import sys

import pytest

from glassbox.activations import extract_prompt_final_hidden_states
from glassbox.config import ProbeConfig
from glassbox.models import load_causal_lm
from glassbox.probes import LogisticProbe, build_probe, sklearn_class_weight


def test_disabled_loader_does_not_import_transformers() -> None:
    with pytest.raises(RuntimeError, match="disabled"):
        load_causal_lm("Qwen/Qwen2.5-3B-Instruct")
    sys.modules.pop("transformers", None)
    with pytest.raises(RuntimeError, match="disabled"):
        load_causal_lm("Qwen/Qwen2.5-3B-Instruct", allow_download=False, revision="b" * 40)
    assert "transformers" not in sys.modules


def test_activation_extractor_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="layers"):
        extract_prompt_final_hidden_states(object(), {"input_ids": [1]}, layers=(-1,))
    with pytest.raises(ValueError, match="required"):
        extract_prompt_final_hidden_states(None, object())


def test_probe_fits_separable_features_and_refuses_an_untrained_score() -> None:
    probe = build_probe(ProbeConfig())
    assert isinstance(probe, LogisticProbe)
    assert sklearn_class_weight("balanced") == "balanced"
    assert sklearn_class_weight("none") is None
    with pytest.raises(ValueError, match="class_weight"):
        sklearn_class_weight("auto")
    with pytest.raises(ValueError, match="required"):
        probe.fit(None, [0])
    with pytest.raises(RuntimeError, match="PROBE NOT TRAINED"):
        probe.predict_proba([[0.0, 1.0]])
    probe.fit([[0.0], [0.1], [1.0], [1.1]], [0, 0, 1, 1])
    probabilities = probe.predict_proba([[0.0], [1.0]])
    assert probabilities[0] < 0.5
    assert probabilities[1] > 0.5
