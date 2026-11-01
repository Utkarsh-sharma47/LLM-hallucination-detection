"""Tests for hidden-state capture overhead, forward calls, and numerical equivalence."""

from __future__ import annotations

import numpy as np
import pytest

from glassbox.analysis.capture import (
    BYTE_LIMIT,
    estimate_transient_hidden_state_bytes,
    select_layers,
    spans_within_budget,
)
from glassbox.analysis.capture_benchmark import run_capture_benchmark
from glassbox.models.backend import (
    GenerationSettings,
    LoadedModel,
    ModelSpec,
    run_generation,
)


def _get_mock_classes():
    torch = pytest.importorskip("torch")

    class MockLM(torch.nn.Module):
        """Mock model that counts forward calls and outputs predictable hidden states."""

        def __init__(self, hidden_dim: int = 32, num_layers: int = 4):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.call_count = 0
            self.linear = torch.nn.Linear(hidden_dim, 100)

        def forward(self, input_ids, output_hidden_states=False, use_cache=False, **kwargs):
            self.call_count += 1
            batch_size, seq_len = input_ids.shape
            hidden_states = tuple(
                torch.ones((batch_size, seq_len, self.hidden_dim), dtype=torch.float32)
                * float(layer + 1.0)
                for layer in range(self.num_layers)
            )
            logits = torch.zeros((batch_size, seq_len, 100), dtype=torch.float32)
            logits[:, :, 5] = 10.0

            output = type("ModelOutput", (), {})()
            output.logits = logits
            output.past_key_values = None
            if output_hidden_states:
                output.hidden_states = hidden_states
            return output

    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 99

        def __call__(self, text, return_tensors="pt"):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

        def decode(self, ids, **kwargs):
            return "mocked answer"

    return torch, MockLM, MockTokenizer


def _make_loaded(mock_model, mock_tokenizer) -> LoadedModel:
    spec = ModelSpec(
        name="mock-model",
        revision="0" * 40,
        device="cpu",
        dtype="float32",
        local_files_only=True,
    )
    return LoadedModel(
        spec=spec,
        model=mock_model,
        tokenizer=mock_tokenizer,
        device="cpu",
        dtype="float32",
        resolved_revision="0" * 40,
    )


def test_forward_call_counts_for_capture_modes():
    """AC #1: Counts forward calls and verifies whether an analysis pass occurs."""
    _, MockLM, MockTokenizer = _get_mock_classes()
    mock_model = MockLM()
    loaded = _make_loaded(mock_model, MockTokenizer())

    # 1. Standard mode: 2 tokens generated + 1 analysis forward pass = 3 calls
    settings_standard = GenerationSettings(max_new_tokens=2, capture_mode="final")
    res_standard = run_generation(loaded, "Test prompt?", settings_standard)
    assert res_standard.analysis_forward_calls == 1
    assert res_standard.hidden_states_available is True
    assert mock_model.call_count == 3

    # Reset call counter
    mock_model.call_count = 0

    # 2. Capture mode "none": Analysis pass is skipped entirely (0 analysis calls)
    settings_none = GenerationSettings(max_new_tokens=2, capture_mode="none")
    res_none = run_generation(loaded, "Test prompt?", settings_none)
    assert res_none.analysis_forward_calls == 0
    assert res_none.hidden_states_available is False
    assert res_none.analysis_latency_seconds == 0.0
    assert mock_model.call_count == 2


def test_generation_token_invariance():
    """AC #5: Generated token IDs and probabilities are identical across capture modes."""
    _, MockLM, MockTokenizer = _get_mock_classes()
    mock_model = MockLM()
    loaded = _make_loaded(mock_model, MockTokenizer())

    res1 = run_generation(
        loaded, "Question?", GenerationSettings(max_new_tokens=3, capture_mode="final")
    )
    res2 = run_generation(
        loaded, "Question?", GenerationSettings(max_new_tokens=3, capture_mode="none")
    )

    assert len(res1.tokens) == len(res2.tokens)
    for t1, t2 in zip(res1.tokens, res2.tokens, strict=True):
        assert t1.token_id == t2.token_id
        assert np.isclose(t1.probability, t2.probability)
    assert res1.answer == res2.answer


def test_numerical_equivalence_on_fixed_inputs():
    """AC #4: Feature vector matches expected layer activation within numerical tolerance."""
    _, MockLM, MockTokenizer = _get_mock_classes()
    mock_model = MockLM(hidden_dim=16, num_layers=4)
    loaded = _make_loaded(mock_model, MockTokenizer())

    settings = GenerationSettings(
        max_new_tokens=2,
        layer_index=-1,
        position="last_generated_token",
        capture_mode="final",
    )
    result = run_generation(loaded, "Question?", settings)

    assert result.feature is not None
    assert result.feature_layer == 3
    expected = np.ones(16, dtype=np.float32) * 4.0
    assert np.allclose(result.feature, expected, atol=1e-5)


def test_transient_vs_retained_memory_distinction():
    """AC #3: Transient all-layer memory is distinguished from retained NumPy bytes."""
    transient = estimate_transient_hidden_state_bytes(
        sequence_length=100, layer_count=25, dimension=896, bytes_per_element=4
    )
    assert transient == 100 * 25 * 896 * 4

    assert select_layers(25, "final", 24) == ()
    assert select_layers(25, "none", 24) == ()


def test_budget_overflow_and_layer_fallback():
    """AC #6: Exceeding 32MB cap narrows capture cleanly with explicit note."""
    selected, note = spans_within_budget(generated=10000, layers=tuple(range(25)), dimension=896)
    assert len(selected) == 1
    assert "Capture was narrowed to one layer" in note
    assert str(BYTE_LIMIT) in note


def test_capture_benchmark_runner():
    """AC #2: Benchmark runner collects records across capture modes."""
    _, MockLM, MockTokenizer = _get_mock_classes()
    mock_model = MockLM()
    loaded = _make_loaded(mock_model, MockTokenizer())

    class MockBackend:
        def __init__(self, loaded):
            self._loaded = loaded
            self.load_count = 1

        def ensure_loaded(self):
            return self._loaded

        def generate(self, question, settings):
            res = run_generation(self._loaded, question, settings)
            res.load_count = self.load_count
            return res

    backend = MockBackend(loaded)
    records = run_capture_benchmark(
        backend,
        "What is the capital?",
        capture_modes=("none", "final"),
        max_new_tokens=2,
    )

    assert len(records) == 2
    assert records[0].capture_mode == "none"
    assert records[0].analysis_forward_calls == 0
    assert records[1].capture_mode == "final"
    assert records[1].analysis_forward_calls == 1
    assert isinstance(records[0].to_dict(), dict)
