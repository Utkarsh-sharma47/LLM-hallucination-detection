"""Benchmark hidden-state capture overhead during generation.

Records wall-clock latency, model forward calls, transient model-side
hidden-state peak bytes, and retained NumPy span bytes across capture modes.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any

from glassbox.analysis.capture import (
    BYTE_LIMIT,
    estimate_transient_hidden_state_bytes,
)
from glassbox.models.backend import GenerationResult, GenerationSettings, HuggingFaceBackend


@dataclass(frozen=True)
class BenchmarkRecord:
    """Resource and timing measurement for one generation capture run."""

    model_name: str
    model_revision: str
    device: str
    dtype: str
    prompt_tokens: int
    generated_tokens: int
    layer_count: int
    selected_layers: list[int]
    capture_mode: str
    generation_latency_seconds: float
    analysis_latency_seconds: float
    total_latency_seconds: float
    analysis_forward_calls: int
    transient_hidden_state_bytes: int
    retained_capture_bytes: int
    retained_cap_bytes: int
    peak_allocated_memory_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_capture_benchmark(
    backend: HuggingFaceBackend,
    question: str,
    *,
    capture_modes: tuple[str, ...] = ("none", "final", "generated", "all"),
    max_new_tokens: int = 32,
) -> list[BenchmarkRecord]:
    """Benchmark different capture modes to quantify analysis forward cost."""
    records: list[BenchmarkRecord] = []
    backend.ensure_loaded()

    for mode in capture_modes:
        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]

        settings = GenerationSettings(
            max_new_tokens=max_new_tokens,
            capture_mode=mode,
        )

        started = time.perf_counter()
        result: GenerationResult = backend.generate(question, settings)
        total_latency = time.perf_counter() - started

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        selected_layers = list(result.token_spans.keys()) if result.token_spans else []
        hidden_count = result.hidden_state_count or 0
        hidden_dim = result.feature.shape[0] if result.feature is not None else 0

        transient_bytes = (
            estimate_transient_hidden_state_bytes(
                sequence_length=result.prompt_tokens + result.generated_tokens,
                layer_count=hidden_count,
                dimension=hidden_dim,
            )
            if result.hidden_states_available
            else 0
        )

        record = BenchmarkRecord(
            model_name=result.model_name,
            model_revision=result.model_revision,
            device=result.device,
            dtype=result.dtype,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            layer_count=hidden_count,
            selected_layers=selected_layers,
            capture_mode=mode,
            generation_latency_seconds=result.latency_seconds,
            analysis_latency_seconds=result.analysis_latency_seconds,
            total_latency_seconds=total_latency,
            analysis_forward_calls=result.analysis_forward_calls,
            transient_hidden_state_bytes=transient_bytes,
            retained_capture_bytes=result.capture_bytes,
            retained_cap_bytes=BYTE_LIMIT,
            peak_allocated_memory_bytes=peak_mem - start_mem,
        )
        records.append(record)

    return records
