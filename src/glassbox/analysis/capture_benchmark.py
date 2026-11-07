"""Benchmark hidden-state capture overhead during generation.

Records wall-clock latency, model forward calls, estimated transient model-side
hidden-state peak bytes, retained NumPy span bytes, Python-traced allocations,
and measured process/accelerator peaks across capture modes.
"""

from __future__ import annotations

import os
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any

from glassbox.analysis.capture import (
    BYTE_LIMIT,
    estimate_transient_hidden_state_bytes,
)
from glassbox.models.backend import GenerationResult, GenerationSettings, HuggingFaceBackend
from glassbox.models.resources import DTYPE_BYTES


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
    estimated_transient_hidden_state_bytes: int
    retained_capture_bytes: int
    retained_cap_bytes: int
    python_traced_peak_bytes: int
    process_resident_sampled_peak_bytes: int | None = None
    accelerator_peak_allocated_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_process_resident_bytes() -> int | None:
    """Return current process RSS in bytes on systems exposing procfs."""
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError, AttributeError):
        return None


class _ProcessResidentSampler:
    """Sample process RSS during one run; brief peaks may fall between samples."""

    def __init__(self) -> None:
        self.peak_bytes = get_process_resident_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int | None:
        self._stop.set()
        self._thread.join()
        self._sample()
        return self.peak_bytes

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(0.01):
            self._sample()

    def _sample(self) -> None:
        current = get_process_resident_bytes()
        if current is not None:
            self.peak_bytes = max(self.peak_bytes or 0, current)


def reset_accelerator_peak_memory(device: str | None) -> str | None:
    """Reset PyTorch's allocated-memory high-water mark for a CUDA device."""
    if not device or not device.startswith("cuda"):
        return None
    try:
        import torch

    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return device


def get_accelerator_peak_allocated_bytes(device: str | None) -> int | None:
    """Return allocated-byte high-water mark since reset for the given device."""
    if device is None:
        return None
    try:
        import torch

    except ImportError:
        return None
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def run_capture_benchmark(
    backend: HuggingFaceBackend,
    question: str,
    *,
    capture_modes: tuple[str, ...] = ("none", "final", "generated", "all"),
    max_new_tokens: int = 32,
) -> list[BenchmarkRecord]:
    """Benchmark different capture modes to quantify analysis forward cost."""
    records: list[BenchmarkRecord] = []
    loaded = backend.ensure_loaded()
    device = getattr(loaded, "device", None)

    for mode in capture_modes:
        settings = GenerationSettings(
            max_new_tokens=max_new_tokens,
            capture_mode=mode,
        )
        accelerator_device = reset_accelerator_peak_memory(device)
        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]
        resident_sampler = _ProcessResidentSampler()
        resident_sampler.start()

        started = time.perf_counter()
        try:
            result: GenerationResult = backend.generate(question, settings)
            total_latency = time.perf_counter() - started
        finally:
            resident_peak_bytes = resident_sampler.stop()
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        selected_layers = list(result.token_spans.keys()) if result.token_spans else []
        hidden_count = result.hidden_state_count or 0
        hidden_dim = result.feature.shape[0] if result.feature is not None else 0

        if result.hidden_states_available:
            if getattr(result, "transient_hidden_state_bytes", 0) > 0:
                transient_bytes = int(result.transient_hidden_state_bytes)
            else:
                elem_bytes = DTYPE_BYTES.get(result.dtype, 4)
                transient_bytes = estimate_transient_hidden_state_bytes(
                    sequence_length=result.prompt_tokens + result.generated_tokens,
                    layer_count=hidden_count,
                    dimension=hidden_dim,
                    bytes_per_element=elem_bytes,
                )
        else:
            transient_bytes = 0

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
            estimated_transient_hidden_state_bytes=transient_bytes,
            retained_capture_bytes=result.capture_bytes,
            retained_cap_bytes=BYTE_LIMIT,
            python_traced_peak_bytes=peak_mem - start_mem,
            process_resident_sampled_peak_bytes=resident_peak_bytes,
            accelerator_peak_allocated_bytes=get_accelerator_peak_allocated_bytes(
                accelerator_device
            ),
        )
        records.append(record)

    return records
