"""Benchmark hidden-state capture overhead during generation.

Records wall-clock latency, model forward calls, estimated transient model-side
hidden-state peak bytes, retained NumPy span bytes, Python-traced allocations,
and measured process/accelerator peaks across capture modes.
"""

from __future__ import annotations

import sys
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
    process_resident_peak_bytes: int | None = None
    accelerator_peak_allocated_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_process_peak_resident_bytes() -> int | None:
    """Return process peak resident set size (RSS) in bytes, if available."""
    try:
        import resource

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        scale = 1024 if sys.platform != "darwin" else 1
        return int(rusage.ru_maxrss * scale)
    except (ImportError, AttributeError):
        pass
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, Exception):
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
    except Exception:
        pass
    return None


def get_accelerator_peak_allocated_bytes() -> int | None:
    """Return accelerator device peak allocated bytes via PyTorch if available."""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except (ImportError, Exception):
        pass
    return None


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
            process_resident_peak_bytes=get_process_peak_resident_bytes(),
            accelerator_peak_allocated_bytes=get_accelerator_peak_allocated_bytes(),
        )
        records.append(record)

    return records
