"""Plan which hidden-state rows to keep from the generation forward pass.

The generation path already runs one forward with hidden states to read the
final answer piece. These helpers slice that result. They do not start another
forward, and they do not treat a cosine change as a hallucination onset.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

CAPTURE_MODES = ("final", "generated", "selected", "all")
BYTE_LIMIT = 32_000_000


def select_layers(hidden_count: int, mode: str, feature_layer: int) -> tuple[int, ...]:
    """Choose layer indexes for a capture mode. ``final`` keeps none of the span."""
    if mode not in CAPTURE_MODES:
        raise ValueError(f"capture mode must be one of {', '.join(CAPTURE_MODES)}")
    if hidden_count < 1:
        raise ValueError("hidden_count must be positive")
    if not 0 <= feature_layer < hidden_count:
        raise ValueError("feature_layer is outside the hidden-state list")
    if mode == "final":
        return ()
    if mode == "generated":
        return (feature_layer,)
    if mode == "all":
        return tuple(range(hidden_count))
    middle = hidden_count // 2
    return tuple(dict.fromkeys((0, middle, feature_layer)))


def estimate_bytes(generated: int, layers: int, dimension: int) -> int:
    """Bytes for one float32 copy of the selected generated rows."""
    if min(generated, layers, dimension) < 0:
        raise ValueError("capture sizes must be non-negative")
    return int(generated) * int(layers) * int(dimension) * 4


def spans_within_budget(
    generated: int, layers: Sequence[int], dimension: int
) -> tuple[tuple[int, ...], str]:
    """Drop to the feature layer when the selected copy would exceed the cap."""
    chosen = tuple(layers)
    if estimate_bytes(generated, len(chosen), dimension) <= BYTE_LIMIT:
        return chosen, ""
    if not chosen:
        return (), ""
    kept = (chosen[-1],)
    return kept, (
        f"Capture was narrowed to one layer. {len(chosen)} layers of {generated} pieces "
        f"would exceed {BYTE_LIMIT} bytes."
    )


def slice_generated_spans(
    hidden_states: Sequence[Any],
    *,
    prompt_tokens: int,
    generated_count: int,
    layers: Sequence[int],
) -> dict[int, np.ndarray]:
    """Copy generated rows only. Prompt rows are not retained."""
    if generated_count < 1 or not layers:
        return {}
    start = prompt_tokens
    end = prompt_tokens + generated_count
    spans: dict[int, np.ndarray] = {}
    for layer in layers:
        state = hidden_states[layer]
        if end > state.shape[1]:
            raise ValueError("generated span is outside the sequence")
        chunk = state[0, start:end].detach().float().cpu().numpy().copy()
        spans[int(layer)] = np.asarray(chunk, dtype=np.float32)
    return spans


def representation_change(span: np.ndarray) -> dict[str, object]:
    """Cosine distance from the first generated row to the last.

    This is a representation statistic. It is not a hallucination score and
    not an onset time.
    """
    if span.ndim != 2 or span.shape[0] < 1:
        return {"available": False, "reason": "This layer has no generated rows."}
    if span.shape[0] == 1:
        return {
            "available": True,
            "cosine_distance_first_to_last": 0.0,
            "interpretation": "One generated piece. No change across generation was measured.",
        }
    first = span[0]
    last = span[-1]
    denom = float(np.linalg.norm(first) * np.linalg.norm(last))
    if denom == 0.0:
        distance = None
    else:
        cosine = float(np.dot(first, last) / denom)
        distance = float(1.0 - cosine)
    return {
        "available": True,
        "cosine_distance_first_to_last": distance,
        "mean_l2_norm": float(np.linalg.norm(span, axis=1).mean()),
        "interpretation": (
            "Cosine distance between the first and last generated piece at this layer. "
            "It is not a learned hallucination signal and not an onset."
        ),
    }
