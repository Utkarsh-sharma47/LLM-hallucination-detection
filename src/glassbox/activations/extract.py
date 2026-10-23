"""Hidden-state extraction from a loaded causal language model.

Layer 0 is the embedding stream when the model returns one. Callers record the
resolved index rather than assuming an architecture.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def feature_index(prompt_tokens: int, answer_tokens: int, position: str) -> int:
    """Return the sequence index used for both training and inference.

    ``last_prompt_token`` is the last prompt token and does not see the answer.
    ``last_generated_token`` is the last answer token. The probe protocol uses
    the second one, and the demo config uses the same name.
    """
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 1:
        raise ValueError("prompt_tokens must be a positive integer")
    if isinstance(answer_tokens, bool) or not isinstance(answer_tokens, int) or answer_tokens < 0:
        raise ValueError("answer_tokens must be a non-negative integer")
    if position == "last_prompt_token":
        return prompt_tokens - 1
    if position == "last_generated_token":
        if answer_tokens < 1:
            raise ValueError("last_generated_token requires at least one answer token")
        return prompt_tokens + answer_tokens - 1
    raise ValueError(f"Unknown hidden-state position {position!r}")


def resolve_layer(layer_count: int, layer_index: int) -> int:
    """Resolve a possibly negative layer index against ``hidden_states``."""
    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise ValueError("layer_index must be an integer")
    if layer_count < 1:
        raise ValueError("layer_count must be positive")
    resolved = layer_count + layer_index if layer_index < 0 else layer_index
    if not 0 <= resolved < layer_count:
        raise ValueError(f"layer_index {layer_index} is outside 0..{layer_count - 1}")
    return resolved


def extract_prompt_final_hidden_states(
    model: Any,
    batch: Mapping[str, Any],
    *,
    layers: Sequence[int] | None = None,
) -> dict[int, np.ndarray]:
    """Return hidden states at the last non-padding prompt token.

    ``layers`` uses the model's ``hidden_states`` indexes and rejects negatives.
    The forward pass sets ``output_hidden_states=True``.
    """
    if model is None or batch is None:
        raise ValueError("model and batch are required")
    _validate_layers(layers)
    torch = _import_torch()
    device = module_device(model)
    tensors = _batch_tensors(batch, device=device, torch=torch)
    with torch.inference_mode():
        outputs = model(**tensors, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    if not hidden_states:
        raise RuntimeError("The model did not return hidden states.")
    position = _last_non_padding_index(tensors["attention_mask"])
    selected = range(len(hidden_states)) if layers is None else layers
    return vectors_at_position(hidden_states, position, layers=tuple(selected))


def vectors_at_position(
    hidden_states: Sequence[Any],
    position: int,
    *,
    layers: Sequence[int] | None = None,
) -> dict[int, np.ndarray]:
    """Copy selected layer vectors at one already-computed sequence position."""
    if not hidden_states:
        raise ValueError("hidden_states is empty")
    selected = list(range(len(hidden_states))) if layers is None else list(layers)
    vectors: dict[int, np.ndarray] = {}
    for layer in selected:
        resolved = resolve_layer(len(hidden_states), layer)
        state = hidden_states[resolved]
        if position < 0 or position >= state.shape[1]:
            raise ValueError("position is outside the sequence")
        vector = state[0, position].detach().float().cpu().numpy().copy()
        vectors[resolved] = vector
    return vectors


def module_device(model: Any) -> Any:
    """Return the device of the first parameter."""
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise TypeError("model must be a PyTorch module with parameters") from exc


def _validate_layers(layers: Sequence[int] | None) -> None:
    if layers is None:
        return
    if len(layers) == 0:
        raise ValueError("layers must be non-negative integer indexes")
    for layer in layers:
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError("layers must be non-negative integer indexes")


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for hidden-state extraction. "
            "Install the CPU build with: python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch


def _batch_tensors(batch: Mapping[str, Any], *, device: Any, torch: Any) -> dict[str, Any]:
    if "input_ids" not in batch:
        raise ValueError("batch must include input_ids")
    input_ids = torch.as_tensor(batch["input_ids"], device=device)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    attention = batch.get("attention_mask")
    if attention is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = torch.as_tensor(attention, device=device)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _last_non_padding_index(attention_mask: Any) -> int:
    row = attention_mask[0].detach().cpu().tolist()
    for index in range(len(row) - 1, -1, -1):
        if int(row[index]) != 0:
            return index
    raise ValueError("attention_mask has no non-padding token")
