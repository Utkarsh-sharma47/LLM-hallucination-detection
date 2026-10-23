"""Model entry points.

Importing this module does not import torch or transformers. A download starts
only when ``allow_download`` is true or ``local_files_only`` reads an existing cache.
"""

from __future__ import annotations

from typing import Any

from glassbox.models.backend import ModelSpec, load_hf_model


def load_causal_lm(
    model_name: str,
    *,
    allow_download: bool = False,
    revision: str = "main",
    local_files_only: bool = False,
    device: str = "auto",
    dtype: str = "auto",
) -> Any:
    """Load a frozen causal language model and its tokenizer.

    Returns a ``LoadedModel``. Tests and the smoke check keep ``allow_download``
    false, so they do not reach the Hugging Face client.
    """
    spec = ModelSpec(
        name=model_name,
        revision=revision,
        device=device,
        dtype=dtype,
        allow_download=allow_download,
        local_files_only=local_files_only,
    )
    return load_hf_model(spec)
