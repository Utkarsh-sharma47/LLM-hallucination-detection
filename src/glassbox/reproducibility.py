"""Checks that keep recorded runs pinned to exact revisions."""

from __future__ import annotations

import importlib.metadata
import re

_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")


def require_commit_hash(value: str, *, field: str) -> str:
    """Return a lowercase 40-hex commit hash.

    Branch names, including ``main``, are rejected. A recorded run has to name
    the exact model revision and the exact code revision it used.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip().lower()
    if _COMMIT_HASH.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be a 40-character hexadecimal commit hash")
    return normalized


def package_versions() -> dict[str, str]:
    """Record installed versions. A missing package is named, not invented."""
    found: dict[str, str] = {}
    for name in ("numpy", "scikit-learn", "torch", "transformers", "huggingface_hub"):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = "not installed"
    return found
