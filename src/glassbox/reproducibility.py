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


def mark_reproduction(*, dirty: bool, finished_reproduction: bool) -> dict[str, bool]:
    """Return the reproduction fields stored on a HaluEval pair record.

    A dirty ``git status --porcelain`` cannot be marked as a finished
    reproduction. The probe may still be saved as an experiment. This function
    does not rewrite an existing artifact.
    """
    if not isinstance(dirty, bool) or not isinstance(finished_reproduction, bool):
        raise ValueError("dirty and finished_reproduction must be booleans")
    if dirty and finished_reproduction:
        raise ValueError("A dirty worktree cannot be marked as a finished reproduction")
    accepted = finished_reproduction and not dirty
    return {
        "reproducible": accepted,
        "finished_reproduction": accepted,
        "worktree_dirty": dirty,
    }


def pair_record_notes(*, dirty: bool, calibration_validated: bool, model_name: str) -> str:
    """Notes for a pair-protocol record. A dirty tree is not called finished."""
    mark_reproduction(dirty=dirty, finished_reproduction=not dirty)
    if dirty:
        status = "This record is not a finished reproduction because the worktree was dirty."
    else:
        status = "Worktree was clean."
    return (
        "HaluEval pair protocol on Qwen2.5-0.5B-Instruct. "
        f"Calibration validated={calibration_validated}. "
        f"{status} "
        "Early detection and intervention were not run. "
        f"Model requested: {model_name}. "
        "The write-up keeps lexical overlap with the gold answer beside the probe AUROC."
    )


def package_versions() -> dict[str, str]:
    """Record installed versions. A missing package is named, not invented."""
    found: dict[str, str] = {}
    for name in ("numpy", "scikit-learn", "torch", "transformers", "huggingface_hub"):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = "not installed"
    return found
