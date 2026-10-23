"""Local run records.

Records live under the artifacts directory, which Git ignores. Building a
record does not mean an experiment has been run.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from glassbox.constants import REQUIRED_METRICS
from glassbox.reproducibility import require_commit_hash

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SECRET_MARKERS = ("SECRET", "PASSWORD", "TOKEN", "CREDENTIAL", "API_KEY", "PRIVATE_KEY")
_NULLABLE_METRICS = {"early_warning_lead"}


def build_run_record(
    *,
    run_id: str,
    git_commit: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, float | None],
    notes: str = "",
) -> dict[str, Any]:
    """Assemble a JSON-ready record of one future run.

    ``early_warning_lead`` may be null when the run is not a sequential
    detection run. Every other required metric needs a finite float supplied
    by the caller. This function does not compute metrics.
    """
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be one path segment of letters, digits, '.', '_' or '-'")
    if not isinstance(notes, str):
        raise ValueError("notes must be a string")
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "git_commit": require_commit_hash(git_commit, field="git_commit"),
        "config": dict(config),
        "metrics": normalize_metrics(metrics),
        "notes": notes,
    }
    assert_no_secret_keys(record)
    return record


def normalize_metrics(metrics: Mapping[str, float | None]) -> dict[str, float | None]:
    """Check the required metric map and coerce numeric values to ``float``."""
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    unknown = [name for name in metrics if name not in REQUIRED_METRICS]
    if missing or unknown:
        raise ValueError(f"metrics missing={missing}; unknown={unknown}")
    normalized: dict[str, float | None] = {}
    for name in REQUIRED_METRICS:
        normalized[name] = _metric_value(name, metrics[name])
    return normalized


def assert_no_secret_keys(payload: Mapping[str, Any]) -> None:
    """Reject mappings whose keys look like credentials."""
    _walk_secrets(payload, prefix="")


def write_json(path: Path, payload: Mapping[str, Any], *, root: Path) -> Path:
    """Write JSON under ``root`` and refuse paths that resolve outside it."""
    assert_no_secret_keys(payload)
    root_resolved = Path(root).resolve()
    destination = Path(path).resolve()
    if not destination.is_relative_to(root_resolved):
        raise ValueError(f"Refusing to write outside {root_resolved}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination.write_text(text, encoding="utf-8")
    return destination


def _metric_value(name: str, value: float | None) -> float | None:
    if value is None:
        if name not in _NULLABLE_METRICS:
            raise ValueError(f"{name} requires a finite float")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite float")
    return number


def _walk_secrets(value: Any, *, prefix: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key)
            upper = name.upper()
            if any(marker in upper for marker in _SECRET_MARKERS):
                raise ValueError(f"Refusing to record a secret-like field: {prefix}{name}")
            _walk_secrets(nested, prefix=f"{prefix}{name}.")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _walk_secrets(nested, prefix=f"{prefix}{index}.")
