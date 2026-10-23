"""Demo configuration, separate from the research scaffold config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from glassbox.config import repo_root
from glassbox.models.backend import GenerationSettings, ModelSpec
from glassbox.models.resources import available_ram_bytes, recommended_model


@dataclass(frozen=True)
class DemoConfig:
    """Settings for the terminal demo."""

    spec: ModelSpec
    generation: GenerationSettings
    references_path: Path
    probe_dir: Path | None
    selection_reason: str


def default_demo_config_path() -> Path:
    """Return the shipped demo config."""
    return repo_root() / "configs" / "demo.yaml"


def default_probe_dir() -> Path:
    """Return the directory where the small training run writes its probe."""
    return repo_root() / "artifacts" / "probes" / "final-hidden-state"


def load_demo_config(
    path: Path | None = None,
    *,
    offline: bool = False,
    model_name: str | None = None,
) -> DemoConfig:
    """Load the demo YAML and apply CLI overrides.

    ``offline=True`` reads the cache and does not arm a download. Otherwise the
    demo is an explicit request to load weights, including a first download.
    """
    config_path = default_demo_config_path() if path is None else Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Demo config must be a mapping: {config_path}")
    model = _mapping(payload.get("model"), "model")
    generation = _mapping(payload.get("generation"), "generation")
    references = payload.get("references")
    if not isinstance(references, str) or not references:
        raise ValueError("references must be a path")
    name = model_name or _string(model.get("name"), "model.name")
    available = available_ram_bytes()
    recommended, reason = recommended_model(available)
    if name == "auto":
        name = recommended
    spec = ModelSpec(
        name=name,
        revision=_string(model.get("revision"), "model.revision"),
        device=_string(model.get("device", "auto"), "model.device"),
        dtype=_string(model.get("dtype", "auto"), "model.dtype"),
        allow_download=not offline,
        local_files_only=offline,
    )
    settings = GenerationSettings(
        max_new_tokens=_int(generation.get("max_new_tokens", 32), "generation.max_new_tokens"),
        temperature=_float(generation.get("temperature", 0.0), "generation.temperature"),
        top_k=_int(generation.get("top_k", 5), "generation.top_k"),
        layer_index=_int(generation.get("layer_index", -1), "generation.layer_index"),
        position=_string(generation.get("position", "last_prompt_token"), "generation.position"),
    )
    probe_dir = _optional_path(payload.get("probe_dir"))
    references_path = Path(references)
    if not references_path.is_absolute():
        references_path = repo_root() / references_path
    return DemoConfig(
        spec=spec,
        generation=settings,
        references_path=references_path,
        probe_dir=probe_dir,
        selection_reason=reason
        if name == recommended
        else (f"Configured model {name}. Recommendation for this machine: {reason}"),
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("probe_dir must be a path or null")
    path = Path(value)
    if not path.is_absolute():
        path = repo_root() / path
    return path
