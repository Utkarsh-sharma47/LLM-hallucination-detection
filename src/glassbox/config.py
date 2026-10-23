"""Typed project configuration.

``configs/default.yaml`` and the dataclass defaults describe the same scaffold.
Constructing a config validates its shape. It does not contact the network.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from glassbox.constants import (
    CLASS_WEIGHTS,
    DATASET_IDS,
    DEFAULT_MODEL_NAME,
    PHASES,
    PROBE_KINDS,
    REQUIRED_METRICS,
    TASK_DESCRIPTION,
    TASK_NAME,
    UNPINNED_REVISION,
)

_TOP_KEYS = {"model", "data", "probe", "task", "metrics", "artifacts_dir", "phase"}
_MODEL_KEYS = {"name", "revision", "allow_download"}
_DATA_KEYS = {"exploration_dataset", "evaluation_datasets", "allow_download", "seed"}
_PROBE_KEYS = {"kind", "freeze_backbone", "class_weight", "max_iter", "random_state"}
_TASK_KEYS = {"name", "description"}


def repo_root() -> Path:
    """Return the checkout root that contains this package and ``pyproject.toml``."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        has_project = (candidate / "pyproject.toml").is_file()
        has_package = (candidate / "src" / "glassbox" / "__init__.py").is_file()
        if has_project and has_package:
            return candidate
    raise FileNotFoundError(
        "Could not locate the Glass Box checkout. Install the package in editable mode "
        "from the repository root."
    )


def default_config_path() -> Path:
    """Return the path to the shipped scaffold config."""
    return repo_root() / "configs" / "default.yaml"


@dataclass(frozen=True)
class ModelConfig:
    """Frozen-backbone model settings."""

    name: str = DEFAULT_MODEL_NAME
    revision: str = UNPINNED_REVISION
    allow_download: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("model.name must be a non-empty string")
        if any(character.isspace() for character in self.name):
            raise ValueError("model.name must not contain whitespace")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("model.revision must be a non-empty string")
        if not isinstance(self.allow_download, bool):
            raise ValueError("model.allow_download must be a boolean")


@dataclass(frozen=True)
class DataConfig:
    """Dataset selection for the planned study. Downloads stay off by default."""

    exploration_dataset: str = "halueval_qa"
    evaluation_datasets: tuple[str, ...] = ("halueval_qa", "truthfulqa")
    allow_download: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluation_datasets", tuple(self.evaluation_datasets))
        self.validate()

    def validate(self) -> None:
        _require_dataset(self.exploration_dataset, "data.exploration_dataset")
        if not isinstance(self.allow_download, bool):
            raise ValueError("data.allow_download must be a boolean")
        _require_int(self.seed, "data.seed", minimum=0)
        datasets = self.evaluation_datasets
        if not datasets:
            raise ValueError("data.evaluation_datasets must be non-empty")
        if len(datasets) != len(set(datasets)):
            raise ValueError("data.evaluation_datasets contains duplicates")
        for dataset_id in datasets:
            _require_dataset(dataset_id, "data.evaluation_datasets")


@dataclass(frozen=True)
class ProbeConfig:
    """Untested initial settings for a frozen-model logistic probe."""

    kind: str = "logistic_regression"
    freeze_backbone: bool = True
    class_weight: str = "balanced"
    max_iter: int = 1000
    random_state: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.kind not in PROBE_KINDS:
            raise ValueError(f"probe.kind must be one of {', '.join(PROBE_KINDS)}")
        if self.freeze_backbone is not True:
            raise ValueError("probe.freeze_backbone must be true; the backbone stays frozen")
        if self.class_weight not in CLASS_WEIGHTS:
            raise ValueError(f"probe.class_weight must be one of {', '.join(CLASS_WEIGHTS)}")
        _require_int(self.max_iter, "probe.max_iter", minimum=1)
        _require_int(self.random_state, "probe.random_state", minimum=0)


@dataclass(frozen=True)
class TaskConfig:
    """The only task in scope for the first study."""

    name: str = TASK_NAME
    description: str = TASK_DESCRIPTION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.name != TASK_NAME:
            raise ValueError(f"task.name must be {TASK_NAME!r} in this scaffold")
        if self.description != TASK_DESCRIPTION:
            raise ValueError("task.description does not match the scaffold task")


@dataclass(frozen=True)
class ProjectConfig:
    """Full scaffold configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    metrics: tuple[str, ...] = REQUIRED_METRICS
    artifacts_dir: str = "artifacts"
    phase: str = "scaffold"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        self.validate()

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()
        self.probe.validate()
        self.task.validate()
        if self.metrics != REQUIRED_METRICS:
            raise ValueError(f"metrics must be {list(REQUIRED_METRICS)}")
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {', '.join(PHASES)}")
        if not isinstance(self.artifacts_dir, str) or not self.artifacts_dir:
            raise ValueError("artifacts_dir must be a non-empty relative path")
        artifacts = Path(self.artifacts_dir)
        if artifacts.is_absolute() or ".." in artifacts.parts:
            raise ValueError("artifacts_dir must be a relative path and must not contain '..'")

    def artifacts_path(self) -> Path:
        """Return the artifacts directory under the checkout root."""
        return repo_root() / self.artifacts_dir

    def is_offline(self) -> bool:
        """Return whether both model and dataset downloads are disarmed."""
        return not self.model.allow_download and not self.data.allow_download


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load and validate a YAML config. The default is ``configs/default.yaml``."""
    config_path = default_config_path() if path is None else Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return project_config_from_mapping(payload)


def project_config_from_mapping(payload: Mapping[str, Any]) -> ProjectConfig:
    """Build a config from a plain mapping with the shipped key set."""
    _require_exact_keys(payload, _TOP_KEYS, "config")
    return ProjectConfig(
        model=_model_from_mapping(_as_mapping(payload["model"], "model")),
        data=_data_from_mapping(_as_mapping(payload["data"], "data")),
        probe=_probe_from_mapping(_as_mapping(payload["probe"], "probe")),
        task=_task_from_mapping(_as_mapping(payload["task"], "task")),
        metrics=_as_str_tuple(payload["metrics"], "metrics"),
        artifacts_dir=_as_str(payload["artifacts_dir"], "artifacts_dir"),
        phase=_as_str(payload["phase"], "phase"),
    )


def _model_from_mapping(payload: dict[str, Any]) -> ModelConfig:
    _require_exact_keys(payload, _MODEL_KEYS, "model")
    return ModelConfig(
        name=_as_str(payload["name"], "model.name"),
        revision=_as_str(payload["revision"], "model.revision"),
        allow_download=_as_bool(payload["allow_download"], "model.allow_download"),
    )


def _data_from_mapping(payload: dict[str, Any]) -> DataConfig:
    _require_exact_keys(payload, _DATA_KEYS, "data")
    return DataConfig(
        exploration_dataset=_as_str(payload["exploration_dataset"], "data.exploration_dataset"),
        evaluation_datasets=_as_str_tuple(
            payload["evaluation_datasets"], "data.evaluation_datasets"
        ),
        allow_download=_as_bool(payload["allow_download"], "data.allow_download"),
        seed=payload["seed"],
    )


def _probe_from_mapping(payload: dict[str, Any]) -> ProbeConfig:
    _require_exact_keys(payload, _PROBE_KEYS, "probe")
    return ProbeConfig(
        kind=_as_str(payload["kind"], "probe.kind"),
        freeze_backbone=_as_bool(payload["freeze_backbone"], "probe.freeze_backbone"),
        class_weight=_as_str(payload["class_weight"], "probe.class_weight"),
        max_iter=payload["max_iter"],
        random_state=payload["random_state"],
    )


def _task_from_mapping(payload: dict[str, Any]) -> TaskConfig:
    _require_exact_keys(payload, _TASK_KEYS, "task")
    return TaskConfig(
        name=_as_str(payload["name"], "task.name"),
        description=_as_str(payload["description"], "task.description"),
    )


def _require_exact_keys(payload: Mapping[str, Any], allowed: set[str], name: str) -> None:
    missing = allowed - set(payload)
    unknown = set(payload) - allowed
    if missing or unknown:
        raise ValueError(
            f"{name} keys must be {sorted(allowed)}; "
            f"missing={sorted(missing)}; unknown={sorted(unknown)}"
        )


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _as_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _as_str_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return tuple(value)


def _require_dataset(dataset_id: str, name: str) -> None:
    if dataset_id not in DATASET_IDS:
        raise ValueError(f"{name} must be one of {', '.join(DATASET_IDS)}")


def _require_int(value: object, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
