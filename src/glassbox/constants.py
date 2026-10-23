"""Shared constants for the Glass Box scaffold.

These values describe the planned study. They are not experimental findings.
"""

from __future__ import annotations

DATASET_IDS: tuple[str, ...] = ("halueval_qa", "truthfulqa")

PROBE_KINDS: tuple[str, ...] = ("logistic_regression",)

CLASS_WEIGHTS: tuple[str, ...] = ("balanced", "none")

PHASES: tuple[str, ...] = (
    "scaffold",
    "data_audit",
    "layerwise_probes",
    "early_detection",
    "activation_intervention",
)

CLASSIFICATION_METRICS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "false_positive_rate",
)

SEQUENTIAL_METRICS: tuple[str, ...] = ("early_warning_lead",)

REQUIRED_METRICS: tuple[str, ...] = CLASSIFICATION_METRICS + SEQUENTIAL_METRICS

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TASK_NAME = "short_factual_qa"

TASK_DESCRIPTION = "Short, verifiable factual question answering."

# Placeholder used by the shipped config. Recorded runs must replace it with a commit hash.
UNPINNED_REVISION = "main"
