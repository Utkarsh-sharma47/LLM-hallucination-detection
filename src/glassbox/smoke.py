"""Offline smoke check for the scaffold.

The check loads the shipped config, exercises local helpers, and confirms that
the download guard is armed. It does not read ``.env`` and does not fetch a
model or a dataset.
"""

from __future__ import annotations

import numpy as np

from glassbox import __version__
from glassbox.caveats import RESEARCH_CAVEATS
from glassbox.config import ProjectConfig, load_config
from glassbox.constants import REQUIRED_METRICS
from glassbox.data.splits import assert_disjoint_ids
from glassbox.evaluation.metrics import compute_binary_metrics, early_warning_lead
from glassbox.guards import DownloadDisabledError, reject_download
from glassbox.reproducibility import require_commit_hash

SMOKE_OK = "glassbox smoke test: ok"


def run_smoke_checks() -> ProjectConfig:
    """Run the local checks and return the shipped config."""
    config = load_config()
    if config != ProjectConfig():
        raise RuntimeError("Shipped YAML defaults and ProjectConfig() disagree")
    if not config.is_offline():
        raise RuntimeError("Shipped config must keep model and dataset downloads disabled")
    if config.phase != "scaffold":
        raise RuntimeError("Shipped config must stay in the scaffold phase")
    if len(RESEARCH_CAVEATS) != 3 or any(not item.strip() for item in RESEARCH_CAVEATS):
        raise RuntimeError("Research caveats are missing")
    if config.metrics != REQUIRED_METRICS:
        raise RuntimeError("Shipped metrics do not match the required set")
    try:
        require_commit_hash(config.model.revision, field="model.revision")
    except ValueError:
        pass
    else:
        raise RuntimeError("Shipped model.revision is a placeholder and must not already be pinned")

    scores = compute_binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
    )
    if tuple(scores) != REQUIRED_METRICS[:6]:
        raise RuntimeError("Classification metrics did not return the expected keys")
    if early_warning_lead((False, True, True), 2) != 1:
        raise RuntimeError("Early-warning lead helper returned an unexpected value")
    assert_disjoint_ids(("halueval_qa:1",), ("halueval_qa:2",))
    try:
        reject_download(False, resource="smoke-check")
    except DownloadDisabledError:
        return config
    raise RuntimeError("Download guard did not refuse an offline call")


def main() -> int:
    """Entry point for ``python scripts/smoke_test.py``."""
    config = run_smoke_checks()
    print(
        f"{SMOKE_OK} "
        f"(version={__version__}, phase={config.phase}, model={config.model.name}, "
        "downloads=disabled)"
    )
    return 0
