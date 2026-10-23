"""Opt-in check for a cached Qwen snapshot.

This script does not download weights. GitHub-hosted runners do not have the
cache or the gitignored datasets, so they exit 0 after saying the evaluation
was skipped.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
REPO = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> int:
    try:
        from huggingface_hub.file_download import try_to_load_from_cache
    except ImportError:
        print("huggingface_hub is not installed. Skipping. No download was attempted.")
        return 0
    cached = try_to_load_from_cache(REPO, "config.json", revision=REVISION)
    root = Path(__file__).resolve().parents[1]
    dataset = root / "artifacts" / "datasets" / "truthfulqa" / "TruthfulQA.csv"
    probe = root / "artifacts" / "probes" / "final-hidden-state" / "probe.joblib"
    if not cached or not dataset.is_file() or not probe.is_file():
        print(
            "Cached Qwen evaluation skipped. "
            "The snapshot, TruthfulQA CSV, or saved probe is absent. "
            "This script does not download weights."
        )
        return 0
    print(f"Using cached config {cached}. Offline evaluation of 8 TruthfulQA questions.")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "glassbox",
            "--offline",
            "eval-truthfulqa",
            "--max-questions",
            "8",
            "--run-id",
            "truthfulqa-ci-cache-check",
        ],
        check=False,
        cwd=root,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
