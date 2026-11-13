"""Read-only AUROC check for a reviewed calibration batch.

Scores each eligible row (FAITHFUL / HALLUCINATED) with the saved pair probe and
reports AUROC with a question-group bootstrap interval. Changes no files except the
optional --out report. The probe file hashes are checked before and after.

Usage:
  python scripts/calibration_batch_auroc.py --run-id calibration-batch-001-qwen0.5b \
      --probe-dir artifacts/probes/final-hidden-state
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from glassbox.config import repo_root
from glassbox.probes.calibration_batch import (
    DEFAULT_BATCH_ID,
    _file_hash,
    _paths,
    analysis_shards,
    read_decisions,
    read_queue,
    read_shard,
    read_vectors,
)
from glassbox.probes.store import experimental_score, load_probe


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    p.add_argument("--run-id", required=True)
    p.add_argument("--probe-dir", required=True)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    artifacts = repo_root() / "artifacts"
    batch_dir, run_dir, review_dir = _paths(artifacts, a.batch_id, a.run_id)
    plan = json.loads((batch_dir / "plan.json").read_text(encoding="utf-8"))
    probe_dir = Path(a.probe_dir)
    files = [probe_dir / "metadata.json", probe_dir / "probe.joblib"]
    before = [_file_hash(f) for f in files]
    probe = load_probe(probe_dir)

    queue = read_queue(review_dir)
    decisions = read_decisions(review_dir, queue)
    shards = analysis_shards(queue, decisions, int(plan["shard_count"]))

    y, s, g = [], [], []
    for shard in shards:
        vectors = read_vectors(run_dir, shard)
        for row in read_shard(run_dir, shard):
            if row.get("label_name") == "EXCLUDED":
                continue
            label = decisions[int(row["plan_index"])]["label"]
            vec = vectors.get(str(row["question_id"]))
            if label not in ("FAITHFUL", "HALLUCINATED") or vec is None:
                continue
            y.append(int(label == "HALLUCINATED"))
            s.append(float(experimental_score(probe, vec)))
            g.append(str(row["group_id"]))
    if [_file_hash(f) for f in files] != before:
        raise RuntimeError("probe files changed")

    y, s, g = np.array(y), np.array(s), np.array(g)
    auroc = roc_auc_score(y, s)  # score as P(hallucinated)

    rng = np.random.default_rng(0)
    groups = np.unique(g)
    idx_by_group = {k: np.flatnonzero(g == k) for k in groups}
    boots = []
    for _ in range(a.resamples):
        pick = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([idx_by_group[k] for k in pick])
        if len(set(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    report = {
        "n": int(len(y)),
        "hallucinated": int(y.sum()),
        "faithful": int((1 - y).sum()),
        "auroc_score_as_p_hallucinated": float(auroc),
        "auroc_95ci": [float(lo), float(hi)],
        "resamples_used": len(boots),
        "mean_score_hallucinated": float(s[y == 1].mean()),
        "mean_score_faithful": float(s[y == 0].mean()),
        "share_scores_above_0.9": float((s > 0.9).mean()),
        "share_scores_below_0.1": float((s < 0.1).mean()),
        "note": "0.5 = chance. Below 0.5 means the score points the wrong way.",
    }
    print(json.dumps(report, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
