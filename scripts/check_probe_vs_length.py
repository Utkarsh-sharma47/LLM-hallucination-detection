"""Does the generation probe add signal beyond answer length? (read-only check)

Uses the saved probe from train_reviewed_generation_probe.py. It does not refit the
probe. Length features (characters, words) are fit on task-7 train; the stacked
"length + probe" model is fit on task-7 validation only. Evaluated on task-7 test and
on calibration batch 001.

Three checks:
  1. paired bootstrap of AUROC(probe) - AUROC(length)
  2. length-stratified AUROC: probe AUROC inside word-count bins (bins from train)
  3. incremental value: AUROC(length + probe) - AUROC(length alone), both fit on validation

Usage:
  python scripts/check_probe_vs_length.py
"""

from __future__ import annotations

import importlib.util
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from glassbox.config import repo_root

ROOT = repo_root()
spec = importlib.util.spec_from_file_location(
    "trainmod", ROOT / "scripts" / "train_reviewed_generation_probe.py"
)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
SEED = 0


def feats(ans):
    return np.array([[len(a), len(a.split())] for a in ans], dtype=float)


def batch_with_answers(art):
    from glassbox.probes.calibration_batch import (
        _paths,
        analysis_shards,
        read_decisions,
        read_queue,
        read_shard,
        read_vectors,
    )

    batch_dir, run_dir, review_dir = _paths(art, "pair-calibration-validation-001", T.BATCH_RUN)
    plan = json.loads((batch_dir / "plan.json").read_text(encoding="utf-8"))
    queue = read_queue(review_dir)
    dec = read_decisions(review_dir, queue)
    X, y, g, ans = [], [], [], []
    for shard in analysis_shards(queue, dec, int(plan["shard_count"])):
        vectors = read_vectors(run_dir, shard)
        for row in read_shard(run_dir, shard):
            lab = dec[int(row["plan_index"])]["label"]
            v = vectors.get(str(row["question_id"]))
            if (
                lab not in ("FAITHFUL", "HALLUCINATED")
                or v is None
                or row.get("label_name") == "EXCLUDED"
            ):
                continue
            X.append(v)
            y.append(int(lab == "HALLUCINATED"))
            g.append(row["group_id"])
            ans.append(str(row["answer"]))
    return {"X": np.stack(X).astype(float), "y": np.array(y), "g": np.array(g), "ans": ans}


def paired_diff(y, a, b, g, n=1000):
    rng = np.random.default_rng(SEED)
    groups = np.unique(g)
    by = {k: np.flatnonzero(g == k) for k in groups}
    d = []
    for _ in range(n):
        idx = np.concatenate([by[k] for k in rng.choice(groups, len(groups), replace=True)])
        if len(set(y[idx])) == 2:
            d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    d = np.array(d)
    return {
        "point": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "share_resamples_above_0": float((d > 0).mean()),
    }


def stratified(y, score, words, edges, min_class=5):
    bins = np.digitize(words, edges)
    out, wsum, wtot = [], 0.0, 0
    for b in np.unique(bins):
        m = bins == b
        nf, nh = int((y[m] == 0).sum()), int((y[m] == 1).sum())
        row = {"bin": int(b), "n": int(m.sum()), "faithful": nf, "hallucinated": nh}
        if nf >= min_class and nh >= min_class:
            row["auroc"] = float(roc_auc_score(y[m], score[m]))
            wsum += row["auroc"] * m.sum()
            wtot += m.sum()
        out.append(row)
    return {
        "bins": out,
        "weighted_auroc_within_bins": float(wsum / wtot) if wtot else None,
        "word_edges": [float(e) for e in edges],
    }


def main() -> int:
    art = ROOT / "artifacts"
    saved = joblib.load(art / "probes" / "generation-reviewed-task7-v1" / "probe.joblib")
    probe = saved["probe"]
    d = T.load_task7(art)
    tr, va, te = d["train"], d["validation"], d["test"]
    sets = {"task7_test": te, "batch_001": batch_with_answers(art)}

    length = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(
        feats(tr["ans"]), tr["y"]
    )
    edges = np.quantile(feats(tr["ans"])[:, 1], [0.25, 0.5, 0.75])

    def stack_x(s):
        return np.column_stack(
            [feats(s["ans"]), T.logit(probe.predict_proba(s["X"])[:, 1]).ravel()]
        )

    len_only_v = make_pipeline(StandardScaler(), LogisticRegression()).fit(
        feats(va["ans"]), va["y"]
    )
    stacked_v = make_pipeline(StandardScaler(), LogisticRegression()).fit(stack_x(va), va["y"])

    report = {}
    for name, s in sets.items():
        y, g = s["y"], s["g"]
        p = probe.predict_proba(s["X"])[:, 1]
        L = length.predict_proba(feats(s["ans"]))[:, 1]
        words = feats(s["ans"])[:, 1]
        report[name] = {
            "n": int(len(y)),
            "faithful": int((1 - y).sum()),
            "auroc_probe": float(roc_auc_score(y, p)),
            "auroc_length": float(roc_auc_score(y, L)),
            "corr_probe_vs_words": float(np.corrcoef(p, words)[0, 1]),
            "1_probe_minus_length": paired_diff(y, p, L, g),
            "2_length_stratified_probe": stratified(y, p, words, edges),
            "2_length_stratified_length": stratified(y, L, words, edges),
            "3_stacked_minus_length_only": paired_diff(
                y,
                stacked_v.predict_proba(stack_x(s))[:, 1],
                len_only_v.predict_proba(feats(s["ans"]))[:, 1],
                g,
            ),
        }
    out = art / "probes" / "generation-reviewed-task7-v1" / "length-check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, r in report.items():
        print(f"\n== {name} (n={r['n']}, faithful={r['faithful']})")
        print(
            f"AUROC probe {r['auroc_probe']:.3f} | length {r['auroc_length']:.3f} | corr(probe, words) {r['corr_probe_vs_words']:.2f}"
        )
        a = r["1_probe_minus_length"]
        print(
            f"1. probe - length   : {a['point']:+.3f}  CI {a['ci95'][0]:+.3f}..{a['ci95'][1]:+.3f}"
        )
        print(
            f"2. within length bins: probe {r['2_length_stratified_probe']['weighted_auroc_within_bins']} | length {r['2_length_stratified_length']['weighted_auroc_within_bins']}"
        )
        c = r["3_stacked_minus_length_only"]
        print(
            f"3. +probe over length: {c['point']:+.3f}  CI {c['ci95'][0]:+.3f}..{c['ci95'][1]:+.3f}"
        )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
