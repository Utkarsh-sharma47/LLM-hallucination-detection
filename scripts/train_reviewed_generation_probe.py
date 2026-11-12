"""Train a generation-style probe on the task-7 reviewed labels and check it.

Data
  labels  : artifacts/reviews/task7-halueval-generation-2400-001-ai-v1/generated/training-index.jsonl
            (FAITHFUL / HALLUCINATED only, splits already group-isolated)
  vectors : artifacts/experiments/task7-halueval-generation-2400-001/features.npz
            (Qwen2.5-0.5B, layer 24, last generated token, 896 dims)

Protocol (fixed before looking at test)
  train       -> fit the probe. Regularisation C is picked by grouped 5-fold CV inside train.
  validation  -> fit the Platt calibrator (the ECE fix). Nothing else.
  test        -> scored once. AUROC, ECE, Brier with question-group bootstrap intervals.
  external    -> calibration batch 001 (reviewed, never trained on), scored once as a
                 second out-of-sample check. Use --no-external to skip.

Baselines on the same test rows: answer length, shuffled labels, and a constant
predictor that always outputs the validation hallucination rate. The probe must
beat these, not just 0.5.

Usage
  python scripts/train_reviewed_generation_probe.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from glassbox.config import repo_root
from glassbox.probes.scoring import expected_calibration_error

RUN = "task7-halueval-generation-2400-001"
INDEX = f"reviews/{RUN}-ai-v1/generated/training-index.jsonl"
FEATURES = f"experiments/{RUN}/features.npz"
QUEUE = f"experiments/{RUN}/review_queue.json"
BATCH_RUN = "calibration-batch-001-qwen0.5b"
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
ECE_LIMIT, MIN_PER_CLASS, SEED = 0.10, 20, 0


def make_probe(c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000),
    )


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def bootstrap(y, s, g, fn, n=1000):
    rng = np.random.default_rng(SEED)
    groups = np.unique(g)
    by = {k: np.flatnonzero(g == k) for k in groups}
    vals = []
    for _ in range(n):
        idx = np.concatenate([by[k] for k in rng.choice(groups, len(groups), replace=True)])
        if len(set(y[idx])) == 2:
            vals.append(fn(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def metrics(y, raw, cal, g):
    return {
        "n": int(len(y)),
        "hallucinated": int(y.sum()),
        "faithful": int((1 - y).sum()),
        "auroc": float(roc_auc_score(y, cal)),
        "auroc_95ci": bootstrap(y, cal, g, roc_auc_score),
        "ece_raw": float(expected_calibration_error(y, raw)),
        "ece_calibrated": float(expected_calibration_error(y, cal)),
        "ece_calibrated_95ci": bootstrap(y, cal, g, expected_calibration_error),
        "brier_calibrated": float(brier_score_loss(y, cal)),
        "mean_score_hallucinated": float(cal[y == 1].mean()),
        "mean_score_faithful": float(cal[y == 0].mean()),
    }


def load_task7(art: Path):
    rows = [
        json.loads(line)
        for line in (art / INDEX).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r["label"] in ("FAITHFUL", "HALLUCINATED")]
    z = np.load(art / FEATURES, allow_pickle=False)
    vec = {str(q): v for q, v in zip(z["question_id"], z["vectors"], strict=False)}
    answers = {
        r["question_id"]: str(r["answer"])
        for r in json.loads((art / QUEUE).read_text(encoding="utf-8"))["rows"]
    }
    missing = [r["question_id"] for r in rows if r["question_id"] not in vec]
    if missing:
        raise ValueError(f"{len(missing)} reviewed rows have no vector, e.g. {missing[:3]}")
    out = {}
    for split in ("train", "validation", "test"):
        sel = [r for r in rows if r["split"] == split]
        out[split] = {
            "X": np.stack([vec[r["question_id"]] for r in sel]).astype(np.float64),
            "y": np.array([int(r["label"] == "HALLUCINATED") for r in sel]),
            "g": np.array([r["group_id"] for r in sel]),
            "ans": [answers.get(r["question_id"], "") for r in sel],
            "ids": {r["question_id"] for r in sel},
            "groups": {r["group_id"] for r in sel},
        }
    a, b, c = (out[s]["groups"] for s in ("train", "validation", "test"))
    if a & b or a & c or b & c:
        raise ValueError("question groups overlap between splits")
    return out


def load_batch(art: Path):
    from glassbox.probes.calibration_batch import (
        _paths,
        analysis_shards,
        read_decisions,
        read_queue,
        read_shard,
        read_vectors,
    )

    batch_dir, run_dir, review_dir = _paths(art, "pair-calibration-validation-001", BATCH_RUN)
    plan = json.loads((batch_dir / "plan.json").read_text(encoding="utf-8"))
    queue = read_queue(review_dir)
    dec = read_decisions(review_dir, queue)
    X, y, g, ids = [], [], [], []
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
            if (int(row["feature_layer"]), row["position"], row["model_name"]) != (
                24,
                "last_generated_token",
                "Qwen/Qwen2.5-0.5B-Instruct",
            ):
                raise ValueError("batch vectors use a different model/layer/position")
            X.append(v)
            y.append(int(lab == "HALLUCINATED"))
            g.append(row["group_id"])
            ids.append(row["question_id"])
    return np.stack(X).astype(np.float64), np.array(y), np.array(g), set(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/probes/generation-reviewed-task7-v1")
    ap.add_argument("--no-external", action="store_true")
    args = ap.parse_args()
    root = repo_root()
    art = root / "artifacts"
    d = load_task7(art)
    tr, va, te = d["train"], d["validation"], d["test"]

    # 1. pick C by grouped CV inside train only
    cv = {}
    for c in C_GRID:
        aucs = []
        for fit_i, hold_i in GroupKFold(n_splits=5).split(tr["X"], tr["y"], tr["g"]):
            m = make_probe(c).fit(tr["X"][fit_i], tr["y"][fit_i])
            aucs.append(roc_auc_score(tr["y"][hold_i], m.predict_proba(tr["X"][hold_i])[:, 1]))
        cv[c] = float(np.mean(aucs))
    best_c = max(cv, key=cv.get)
    probe = make_probe(best_c).fit(tr["X"], tr["y"])

    # 2. Platt calibrator on validation only
    raw_va = probe.predict_proba(va["X"])[:, 1]
    platt = LogisticRegression(C=1e6, max_iter=5000).fit(logit(raw_va), va["y"])

    def cal(raw):
        return platt.predict_proba(logit(raw))[:, 1]

    # 3. test, once
    raw_te = probe.predict_proba(te["X"])[:, 1]
    test = metrics(te["y"], raw_te, cal(raw_te), te["g"])

    # baselines on the same test rows
    def feats(ans):
        return np.array([[len(a), len(a.split())] for a in ans], dtype=float)

    length = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(
        feats(tr["ans"]), tr["y"]
    )
    shuffled = make_probe(best_c).fit(tr["X"], np.random.default_rng(SEED).permutation(tr["y"]))
    base_rate = float(va["y"].mean())
    baselines = {
        "answer_length_auroc": float(
            roc_auc_score(te["y"], length.predict_proba(feats(te["ans"]))[:, 1])
        ),
        "shuffled_label_auroc": float(
            roc_auc_score(te["y"], shuffled.predict_proba(te["X"])[:, 1])
        ),
        "constant_base_rate": base_rate,
        "constant_base_rate_ece": float(
            expected_calibration_error(te["y"], np.full(len(te["y"]), base_rate))
        ),
        "note": "A constant score has AUROC 0.5 by definition. Its ECE shows how low ECE can go with zero detection.",
    }

    external = None
    if not args.no_external:
        Xb, yb, gb, idb = load_batch(art)
        overlap = idb & (tr["ids"] | va["ids"] | te["ids"])
        if overlap:
            raise ValueError(f"batch shares {len(overlap)} questions with task 7")
        raw_b = probe.predict_proba(Xb)[:, 1]
        external = metrics(yb, raw_b, cal(raw_b), gb)

    enough = test["faithful"] >= MIN_PER_CLASS and test["hallucinated"] >= MIN_PER_CLASS
    verdict = {
        "detects": test["auroc_95ci"][0] > 0.5,
        "beats_length_baseline": test["auroc"] > baselines["answer_length_auroc"],
        "ece_within_limit": test["ece_calibrated"] <= ECE_LIMIT,
        "enough_per_class": enough,
    }
    report = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "labels": INDEX,
        "vectors": FEATURES,
        "reviewer_type": "AI",
        "human_reviewed": False,
        "train": {
            "n": int(len(tr["y"])),
            "faithful": int((1 - tr["y"]).sum()),
            "hallucinated": int(tr["y"].sum()),
        },
        "validation": {
            "n": int(len(va["y"])),
            "faithful": int((1 - va["y"]).sum()),
            "hallucinated": int(va["y"].sum()),
            "used_for": "Platt calibrator only",
        },
        "cv_auroc_by_C": {str(k): v for k, v in cv.items()},
        "chosen_C": best_c,
        "test": test,
        "baselines": baselines,
        "external_batch_001": external,
        "verdict": verdict,
        "note": "Single AI reviewer. Test and external sets were scored once; do not retune on them.",
    }
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"probe": probe, "platt": platt}, out / "probe.joblib")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("chosen_C", "test", "baselines", "external_batch_001", "verdict")
            },
            indent=2,
        )
    )
    print(f"Wrote {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
