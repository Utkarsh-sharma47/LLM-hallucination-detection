"""Export the task-7 generation probe in the format `python -m glassbox` loads.

Reads the probe trained by train_reviewed_generation_probe.py, then writes
artifacts/probes/generation-reviewed-task7-v1-glassbox/{probe.joblib, metadata.json}.

What changes versus the training script
  * The app applies Platt scaling to the raw probe score (coef*raw + intercept), not to its
    logit. So the calibrator is refit in that form on task-7 VALIDATION only.
  * The new calibrator is re-scored on task-7 test and batch 001. The export is refused if
    either ECE is above 0.10 or the probe cannot be reloaded to the same scores.
  * The decision threshold is picked on validation (Youden J) and only recorded; it changes
    the faithful/hallucinated label the app prints, not the probability.
The probe weights are NOT retrained.

Usage
  python scripts/export_generation_probe.py
  python -m glassbox --offline --probe artifacts/probes/generation-reviewed-task7-v1-glassbox
"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from glassbox.config import repo_root
from glassbox.probes.logistic import LogisticProbe
from glassbox.probes.scoring import apply_platt, expected_calibration_error
from glassbox.probes.store import experimental_score, load_probe, save_probe

ROOT = repo_root()
ART = ROOT / "artifacts"
SRC = ART / "probes" / "generation-reviewed-task7-v1"
DEST = ART / "probes" / "generation-reviewed-task7-v1-glassbox"
GENERATION_PROTOCOL = "halueval_generation_evidence_v1"

spec = importlib.util.spec_from_file_location(
    "trainmod", ROOT / "scripts" / "train_reviewed_generation_probe.py"
)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def contract():
    rows = json.loads((ART / T.QUEUE).read_text(encoding="utf-8"))["rows"]
    keys = (
        "model_name",
        "model_revision",
        "feature_layer",
        "hidden_state_count",
        "prompt_mode",
        "dtype",
        "position",
        "feature_dim",
    )
    seen = {tuple(str(r[k]) for k in keys) for r in rows}
    if len(seen) != 1:
        raise ValueError(f"task-7 rows disagree on the model contract: {seen}")
    return dict(zip(keys, next(iter(seen)), strict=False))


def main() -> int:
    saved = joblib.load(SRC / "probe.joblib")
    pipe = saved["probe"]
    scaler, clf = pipe.steps[0][1], pipe.steps[1][1]
    c = contract()
    d = T.load_task7(ART)
    tr, va, te = d["train"], d["validation"], d["test"]
    Xb, yb, gb, idb = T.load_batch(ART)
    if idb & (tr["ids"] | va["ids"] | te["ids"]):
        raise ValueError("batch 001 overlaps task 7")

    def raw(X):
        return pipe.predict_proba(X)[:, 1]

    # Platt in the app's form, fit on validation only
    platt = LogisticRegression(C=1e6, max_iter=5000).fit(raw(va["X"]).reshape(-1, 1), va["y"])
    coef, intercept = float(platt.coef_[0, 0]), float(platt.intercept_[0])

    def cal(r):
        return apply_platt(r, coef, intercept)

    # threshold on validation only (Youden J on calibrated score)
    cv = cal(raw(va["X"]))
    fpr, tpr, thr = roc_curve(va["y"], cv)
    threshold = float(thr[np.argmax(tpr - fpr)])

    def ev(X, y):
        s = cal(raw(X))
        return {
            "n": int(len(y)),
            "auroc": float(roc_auc_score(y, s)),
            "ece": float(expected_calibration_error(y, s)),
            "brier": float(brier_score_loss(y, s)),
            "faithful": int((1 - y).sum()),
            "hallucinated": int(y.sum()),
        }

    test, batch = ev(te["X"], te["y"]), ev(Xb, yb)
    if test["ece"] > T.ECE_LIMIT or batch["ece"] > T.ECE_LIMIT:
        raise SystemExit(
            f"Refusing to export: ECE test {test['ece']:.3f}, batch {batch['ece']:.3f} (limit {T.ECE_LIMIT})"
        )

    probe = LogisticProbe().attach(clf, int(c["feature_dim"]))
    metadata = {
        "label_protocol": GENERATION_PROTOCOL,
        "model_name": c["model_name"],
        "model_revision": c["model_revision"],
        "layer_index": int(c["feature_layer"]),
        "hidden_state_count": int(c["hidden_state_count"]),
        "position": c["position"],
        "prompt_mode": c["prompt_mode"],
        "dtype": c["dtype"],
        "threshold": threshold,
        "calibration": "platt",
        "calibration_validated": True,
        "platt": {
            "coef": coef,
            "intercept": intercept,
            "fit_on": "task-7 validation only",
            "fit_rows": int(len(va["y"])),
        },
        "calibration_rows": int(len(va["y"])),
        "base_rate": float(va["y"].mean()),
        "test_ece": test["ece"],
        "brier": test["brier"],
        "evaluation": {"task7_test": test, "calibration_batch_001": batch},
        "training": {
            "rows": int(len(tr["y"])),
            "labels": "task-7 AI-reviewed (ai_passage_review_v1)",
            "reviewer_type": "AI",
            "human_reviewed": False,
            "C": T.C_GRID and float(clf.C),
        },
        "metrics_note": (
            "Generation probe trained on AI-reviewed Qwen answers (HaluEval-style questions, "
            f"Qwen2.5-0.5B only). Held-out AUROC {test['auroc']:.2f} (task-7 test) and "
            f"{batch['auroc']:.2f} (batch 001); adds about +0.05 AUROC over answer length alone. "
            "Labels come from one AI reviewer, not a human."
        ),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_probe": str(SRC.relative_to(ROOT)),
    }
    save_probe(DEST, probe, metadata=metadata, mean=scaler.mean_, scale=scaler.scale_)

    # round trip through the app's own loader
    loaded = load_probe(DEST)
    sample = te["X"][:50]
    app = np.array([experimental_score(loaded, v) for v in sample])
    if not loaded.calibrated or not np.allclose(app, cal(raw(sample)), atol=1e-6):
        raise SystemExit("Round trip failed: the app loader does not reproduce the trained scores")
    print(
        json.dumps(
            {
                "saved_to": str(DEST),
                "threshold": threshold,
                "platt": [coef, intercept],
                "task7_test": test,
                "calibration_batch_001": batch,
                "round_trip": "ok (50 test vectors)",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
