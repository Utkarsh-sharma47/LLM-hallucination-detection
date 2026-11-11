"""Train and calibrate a generation probe on the AI-assisted reviewed targets.

Handoff implementation for task7-halueval-generation-2400-001-ai-v1.
Reads the 1,271 binary candidate targets from training-index.jsonl,
joins with frozen Qwen hidden-state vectors from features.npz,
fits StandardScaler and LogisticRegression, applies Platt scaling,
and evaluates the Issue #18 calibration gate (ECE <= 0.10).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = (
    REPO_ROOT
    / "artifacts"
    / "reviews"
    / "task7-halueval-generation-2400-001-ai-v1"
    / "generated"
    / "training-index.jsonl"
)
CALIBRATION_ECE_LIMIT = 0.10
ECE_BINS = 10


def expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, *, bins: int = ECE_BINS
) -> float:
    """Calculate Expected Calibration Error across equal-width confidence bins."""
    truth = np.asarray(labels, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if truth.shape != values.shape or truth.size == 0:
        raise ValueError("labels and scores must be non-empty and aligned")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(truth.size)
    error = 0.0
    for idx, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        if idx == bins - 1:
            mask = (values >= start) & (values <= end)
        else:
            mask = (values >= start) & (values < end)
        count = int(np.sum(mask))
        if count == 0:
            continue
        confidence = float(values[mask].mean())
        accuracy = float(truth[mask].mean())
        error += (count / total) * abs(accuracy - confidence)
    return float(error)


def select_best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Pick validation threshold with maximum F1, tie-breaking on lowest FPR."""
    best_thresh = 0.5
    best_key = (-1.0, -1.0)
    truth = np.asarray(labels)
    for step in range(1, 20):
        t = step / 20.0
        pred = (scores >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            truth, pred, average="binary", zero_division=0
        )
        fp = int(np.sum((truth == 0) & (pred == 1)))
        tn = int(np.sum((truth == 0) & (pred == 0)))
        fpr = fp / max(1, fp + tn)
        key = (f1, -fpr)
        if key > best_key:
            best_key = key
            best_thresh = t
    return best_thresh


def fit_platt_scaling(scores: np.ndarray, labels: np.ndarray, seed: int) -> tuple[float, float]:
    """Fit a 1D logistic map on validation scores."""
    model = LogisticRegression(random_state=seed, max_iter=1000)
    model.fit(scores.reshape(-1, 1), labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def apply_platt_scaling(scores: np.ndarray, coef: float, intercept: float) -> np.ndarray:
    logits = coef * scores + intercept
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to task7 features.npz (contains question_id and vectors)",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="Path to training-index.jsonl (default: task7 v1 index)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "experiments" / "task7-generation-reviewed-probe",
        help="Where to write the results and probe model",
    )
    args = parser.parse_args()

    if not args.features.is_file():
        print(f"Error: Features file not found at {args.features}", file=sys.stderr)
        sys.exit(1)
    if not args.index.is_file():
        print(f"Error: Index file not found at {args.index}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading candidate index from: {args.index}")
    rows_by_id = {}
    with args.index.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows_by_id[str(item["question_id"])] = item

    print(f"Loading features from: {args.features}")
    feat_data = np.load(args.features)
    qids = [str(q) for q in feat_data["question_id"]]
    vectors = feat_data["vectors"]
    vectors_by_id = {qid: vec for qid, vec in zip(qids, vectors)}

    data_by_split = {"train": ([], []), "validation": ([], []), "test": ([], [])}
    for qid, row in rows_by_id.items():
        if qid not in vectors_by_id:
            raise KeyError(f"Question ID {qid} from index missing in features.npz")
        split = row["split"]
        target = int(row["target"])
        data_by_split[split][0].append(vectors_by_id[qid])
        data_by_split[split][1].append(target)

    X_train = np.asarray(data_by_split["train"][0], dtype=np.float32)
    y_train = np.asarray(data_by_split["train"][1], dtype=int)
    X_val = np.asarray(data_by_split["validation"][0], dtype=np.float32)
    y_val = np.asarray(data_by_split["validation"][1], dtype=int)
    X_test = np.asarray(data_by_split["test"][0], dtype=np.float32)
    y_test = np.asarray(data_by_split["test"][1], dtype=int)

    print(f"\nSplit counts (Binary Candidates):")
    print(f"  Train:      {len(y_train)} rows (Faithful={np.sum(y_train==0)}, Hallucinated={np.sum(y_train==1)})")
    print(f"  Validation: {len(y_val)} rows (Faithful={np.sum(y_val==0)}, Hallucinated={np.sum(y_val==1)})")
    print(f"  Test:       {len(y_test)} rows (Faithful={np.sum(y_test==0)}, Hallucinated={np.sum(y_test==1)})")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("\nFitting LogisticRegression(class_weight='balanced')...")
    clf = LogisticRegression(class_weight="balanced", random_state=args.seed, max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    val_raw_scores = clf.predict_proba(X_val_scaled)[:, 1]
    best_thresh = select_best_threshold(y_val, val_raw_scores)
    print(f"Selected Validation Threshold (max F1): {best_thresh:.2f}")

    coef, intercept = fit_platt_scaling(val_raw_scores, y_val, seed=args.seed)
    val_calibrated = apply_platt_scaling(val_raw_scores, coef, intercept)
    val_ece = expected_calibration_error(y_val, val_calibrated)
    print(f"Validation ECE: {val_ece:.4f} (Limit: <= {CALIBRATION_ECE_LIMIT})")

    test_raw_scores = clf.predict_proba(X_test_scaled)[:, 1]
    test_calibrated = apply_platt_scaling(test_raw_scores, coef, intercept)
    test_preds = (test_raw_scores >= best_thresh).astype(int)

    test_auroc = float(roc_auc_score(y_test, test_raw_scores))
    test_auprc = float(average_precision_score(y_test, test_raw_scores))
    test_ece = expected_calibration_error(y_test, test_calibrated)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, test_preds, average="binary", zero_division=0
    )

    tn = int(np.sum((y_test == 0) & (test_preds == 0)))
    fp = int(np.sum((y_test == 0) & (test_preds == 1)))
    fn = int(np.sum((y_test == 1) & (test_preds == 0)))
    tp = int(np.sum((y_test == 1) & (test_preds == 1)))
    fpr = fp / max(1, fp + tn)

    calibration_accepted = test_ece <= CALIBRATION_ECE_LIMIT

    print("\n" + "=" * 50)
    print("TEST EVALUATION RESULTS (188 Candidate Rows)")
    print("=" * 50)
    print(f"AUROC:               {test_auroc:.4f}")
    print(f"AUPRC:               {test_auprc:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print(f"Precision:           {prec:.4f}")
    print(f"Recall:              {rec:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"Confusion Matrix:    tn={tn}, fp={fp}, fn={fn}, tp={tp}")
    print("-" * 50)
    print(f"Test ECE:            {test_ece:.4f}")
    print(f"Calibration Gate:    {'ACCEPTED' if calibration_accepted else 'REJECTED'}")
    print("=" * 50)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "COMPLETED",
        "binary_candidate_count": len(rows_by_id),
        "split_counts": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
        },
        "threshold": best_thresh,
        "platt_scaling": {"coef": coef, "intercept": intercept},
        "validation_ece": val_ece,
        "test_metrics": {
            "auroc": test_auroc,
            "auprc": test_auprc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "false_positive_rate": fpr,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "ece": test_ece,
            "calibration_validated": calibration_accepted,
        },
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {args.output_dir / 'report.json'}")

    # Save trained probe artifact (probe.joblib and metadata.json)
    import joblib
    payload = {
        "classifier": clf,
        "mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scale": np.asarray(scaler.scale_, dtype=np.float64),
        "feature_dim": 896,
    }
    joblib.dump(payload, args.output_dir / "probe.joblib")

    metadata = {
        "task": "qwen_generation_factuality",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "position": "last_generated_token",
        "feature_dim": 896,
        "threshold": best_thresh,
        "calibration": "platt",
        "calibration_validated": calibration_accepted,
        "test_ece": test_ece,
        "platt": {"coef": coef, "intercept": intercept},
        "preprocessing": {
            "type": "standard_scaler",
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "metrics": {
            "auroc": test_auroc,
            "auprc": test_auprc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "false_positive_rate": fpr,
        },
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Probe artifact written to: {args.output_dir / 'probe.joblib'}")


if __name__ == "__main__":
    main()

