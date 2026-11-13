"""Train and evaluate probes at every Qwen layer on the reviewed Task 7 set.

This replays the frozen questions and answers through the exact pinned model to recover
per-layer last-answer-token vectors. The reviewed train/validation/test split is preserved.
Each layer gets its own grouped-CV-selected logistic probe and validation-only Platt map.
No layer is selected using test metrics, and the active demo probe is never overwritten.

Run offline from the repository root::

    .venv/Scripts/python.exe scripts/layerwise_reviewed_generation_probe.py

The model checkpoint must already be cached. Pass --allow-model-download to opt in to
downloading the pinned checkpoint if it is not cached.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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
from glassbox.models.backend import (
    GenerationSettings,
    HuggingFaceBackend,
    ModelSpec,
    completion_hidden_states,
)
from glassbox.probes.generation_ai_review import _verify_bundle
from glassbox.probes.scoring import apply_platt, expected_calibration_error

SOURCE_RUN = "task7-halueval-generation-2400-001"
REVIEW_DIR = f"{SOURCE_RUN}-ai-v1"
DEFAULT_RUN_ID = "layerwise-task7-ai-v1-001"
INDEX_REL = Path("artifacts/reviews") / REVIEW_DIR / "generated/training-index.jsonl"
ANNOTATIONS_REL = Path("artifacts/reviews") / REVIEW_DIR / "annotations.jsonl"
SOURCE_REL = Path("artifacts/experiments") / SOURCE_RUN
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SEED = 0
BOOTSTRAP_REPLICATES = 1000


def make_probe(c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=SEED),
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_reviewed_rows(artifacts: Path) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    """Validate the frozen review bundle and join binary targets by question id."""
    source = artifacts / "experiments" / SOURCE_RUN
    review = artifacts / "reviews" / REVIEW_DIR
    predictions_path = source / "predictions.json"
    vectors_path = source / "features.npz"
    index_path = review / "generated" / "training-index.jsonl"
    annotations_path = review / "annotations.jsonl"
    for path in (predictions_path, vectors_path, index_path, annotations_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Task 7 input is missing: {path}")

    # The bundle's own validator checks frozen source and feature hashes before use.
    _verify_bundle(review, predictions_path, vectors_path)
    index = read_jsonl(index_path)
    annotations = {str(row["question_id"]): row for row in read_jsonl(annotations_path)}
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))["rows"]
    prediction_by_id = {str(row["question_id"]): row for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("Frozen Task 7 predictions repeat a question_id")

    saved = np.load(vectors_path, allow_pickle=False)
    saved_ids = [str(value) for value in saved["question_id"]]
    if len(set(saved_ids)) != len(saved_ids):
        raise ValueError("Frozen Task 7 vectors repeat a question_id")
    saved_vectors = {
        qid: np.asarray(vector, dtype=np.float32)
        for qid, vector in zip(saved_ids, saved["vectors"], strict=True)
    }

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in index:
        qid = str(item["question_id"])
        if qid in seen:
            raise ValueError(f"Reviewed training index repeats {qid}")
        seen.add(qid)
        if item.get("reviewer_type") != "AI" or item.get("human_reviewed") is not False:
            raise ValueError(f"Unexpected review provenance for {qid}")
        if (
            item.get("policy_version") != "ai_passage_review_v1"
            or item.get("adjudicated") is not False
        ):
            raise ValueError(f"Unexpected review policy or adjudication status for {qid}")
        qid_row = prediction_by_id.get(qid)
        annotation = annotations.get(qid)
        if qid_row is None or annotation is None or qid not in saved_vectors:
            raise ValueError(
                f"Reviewed row {qid} is missing source, annotation, or layer-24 vector"
            )
        if item.get("split") != qid_row.get("split") or item.get("split") != annotation.get(
            "split"
        ):
            raise ValueError(f"Frozen split mismatch for {qid}")
        if item.get("group_id") != qid_row.get("group_id"):
            raise ValueError(f"Frozen group mismatch for {qid}")
        if item.get("label") != annotation.get("label"):
            raise ValueError(f"Review label mismatch for {qid}")
        if item.get("label") not in {"FAITHFUL", "HALLUCINATED"}:
            continue
        if (
            qid_row.get("model_name") != MODEL_NAME
            or qid_row.get("model_revision") != MODEL_REVISION
        ):
            raise ValueError(f"Unexpected model provenance for {qid}")
        rows.append(
            {
                "question_id": qid,
                "group_id": str(item["group_id"]),
                "split": str(item["split"]),
                "label": int(item["target"]),
                "question": str(qid_row["question"]),
                "answer": str(qid_row["answer"]),
                "knowledge": str(qid_row["knowledge"]),
                "right_answer": str(qid_row["right_answer"]),
                "hallucinated_answer": str(qid_row["hallucinated_answer"]),
                "prompt_piece_count": int(qid_row["prompt_piece_count"]),
                "generated_piece_count": int(qid_row["generated_piece_count"]),
                "saved_layer24": saved_vectors[qid],
            }
        )

    for split in ("train", "validation", "test"):
        part = [row for row in rows if row["split"] == split]
        if not part or {int(row["label"]) for row in part} != {0, 1}:
            raise ValueError(f"Reviewed split {split} is empty or lacks a binary class")
    groups = {
        split: {str(row["group_id"]) for row in rows if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    if (
        groups["train"] & groups["validation"]
        or groups["train"] & groups["test"]
        or groups["validation"] & groups["test"]
    ):
        raise ValueError("Knowledge groups overlap between the frozen splits")
    return rows, saved_vectors


def extract_layer_features(
    rows: list[dict[str, object]], *, allow_download: bool
) -> tuple[np.ndarray, dict[str, object]]:
    """Replay exact frozen answers and return float32 [rows, layers, dimensions]."""
    spec = ModelSpec(
        name=MODEL_NAME,
        revision=MODEL_REVISION,
        device="auto",
        dtype="bfloat16",
        allow_download=allow_download,
        local_files_only=not allow_download,
    )
    backend = HuggingFaceBackend(spec)
    print(f"Loading pinned model {MODEL_NAME}@{MODEL_REVISION} ...", flush=True)
    loaded = backend.ensure_loaded()
    if loaded.resolved_revision != MODEL_REVISION:
        raise ValueError(
            f"Loaded checkpoint revision {loaded.resolved_revision} differs from frozen run"
        )
    settings = GenerationSettings(
        max_new_tokens=32,
        temperature=0.0,
        top_k=5,
        layer_index=-1,
        position="last_generated_token",
        capture_all_layers=True,
        capture_mode="final",
    )

    all_vectors: list[np.ndarray] = []
    alignment_cosines: list[float] = []
    answer_token_count_mismatches = 0
    layer_count = dimension = None
    for index, row in enumerate(rows, start=1):
        vectors, found_layers, feature_layer, _prompt_mode, prompt_text = completion_hidden_states(
            loaded,
            str(row["question"]),
            str(row["answer"]),
            settings,
        )
        prompt_count = len(loaded.tokenizer(prompt_text, add_special_tokens=True)["input_ids"])
        answer_count = len(
            loaded.tokenizer.encode(str(row["answer"]).strip(), add_special_tokens=False)
        )
        if prompt_count != row["prompt_piece_count"]:
            raise ValueError(
                f"Prompt tokenization differs from frozen Task 7 row {row['question_id']}: "
                f"saved={row['prompt_piece_count']}, replayed={prompt_count}"
            )
        exact_answer_tokenization = answer_count == row["generated_piece_count"]
        if not exact_answer_tokenization:
            # Original generated token IDs were not retained. The decoded answer text can
            # therefore re-tokenize to a different number of pieces; it remains usable for
            # an across-layer comparison, but cannot be aligned to that row's cached vector.
            answer_token_count_mismatches += 1
        if feature_layer != 24 or found_layers != 25 or set(vectors) != set(range(found_layers)):
            raise ValueError(
                "Expected all 25 hidden-state streams with layer 24 as the final layer"
            )
        matrix = np.stack([vectors[layer] for layer in range(found_layers)]).astype(np.float32)
        if layer_count is None:
            layer_count, dimension = matrix.shape
        elif matrix.shape != (layer_count, dimension):
            raise ValueError(f"Hidden-state shape changed at row {row['question_id']}")
        reference = np.asarray(row["saved_layer24"], dtype=np.float32)
        if exact_answer_tokenization:
            norm = float(np.linalg.norm(matrix[24]) * np.linalg.norm(reference))
            cosine = float(np.dot(matrix[24], reference) / norm) if norm else 0.0
            alignment_cosines.append(cosine)
        all_vectors.append(matrix)
        if index % 50 == 0 or index == len(rows):
            print(f"Replayed {index}/{len(rows)} reviewed answers", flush=True)

    if len(alignment_cosines) < 100:
        raise ValueError(
            "Fewer than 100 rows retained the original answer token count, so layer-24 "
            "replay alignment could not be checked reliably"
        )
    cosine_median = float(np.median(alignment_cosines))
    cosine_p05 = float(np.percentile(alignment_cosines, 5))
    if cosine_median < 0.999 or cosine_p05 < 0.95:
        raise ValueError(
            "Recomputed layer-24 vectors do not align with the frozen probe features "
            f"(median cosine={cosine_median:.6f}, 5th percentile={cosine_p05:.6f}); "
            "refusing to compare layers"
        )
    return np.stack(all_vectors), {
        "exact_answer_token_count_rows": len(alignment_cosines),
        "answer_token_count_mismatch_rows": answer_token_count_mismatches,
        "layer24_alignment_median_cosine": cosine_median,
        "layer24_alignment_p05_cosine": cosine_p05,
        "model_name": loaded.spec.name,
        "model_revision": loaded.resolved_revision,
        "dtype": loaded.dtype,
        "device": loaded.device,
        "prompt_protocol": "render_prompt(question); knowledge passage is not included",
        "position": "last_generated_token",
        "replay_note": (
            "Frozen answers were retained as text, not original token IDs. Layer-24 alignment "
            "is checked only where the re-tokenized answer has the original token count."
        ),
    }


def grouped_cv_c(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    splitter = GroupKFold(n_splits=5)
    for c in C_GRID:
        fold_scores = []
        for train_idx, valid_idx in splitter.split(x, y, groups):
            if set(y[train_idx]) != {0, 1} or set(y[valid_idx]) != {0, 1}:
                continue
            estimator = make_probe(c).fit(x[train_idx], y[train_idx])
            fold_scores.append(
                float(roc_auc_score(y[valid_idx], estimator.predict_proba(x[valid_idx])[:, 1]))
            )
        if not fold_scores:
            raise ValueError("Grouped CV produced no two-class folds")
        scores[str(c)] = float(np.mean(fold_scores))
    chosen = max(C_GRID, key=lambda c: (scores[str(c)], -c))
    return chosen, scores


def group_bootstrap_auc(
    y: np.ndarray, scores: np.ndarray, groups: np.ndarray, *, seed: int
) -> list[float] | None:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique_groups}
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([members[group] for group in chosen])
        if len(np.unique(y[indices])) == 2:
            values.append(float(roc_auc_score(y[indices], scores[indices])))
    if not values:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def fit_layer(
    x_by_split: dict[str, np.ndarray],
    y_by_split: dict[str, np.ndarray],
    groups_by_split: dict[str, np.ndarray],
    layer: int,
) -> tuple[dict[str, object], object]:
    x_train, y_train, g_train = x_by_split["train"], y_by_split["train"], groups_by_split["train"]
    x_val, y_val = x_by_split["validation"], y_by_split["validation"]
    x_test, y_test, g_test = x_by_split["test"], y_by_split["test"], groups_by_split["test"]
    chosen_c, cv_scores = grouped_cv_c(x_train, y_train, g_train)
    probe = make_probe(chosen_c).fit(x_train, y_train)
    raw_val = probe.predict_proba(x_val)[:, 1]
    val_logits = np.log(np.clip(raw_val, 1e-6, 1 - 1e-6) / np.clip(1 - raw_val, 1e-6, 1)).reshape(
        -1, 1
    )
    platt = LogisticRegression(C=1e6, max_iter=5000, random_state=SEED).fit(val_logits, y_val)
    raw_test = probe.predict_proba(x_test)[:, 1]
    test_logits = np.log(np.clip(raw_test, 1e-6, 1 - 1e-6) / np.clip(1 - raw_test, 1e-6, 1))
    calibrated = apply_platt(
        test_logits,
        float(platt.coef_[0, 0]),
        float(platt.intercept_[0]),
    )
    report = {
        "layer_index": layer,
        "selected_C": chosen_c,
        "train_cv_auroc_by_C": cv_scores,
        "train": {
            "n": int(len(y_train)),
            "faithful": int((y_train == 0).sum()),
            "hallucinated": int((y_train == 1).sum()),
        },
        "validation": {
            "n": int(len(y_val)),
            "faithful": int((y_val == 0).sum()),
            "hallucinated": int((y_val == 1).sum()),
            "used_for": "Platt calibration only",
        },
        "test": {
            "n": int(len(y_test)),
            "faithful": int((y_test == 0).sum()),
            "hallucinated": int((y_test == 1).sum()),
            "auroc": float(roc_auc_score(y_test, calibrated)),
            "auroc_95ci_group_bootstrap": group_bootstrap_auc(
                y_test, calibrated, g_test, seed=SEED + layer
            ),
            "ece_calibrated": float(expected_calibration_error(y_test, calibrated)),
            "brier_calibrated": float(brier_score_loss(y_test, calibrated)),
        },
    }
    bundle = {
        "probe": probe,
        "platt_coef": float(platt.coef_[0, 0]),
        "platt_intercept": float(platt.intercept_[0]),
        "selected_C": chosen_c,
    }
    return report, bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}", args.run_id):
            raise ValueError(
                "run id must contain only letters, numbers, dot, underscore, or hyphen"
            )
        root = repo_root()
        artifacts = root / "artifacts"
        out = artifacts / "probes" / args.run_id
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite {out}")

        rows, _ = load_reviewed_rows(artifacts)
        vectors, model_info = extract_layer_features(rows, allow_download=args.allow_model_download)
        y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        split = np.asarray([str(row["split"]) for row in rows])
        group = np.asarray([str(row["group_id"]) for row in rows])
        qids = np.asarray([str(row["question_id"]) for row in rows])
        layers: dict[str, object] = {}
        fitted: dict[int, object] = {}
        for layer in range(vectors.shape[1]):
            x_by_split = {
                name: vectors[split == name, layer, :] for name in ("train", "validation", "test")
            }
            y_by_split = {name: y[split == name] for name in ("train", "validation", "test")}
            groups_by_split = {
                name: group[split == name] for name in ("train", "validation", "test")
            }
            layers[str(layer)], fitted[layer] = fit_layer(
                x_by_split, y_by_split, groups_by_split, layer
            )
            result = layers[str(layer)]["test"]
            print(
                f"layer={layer:02d} AUROC={result['auroc']:.3f} ECE={result['ece_calibrated']:.3f}",
                flush=True,
            )

        # Fresh path only; records enough data to reuse the extracted features and probes.
        out.mkdir(parents=True)
        np.savez_compressed(
            out / "layer_features.npz",
            question_id=qids,
            split=split,
            group_id=group,
            label=y,
            features=vectors,
        )
        joblib.dump(fitted, out / "layer_probes.joblib")
        payload = {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "run_id": args.run_id,
            "task": "reviewed_task7_layerwise_generation_probe",
            "labels": str(INDEX_REL).replace("\\", "/"),
            "source_run": SOURCE_RUN,
            "reviewer_type": "AI",
            "human_reviewed": False,
            "adjudicated": False,
            "label_policy": "ai_passage_review_v1",
            "feature_protocol": "per-layer hidden state at final generated token",
            "splits": {
                name: {
                    "n": int(np.sum(split == name)),
                    "faithful": int(np.sum((split == name) & (y == 0))),
                    "hallucinated": int(np.sum((split == name) & (y == 1))),
                }
                for name in ("train", "validation", "test")
            },
            "model": model_info,
            "layer_count": int(vectors.shape[1]),
            "feature_dim": int(vectors.shape[2]),
            "selection_protocol": "C selected by five-fold grouped CV within train; Platt map fit on validation; test evaluated once per layer",
            "test_layer_metrics_are_descriptive": True,
            "note": "Do not select a layer based on this test table and report its test score as an unbiased estimate. All labels are one AI-assisted review pass; no external batch is evaluated because it has no layerwise feature cache.",
            "layers": layers,
        }
        (out / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out / 'report.json'}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, json.JSONDecodeError) as exc:
        print(f"Layerwise probe stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
