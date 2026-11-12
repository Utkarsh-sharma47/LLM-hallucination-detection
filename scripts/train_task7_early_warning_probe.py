"""Train and evaluate a prefix-level warning probe on reviewed Task 7 answers.

Prefix labels are derived with the existing passage-grounded generation evidence rule.
Knowledge groups remain isolated across train, validation, and test. The threshold is
chosen on validation prefix labels; lead time is reported only on the held-out test set.
The labels are one AI-reviewed Task 7 pass, so this is a research measurement, not a
validated real-time warning system.

Run offline from the repository root::

    .venv/Scripts/python.exe scripts/train_task7_early_warning_probe.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime

import joblib
import numpy as np
from layerwise_reviewed_generation_probe import (
    MODEL_NAME,
    MODEL_REVISION,
    load_reviewed_rows,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from glassbox.config import repo_root
from glassbox.data.evidence import label_halueval_generation
from glassbox.evaluation.early_warning import (
    EVENT_DEFINITION,
    EVENT_PROTOCOL_ID,
    event_index_from_prefix_labels,
)
from glassbox.evaluation.metrics import early_warning_lead
from glassbox.models.backend import (
    GenerationSettings,
    HuggingFaceBackend,
    ModelSpec,
    completion_prefix_hidden_states,
)
from glassbox.probes.generation_ai_review import SOURCE_RUN
from glassbox.probes.scoring import select_threshold

DEFAULT_RUN_ID = "early-warning-task7-ai-v1-001"
LAYER_INDEX = 24
SEED = 0
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def _binary_prefix_data(rows: list[dict[str, object]], prefix_records: list[dict[str, object]]):
    vectors: dict[str, list[np.ndarray]] = {name: [] for name in ("train", "validation", "test")}
    labels: dict[str, list[int]] = {name: [] for name in vectors}
    groups: dict[str, list[str]] = {name: [] for name in vectors}
    for row, record in zip(rows, prefix_records, strict=True):
        split = str(row["split"])
        binary = [
            index
            for index, value in enumerate(record["prefix_labels"])
            if value in {"FAITHFUL", "HALLUCINATED"}
        ]
        for index in binary:
            vectors[split].append(np.asarray(record["vectors"][index], dtype=np.float32))
            labels[split].append(int(record["prefix_labels"][index] == "HALLUCINATED"))
            groups[split].append(str(row["group_id"]))
    return (
        {
            name: np.stack(values) if values else np.empty((0, 896), dtype=np.float32)
            for name, values in vectors.items()
        },
        {name: np.asarray(values, dtype=np.int64) for name, values in labels.items()},
        {name: np.asarray(values) for name, values in groups.items()},
    )


def _make_probe(c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=SEED),
    )


def _select_c(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, dict[str, float]]:
    split = GroupKFold(n_splits=5)
    scores: dict[str, float] = {}
    for c in C_GRID:
        aucs = []
        for train_index, valid_index in split.split(x, y, groups):
            if set(y[train_index]) != {0, 1} or set(y[valid_index]) != {0, 1}:
                continue
            model = _make_probe(c).fit(x[train_index], y[train_index])
            aucs.append(
                float(roc_auc_score(y[valid_index], model.predict_proba(x[valid_index])[:, 1]))
            )
        if not aucs:
            raise ValueError("Grouped cross-validation produced no two-class prefix folds")
        scores[str(c)] = float(np.mean(aucs))
    chosen = max(C_GRID, key=lambda c: (scores[str(c)], -c))
    return chosen, scores


def _prefix_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    return {
        "n_binary_prefixes": int(len(y)),
        "faithful_prefixes": int((y == 0).sum()),
        "hallucinated_prefixes": int((y == 1).sum()),
        "auroc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "f1_at_validation_threshold": float(f1_score(y, scores >= threshold, zero_division=0)),
        "accuracy_at_validation_threshold": float(accuracy_score(y, scores >= threshold)),
    }


def _evaluate_events(
    rows: list[dict[str, object]],
    prefix_records: list[dict[str, object]],
    probe,
    threshold: float,
) -> dict[str, object]:
    event_examples = []
    faithful_alerts = []
    missing_events = 0
    label_disagreements = 0
    for row, record in zip(rows, prefix_records, strict=True):
        prefix_labels = list(record["prefix_labels"])
        if not prefix_labels:
            continue
        event_index = event_index_from_prefix_labels(prefix_labels)
        final_auto = prefix_labels[-1]
        if final_auto != str(record["reviewed_label"]):
            label_disagreements += 1
        scores = probe.predict_proba(np.stack(record["vectors"]))[:, 1]
        flags = [bool(value >= threshold) for value in scores]
        reviewed = str(record["reviewed_label"])
        if reviewed == "FAITHFUL":
            faithful_alerts.append(any(flags))
        if reviewed != "HALLUCINATED":
            continue
        if event_index is None:
            missing_events += 1
            continue
        lead = early_warning_lead(flags, event_index)
        event_examples.append(
            {
                "question_id": row["question_id"],
                "event_index": event_index,
                "first_flag_index": event_index - lead if lead is not None else None,
                "lead_tokens": lead,
                "timely_warning": lead is not None,
            }
        )
    leads = [int(item["lead_tokens"]) for item in event_examples if item["lead_tokens"] is not None]
    return {
        # load_reviewed_rows encodes reviewed labels as 0=FAITHFUL, 1=HALLUCINATED.
        "reviewed_hallucinated_answers": sum(int(row["label"]) == 1 for row in rows),
        "hallucinated_answers_with_prefix_event": len(event_examples),
        "hallucinated_answers_without_prefix_event": missing_events,
        "events_flagged_at_first_prefix": sum(
            item["first_flag_index"] == 0 for item in event_examples
        ),
        "timely_warning_count": len(leads),
        "timely_warning_coverage": float(len(leads) / len(event_examples))
        if event_examples
        else None,
        "mean_lead_tokens_among_timely_warnings": float(np.mean(leads)) if leads else None,
        "median_lead_tokens_among_timely_warnings": float(np.median(leads)) if leads else None,
        "faithful_answer_false_warning_rate": (
            float(np.mean(faithful_alerts)) if faithful_alerts else None
        ),
        "reviewed_vs_prefix_final_label_disagreements": label_disagreements,
        "event_examples": event_examples,
    }


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
        sidecar = out / "early_warning_evaluation.json"
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite {out}")

        rows, _ = load_reviewed_rows(artifacts)
        backend = HuggingFaceBackend(
            ModelSpec(
                name=MODEL_NAME,
                revision=MODEL_REVISION,
                device="auto",
                dtype="bfloat16",
                allow_download=args.allow_model_download,
                local_files_only=not args.allow_model_download,
            )
        )
        print(f"Loading pinned model {MODEL_NAME}@{MODEL_REVISION} ...", flush=True)
        loaded = backend.ensure_loaded()
        if loaded.resolved_revision != MODEL_REVISION:
            raise ValueError("Loaded model revision differs from frozen Task 7")
        settings = GenerationSettings(
            max_new_tokens=32,
            temperature=0.0,
            top_k=5,
            layer_index=LAYER_INDEX,
            position="last_generated_token",
        )

        prefix_records = []
        answer_token_count_mismatches = 0
        layer24_alignment_cosines = []
        prompt_mode = None
        for index, row in enumerate(rows, start=1):
            vectors, prefix_texts, prompt_count, token_count, hidden_count, layer, mode, _prompt = (
                completion_prefix_hidden_states(
                    loaded, str(row["question"]), str(row["answer"]), settings
                )
            )
            if prompt_count != row["prompt_piece_count"]:
                raise ValueError(f"Prompt token count mismatch for {row['question_id']}")
            if token_count != row["generated_piece_count"]:
                answer_token_count_mismatches += 1
            else:
                original = np.asarray(row["saved_layer24"], dtype=np.float32)
                final_prefix = np.asarray(vectors[-1], dtype=np.float32)
                norm = float(np.linalg.norm(original) * np.linalg.norm(final_prefix))
                layer24_alignment_cosines.append(
                    float(np.dot(original, final_prefix) / norm) if norm else 0.0
                )
            if layer != LAYER_INDEX or hidden_count != 25:
                raise ValueError("Unexpected layer or hidden-state count")
            prefix_labels = [
                label_halueval_generation(
                    question=str(row["question"]),
                    knowledge=str(row["knowledge"]),
                    right_answer=str(row["right_answer"]),
                    hallucinated_answer=str(row["hallucinated_answer"]),
                    answer=text,
                ).label.name
                for text in prefix_texts
            ]
            prefix_records.append(
                {
                    "question_id": str(row["question_id"]),
                    "split": str(row["split"]),
                    "reviewed_label": "HALLUCINATED" if int(row["label"]) == 1 else "FAITHFUL",
                    "prefix_texts": prefix_texts,
                    "prefix_labels": prefix_labels,
                    "vectors": vectors,
                    "token_count": token_count,
                }
            )
            prompt_mode = mode
            if index % 50 == 0 or index == len(rows):
                print(f"Scored prefix evidence for {index}/{len(rows)} answers", flush=True)

        if len(layer24_alignment_cosines) < 100:
            raise ValueError(
                "Fewer than 100 rows could be used to verify layer-24 replay alignment"
            )
        alignment_median = float(np.median(layer24_alignment_cosines))
        alignment_p05 = float(np.percentile(layer24_alignment_cosines, 5))
        if alignment_median < 0.999 or alignment_p05 < 0.95:
            raise ValueError(
                "Replayed prefix states failed the layer-24 alignment check "
                f"(median={alignment_median:.6f}, p05={alignment_p05:.6f})"
            )

        x, y, groups = _binary_prefix_data(rows, prefix_records)
        for split in ("train", "validation", "test"):
            if len(x[split]) < 1 or set(y[split]) != {0, 1}:
                raise ValueError(f"Prefix split {split} is empty or lacks both classes")

        chosen_c, cv_scores = _select_c(x["train"], y["train"], groups["train"])
        probe = _make_probe(chosen_c).fit(x["train"], y["train"])
        validation_scores = probe.predict_proba(x["validation"])[:, 1]
        threshold = float(select_threshold(y["validation"], validation_scores))
        test_scores = probe.predict_proba(x["test"])[:, 1]
        prefix_test = _prefix_metrics(y["test"], test_scores, threshold)
        test_rows = [row for row in rows if row["split"] == "test"]
        test_records = [record for record in prefix_records if record["split"] == "test"]
        event_metrics = _evaluate_events(test_rows, test_records, probe, threshold)

        flattened_vectors = []
        flattened_labels = []
        flattened_ids = []
        flattened_splits = []
        flattened_groups = []
        flattened_steps = []
        all_prefix_vectors = []
        all_prefix_labels = []
        all_prefix_ids = []
        all_prefix_splits = []
        all_prefix_groups = []
        all_prefix_steps = []
        all_prefix_texts = []
        prefix_event_rows = []
        for row, record in zip(rows, prefix_records, strict=True):
            all_scores = probe.predict_proba(np.stack(record["vectors"]))[:, 1]
            all_flags = [bool(score >= threshold) for score in all_scores]
            event_index = event_index_from_prefix_labels(record["prefix_labels"])
            for step, (vector, label, text_value, _score) in enumerate(
                zip(
                    record["vectors"],
                    record["prefix_labels"],
                    record["prefix_texts"],
                    all_scores,
                    strict=True,
                )
            ):
                all_prefix_vectors.append(vector)
                all_prefix_labels.append(label)
                all_prefix_ids.append(str(row["question_id"]))
                all_prefix_splits.append(str(row["split"]))
                all_prefix_groups.append(str(row["group_id"]))
                all_prefix_steps.append(step)
                all_prefix_texts.append(text_value)
            for step, (vector, label) in enumerate(
                zip(record["vectors"], record["prefix_labels"], strict=True)
            ):
                if label not in {"FAITHFUL", "HALLUCINATED"}:
                    continue
                flattened_vectors.append(vector)
                flattened_labels.append(int(label == "HALLUCINATED"))
                flattened_ids.append(str(row["question_id"]))
                flattened_splits.append(str(row["split"]))
                flattened_groups.append(str(row["group_id"]))
                flattened_steps.append(step)
            lead = early_warning_lead(all_flags, event_index) if event_index is not None else None
            prefix_event_rows.append(
                {
                    "question_id": str(row["question_id"]),
                    "split": str(row["split"]),
                    "reviewed_label": str(record["reviewed_label"]),
                    "event_index": event_index,
                    "early_warning_lead": lead,
                    "prefixes": [
                        {
                            "index": step,
                            "text": text_value,
                            "evidence_label": str(label),
                            "probe_score": float(score),
                            "flagged": all_flags[step],
                        }
                        for step, (text_value, label, score) in enumerate(
                            zip(
                                record["prefix_texts"],
                                record["prefix_labels"],
                                all_scores,
                                strict=True,
                            )
                        )
                    ],
                }
            )

        payload = {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "run_id": args.run_id,
            "task": "task7_reviewed_prefix_early_warning",
            "source_run": SOURCE_RUN,
            "label_protocol": EVENT_PROTOCOL_ID,
            "event_definition": EVENT_DEFINITION,
            "prefix_label_source": "label_halueval_generation applied to each decoded answer prefix",
            "reviewer_type": "AI",
            "human_reviewed": False,
            "layer_index": LAYER_INDEX,
            "position": "each generated-token prefix",
            "model_name": loaded.spec.name,
            "model_revision": loaded.resolved_revision,
            "dtype": loaded.dtype,
            "prompt_mode": prompt_mode,
            "hidden_state_count": 25,
            "feature_dim": int(x["train"].shape[1]),
            "answer_token_count_mismatches": answer_token_count_mismatches,
            "exact_answer_token_count_rows": len(layer24_alignment_cosines),
            "layer24_alignment_median_cosine": alignment_median,
            "layer24_alignment_p05_cosine": alignment_p05,
            "training": {
                "C": chosen_c,
                "cv_auroc_by_C": cv_scores,
                "train_binary_prefixes": int(len(y["train"])),
                "validation_binary_prefixes": int(len(y["validation"])),
                "test_binary_prefixes": int(len(y["test"])),
                "threshold": threshold,
                "threshold_selected_on": "validation prefix labels only",
            },
            "test_prefix_metrics": prefix_test,
            "test_early_warning": event_metrics,
            "note": (
                "Research evaluation only. Prefix labels use the existing automatic evidence protocol; "
                "final answer review labels are one AI-assisted pass. Scores are not calibrated risks. "
                "Original generated token IDs were not saved, so answers were re-tokenized from text."
            ),
        }

        out.mkdir(parents=True)
        np.savez_compressed(
            out / "prefix_features.npz",
            vectors=np.stack(flattened_vectors).astype(np.float32),
            labels=np.asarray(flattened_labels, dtype=np.int8),
            question_id=np.asarray(flattened_ids),
            split=np.asarray(flattened_splits),
            group_id=np.asarray(flattened_groups),
            prefix_index=np.asarray(flattened_steps, dtype=np.int32),
        )
        np.savez_compressed(
            out / "prefix_records.npz",
            vectors=np.stack(all_prefix_vectors).astype(np.float32),
            evidence_label=np.asarray(all_prefix_labels),
            question_id=np.asarray(all_prefix_ids),
            split=np.asarray(all_prefix_splits),
            group_id=np.asarray(all_prefix_groups),
            prefix_index=np.asarray(all_prefix_steps, dtype=np.int32),
            prefix_text=np.asarray(all_prefix_texts),
        )
        (out / "prefix_evidence.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in prefix_event_rows),
            encoding="utf-8",
        )
        joblib.dump(
            {"probe": probe, "threshold": threshold, "layer_index": LAYER_INDEX},
            out / "prefix_probe.joblib",
        )
        (out / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "threshold": threshold,
                    "test_prefix_metrics": prefix_test,
                    "test_early_warning": event_metrics,
                },
                indent=2,
            )
        )
        print(f"Wrote {out / 'report.json'}")
        print(f"Wrote HTML-report data {sidecar}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, json.JSONDecodeError) as exc:
        print(f"Early-warning evaluation stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
