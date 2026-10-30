"""Compare last-token, mean-pool, length, and TF-IDF on one frozen split.

The live pair probe and ``halueval-generation-001`` are not replaced.
Thresholds for new variants are chosen on validation only. Last-token scores
from a historical run stay frozen when supplied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from glassbox.config import ProbeConfig, repo_root
from glassbox.data.schema import Label
from glassbox.evaluation.generation_calibration import (
    PAIR_THRESHOLD,
    evaluate_generation_calibration_gate,
)
from glassbox.evaluation.length_shortcut import summarize_length_effects
from glassbox.evaluation.metrics import BOOTSTRAP_MIN_PER_CLASS
from glassbox.probes.generation_eval import (
    PROTECTED_PROBE_NAMES,
    generation_experiment_dir,
    generation_probe_dir,
    generation_text_baselines,
    interpret_detector,
    score_binary,
)
from glassbox.probes.logistic import LogisticProbe
from glassbox.probes.preprocess import apply_standard_scaler, fit_standard_scaler
from glassbox.probes.scoring import select_threshold

VARIANT_NAMES = ("frozen_final_piece", "mean_pool", "answer_length", "answer_tfidf")


def evaluate_generation_variants(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int = 0,
    frozen_threshold: float | None = None,
    pair_probe_path: Path | None = None,
) -> dict[str, object]:
    """Fit non-last-token variants on train/validation and score test once."""
    binary = [_binary_row(row) for row in rows if _is_binary(row)]
    grouped = {
        name: [row for row in binary if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    counts = {name: _counts(group) for name, group in grouped.items()}
    report: dict[str, object] = {
        "task": "generation_variant_comparison",
        "live_probe_untouched": True,
        "historical_generation_probe_untouched": True,
        "pair_threshold_unchanged": PAIR_THRESHOLD,
        "class_counts": counts,
        "target_per_class": BOOTSTRAP_MIN_PER_CLASS,
        "target_met": all(
            counts[split]["faithful"] >= BOOTSTRAP_MIN_PER_CLASS
            and counts[split]["hallucinated"] >= BOOTSTRAP_MIN_PER_CLASS
            for split in ("validation", "test")
        ),
        "variants": {},
        "successful_detector": False,
        "pair_probe_path": None if pair_probe_path is None else str(pair_probe_path),
        "pair_probe_rewritten": False,
    }
    if any(set(int(row["label"]) for row in grouped[name]) != {0, 1} for name in grouped):
        report["note"] = (
            "A binary split does not contain both classes, so variant metrics were not fit."
        )
        report["detector_conclusion"] = {
            "successful_detector": False,
            "reason": report["note"],
        }
        return report
    baselines = generation_text_baselines(_baseline_rows(binary), seed=seed)
    report["text_baselines"] = baselines
    report["variants"]["answer_length"] = baselines.get("answer_length")
    report["variants"]["answer_tfidf"] = baselines.get("answer_text_tfidf")
    if any(row.get("probe_score") is not None for row in grouped["test"]):
        cutoff = 0.5 if frozen_threshold is None else float(frozen_threshold)
        report["variants"]["frozen_final_piece"] = {
            **score_binary(
                [int(row["label"]) for row in grouped["test"]],
                [float(row["probe_score"]) for row in grouped["test"]],
                [str(row["question_id"]) for row in grouped["test"]],
                threshold=cutoff,
                seed=seed,
            ),
            "threshold": cutoff,
            "threshold_was_refit": False,
            "detail": (
                "Frozen last-generated-token scores from the historical generation run. "
                "The threshold was not chosen on this test split."
            ),
        }
    mean_rows = [row for row in binary if row.get("answer_mean") is not None]
    if mean_rows:
        report["variants"]["mean_pool"] = _fit_vector_variant(
            [row for row in mean_rows if row["split"] == "train"],
            [row for row in mean_rows if row["split"] == "validation"],
            [row for row in mean_rows if row["split"] == "test"],
            field="answer_mean",
            seed=seed,
        )
    else:
        report["variants"]["mean_pool"] = {
            "available": False,
            "note": "Mean-pooled generated hidden states were not stored on these rows.",
        }
    last_token = report["variants"].get("frozen_final_piece")
    probe_metrics = None
    if isinstance(last_token, dict) and isinstance(last_token.get("metrics"), dict):
        probe_metrics = last_token["metrics"]
    conclusion = interpret_detector(
        probe_metrics if isinstance(probe_metrics, dict) else None,
        baselines if isinstance(baselines, dict) else {},
    )
    conclusion["successful_detector"] = False
    report["detector_conclusion"] = conclusion
    report["length_shortcut"] = summarize_length_effects(
        grouped["test"],
        threshold=PAIR_THRESHOLD if frozen_threshold is None else float(frozen_threshold),
    )
    gate_rows = [
        {
            "question_id": row["question_id"],
            "split": row["split"],
            "label": "HALLUCINATED" if int(row["label"]) == 1 else "FAITHFUL",
            "probe_score": row.get("probe_score"),
            "split_overlap": False,
        }
        for row in binary
    ]
    report["generation_calibration_gate"] = evaluate_generation_calibration_gate(
        gate_rows,
        threshold=PAIR_THRESHOLD,
        pair_probe_path=pair_probe_path,
    )
    if not report["target_met"]:
        report["note"] = (
            "Validation or test is below 20 rows per class. Variant metrics are "
            "diagnostic and inconclusive for a detector claim."
        )
    return report


def _fit_vector_variant(
    train: Sequence[Mapping[str, object]],
    validation: Sequence[Mapping[str, object]],
    test: Sequence[Mapping[str, object]],
    *,
    field: str,
    seed: int,
) -> dict[str, object]:
    if any(
        set(int(row["label"]) for row in group) != {0, 1} for group in (train, validation, test)
    ):
        return {
            "available": False,
            "note": "Mean-pool variant was not fit because a split lacked both classes.",
        }
    train_matrix = np.vstack([np.asarray(row[field], dtype=np.float64) for row in train])
    mean, scale = fit_standard_scaler(train_matrix)
    probe = LogisticProbe(ProbeConfig(random_state=seed))
    probe.fit(
        apply_standard_scaler(train_matrix, mean, scale),
        np.asarray([int(row["label"]) for row in train], dtype=int),
    )
    validation_matrix = apply_standard_scaler(
        np.vstack([np.asarray(row[field], dtype=np.float64) for row in validation]),
        mean,
        scale,
    )
    threshold = select_threshold(
        np.asarray([int(row["label"]) for row in validation], dtype=int),
        probe.predict_proba(validation_matrix),
    )
    test_matrix = apply_standard_scaler(
        np.vstack([np.asarray(row[field], dtype=np.float64) for row in test]),
        mean,
        scale,
    )
    scores = probe.predict_proba(test_matrix)
    scored = score_binary(
        [int(row["label"]) for row in test],
        scores.tolist(),
        [str(row["question_id"]) for row in test],
        threshold=threshold,
        seed=seed,
    )
    scored["threshold"] = threshold
    scored["threshold_was_refit"] = True
    scored["available"] = True
    scored["saved_as_live_probe"] = False
    scored["detail"] = (
        "Mean of generated pieces at the registered layer. Scaler and threshold used "
        "training and validation only. This artifact is not installed as the live probe."
    )
    return scored


def _binary_row(row: Mapping[str, object]) -> dict[str, object]:
    label = row.get("label")
    name = str(row.get("label_name") or "")
    if name == "FAITHFUL":
        value = 0
    elif name == "HALLUCINATED":
        value = 1
    else:
        value = int(label)
    payload = dict(row)
    payload["label"] = value
    payload["answer"] = str(row.get("answer") or "")
    payload["question_id"] = str(row["question_id"])
    payload["split"] = str(row["split"])
    payload["group_id"] = str(row.get("group_id") or row["question_id"])
    return payload


def _is_binary(row: Mapping[str, object]) -> bool:
    name = str(row.get("label_name") or "")
    if name in {"FAITHFUL", "HALLUCINATED"}:
        return True
    label = row.get("label")
    return label in {0, 1, Label.FAITHFUL, Label.HALLUCINATED}


def _counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {
        "questions": len(rows),
        "faithful": sum(int(row["label"]) == 0 for row in rows),
        "hallucinated": sum(int(row["label"]) == 1 for row in rows),
    }


def _baseline_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    converted = []
    for row in rows:
        item = dict(row)
        item["label"] = int(row["label"])
        item["right_answer"] = str(row.get("right_answer") or row.get("reference_answer") or "")
        item["false_spans"] = list(row.get("false_spans") or [])
        converted.append(item)
    return converted


def variants_command(args: object) -> int:
    """Compare variants on a frozen generation split. Does not replace saved probes."""
    import json
    import sys

    from glassbox.evaluation.records import write_json

    try:
        run_id = str(args.run_id)
        if run_id in PROTECTED_PROBE_NAMES or run_id.startswith("final-hidden-state"):
            raise ValueError(f"Refusing protected run id {run_id}")
        if run_id == "halueval-generation-001":
            raise ValueError("Refusing to overwrite the historical generation run")
        artifacts = repo_root() / "artifacts"
        directory = generation_experiment_dir(run_id)
        probe_dir = generation_probe_dir(run_id)
        if directory.exists() or probe_dir.exists():
            raise FileExistsError("Refusing to overwrite an existing variant run")
        predictions = Path(args.predictions)
        payload = json.loads(predictions.read_text(encoding="utf-8"))
        rows = payload.get("scored_rows") or payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("predictions file has no rows")
        joined = [dict(row) for row in rows if isinstance(row, dict)]
        if args.features:
            joined = _attach_vectors(joined, Path(args.features))
        pair_before = None
        pair_path = Path(args.pair_probe) if args.pair_probe else None
        if pair_path is not None and (pair_path / "metadata.json").is_file():
            pair_before = (pair_path / "metadata.json").read_bytes()
        report = evaluate_generation_variants(
            joined,
            seed=int(args.seed),
            frozen_threshold=None if args.threshold is None else float(args.threshold),
            pair_probe_path=pair_path,
        )
        write_json(directory / "report.json", report, root=artifacts)
        if pair_before is not None and (pair_path / "metadata.json").read_bytes() != pair_before:
            raise RuntimeError("Variant comparison rewrote the pair-probe metadata")
        if probe_dir.exists():
            raise RuntimeError("Variant comparison wrote a live probe directory")
        print(report["detector_conclusion"]["reason"])
        print(f"Wrote {directory / 'report.json'}")
        return 0
    except (OSError, ValueError, FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Variant comparison stopped: {exc}", file=sys.stderr)
        return 1


def _attach_vectors(rows: list[dict[str, object]], path: Path) -> list[dict[str, object]]:
    data = np.load(path)
    by_id = {
        str(question_id): np.asarray(vector, dtype=np.float64)
        for question_id, vector in zip(data["question_id"], data["vectors"], strict=True)
    }
    attached = []
    for row in rows:
        copied = dict(row)
        vector = by_id.get(str(row.get("question_id")))
        if vector is not None:
            copied["vector"] = vector
        attached.append(copied)
    return attached
