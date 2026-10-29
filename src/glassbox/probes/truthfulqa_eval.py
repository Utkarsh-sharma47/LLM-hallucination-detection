"""Score the saved HaluEval probe on TruthfulQA generations.

TruthfulQA is not added to training. The operating threshold stays the one
frozen on the HaluEval validation split.
"""

from __future__ import annotations

import argparse
import resource
import sys
from pathlib import Path

import numpy as np

from glassbox.config import repo_root
from glassbox.data.schema import Label
from glassbox.data.truthfulqa import (
    LABEL_SOURCE,
    PINNED_SHA256,
    TruthfulItem,
    label_generated_answer,
    read_pinned_snapshot,
    select_items,
)
from glassbox.demo_config import default_probe_dir, load_demo_config
from glassbox.evaluation.metrics import BOOTSTRAP_MIN_PER_CLASS, compute_binary_metrics
from glassbox.evaluation.records import write_json
from glassbox.models.backend import HuggingFaceBackend
from glassbox.probes.logistic import PROBE_NOT_TRAINED
from glassbox.probes.scoring import confusion_counts, expected_calibration_error, overlap_score
from glassbox.probes.store import check_compatibility, experimental_score, load_probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glassbox eval-truthfulqa")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--probe", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--max-questions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="truthfulqa-generation-001")
    args = parser.parse_args(argv)
    return evaluate_command(args)


def evaluate_command(args: argparse.Namespace) -> int:
    try:
        snapshot = args.snapshot or (
            repo_root() / "artifacts" / "datasets" / "truthfulqa" / "TruthfulQA.csv"
        )
        if not Path(snapshot).is_file():
            raise FileNotFoundError(
                f"TruthfulQA CSV is not at {snapshot}. Download it before evaluation. "
                "This command does not train on TruthfulQA."
            )
        items = read_pinned_snapshot(Path(snapshot))
        chosen = select_items(items, max_questions=args.max_questions, seed=args.seed)
        probe = load_probe(args.probe or default_probe_dir())
        config = load_demo_config(args.config, offline=args.offline, model_name=args.model)
        backend = HuggingFaceBackend(config.spec)
        print(f"Loading {config.spec.name} once for TruthfulQA generation...", flush=True)
        loaded = backend.ensure_loaded()
        if any(parameter.requires_grad for parameter in loaded.model.parameters()):
            raise RuntimeError("Qwen parameters were not frozen")
        print(
            f"TruthfulQA rows in file={len(items)} evaluating={len(chosen)} "
            f"revision={loaded.resolved_revision} dtype={loaded.dtype}",
            flush=True,
        )
        rows = [_one(backend, probe, item, config.generation) for item in chosen]
        report = _report(rows, probe.metadata, loaded)
        root = repo_root() / "artifacts"
        destination = root / "experiments" / args.run_id / "report.json"
        write_json(destination, report, root=root)
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Wrote {destination}")
        print(f"Peak RSS: {peak_kb / 1024:.0f} MiB")
        _print(report)
    except (OSError, ValueError, RuntimeError, MemoryError, ImportError) as exc:
        print(f"{PROBE_NOT_TRAINED}: {exc}", file=sys.stderr)
        return 1
    return 0


def _one(backend, probe, item: TruthfulItem, settings) -> dict[str, object]:
    result = backend.generate(item.question, settings)
    label, detail = label_generated_answer(result.answer, item)
    score = None
    score_note = "probe not applied"
    if result.feature is None or result.feature_layer is None or result.hidden_state_count is None:
        score_note = result.note or "hidden state unavailable"
    else:
        try:
            check_compatibility(
                probe,
                model_name=result.model_name,
                model_revision=result.model_revision,
                layer_index=result.feature_layer,
                hidden_state_count=result.hidden_state_count,
                position=result.position,
                prompt_mode=result.prompt_mode,
                feature_dim=int(result.feature.shape[0]),
                dtype=result.dtype,
            )
            score = experimental_score(probe, result.feature)
            score_note = "experimental score from the HaluEval probe"
        except ValueError as exc:
            score_note = str(exc)
    print(f"{item.question_id} {label.name}: {result.answer}", flush=True)
    return {
        "question_id": item.question_id,
        "category": item.category,
        "question": item.question,
        "answer": result.answer,
        "label": label.name,
        "label_detail": detail,
        "probe_score": score,
        "score_note": score_note,
        "model_revision": result.model_revision,
        "dtype": result.dtype,
        "overlap_baseline": _overlap_baseline(result.answer, item),
    }


def _overlap_baseline(answer: str, item: TruthfulItem) -> float:
    incorrect = max((overlap_score(answer, phrase) for phrase in item.incorrect), default=0.0)
    correct = max((overlap_score(answer, phrase) for phrase in item.correct), default=0.0)
    # overlap_score is high when the strings share few tokens. A larger gap means
    # the answer is closer to a listed incorrect reference than to a correct one.
    return float(correct - incorrect)


def _report(rows: list[dict[str, object]], metadata: dict, loaded) -> dict[str, object]:
    labeled = [
        row
        for row in rows
        if row["label"] in {Label.FAITHFUL.name, Label.HALLUCINATED.name}
        and row["probe_score"] is not None
    ]
    counts = {
        "faithful": sum(row["label"] == Label.FAITHFUL.name for row in rows),
        "hallucinated": sum(row["label"] == Label.HALLUCINATED.name for row in rows),
        "ambiguous": sum(row["label"] == Label.AMBIGUOUS.name for row in rows),
        "unvalidated": sum(row["label"] == Label.UNVALIDATED.name for row in rows),
        "scored": len(labeled),
    }
    threshold = metadata.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        threshold = 0.5
    metrics = None
    confusion = None
    ece = None
    baseline = None
    note = (
        "TruthfulQA is not the HaluEval pair task. Labels come only from asserted reference answers. "
        "Unvalidated generations are excluded from the metrics. The threshold was frozen on HaluEval validation."
    )
    labeled_faithful = sum(row["label"] == Label.FAITHFUL.name for row in labeled)
    labeled_hallucinated = sum(row["label"] == Label.HALLUCINATED.name for row in labeled)
    both_classes = {row["label"] for row in labeled} == {
        Label.FAITHFUL.name,
        Label.HALLUCINATED.name,
    }
    smaller = min(labeled_faithful, labeled_hallucinated)
    if not both_classes:
        note = f"{note} Both classes were not present in the scored labeled rows, so AUROC was not computed."
    elif smaller < BOOTSTRAP_MIN_PER_CLASS:
        note = (
            f"{note} Classification metrics and ECE were not published. "
            f"The smaller labeled class has {smaller} scored rows, below {BOOTSTRAP_MIN_PER_CLASS}."
        )
    else:
        y_true = np.asarray(
            [1 if row["label"] == Label.HALLUCINATED.name else 0 for row in labeled], dtype=int
        )
        scores = np.asarray([float(row["probe_score"]) for row in labeled], dtype=np.float64)
        try:
            metrics = compute_binary_metrics(y_true, scores, threshold=float(threshold))
            confusion = confusion_counts(y_true, scores, threshold=float(threshold))
            ece = expected_calibration_error(y_true, scores)
        except ValueError as exc:
            note = f"{note} Metrics were not computed: {exc}"
        baseline_scores = np.asarray(
            [float(row["overlap_baseline"]) for row in labeled], dtype=np.float64
        )
        # Map the signed overlap gap into [0, 1] without fitting on these labels.
        baseline_probability = 1.0 / (1.0 + np.exp(-baseline_scores))
        try:
            baseline = compute_binary_metrics(y_true, baseline_probability, threshold=0.5)
        except ValueError as exc:
            baseline = {"note": str(exc)}
    return {
        "schema_version": 1,
        "protocol": LABEL_SOURCE,
        "dataset_sha256": PINNED_SHA256,
        "model_name": loaded.spec.name,
        "model_revision": loaded.resolved_revision,
        "dtype": loaded.dtype,
        "probe_calibration": metadata.get("calibration"),
        "probe_calibration_validated": metadata.get("calibration_validated") is True,
        "threshold": float(threshold),
        "class_counts": counts,
        "metrics": metrics,
        "confusion": confusion,
        "ece": ece,
        "overlap_baseline": baseline,
        "note": note,
        "rows": rows,
    }


def _print(report: dict[str, object]) -> None:
    print(f"Class counts: {report['class_counts']}")
    print(f"Threshold from HaluEval validation: {report['threshold']}")
    metrics = report["metrics"]
    if isinstance(metrics, dict):
        for name, value in metrics.items():
            print(f"  truthfulqa {name}: {value:.3f}")
    else:
        print("  truthfulqa metrics: not computed")
    if report["confusion"] is not None:
        print(f"  confusion: {report['confusion']}")
    if report["ece"] is not None:
        print(f"  ECE of experimental scores: {report['ece']:.3f}")
    print(report["note"])


if __name__ == "__main__":
    raise SystemExit(main())
