"""Text-only baselines for the HaluEval pair protocol.

This command does not load Qwen and does not write a probe. The labels are
still the benchmark's supplied right answer and hallucinated answer. A high
score here is evidence of a shortcut, not evidence of hallucination detection
on answers Qwen generates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from glassbox.config import ProbeConfig, repo_root
from glassbox.data.halueval import LABEL_SOURCE, prepare_pairs, read_pinned_snapshot
from glassbox.data.schema import Label
from glassbox.evaluation.metrics import bootstrap_interval, compute_binary_metrics
from glassbox.evaluation.records import write_json
from glassbox.probes.logistic import PROBE_NOT_TRAINED, LogisticProbe
from glassbox.probes.scoring import confusion_counts, overlap_score, select_threshold
from glassbox.reproducibility import package_versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glassbox text-baselines")
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--max-questions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="halueval-text-baselines-001")
    args = parser.parse_args(argv)
    return text_baseline_command(args)


def text_baseline_command(args: argparse.Namespace) -> int:
    try:
        snapshot = args.snapshot or (
            repo_root() / "artifacts" / "datasets" / "halueval" / "qa_data.json"
        )
        if not Path(snapshot).is_file():
            raise FileNotFoundError(f"HaluEval snapshot is not at {snapshot}")
        prepared = prepare_pairs(read_pinned_snapshot(Path(snapshot)), seed=args.seed)
        selected = prepared.assignment.subset(args.max_questions)
        rows = _rows(prepared.pairs, selected)
        report = score_text_rows(rows, seed=args.seed)
        report["protocol"] = LABEL_SOURCE
        report["task"] = "paired-answer discrimination"
        report["not_a_generation_detector"] = True
        report["dataset_sha256"] = prepared.sha256
        report["seed"] = args.seed
        report["max_questions"] = args.max_questions
        report["split_sizes"] = {
            "train": len(selected.train),
            "validation": len(selected.validation),
            "test": len(selected.test),
        }
        report["split_ids"] = {
            "train": list(selected.train),
            "validation": list(selected.validation),
            "test": list(selected.test),
        }
        report["packages"] = package_versions()
        report["filter"] = prepared.report.as_dict()
        destination = repo_root() / "artifacts" / "experiments" / args.run_id / "report.json"
        write_json(destination, report, root=repo_root() / "artifacts")
        _print(report)
        print(f"Wrote {destination}")
    except (OSError, ValueError, RuntimeError, MemoryError, ImportError) as exc:
        print(f"{PROBE_NOT_TRAINED}: {exc}", file=sys.stderr)
        return 1
    return 0


def score_text_rows(rows: list[dict[str, object]], *, seed: int) -> dict[str, object]:
    """Fit text baselines on train, choose thresholds on validation, score test once."""
    grouped = {
        name: [row for row in rows if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    for name, group in grouped.items():
        if {int(row["label"]) for row in group} != {0, 1}:
            raise ValueError(f"{name} split lacks both pair classes")
    length = _fit_numeric(_length_matrix, grouped, seed)
    overlap = _overlap_report(grouped)
    text = _fit_tfidf(grouped, seed)
    shuffled = _shuffled_tfidf(grouped, seed)
    lengths = _length_by_class(grouped["test"])
    return {
        "class_counts": {name: _counts(group) for name, group in grouped.items()},
        "mean_characters_on_test": lengths,
        "answer_length": length,
        "lexical_overlap": overlap,
        "answer_text_tfidf": text,
        "shuffled_label_tfidf": shuffled,
        "note": (
            "Lexical overlap uses the gold answer, which is the faithful string under this protocol. "
            "It is a leaking shortcut, not a fair detector. "
            "The TF-IDF classifier sees only the supplied answer text. "
            "None of these scores are labels of a Qwen generation."
        ),
    }


def _rows(pairs: dict, assignment) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        for question_id in assignment.ids(split):
            pair = pairs[question_id]
            rows.append(
                {
                    "question_id": question_id,
                    "split": split,
                    "label": int(Label.FAITHFUL.value),
                    "answer": pair.right_answer,
                    "reference": pair.right_answer,
                }
            )
            rows.append(
                {
                    "question_id": question_id,
                    "split": split,
                    "label": int(Label.HALLUCINATED.value),
                    "answer": pair.hallucinated_answer,
                    "reference": pair.right_answer,
                }
            )
    return rows


def _fit_numeric(
    matrix_fn, grouped: dict[str, list[dict[str, object]]], seed: int
) -> dict[str, object]:
    probe = LogisticProbe(ProbeConfig(random_state=seed))
    probe.fit(matrix_fn(grouped["train"]), _labels(grouped["train"]))
    threshold = select_threshold(
        _labels(grouped["validation"]),
        probe.predict_proba(matrix_fn(grouped["validation"])),
    )
    scores = probe.predict_proba(matrix_fn(grouped["test"]))
    labels = _labels(grouped["test"])
    return _scored(
        labels, scores, threshold, "Character length and word count. Threshold fit on validation."
    )


def _fit_tfidf(grouped: dict[str, list[dict[str, object]]], seed: int) -> dict[str, object]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
    train_matrix = vectorizer.fit_transform(
        [str(row["answer"]) for row in grouped["train"]]
    ).toarray()
    probe = LogisticProbe(ProbeConfig(random_state=seed))
    probe.fit(train_matrix, _labels(grouped["train"]))
    validation = vectorizer.transform(
        [str(row["answer"]) for row in grouped["validation"]]
    ).toarray()
    threshold = select_threshold(_labels(grouped["validation"]), probe.predict_proba(validation))
    test = vectorizer.transform([str(row["answer"]) for row in grouped["test"]]).toarray()
    scores = probe.predict_proba(test)
    return _scored(
        _labels(grouped["test"]),
        scores,
        threshold,
        "Word and bigram TF-IDF fit on training answers only. The gold answer is not a feature.",
    )


def _shuffled_tfidf(grouped: dict[str, list[dict[str, object]]], seed: int) -> dict[str, object]:
    labels = _labels(grouped["train"]).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
    train_matrix = vectorizer.fit_transform(
        [str(row["answer"]) for row in grouped["train"]]
    ).toarray()
    probe = LogisticProbe(ProbeConfig(random_state=seed))
    probe.fit(train_matrix, labels)
    test = vectorizer.transform([str(row["answer"]) for row in grouped["test"]]).toarray()
    scores = probe.predict_proba(test)
    metrics = compute_binary_metrics(_labels(grouped["test"]), scores, threshold=0.5)
    return {
        "metrics": metrics,
        "threshold": 0.5,
        "note": "Training labels were permuted. The threshold stayed 0.5 and was not tuned on the test split.",
    }


def _overlap_report(grouped: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    labels = _labels(grouped["test"])
    scores = np.asarray(
        [overlap_score(str(row["answer"]), str(row["reference"])) for row in grouped["test"]],
        dtype=np.float64,
    )
    scored = _scored(
        labels,
        scores,
        0.5,
        "Leaking shortcut: the faithful answer is the reference string, so low overlap identifies it.",
    )
    scored["fair_baseline"] = False
    return scored


def _scored(
    labels: np.ndarray, scores: np.ndarray, threshold: float, note: str
) -> dict[str, object]:
    metrics = compute_binary_metrics(labels, scores, threshold=threshold)
    return {
        "metrics": metrics,
        "confusion": confusion_counts(labels, scores, threshold=threshold),
        "threshold": threshold,
        "auroc_interval": bootstrap_interval(
            labels, scores, metric="auroc", threshold=threshold, seed=0
        ),
        "n": int(labels.size),
        "note": note,
    }


def _length_matrix(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray(
        [[len(str(row["answer"])), len(str(row["answer"]).split())] for row in rows],
        dtype=np.float64,
    )


def _labels(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=int)


def _counts(rows: list[dict[str, object]]) -> dict[str, int]:
    labels = _labels(rows)
    return {
        "faithful": int(np.sum(labels == 0)),
        "hallucinated": int(np.sum(labels == 1)),
        "n": int(labels.size),
    }


def _length_by_class(rows: list[dict[str, object]]) -> dict[str, float]:
    faithful = [len(str(row["answer"])) for row in rows if int(row["label"]) == 0]
    hallucinated = [len(str(row["answer"])) for row in rows if int(row["label"]) == 1]
    return {
        "faithful": float(np.mean(faithful)) if faithful else 0.0,
        "hallucinated": float(np.mean(hallucinated)) if hallucinated else 0.0,
    }


def _print(report: dict[str, object]) -> None:
    print("Paired-answer text baselines. These are not generation-detector scores.")
    print(f"Split sizes: {report['split_sizes']}")
    for name in ("answer_length", "lexical_overlap", "answer_text_tfidf", "shuffled_label_tfidf"):
        block = report[name]
        metrics = block.get("metrics") if isinstance(block, dict) else None
        if not isinstance(metrics, dict):
            print(f"  {name}: not scored")
            continue
        print(
            f"  {name}: precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} auroc={metrics['auroc']:.3f} "
            f"auprc={metrics['auprc']:.3f} fpr={metrics['false_positive_rate']:.3f} n={block.get('n')}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
