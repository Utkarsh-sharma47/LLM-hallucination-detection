"""Diagnostics for length and wording shortcuts in probe scores.

This module does not train, does not change threshold 0.65, and does not
rewrite the live pair probe. High scores on long faithful answers are
recorded as a format effect, not as a factual-risk percentage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from glassbox.evaluation.generation_calibration import PAIR_THRESHOLD
from glassbox.evaluation.metrics import (
    BOOTSTRAP_MIN_PER_CLASS,
    bootstrap_grouped_interval,
    compute_binary_metrics,
)
from glassbox.probes.scoring import confusion_counts
from glassbox.verification import CuratedFact

FORMATS = (
    "short_faithful",
    "sentence_faithful",
    "short_hallucinated",
    "sentence_hallucinated",
)
LENGTH_MATCH_TOLERANCE = 15


def controlled_variants(fact: CuratedFact) -> list[dict[str, object]]:
    """Four length-controlled answers for one curated question."""
    if not fact.accepted or not fact.false_claims:
        return []
    supported = str(fact.accepted[0]).strip()
    contradicted = str(fact.false_claims[0]).strip()
    return [
        _variant(fact, "short_faithful", supported, 0),
        _variant(fact, "sentence_faithful", f"The answer is {supported}.", 0),
        _variant(fact, "short_hallucinated", contradicted, 1),
        _variant(fact, "sentence_hallucinated", f"The answer is {contradicted}.", 1),
    ]


def summarize_length_effects(
    rows: Sequence[Mapping[str, object]],
    *,
    threshold: float = PAIR_THRESHOLD,
    seed: int = 0,
) -> dict[str, object]:
    """Score distributions, format FPR, correlation, and length-matched metrics."""
    binary = [
        row
        for row in rows
        if row.get("label") in {0, 1}
        and row.get("probe_score") is not None
        and not isinstance(row.get("probe_score"), bool)
    ]
    report: dict[str, object] = {
        "threshold": threshold,
        "threshold_was_fit_on_these_rows": False,
        "n": len(binary),
        "class_counts": {
            "faithful": sum(int(row["label"]) == 0 for row in binary),
            "hallucinated": sum(int(row["label"]) == 1 for row in binary),
        },
        "formats": _format_table(binary, threshold),
        "length_by_class": _length_by_class(binary),
        "score_length_correlation": pearson(
            [len(str(row["answer"])) for row in binary],
            [float(row["probe_score"]) for row in binary],
        ),
        "faithful_score_length_correlation": pearson(
            [len(str(row["answer"])) for row in binary if int(row["label"]) == 0],
            [float(row["probe_score"]) for row in binary if int(row["label"]) == 0],
        ),
        "pair_probe_rewritten": False,
        "calibration_validated": False,
        "successful_detector": False,
    }
    labels = [int(row["label"]) for row in binary]
    scores = [float(row["probe_score"]) for row in binary]
    groups = [str(row.get("question_id") or index) for index, row in enumerate(binary)]
    splits = {str(row.get("split")) for row in binary if row.get("split")}
    report["splits_present"] = sorted(splits)
    report["includes_training_rows"] = "train" in splits
    report["overall"] = _safe_metrics(labels, scores, threshold, groups=groups, seed=seed)
    if report["includes_training_rows"]:
        overall = report["overall"]
        if isinstance(overall, dict):
            note = str(overall.get("note") or "")
            overall["note"] = (
                f"{note} These pooled rows include the training split and are not a held-out result."
            ).strip()
            overall["includes_training_rows"] = True
        test_rows = [row for row in binary if row.get("split") == "test"]
        report["held_out_test"] = _safe_metrics(
            [int(row["label"]) for row in test_rows],
            [float(row["probe_score"]) for row in test_rows],
            threshold,
            groups=[str(row.get("question_id") or index) for index, row in enumerate(test_rows)],
            seed=seed,
        )
    matched = length_matched_subset(binary, tolerance=LENGTH_MATCH_TOLERANCE)
    report["length_matched"] = {
        "tolerance_characters": LENGTH_MATCH_TOLERANCE,
        "n": len(matched),
        "class_counts": {
            "faithful": sum(int(row["label"]) == 0 for row in matched),
            "hallucinated": sum(int(row["label"]) == 1 for row in matched),
        },
        "metrics": _safe_metrics(
            [int(row["label"]) for row in matched],
            [float(row["probe_score"]) for row in matched],
            threshold,
            groups=[str(row.get("question_id") or index) for index, row in enumerate(matched)],
            seed=seed,
        ),
    }
    report["interpretation"] = interpret_length_shortcut(report)
    return report


def length_matched_subset(
    rows: Sequence[Mapping[str, object]], *, tolerance: int = LENGTH_MATCH_TOLERANCE
) -> list[Mapping[str, object]]:
    """Pair hallucinated rows with similar-length faithful rows.

    Unmatched rows are dropped. This is a diagnostic cut, not a new training set.
    """
    faithful = [row for row in rows if int(row["label"]) == 0]
    hallucinated = [row for row in rows if int(row["label"]) == 1]
    used: set[int] = set()
    matched: list[Mapping[str, object]] = []
    for hall in sorted(hallucinated, key=lambda row: len(str(row["answer"]))):
        target = len(str(hall["answer"]))
        best: tuple[int, int, Mapping[str, object]] | None = None
        for index, faith in enumerate(faithful):
            if index in used:
                continue
            distance = abs(len(str(faith["answer"])) - target)
            if best is None or distance < best[0]:
                best = (distance, index, faith)
        if best is None or best[0] > tolerance:
            continue
        used.add(best[1])
        matched.extend([hall, best[2]])
    return matched


def pearson(left: Sequence[float], right: Sequence[float]) -> dict[str, object]:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    if x.size != y.size or x.size < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return {
            "r": None,
            "n": int(x.size),
            "note": "Correlation was not computed. The series is too small or has no variance.",
        }
    return {"r": float(np.corrcoef(x, y)[0, 1]), "n": int(x.size), "note": None}


def interpret_length_shortcut(report: Mapping[str, object]) -> dict[str, object]:
    """Say whether scores track format/length more than the label."""
    formats = report.get("formats") if isinstance(report.get("formats"), dict) else {}
    short_fpr = _format_fpr(formats, "short_faithful")
    sentence_fpr = _format_fpr(formats, "sentence_faithful")
    correlation = report.get("faithful_score_length_correlation")
    r_value = correlation.get("r") if isinstance(correlation, dict) else None
    overall = report.get("overall") if isinstance(report.get("overall"), dict) else {}
    matched = report.get("length_matched") if isinstance(report.get("length_matched"), dict) else {}
    matched_metrics = matched.get("metrics") if isinstance(matched, dict) else {}
    format_gap = (
        sentence_fpr - short_fpr if short_fpr is not None and sentence_fpr is not None else None
    )
    length_driven = (format_gap is not None and format_gap >= 0.25) or (
        isinstance(r_value, float) and r_value >= 0.4
    )
    overall_auroc = None
    if isinstance(overall.get("metrics"), dict):
        overall_auroc = overall["metrics"].get("auroc")
    matched_auroc = None
    if isinstance(matched_metrics, dict) and isinstance(matched_metrics.get("metrics"), dict):
        matched_auroc = matched_metrics["metrics"].get("auroc")
    drop = (
        float(overall_auroc) - float(matched_auroc)
        if isinstance(overall_auroc, (int, float)) and isinstance(matched_auroc, (int, float))
        else None
    )
    if length_driven:
        reason = (
            "Probe scores track answer format or length more than the factuality label. "
            "Sentence-form faithful answers are flagged more often than short faithful "
            "answers, or score correlates with length among faithful rows. "
            "This is not evidence that the probe reads factual correctness."
        )
        learns = "superficial_answer_patterns"
    else:
        reason = (
            "This slice does not show a large length or format gap. That is not a "
            "validated detector result. Sample size and calibration still have to pass."
        )
        learns = "inconclusive_or_not_length_dominated"
    return {
        "learns": learns,
        "sentence_minus_short_faithful_fpr": format_gap,
        "length_matched_auroc_drop": drop,
        "reason": reason,
        "successful_detector": False,
    }


def _variant(fact: CuratedFact, fmt: str, answer: str, label: int) -> dict[str, object]:
    return {
        "case_id": f"{fact.item_id}:{fmt}",
        "question_id": fact.item_id,
        "question": fact.question,
        "answer": answer,
        "format": fmt,
        "label": label,
        "label_name": "HALLUCINATED" if label == 1 else "FAITHFUL",
        "source": "controlled_completion",
        "probe_score": None,
    }


def _format_table(
    rows: Sequence[Mapping[str, object]], threshold: float
) -> dict[str, dict[str, object]]:
    table: dict[str, dict[str, object]] = {}
    for name in FORMATS:
        group = [row for row in rows if row.get("format") == name]
        labels = [int(row["label"]) for row in group]
        scores = [float(row["probe_score"]) for row in group]
        predicted = [score >= threshold for score in scores]
        faithful = [row for row, label in zip(group, labels, strict=True) if label == 0]
        table[name] = {
            "n": len(group),
            "mean_characters": (
                None if not group else float(np.mean([len(str(row["answer"])) for row in group]))
            ),
            "mean_score": None if not scores else float(np.mean(scores)),
            "false_positive_rate": (
                None
                if not faithful
                else float(np.mean([float(row["probe_score"]) >= threshold for row in faithful]))
            ),
            "predicted_hallucinated": int(sum(predicted)),
        }
    if not any(table[name]["n"] for name in FORMATS):
        # Rows without a format field still get an overall length split.
        short = [row for row in rows if len(str(row["answer"]).split()) <= 3]
        long = [row for row in rows if len(str(row["answer"]).split()) > 3]
        table["short_unspecified"] = _plain_group(short, threshold)
        table["sentence_unspecified"] = _plain_group(long, threshold)
    return table


def _plain_group(rows: Sequence[Mapping[str, object]], threshold: float) -> dict[str, object]:
    faithful = [row for row in rows if int(row["label"]) == 0]
    scores = [float(row["probe_score"]) for row in rows]
    return {
        "n": len(rows),
        "mean_characters": (
            None if not rows else float(np.mean([len(str(row["answer"])) for row in rows]))
        ),
        "mean_score": None if not scores else float(np.mean(scores)),
        "false_positive_rate": (
            None
            if not faithful
            else float(np.mean([float(row["probe_score"]) >= threshold for row in faithful]))
        ),
    }


def _length_by_class(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, label in (("faithful", 0), ("hallucinated", 1)):
        group = [row for row in rows if int(row["label"]) == label]
        lengths = [len(str(row["answer"])) for row in group]
        scores = [float(row["probe_score"]) for row in group]
        result[name] = {
            "n": len(group),
            "mean_characters": None if not lengths else float(np.mean(lengths)),
            "mean_score": None if not scores else float(np.mean(scores)),
        }
    return result


def _safe_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    *,
    groups: Sequence[str] | None = None,
    seed: int = 0,
) -> dict[str, object]:
    label_array = np.asarray(list(labels), dtype=int)
    score_array = np.asarray(list(scores), dtype=np.float64)
    counts = {
        "faithful": int(np.sum(label_array == 0)),
        "hallucinated": int(np.sum(label_array == 1)),
    }
    smaller = min(counts["faithful"], counts["hallucinated"])
    payload: dict[str, object] = {
        "n": int(label_array.size),
        "class_counts": counts,
        "interval": None,
        "inconclusive": smaller < BOOTSTRAP_MIN_PER_CLASS,
    }
    if label_array.size == 0 or set(label_array.tolist()) != {0, 1}:
        payload["metrics"] = None
        payload["confusion"] = None
        payload["note"] = "Metrics were not computed. Both classes were not present."
        return payload
    payload["metrics"] = compute_binary_metrics(label_array, score_array, threshold=threshold)
    payload["confusion"] = confusion_counts(label_array, score_array, threshold=threshold)
    if payload["inconclusive"]:
        payload["note"] = (
            f"Point metrics are shown. The smaller class has {smaller} rows, "
            f"below {BOOTSTRAP_MIN_PER_CLASS}, so no confidence interval is reported "
            "and this is not a detector result."
        )
    else:
        payload["interval"] = bootstrap_grouped_interval(
            label_array,
            score_array,
            list(groups)
            if groups is not None
            else [str(index) for index in range(label_array.size)],
            metric="auroc",
            threshold=threshold,
            seed=seed,
        )
        payload["note"] = "Both classes meet the interval minimum on this slice."
    return payload


def _format_fpr(formats: Mapping[str, object], name: str) -> float | None:
    block = formats.get(name)
    if not isinstance(block, dict):
        return None
    value = block.get("false_positive_rate")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
