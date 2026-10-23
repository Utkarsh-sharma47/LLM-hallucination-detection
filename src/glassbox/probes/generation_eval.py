"""Scoring helpers for the generation-task probe.

These functions do not load Qwen. Training calls them after hidden states
exist. Tests call them with in-memory rows.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from glassbox.config import ProbeConfig, repo_root
from glassbox.data.evidence import GENERATION_LABEL_SOURCE
from glassbox.data.schema import Label
from glassbox.evaluation.metrics import bootstrap_grouped_interval, compute_binary_metrics
from glassbox.evaluation.records import assert_no_secret_keys
from glassbox.probes.logistic import LogisticProbe
from glassbox.probes.scoring import (
    CALIBRATION_ECE_LIMIT,
    CALIBRATION_MIN_PER_CLASS,
    confusion_counts,
    overlap_score,
    select_threshold,
)
from glassbox.reproducibility import require_commit_hash

PROTECTED_PROBE_NAMES = (
    "final-hidden-state",
    "final-hidden-state.previous",
    "final-hidden-state.staging",
    "final-hidden-state.incoming",
)
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}")


def halueval_snapshot_path() -> Path:
    return repo_root() / "artifacts" / "datasets" / "halueval" / "qa_data.json"


def truthfulqa_snapshot_path() -> Path:
    return repo_root() / "artifacts" / "datasets" / "truthfulqa" / "TruthfulQA.csv"


def generation_probe_dir(run_id: str) -> Path:
    return repo_root() / "artifacts" / "probes" / "generation-runs" / run_id


def generation_experiment_dir(run_id: str) -> Path:
    return repo_root() / "artifacts" / "experiments" / run_id


def assert_output_allowed(artifacts: Path, run_id: str, probe_dir: Path) -> None:
    """Refuse protected probe directories and destinations that already exist."""
    if _RUN_ID.fullmatch(run_id) is None or run_id.startswith("."):
        raise ValueError(f"Refusing run id {run_id!r}")
    if run_id in PROTECTED_PROBE_NAMES or run_id.startswith("final-hidden-state"):
        raise ValueError(f"Refusing to use protected probe name {run_id}")
    root = Path(artifacts).resolve()
    experiment = (root / "experiments" / run_id).resolve()
    probe = Path(probe_dir).resolve()
    for path in (experiment, probe):
        if not path.is_relative_to(root):
            raise ValueError(f"Refusing to write outside {root}")
        _refuse_protected(path, root)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")


def score_binary(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    groups: list[str],
    *,
    threshold: float,
    seed: int = 0,
) -> dict[str, object]:
    """Score one split. A single-class split does not receive invented metrics."""
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(scores, dtype=np.float64)
    if y_true.size == 0 or set(y_true.tolist()) != {0, 1}:
        return {
            "metrics": None,
            "confusion": None,
            "auroc_interval": None,
            "n": int(y_true.size),
            "note": "Metrics were not computed. The evaluation set is empty or has only one class.",
        }
    metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)
    return {
        "metrics": metrics,
        "confusion": confusion_counts(y_true, y_score, threshold=threshold),
        "auroc_interval": bootstrap_grouped_interval(
            y_true,
            y_score,
            groups,
            metric="auroc",
            threshold=threshold,
            seed=seed,
        ),
        "n": int(y_true.size),
        "note": "Positive class is HALLUCINATED. The interval resamples question groups.",
    }


def calibration_decision(
    validation_counts: dict[str, int],
    test_counts: dict[str, int],
    test_ece: float | None,
) -> dict[str, object]:
    """Apply the project's existing ECE rule. Failure stays rejected."""
    enough = _enough(validation_counts) and _enough(test_counts)
    if not enough or test_ece is None:
        return {
            "status": "rejected",
            "calibration_validated": False,
            "test_ece": test_ece,
            "reason": (
                "Calibration was rejected. Validation and test each need at least "
                f"{CALIBRATION_MIN_PER_CLASS} faithful and hallucinated rows, and a finite test ECE."
            ),
        }
    if test_ece > CALIBRATION_ECE_LIMIT:
        return {
            "status": "rejected",
            "calibration_validated": False,
            "test_ece": test_ece,
            "reason": (
                f"Calibration was rejected. Test ECE {test_ece:.3f} is above {CALIBRATION_ECE_LIMIT}."
            ),
        }
    return {
        "status": "accepted",
        "calibration_validated": True,
        "test_ece": test_ece,
        "reason": "Platt scaling met the pre-registered validation and test checks.",
    }


def generation_text_baselines(rows: list[dict[str, object]], *, seed: int) -> dict[str, object]:
    """Length and TF-IDF on generated answers. Reference overlap is marked as a shortcut."""
    grouped = {
        name: [row for row in rows if row["split"] == name and _binary(row)]
        for name in ("train", "validation", "test")
    }
    report: dict[str, object] = {"protocol": GENERATION_LABEL_SOURCE}
    if any(set(int(row["label"]) for row in group) != {0, 1} for group in grouped.values()):
        report["note"] = (
            "Text baselines were not fit. A binary split does not contain both classes."
        )
        return report
    report["answer_length"] = _fit_length(grouped, seed)
    report["answer_text_tfidf"] = _fit_tfidf(grouped, seed)
    report["right_answer_overlap"] = _overlap_baseline(
        grouped["test"],
        "right_answer",
        (
            "Shortcut: dissimilarity from the benchmark right answer. "
            "The evidence label often fires when that answer is asserted, so this is not a fair detector."
        ),
    )
    report["false_span_similarity"] = _false_span_baseline(grouped["test"])
    report["mean_characters_on_test"] = {
        "faithful": _mean_length(grouped["test"], 0),
        "hallucinated": _mean_length(grouped["test"], 1),
    }
    report["note"] = (
        "Length and TF-IDF see the generated answer only and are fit without the test labels. "
        "Overlap with the right answer or a contradicted span can recover the evidence rule."
    )
    return report


def interpret_detector(
    probe_metrics: dict[str, float] | None,
    baselines: dict[str, object],
) -> dict[str, object]:
    """Say whether the probe beat text baselines. A high AUROC is not enough."""
    if probe_metrics is None:
        return {
            "successful_detector": False,
            "reason": "Held-out probe metrics are unavailable, so no detector claim is made.",
        }
    probe_auroc = float(probe_metrics["auroc"])
    length = _auroc(baselines.get("answer_length"))
    tfidf = _auroc(baselines.get("answer_text_tfidf"))
    overlap = _auroc(baselines.get("right_answer_overlap"))
    false_span = _auroc(baselines.get("false_span_similarity"))
    text_ceiling = (
        max(value for value in (length, tfidf) if value is not None)
        if any(value is not None for value in (length, tfidf))
        else None
    )
    beats_text = text_ceiling is not None and probe_auroc > text_ceiling + 0.02
    shortcut = (overlap is not None and overlap >= probe_auroc - 0.02) or (
        false_span is not None and false_span >= probe_auroc - 0.02
    )
    if not beats_text or shortcut:
        reason = (
            "The hidden-state probe is not reported as a successful hallucination detector. "
            "It does not clearly beat the answer-length and TF-IDF baselines, or a reference-overlap "
            "shortcut matches it. A high AUROC on this evidence rule is not useful detection by itself."
        )
        success = False
    else:
        reason = (
            "On this split the probe's AUROC is above the length and TF-IDF baselines and above "
            "the reference-overlap shortcuts. That is still not a calibrated probability, and it "
            "has to survive the external dataset before it is treated as a detector."
        )
        success = True
    return {
        "successful_detector": success,
        "probe_auroc": probe_auroc,
        "length_auroc": length,
        "tfidf_auroc": tfidf,
        "right_answer_overlap_auroc": overlap,
        "false_span_similarity_auroc": false_span,
        "reason": reason,
    }


def validate_provenance(metadata: dict[str, object]) -> None:
    """Refuse a record that claims a reproducible run without the pins."""
    required = (
        "model_revision",
        "encoder_revision",
        "dataset_sha256",
        "seed",
        "position",
        "preprocessing",
        "label_protocol",
    )
    missing = [key for key in required if metadata.get(key) in (None, "", [])]
    if missing:
        raise ValueError(
            "Refusing an incomplete generation artifact. Missing " + ", ".join(missing)
        )
    require_commit_hash(str(metadata["model_revision"]), field="model_revision")
    require_commit_hash(str(metadata["encoder_revision"]), field="encoder_revision")
    if metadata.get("label_protocol") != GENERATION_LABEL_SOURCE:
        raise ValueError("Generation artifact names a different label protocol")
    if (
        metadata.get("publishable_as_reproducible") is True
        and metadata.get("worktree_dirty") is True
    ):
        raise ValueError("A dirty worktree cannot be published as a reproducible result")
    if metadata.get("publishable_as_reproducible") is True and missing:
        raise ValueError("Missing provenance cannot be published as reproducible")
    assert_no_secret_keys(metadata)


def overlap_questions(left: list[str], right: list[str]) -> list[str]:
    """Normalized questions that appear in both datasets."""
    right_keys = {_question_key(item) for item in right}
    return [item for item in left if _question_key(item) in right_keys]


def external_score_report(
    rows: list[dict[str, object]], *, threshold: float, seed: int
) -> dict[str, object]:
    """Score an external generation set with a threshold frozen elsewhere."""
    labeled = [
        row
        for row in rows
        if row.get("label") in {Label.FAITHFUL.name, Label.HALLUCINATED.name}
        and row.get("probe_score") is not None
    ]
    counts = {
        "faithful": sum(row.get("label") == Label.FAITHFUL.name for row in rows),
        "hallucinated": sum(row.get("label") == Label.HALLUCINATED.name for row in rows),
        "ambiguous": sum(row.get("label") == Label.AMBIGUOUS.name for row in rows),
        "unvalidated": sum(row.get("label") == Label.UNVALIDATED.name for row in rows),
        "scored": len(labeled),
        "coverage_upgrades": sum(bool(row.get("coverage_upgraded")) for row in rows),
    }
    labels = [1 if row["label"] == Label.HALLUCINATED.name else 0 for row in labeled]
    scores = [float(row["probe_score"]) for row in labeled]
    groups = [str(row.get("question_id")) for row in labeled]
    scored = score_binary(labels, scores, groups, threshold=threshold, seed=seed)
    return {
        "dataset": "truthfulqa",
        "task": "qwen_generation",
        "not_used_for_training_threshold_or_calibration": True,
        "threshold_frozen_on": "halueval_generation_validation",
        "threshold": threshold,
        "class_counts": counts,
        "scored": scored,
        "note": (
            "TruthfulQA labels use assertion plus in-order content coverage. "
            "Unvalidated answers stay out of the metrics. This split did not choose the threshold."
        ),
    }


def _enough(counts: dict[str, int]) -> bool:
    return (
        int(counts.get("faithful", 0)) >= CALIBRATION_MIN_PER_CLASS
        and int(counts.get("hallucinated", 0)) >= CALIBRATION_MIN_PER_CLASS
    )


def _binary(row: dict[str, object]) -> bool:
    return int(row["label"]) in {0, 1}


def _fit_length(grouped: dict[str, list[dict[str, object]]], seed: int) -> dict[str, object]:
    probe = LogisticProbe(ProbeConfig(random_state=seed))
    probe.fit(_length_matrix(grouped["train"]), _labels(grouped["train"]))
    threshold = select_threshold(
        _labels(grouped["validation"]), probe.predict_proba(_length_matrix(grouped["validation"]))
    )
    scores = probe.predict_proba(_length_matrix(grouped["test"]))
    scored = score_binary(
        _labels(grouped["test"]).tolist(),
        scores.tolist(),
        [str(row["question_id"]) for row in grouped["test"]],
        threshold=threshold,
        seed=seed,
    )
    scored["threshold"] = threshold
    scored["fair_baseline"] = True
    scored["detail"] = (
        "Character length and word count of the generated answer. Threshold fit on validation."
    )
    return scored


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
    scored = score_binary(
        _labels(grouped["test"]).tolist(),
        scores.tolist(),
        [str(row["question_id"]) for row in grouped["test"]],
        threshold=threshold,
        seed=seed,
    )
    scored["threshold"] = threshold
    scored["fair_baseline"] = True
    scored["detail"] = "Word and bigram TF-IDF fit on training generations only."
    return scored


def _overlap_baseline(rows: list[dict[str, object]], field: str, detail: str) -> dict[str, object]:
    scores = [overlap_score(str(row["answer"]), str(row.get(field) or "")) for row in rows]
    scored = score_binary(
        _labels(rows).tolist(),
        scores,
        [str(row["question_id"]) for row in rows],
        threshold=0.5,
    )
    scored["fair_baseline"] = False
    scored["detail"] = detail
    return scored


def _false_span_baseline(rows: list[dict[str, object]]) -> dict[str, object]:
    scores = []
    for row in rows:
        spans = row.get("false_spans")
        if not isinstance(spans, list) or not spans:
            scores.append(0.0)
            continue
        scores.append(max(1.0 - overlap_score(str(row["answer"]), str(span)) for span in spans))
    scored = score_binary(
        _labels(rows).tolist(),
        scores,
        [str(row["question_id"]) for row in rows],
        threshold=0.5,
    )
    scored["fair_baseline"] = False
    scored["detail"] = (
        "Similarity to a contradicted substitute. The evidence label uses those spans, so this can shortcut the task."
    )
    return scored


def _labels(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=int)


def _length_matrix(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray(
        [[len(str(row["answer"])), len(str(row["answer"]).split())] for row in rows],
        dtype=np.float64,
    )


def _mean_length(rows: list[dict[str, object]], label: int) -> float | None:
    values = [len(str(row["answer"])) for row in rows if int(row["label"]) == label]
    if not values:
        return None
    return float(np.mean(values))


def _auroc(block: object) -> float | None:
    if not isinstance(block, dict):
        return None
    metrics = block.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("auroc") is None:
        return None
    return float(metrics["auroc"])


def _question_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip("?.!")


def _refuse_protected(path: Path, artifacts: Path) -> None:
    for name in PROTECTED_PROBE_NAMES:
        protected = (artifacts / "probes" / name).resolve()
        if path == protected or path.is_relative_to(protected) or protected.is_relative_to(path):
            raise ValueError(f"Refusing to write into protected probe path {protected}")
