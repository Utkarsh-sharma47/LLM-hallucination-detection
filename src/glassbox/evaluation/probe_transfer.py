"""Score the saved HaluEval pair probe on Qwen's own TruthfulQA answers.

The probe was trained to discriminate supplied HaluEval answer strings. This
module does not retrain it, does not choose a new threshold, and does not treat
a pair-protocol score as a generation result. Factual labels come from asserted
TruthfulQA references. A probe prediction is never the label.

``AMBIGUOUS`` and ``UNVALIDATED`` stay in the record and stay out of binary
metrics. A question whose text also appears in the probe's HaluEval splits
stays in the record and stays out of those metrics too. Generation ECE is
reported and does not set ``calibration_validated``.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import resource
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from glassbox.config import repo_root
from glassbox.data.generation_dataset import (
    FrozenSplit,
    assert_dataset_destination,
    build_generation_dataset,
    freeze_question_split,
    write_dataset_file,
)
from glassbox.data.halueval import filter_pairs, read_pinned_snapshot
from glassbox.data.schema import Label
from glassbox.data.truthfulqa import (
    LABEL_SOURCE,
    PINNED_SHA256,
    TruthfulItem,
    label_with_phrases,
)
from glassbox.data.truthfulqa import read_pinned_snapshot as read_truthfulqa
from glassbox.evaluation.generation_record import (
    build_generation_evaluation_record,
    write_generation_evaluation,
)
from glassbox.evaluation.metrics import (
    BOOTSTRAP_MIN_PER_CLASS,
    bootstrap_interval,
    compute_binary_metrics,
)
from glassbox.evaluation.records import write_json
from glassbox.probes.scoring import confusion_counts, expected_calibration_error, overlap_score
from glassbox.probes.store import (
    ProbeCompatibilityError,
    check_compatibility,
    experimental_score,
    load_probe,
)
from glassbox.reproducibility import require_commit_hash
from glassbox.verification import normalize_question

# Frozen on the HaluEval validation split. Do not replace it after seeing
# generation scores. A generation-trained threshold such as 0.25 is refused.
SAVED_PAIR_THRESHOLD = 0.65
# Declared before any generation in this protocol. Baseline F1 uses this cut.
# It is not fit on the generation labels.
BASELINE_THRESHOLD = 0.5
# word_count / (word_count + LENGTH_SMOOTHING). Not fit on labels.
LENGTH_SMOOTHING = 8
PAIR_PROTOCOL = "halueval_qa_pair_v1"
SCORE_NAME = "EXPERIMENTAL PROBE SCORE"
RISK_STATUS = "RISK NOT ASSESSED"
# Pilot labeled 2 of 40 answers as hallucinated. 200 is the planned sample for
# this transfer run. It is not raised or lowered after the scores are known.
PLANNED_MAX_QUESTIONS = 200
RUN_ID = "truthfulqa-probe-transfer-001"
_BINARY = {Label.FAITHFUL.name, Label.HALLUCINATED.name}
_PROBE_SPLITS = ("train", "validation", "test")


def require_pair_probe(metadata: Mapping[str, object]) -> float:
    """Accept only the saved HaluEval pair probe and return its frozen threshold.

    A missing field or a different threshold raises. This does not read a
    generation score and does not call ``select_threshold``.
    """
    threshold = metadata.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Saved probe is missing the HaluEval threshold")
    if not math.isclose(float(threshold), SAVED_PAIR_THRESHOLD, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Refusing to replace the HaluEval threshold {SAVED_PAIR_THRESHOLD} with {threshold}"
        )
    if metadata.get("position") != "last_generated_token":
        raise ValueError("Saved probe position is not last_generated_token")
    if metadata.get("label_protocol") != PAIR_PROTOCOL:
        raise ValueError(
            "Saved probe was not trained for HaluEval paired-answer discrimination, "
            f"or its protocol is missing: {metadata.get('label_protocol')!r}"
        )
    if metadata.get("positive_class") != "hallucinated":
        raise ValueError("Saved probe positive class must be hallucinated")
    calibration = metadata.get("calibration_validated")
    if not isinstance(calibration, bool):
        raise ValueError("Saved probe calibration_validated must be a boolean")
    layer = metadata.get("layer_index")
    width = metadata.get("feature_dim")
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise ValueError("Saved probe layer_index must be an integer")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("Saved probe feature_dim must be a positive integer")
    require_commit_hash(str(metadata.get("model_revision", "")), field="model_revision")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("Saved probe is missing dataset.used_question_ids")
    used = dataset.get("used_question_ids")
    if not isinstance(used, Mapping):
        raise ValueError("Saved probe is missing dataset.used_question_ids")
    for split_name in _PROBE_SPLITS:
        ids = used.get(split_name)
        if (
            not isinstance(ids, list)
            or not ids
            or not all(isinstance(item, str) and item for item in ids)
        ):
            raise ValueError(f"Saved probe split {split_name} needs question ids")
    return SAVED_PAIR_THRESHOLD


def training_question_texts(
    pairs: Sequence[object], used_ids: Mapping[str, Sequence[str]]
) -> dict[str, tuple[str, ...]]:
    """Map each probe split to the question text of its ids.

    Every saved id must be found. A missing id would hide a leaked question.
    """
    by_id = {str(pair.question_id): str(pair.question) for pair in pairs}
    found: dict[str, tuple[str, ...]] = {}
    for split_name in _PROBE_SPLITS:
        texts: list[str] = []
        missing: list[str] = []
        for question_id in used_ids[split_name]:
            text = by_id.get(question_id)
            if text is None:
                missing.append(question_id)
            else:
                texts.append(text)
        if missing:
            raise ValueError(f"Probe {split_name} ids are not in the HaluEval snapshot: {missing}")
        found[split_name] = tuple(texts)
    return found


def question_overlaps(
    items: Sequence[TruthfulItem], training_questions: Mapping[str, Sequence[str]]
) -> dict[str, tuple[str, ...]]:
    """Return HaluEval splits whose question text matches each evaluation item.

    Comparison uses normalized question text. TruthfulQA ids such as ``row0``
    are not HaluEval ids, so an id match is not evidence of leakage.
    """
    buckets: dict[str, set[str]] = {}
    for split_name, questions in training_questions.items():
        for question in questions:
            buckets.setdefault(normalize_question(question), set()).add(split_name)
    overlaps: dict[str, tuple[str, ...]] = {}
    seen: dict[str, str] = {}
    for item in items:
        key = normalize_question(item.question)
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(f"Question text repeats across {previous} and {item.question_id}")
        seen[key] = item.question_id
        overlaps[item.question_id] = tuple(
            split_name for split_name in _PROBE_SPLITS if split_name in buckets.get(key, set())
        )
    return overlaps


def length_score(answer: str) -> float:
    """Map answer length into ``[0, 1]`` with a fixed denominator.

    The map is monotone in the whitespace-separated word count. It is not fit
    on the generation labels.
    """
    count = len([part for part in answer.split() if part])
    return _unit(count / (count + LENGTH_SMOOTHING))


def reference_overlap_score(answer: str, correct: Sequence[str], incorrect: Sequence[str]) -> float:
    """Map closeness to listed phrases into ``[0, 1]``.

    ``overlap_score`` is high when two strings share few tokens. This helper
    turns that into similarity and scores the answer higher when it is closer
    to a listed incorrect phrase than to a listed correct phrase. It does not
    use a HaluEval gold answer.
    """
    return _unit(
        (_best_similarity(answer, incorrect) - _best_similarity(answer, correct) + 1.0) / 2.0
    )


def reference_tfidf_scores(
    rows: Sequence[tuple[str, str, tuple[str, ...], tuple[str, ...]]],
) -> dict[str, float]:
    """Score each answer from a TF-IDF model fit only on listed reference phrases.

    Generated answers and their labels are not in the fit. The score uses the
    same incorrect-minus-correct direction as :func:`reference_overlap_score`.
    """
    corpus: list[str] = []
    for _, _, correct, incorrect in rows:
        corpus.extend(correct)
        corpus.extend(incorrect)
    if not any(phrase.strip() for phrase in corpus):
        return {question_id: BASELINE_THRESHOLD for question_id, _, _, _ in rows}
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform(corpus)
    offset = 0
    phrase_rows: dict[str, np.ndarray] = {}
    for _, _, correct, incorrect in rows:
        phrases = (*correct, *incorrect)
        for phrase in phrases:
            phrase_rows[phrase] = matrix[offset]
            offset += 1
    scores: dict[str, float] = {}
    for question_id, answer, correct, incorrect in rows:
        answer_vector = vectorizer.transform([answer])
        scores[question_id] = _unit(
            (
                _best_cosine(answer_vector, incorrect, phrase_rows)
                - _best_cosine(answer_vector, correct, phrase_rows)
                + 1.0
            )
            / 2.0
        )
    return scores


def score_feature(loaded: object, feature: np.ndarray, *, live: Mapping[str, object]) -> float:
    """Apply the saved probe to one hidden state, or raise on a mismatch.

    An incompatible feature does not become a score of zero or a guessed class.
    """
    check_compatibility(
        loaded,
        model_name=str(live["model_name"]),
        model_revision=str(live["model_revision"]),
        layer_index=int(live["layer_index"]),
        hidden_state_count=int(live["hidden_state_count"]),
        position=str(live["position"]),
        prompt_mode=str(live["prompt_mode"]),
        feature_dim=int(live["feature_dim"]),
        dtype=str(live["dtype"]) if live.get("dtype") is not None else None,
    )
    return float(experimental_score(loaded, feature))


def build_transfer(
    *,
    split: FrozenSplit,
    items: Mapping[str, TruthfulItem],
    answers: Mapping[str, str],
    probe_scores: Mapping[str, float | None],
    score_notes: Mapping[str, str],
    training_questions: Mapping[str, Sequence[str]],
    provenance: Mapping[str, object],
    probe_metadata_calibration_validated: bool,
    generation_settings: Mapping[str, object],
    probe_metadata_sha256: str,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Label, score, and summarize one frozen split. Does not write a file."""
    if not isinstance(probe_metadata_calibration_validated, bool):
        raise ValueError("probe calibration_validated must be a boolean")
    ordered = [items[question_id] for question_id in split.question_ids]
    overlaps = question_overlaps(ordered, training_questions)
    tfidf = reference_tfidf_scores(
        tuple(
            (item.question_id, answers[item.question_id], item.correct, item.incorrect)
            for item in ordered
        )
    )
    record_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for item in ordered:
        answer = answers[item.question_id]
        label, detail, phrases = label_with_phrases(answer, item)
        score = probe_scores.get(item.question_id)
        note = score_notes.get(item.question_id, "").strip()
        if score is None and not note:
            raise ValueError(f"{item.question_id} is missing a probe score")
        if score is not None and not note:
            note = "experimental score from the saved HaluEval pair probe"
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) < 0.0
            or float(score) > 1.0
        ):
            raise ValueError(f"{item.question_id} probe score must be a probability in [0, 1]")
        prediction = None
        if score is not None:
            prediction = (
                Label.HALLUCINATED.name
                if float(score) >= SAVED_PAIR_THRESHOLD
                else Label.FAITHFUL.name
            )
        leaked = overlaps[item.question_id]
        reasons: list[str] = []
        if score is None:
            reasons.append("missing_score")
        if label.name not in _BINARY:
            reasons.append(label.name.lower())
        if leaked:
            reasons.append("split_overlap")
        eligible = not reasons
        record_rows.append(
            {
                "question_id": item.question_id,
                "answer": answer,
                "label": label.name,
                "reference_phrases": list(phrases),
                "probe_score": None if score is None else float(score),
                "missing_score_reason": None if score is not None else note,
            }
        )
        detail_rows.append(
            {
                "question_id": item.question_id,
                "question": item.question,
                "answer": answer,
                "label": label.name,
                "label_detail": detail,
                "probe_score": None if score is None else float(score),
                "probe_prediction": prediction,
                "score_note": note,
                "length_score": length_score(answer),
                "reference_overlap_score": reference_overlap_score(
                    answer, item.correct, item.incorrect
                ),
                "reference_tfidf_score": tfidf[item.question_id],
                "split_overlap": list(leaked),
                "in_binary_evaluation": eligible,
                "exclusion_reasons": reasons,
            }
        )
    dataset = build_generation_dataset(
        split=split,
        items=items,
        answers=answers,
        model_name=str(provenance["model_name"]),
        model_revision=str(provenance["model_revision"]),
        dtype=str(provenance["dtype"]),
    )
    record_provenance = dict(provenance)
    # The fail-closed writer accepts only false. Generation ECE cannot flip it,
    # and a true flag on the pair probe is not a generation calibration.
    record_provenance["calibration_validated"] = False
    record = build_generation_evaluation_record(
        record_provenance,
        record_rows,
        score_requested=True,
        probe_available=True,
    )
    metrics = _metrics(
        detail_rows,
        probe_metadata_calibration_validated=probe_metadata_calibration_validated,
        generation_settings=generation_settings,
        probe_metadata_sha256=probe_metadata_sha256,
        provenance=provenance,
    )
    return dataset, record["rows"], metrics


def publish_transfer(
    *,
    split: FrozenSplit,
    items: Mapping[str, TruthfulItem],
    answers: Mapping[str, str],
    probe_scores: Mapping[str, float | None],
    score_notes: Mapping[str, str],
    training_questions: Mapping[str, Sequence[str]],
    provenance: Mapping[str, object],
    destination: Path,
    artifacts_root: Path,
    probe_metadata_calibration_validated: bool,
    generation_settings: Mapping[str, object],
    probe_metadata_sha256: str,
) -> dict[str, object]:
    """Write split, dataset, record, and metrics. A failed build writes nothing."""
    run_dir = Path(destination)
    if run_dir.exists():
        raise ValueError(f"Refusing to overwrite {run_dir}")
    for name in ("split.json", "dataset.json", "record.json", "metrics.json"):
        assert_dataset_destination(run_dir / name, artifacts_root)
    dataset, rows, metrics = build_transfer(
        split=split,
        items=items,
        answers=answers,
        probe_scores=probe_scores,
        score_notes=score_notes,
        training_questions=training_questions,
        provenance=provenance,
        probe_metadata_calibration_validated=probe_metadata_calibration_validated,
        generation_settings=generation_settings,
        probe_metadata_sha256=probe_metadata_sha256,
    )
    root = Path(artifacts_root)
    written: list[Path] = []
    try:
        written.append(
            write_dataset_file(split.as_dict(), run_dir / "split.json", artifacts_root=root)
        )
        written.append(write_dataset_file(dataset, run_dir / "dataset.json", artifacts_root=root))
        written.append(
            write_generation_evaluation(
                {**provenance, "calibration_validated": False},
                rows,
                run_dir / "record.json",
                artifacts_root=root,
                score_requested=True,
                probe_available=True,
            )
        )
        written.append(write_json(run_dir / "metrics.json", metrics, root=root))
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        if run_dir.exists() and not any(run_dir.iterdir()):
            run_dir.rmdir()
        raise
    return metrics


def transfer_command(args: argparse.Namespace) -> int:
    """Generate with cached Qwen, score the saved pair probe, and write a new run."""
    try:
        return _transfer_command(args)
    except (
        OSError,
        ValueError,
        RuntimeError,
        MemoryError,
        ImportError,
        ProbeCompatibilityError,
    ) as exc:
        print(f"Generation transfer was not recorded: {exc}", file=sys.stderr)
        return 1


def _transfer_command(args: argparse.Namespace) -> int:
    snapshot = Path(
        args.snapshot or (repo_root() / "artifacts" / "datasets" / "truthfulqa" / "TruthfulQA.csv")
    )
    halueval = Path(
        args.halueval or (repo_root() / "artifacts" / "datasets" / "halueval" / "qa_data.json")
    )
    if args.probe is None:
        probe_dir = repo_root() / "artifacts" / "probes" / "final-hidden-state"
    else:
        probe_dir = Path(args.probe)
    if args.artifacts_root is None:
        artifacts_root = repo_root() / "artifacts"
    else:
        artifacts_root = Path(args.artifacts_root)
    if not snapshot.is_file():
        raise FileNotFoundError(
            f"TruthfulQA CSV is not at {snapshot}. This command does not download it."
        )
    if not halueval.is_file():
        raise FileNotFoundError(
            f"HaluEval snapshot is not at {halueval}. This command does not download it."
        )
    if not (probe_dir / "metadata.json").is_file():
        raise FileNotFoundError(f"Saved probe metadata is not at {probe_dir}.")
    max_questions = PLANNED_MAX_QUESTIONS if args.max_questions is None else args.max_questions
    run_id = RUN_ID if args.run_id is None else args.run_id
    if isinstance(max_questions, bool) or not isinstance(max_questions, int) or max_questions < 1:
        raise ValueError("max_questions must be a positive integer")
    run_dir = artifacts_root / "experiments" / run_id
    if run_dir.exists():
        raise ValueError(f"Refusing to overwrite {run_dir}")
    metadata_bytes = (probe_dir / "metadata.json").read_bytes()
    metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
    loaded = load_probe(probe_dir)
    require_pair_probe(loaded.metadata)
    commit, dirty = _git_state()
    if dirty:
        raise ValueError("A dirty worktree cannot be published as a finished generation evaluation")
    items = read_truthfulqa(snapshot)
    split = freeze_question_split(
        items,
        max_questions=max_questions,
        seed=args.seed,
        dataset_sha256=PINNED_SHA256,
    )
    by_id = {item.question_id: item for item in items}
    chosen = [by_id[question_id] for question_id in split.question_ids]
    pairs, _report = filter_pairs(read_pinned_snapshot(halueval))
    used = loaded.metadata["dataset"]["used_question_ids"]
    training_questions = training_question_texts(pairs, used)
    from glassbox.demo_config import load_demo_config
    from glassbox.models.backend import HuggingFaceBackend

    config = load_demo_config(args.config, offline=args.offline, model_name=args.model)
    if not config.spec.local_files_only:
        raise ValueError("This evaluation requires --offline so Qwen is not downloaded")
    backend = HuggingFaceBackend(config.spec)
    print(f"Loading {config.spec.name} once for generation transfer...", flush=True)
    model = backend.ensure_loaded()
    if any(parameter.requires_grad for parameter in model.model.parameters()):
        raise RuntimeError("Qwen parameters were not frozen")
    if model.resolved_revision != loaded.metadata.get("model_revision"):
        raise ProbeCompatibilityError("Live model revision does not match the saved probe revision")
    answers: dict[str, str] = {}
    scores: dict[str, float | None] = {}
    notes: dict[str, str] = {}
    live: dict[str, object] | None = None
    for item in chosen:
        result = backend.generate(item.question, config.generation)
        answers[item.question_id] = result.answer
        if (
            result.feature is None
            or result.feature_layer is None
            or result.hidden_state_count is None
        ):
            raise ProbeCompatibilityError(result.note or f"{item.question_id} has no hidden state")
        current = {
            "model_name": result.model_name,
            "model_revision": result.model_revision,
            "layer_index": result.feature_layer,
            "hidden_state_count": result.hidden_state_count,
            "position": result.position,
            "prompt_mode": result.prompt_mode,
            "feature_dim": int(result.feature.shape[0]),
            "dtype": result.dtype,
        }
        if live is None:
            live = current
        elif live != current:
            raise ProbeCompatibilityError(
                f"{item.question_id} representation does not match the first answer"
            )
        scores[item.question_id] = score_feature(loaded, result.feature, live=current)
        notes[item.question_id] = "experimental score from the saved HaluEval pair probe"
        print(f"{item.question_id}: {result.answer}", flush=True)
    if live is None:
        raise ValueError("No answers were generated")
    commit_after, dirty_after = _git_state()
    if dirty_after or commit_after != commit:
        raise ValueError("The worktree changed during generation, so the run was not published")
    after = hashlib.sha256((probe_dir / "metadata.json").read_bytes()).hexdigest()
    if after != metadata_sha:
        raise RuntimeError("Probe metadata changed during the evaluation")
    provenance = {
        "model_name": str(live["model_name"]),
        "model_revision": str(live["model_revision"]),
        "dtype": str(live["dtype"]),
        "dataset_id": "truthfulqa",
        "dataset_sha256": PINNED_SHA256,
        "split_id": f"truthfulqa-seed{args.seed}-n{max_questions}",
        "seed": args.seed,
        "probe_directory": str(probe_dir),
        "probe_model_revision": str(loaded.metadata["model_revision"]),
        "layer": int(live["layer_index"]),
        "position": str(live["position"]),
        "prompt_mode": str(live["prompt_mode"]),
        "calibration_validated": False,
        "config_path": str(args.config or "configs/demo.yaml"),
        "git_commit": commit,
        "worktree_clean": True,
    }
    settings = {
        "answer_piece_limit": config.generation.max_new_tokens,
        "temperature": config.generation.temperature,
        "top_k": config.generation.top_k,
        "layer_index": config.generation.layer_index,
        "position": config.generation.position,
        "seed": args.seed,
        "max_questions": max_questions,
        "threshold": SAVED_PAIR_THRESHOLD,
        "baseline_threshold": BASELINE_THRESHOLD,
        "length_smoothing": LENGTH_SMOOTHING,
    }
    metrics = publish_transfer(
        split=split,
        items=by_id,
        answers=answers,
        probe_scores=scores,
        score_notes=notes,
        training_questions=training_questions,
        provenance=provenance,
        destination=run_dir,
        artifacts_root=artifacts_root,
        probe_metadata_calibration_validated=loaded.metadata.get("calibration_validated") is True,
        generation_settings=settings,
        probe_metadata_sha256=metadata_sha,
    )
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Wrote {run_dir}")
    print(f"Peak RSS: {peak_kb / 1024:.0f} MiB")
    print(format_summary(metrics))
    return 0


def format_summary(metrics: Mapping[str, object]) -> str:
    """One block of the counts and the ranking result. Does not add a claim."""
    counts = metrics["sample_counts"]
    assert isinstance(counts, Mapping)
    probe = metrics["probe"]
    assert isinstance(probe, Mapping)
    interval = probe["auroc_interval"]
    assert isinstance(interval, Mapping)
    return "\n".join(
        [
            f"generated={counts['generated']}",
            (
                f"faithful={counts['faithful']} hallucinated={counts['hallucinated']} "
                f"ambiguous={counts['ambiguous']} unvalidated={counts['unvalidated']}"
            ),
            f"binary_eligible={counts['binary_eligible']}",
            f"probe_auroc={probe['auroc']}",
            f"auroc_interval_low={interval['low']} auroc_interval_high={interval['high']}",
            f"calibration_validated={metrics['calibration_validated']}",
            f"score_name={metrics['score_name']}",
            f"risk_status={metrics['risk_status']}",
            f"detector_claim={metrics['detector_claim']}",
        ]
    )


def _metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    probe_metadata_calibration_validated: bool,
    generation_settings: Mapping[str, object],
    probe_metadata_sha256: str,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    counts = {
        "generated": len(rows),
        "faithful": sum(row["label"] == Label.FAITHFUL.name for row in rows),
        "hallucinated": sum(row["label"] == Label.HALLUCINATED.name for row in rows),
        "ambiguous": sum(row["label"] == Label.AMBIGUOUS.name for row in rows),
        "unvalidated": sum(row["label"] == Label.UNVALIDATED.name for row in rows),
        "split_overlap": sum(bool(row["split_overlap"]) for row in rows),
        "binary_eligible": sum(bool(row["in_binary_evaluation"]) for row in rows),
    }
    eligible = [row for row in rows if row["in_binary_evaluation"]]
    labels = np.asarray(
        [1 if row["label"] == Label.HALLUCINATED.name else 0 for row in eligible], dtype=int
    )
    eligible_counts = {
        "faithful": int(np.sum(labels == 0)) if labels.size else 0,
        "hallucinated": int(np.sum(labels == 1)) if labels.size else 0,
    }
    counts["eligible_faithful"] = eligible_counts["faithful"]
    counts["eligible_hallucinated"] = eligible_counts["hallucinated"]
    both = set(labels.tolist()) == {0, 1} if labels.size else False
    probe = _ranking(eligible, labels, "probe_score", SAVED_PAIR_THRESHOLD, both=both)
    length = _ranking(eligible, labels, "length_score", BASELINE_THRESHOLD, both=both)
    overlap = _ranking(eligible, labels, "reference_overlap_score", BASELINE_THRESHOLD, both=both)
    tfidf = _ranking(eligible, labels, "reference_tfidf_score", BASELINE_THRESHOLD, both=both)
    generation_ece = None
    if both:
        generation_ece = expected_calibration_error(
            labels, np.asarray([float(row["probe_score"]) for row in eligible], dtype=np.float64)
        )
    interval_ready = (
        min(eligible_counts["faithful"], eligible_counts["hallucinated"]) >= BOOTSTRAP_MIN_PER_CLASS
    )
    probe_auroc = probe["auroc"]
    baseline_aurocs = [length["auroc"], overlap["auroc"], tfidf["auroc"]]
    outperformed: bool | None
    if not interval_ready or probe_auroc is None or any(value is None for value in baseline_aurocs):
        outperformed = None
    else:
        outperformed = all(float(probe_auroc) > float(value) for value in baseline_aurocs)
    errors = _errors(eligible)
    return {
        "schema_version": 1,
        "protocol": LABEL_SOURCE,
        "pair_protocol": PAIR_PROTOCOL,
        "sample_counts": counts,
        "interval_minimum": BOOTSTRAP_MIN_PER_CLASS,
        "score_name": SCORE_NAME,
        "risk_status": RISK_STATUS,
        "detector_claim": False,
        "calibration_validated": False,
        "probe_metadata_calibration_validated": probe_metadata_calibration_validated,
        "generation_ece": generation_ece,
        "generation_ece_set_calibration_validated": False,
        "outperformed_text_baselines": outperformed,
        "probe": probe,
        "length_baseline": length,
        "reference_overlap_baseline": overlap,
        "reference_tfidf_baseline": tfidf,
        "false_positives": errors["false_positives"],
        "false_negatives": errors["false_negatives"],
        "generation_settings": dict(generation_settings),
        "probe_metadata_sha256": probe_metadata_sha256,
        "git_commit": provenance.get("git_commit"),
        "split_id": provenance.get("split_id"),
        "dataset_sha256": provenance.get("dataset_sha256"),
        "rows": [dict(row) for row in rows],
        "limitation": (
            "This probe was trained to discriminate supplied HaluEval answer pairs "
            f"({PAIR_PROTOCOL}). It was not trained to detect factual errors in Qwen generations. "
            "Pair-protocol scores are a different experiment. "
            f"The displayed score stays {SCORE_NAME} and the risk line stays {RISK_STATUS}. "
            "A generation ECE does not validate calibration. "
            f"A confidence interval is omitted when the smaller eligible class has fewer than "
            f"{BOOTSTRAP_MIN_PER_CLASS} rows. "
            "Reference overlap and TF-IDF use listed TruthfulQA phrases, not a HaluEval gold answer. "
            f"Probe decisions use the saved threshold {SAVED_PAIR_THRESHOLD}. "
            f"Baseline decisions use the predeclared threshold {BASELINE_THRESHOLD}."
        ),
    }


def _ranking(
    rows: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    score_key: str,
    threshold: float,
    *,
    both: bool,
) -> dict[str, object]:
    if not both:
        interval = {
            "metric": "auroc",
            "point": None,
            "low": None,
            "high": None,
            "n": int(labels.size),
            "note": "Both classes were not present in the eligible rows, so AUROC was not computed.",
        }
        return {
            "threshold": threshold,
            "auroc": None,
            "auprc": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "false_positive_rate": None,
            "confusion": None,
            "auroc_interval": interval,
        }
    scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
    metrics = compute_binary_metrics(labels, scores, threshold=threshold)
    interval = bootstrap_interval(labels, scores, metric="auroc", threshold=threshold, seed=0)
    confusion = None
    if score_key == "probe_score":
        confusion = confusion_counts(labels, scores, threshold=threshold)
    return {
        "threshold": threshold,
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "false_positive_rate": metrics["false_positive_rate"],
        "confusion": confusion,
        "auroc_interval": interval,
    }


def _errors(rows: Sequence[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    false_positives: list[dict[str, object]] = []
    false_negatives: list[dict[str, object]] = []
    for row in rows:
        score = float(row["probe_score"])
        predicted = score >= SAVED_PAIR_THRESHOLD
        payload = {
            "question_id": row["question_id"],
            "question": row["question"],
            "answer": row["answer"],
            "label": row["label"],
            "probe_score": score,
            "probe_prediction": row["probe_prediction"],
        }
        if row["label"] == Label.FAITHFUL.name and predicted:
            false_positives.append(payload)
        if row["label"] == Label.HALLUCINATED.name and not predicted:
            false_negatives.append(payload)
    return {"false_positives": false_positives, "false_negatives": false_negatives}


def _unit(value: float) -> float:
    """Clip a baseline into ``[0, 1]``.

    Cosine similarity can be a fraction below zero from floating-point noise.
    The clip is not a threshold and is not fit on labels.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("baseline score must be a finite number")
    return float(min(1.0, max(0.0, value)))


def _best_similarity(answer: str, phrases: Sequence[str]) -> float:
    if not phrases:
        return 0.0
    return max(1.0 - overlap_score(answer, phrase) for phrase in phrases)


def _best_cosine(
    answer_vector: object, phrases: Sequence[str], phrase_rows: Mapping[str, np.ndarray]
) -> float:
    if not phrases:
        return 0.0
    best = 0.0
    for phrase in phrases:
        similarity = cosine_similarity(answer_vector, phrase_rows[phrase])
        best = max(best, float(similarity[0, 0]))
    return best


def _git_state() -> tuple[str, bool]:
    root = repo_root()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return require_commit_hash(commit, field="git_commit"), dirty
