"""Train and evaluate a logistic probe on HaluEval QA pairs.

The saved probe is the pre-registered final layer. Validation chooses the
threshold and the Platt map. The test split is scored once and is never added
to training when a class is missing.
"""

from __future__ import annotations

import argparse
import resource
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from glassbox.activations.extract import resolve_layer
from glassbox.config import ProbeConfig, repo_root
from glassbox.data.halueval import (
    LABEL_SOURCE,
    PINNED_COMMIT,
    PINNED_SHA256,
    PreparedPairs,
    download_snapshot,
    prepare_pairs,
    read_pinned_snapshot,
)
from glassbox.data.schema import Label
from glassbox.data.splits import assert_disjoint_ids
from glassbox.demo_config import default_probe_dir, load_demo_config
from glassbox.evaluation.metrics import compute_binary_metrics
from glassbox.evaluation.records import build_run_record, write_json
from glassbox.models.backend import GenerationSettings, HuggingFaceBackend, completion_hidden_states
from glassbox.probes.logistic import PROBE_NOT_TRAINED, LogisticProbe
from glassbox.probes.preprocess import apply_standard_scaler, fit_standard_scaler
from glassbox.probes.scoring import (
    CALIBRATION_ECE_LIMIT,
    CALIBRATION_MIN_PER_CLASS,
    apply_platt,
    class_counts,
    confusion_counts,
    expected_calibration_error,
    fit_platt,
    majority_baseline,
    overlap_score,
    select_threshold,
)
from glassbox.probes.store import save_probe
from glassbox.reproducibility import require_commit_hash

MIN_TRAIN_PER_CLASS = 20
REGISTERED_POSITION = "last_generated_token"


@dataclass
class FeatureExample:
    """One supplied answer and the hidden states at its last token."""

    example_id: str
    question_id: str
    split: str
    label: Label
    answer: str
    reference_answer: str
    vectors: dict[int, np.ndarray]
    prompt_mode: str
    position: str
    hidden_state_count: int
    model_name: str
    model_revision: str
    dtype: str = "unrecorded"


@dataclass
class TrainingOutcome:
    """What the training routine actually saved."""

    saved: bool
    reason: str
    directory: Path | None = None
    metrics: dict[str, float] | None = None
    threshold: float | None = None
    calibration_validated: bool = False
    report: dict[str, object] = field(default_factory=dict)


def train_from_examples(
    examples: list[FeatureExample],
    *,
    layer_index: int,
    output_dir: Path,
    seed: int = 0,
    min_per_class: int = MIN_TRAIN_PER_CLASS,
    dataset_record: dict[str, object] | None = None,
    label_protocol: str | None = None,
    metrics_note: str | None = None,
    overlap_note: str | None = None,
) -> TrainingOutcome:
    """Fit the registered layer and score validation and test separately."""
    if not examples:
        return TrainingOutcome(
            saved=False, reason=f"{PROBE_NOT_TRAINED}: no examples were extracted."
        )
    _require_same_model(examples)
    _require_position(examples)
    grouped = _by_split(examples)
    train = _labeled(grouped.get("train", []))
    validation = _labeled(grouped.get("validation", []))
    test = _labeled(grouped.get("test", []))
    assert_disjoint_ids(
        [item.example_id for item in train],
        [item.example_id for item in validation],
        [item.example_id for item in test],
    )
    _require_one_split_per_question(train + validation + test)
    train_counts = _counts(train)
    if train_counts["faithful"] < min_per_class or train_counts["hallucinated"] < min_per_class:
        return TrainingOutcome(
            saved=False,
            reason=(
                f"{PROBE_NOT_TRAINED}: the training split needs at least {min_per_class} "
                "faithful and hallucinated rows. Test and validation rows were not moved. "
                f"Got faithful={train_counts['faithful']}, hallucinated={train_counts['hallucinated']}."
            ),
            report={"train_class_counts": train_counts},
        )
    resolved_layer = resolve_layer(examples[0].hidden_state_count, layer_index)
    probe, mean, scale, train_scores = _fit(train, resolved_layer, seed=seed)
    if probe is None or mean is None or scale is None or train_scores is None:
        return TrainingOutcome(
            saved=False,
            reason=f"{PROBE_NOT_TRAINED}: layer {resolved_layer} could not be fit on the training split.",
        )
    validation_scores = _scores(probe, mean, scale, validation, resolved_layer)
    test_scores = _scores(probe, mean, scale, test, resolved_layer)
    validation_labels = _labels(validation)
    test_labels = _labels(test)
    report: dict[str, object] = {
        "train_class_counts": train_counts,
        "validation_class_counts": _counts(validation),
        "test_class_counts": _counts(test),
        "layer_index": resolved_layer,
        "validation_layer_metrics": _validation_layer_table(
            train, validation, examples[0].hidden_state_count, seed
        ),
    }
    if validation_scores is None or set(validation_labels.tolist()) != {0, 1}:
        return _save_without_test(
            output_dir,
            probe,
            mean,
            scale,
            examples,
            resolved_layer,
            report,
            dataset_record,
            reason=(
                f"{PROBE_NOT_TRAINED}: validation does not contain both classes, "
                "so the threshold was not chosen and the test split was not scored."
            ),
            label_protocol=label_protocol,
            metrics_note=metrics_note,
            seed=seed,
        )
    threshold = select_threshold(validation_labels, validation_scores)
    report["threshold"] = threshold
    report["threshold_rule"] = (
        "Maximum validation F1 on the grid 0.05..0.95. Ties keep the lower false-positive rate."
    )
    report["baselines"] = _baselines(
        train, validation, test, resolved_layer, seed, overlap_note=overlap_note
    )
    platt = fit_platt(validation_scores, validation_labels, seed=seed)
    calibration_validated = False
    test_metrics = None
    test_confusion = None
    test_ece = None
    if test_scores is None or set(test_labels.tolist()) != {0, 1}:
        report["test_evaluation"] = "incomplete: the test split does not contain both classes"
    else:
        test_metrics = compute_binary_metrics(test_labels, test_scores, threshold=threshold)
        test_confusion = confusion_counts(test_labels, test_scores, threshold=threshold)
        report["test_metrics"] = test_metrics
        report["test_confusion"] = test_confusion
        report["metrics_scale"] = "Uncalibrated logistic probabilities at the validation threshold."
        if (
            platt is not None
            and _enough_for_calibration(validation_labels)
            and _enough_for_calibration(test_labels)
        ):
            coef, intercept = platt
            calibrated_validation = apply_platt(validation_scores, coef, intercept)
            calibrated_test = apply_platt(test_scores, coef, intercept)
            test_ece = expected_calibration_error(test_labels, calibrated_test)
            report["platt"] = {"coef": coef, "intercept": intercept}
            report["test_ece"] = test_ece
            report["validation_ece"] = expected_calibration_error(
                validation_labels, calibrated_validation
            )
            report["calibration_rule"] = (
                f"Platt scaling is fit on validation scores only. It is accepted when each of "
                f"validation and test has at least {CALIBRATION_MIN_PER_CLASS} rows of each class "
                f"and the test ECE is at most {CALIBRATION_ECE_LIMIT}."
            )
            if test_ece <= CALIBRATION_ECE_LIMIT:
                calibration_validated = True
            else:
                report["calibration_rejected"] = (
                    f"Test ECE {test_ece:.3f} is above {CALIBRATION_ECE_LIMIT}. "
                    "The saved score stays the uncalibrated logistic output."
                )
        else:
            report["calibration_rejected"] = (
                "Calibration was not accepted. Validation or test does not have "
                f"{CALIBRATION_MIN_PER_CLASS} rows of each class, or Platt scaling did not fit."
            )
    metadata = _metadata(
        examples[0],
        layer_index=resolved_layer,
        threshold=threshold,
        seed=seed,
        report=report,
        dataset_record=dataset_record or {},
        calibration_validated=calibration_validated,
        examples=examples,
        label_protocol=label_protocol,
        metrics_note=metrics_note,
    )
    directory = save_probe(output_dir, probe, metadata=metadata, mean=mean, scale=scale)
    if test_metrics is None:
        reason = f"Saved an experimental probe to {directory}. Held-out metrics are incomplete."
    elif calibration_validated:
        reason = (
            f"Saved a probe to {directory}. Platt calibration passed the pre-registered ECE check. "
            "The percentage is the calibrated probability of the HaluEval hallucinated-answer class, "
            "not a measured accuracy on new generations."
        )
    else:
        reason = (
            f"Saved an experimental probe to {directory}. "
            "Calibration was not validated, so the demo must not present the score as a factual-risk percentage."
        )
    return TrainingOutcome(
        saved=True,
        reason=reason,
        directory=directory,
        metrics=test_metrics,
        threshold=threshold,
        calibration_validated=calibration_validated,
        report=report,
    )


def extract_examples(
    backend: HuggingFaceBackend,
    prepared: PreparedPairs,
    settings: GenerationSettings,
) -> tuple[list[FeatureExample], dict[str, int]]:
    """Encode both answers for the selected questions. A failed pair is dropped entirely."""
    examples: list[FeatureExample] = []
    skipped = {"too_long_or_empty": 0, "hidden_state_failed": 0}
    for split in ("train", "validation", "test"):
        for question_id in prepared.assignment.ids(split):
            pair = prepared.pairs[question_id]
            try:
                faithful = _one_example(
                    backend,
                    pair.question,
                    pair.right_answer,
                    pair.right_answer,
                    "faithful",
                    pair.question_id,
                    split,
                    settings,
                )
                hallucinated = _one_example(
                    backend,
                    pair.question,
                    pair.hallucinated_answer,
                    pair.right_answer,
                    "hallucinated",
                    pair.question_id,
                    split,
                    settings,
                )
            except ValueError:
                skipped["too_long_or_empty"] += 1
                continue
            except RuntimeError:
                skipped["hidden_state_failed"] += 1
                continue
            examples.extend((faithful, hallucinated))
            if len(examples) % 40 == 0:
                print(f"Extracted {len(examples)} answer states...", flush=True)
    return examples, skipped


def main(argv: list[str] | None = None) -> int:
    """Train from the pinned HaluEval snapshot and the demo model."""
    parser = argparse.ArgumentParser(prog="glassbox train-probe")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--max-questions", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="halueval-pair-001")
    args = parser.parse_args(argv)
    return train_command(args)


def train_command(args: argparse.Namespace) -> int:
    """Run the command and return a process status."""
    try:
        snapshot = _snapshot_path(args.snapshot)
        if args.download_data:
            if args.offline:
                raise ValueError("--offline refuses --download-data")
            download_snapshot(snapshot, allow_download=True)
        if not snapshot.is_file():
            raise FileNotFoundError(
                f"HaluEval snapshot is not at {snapshot}. "
                "Pass --download-data to fetch the pinned file, or run again after it is cached."
            )
        rows = read_pinned_snapshot(snapshot)
        prepared = prepare_pairs(rows, seed=args.seed)
        selected = prepared.assignment.subset(args.max_questions)
        prepared.assignment = selected
        if len(selected.train) < MIN_TRAIN_PER_CLASS:
            raise ValueError(
                f"{PROBE_NOT_TRAINED}: --max-questions leaves {len(selected.train)} training questions. "
                f"Need at least {MIN_TRAIN_PER_CLASS} so both classes can be adequate."
            )
        config = load_demo_config(args.config, offline=args.offline, model_name=args.model)
        if config.generation.position != REGISTERED_POSITION:
            raise ValueError(
                f"Demo position is {config.generation.position}. "
                f"This probe requires {REGISTERED_POSITION} so train and inference match."
            )
        settings = GenerationSettings(
            max_new_tokens=config.generation.max_new_tokens,
            temperature=0.0,
            top_k=config.generation.top_k,
            layer_index=config.generation.layer_index,
            position=config.generation.position,
            capture_all_layers=False,
        )
        backend = HuggingFaceBackend(config.spec)
        print(f"Loading {config.spec.name} once for feature extraction...", flush=True)
        loaded = backend.ensure_loaded()
        require_commit_hash(loaded.resolved_revision, field="model.revision")
        print(
            f"Loaded {loaded.spec.name} dtype={loaded.dtype} device={loaded.device} "
            f"revision={loaded.resolved_revision}",
            flush=True,
        )
        print(
            f"Filtered questions kept={prepared.report.kept} dropped={prepared.report.dropped}. "
            f"This run uses train={len(selected.train)} validation={len(selected.validation)} test={len(selected.test)}.",
            flush=True,
        )
        examples, skipped = extract_examples(backend, prepared, settings)
        print(f"Skipped pairs: {skipped}", flush=True)
        dataset_record = _dataset_record(prepared, skipped, args)
        final_dir = args.output or default_probe_dir()
        staging = final_dir.with_name(final_dir.name + ".staging")
        if staging.exists():
            shutil.rmtree(staging)
        outcome = train_from_examples(
            examples,
            layer_index=settings.layer_index,
            output_dir=staging,
            seed=args.seed,
            dataset_record=dataset_record,
        )
        if outcome.saved and outcome.directory is not None:
            publish_probe(outcome.directory, final_dir)
            outcome.reason = outcome.reason.replace(str(staging), str(final_dir))
            outcome.directory = final_dir
        _write_experiment(args.run_id, outcome, dataset_record, config.spec.name)
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Peak RSS: {peak_kb / 1024:.0f} MiB", flush=True)
    except (OSError, ValueError, RuntimeError, MemoryError, ImportError) as exc:
        print(f"{PROBE_NOT_TRAINED}: {exc}", file=sys.stderr)
        return 1
    print(outcome.reason)
    _print_report(outcome)
    return 0 if outcome.saved else 1


def publish_probe(staging: Path, destination: Path) -> None:
    """Replace the live probe only after the staging directory is complete."""
    incoming = destination.with_name(destination.name + ".incoming")
    if incoming.exists():
        shutil.rmtree(incoming)
    staging.rename(incoming)
    backup = destination.with_name(destination.name + ".previous")
    if destination.exists():
        if backup.exists():
            shutil.rmtree(backup)
        destination.rename(backup)
    incoming.rename(destination)


def _one_example(
    backend: HuggingFaceBackend,
    question: str,
    answer: str,
    reference: str,
    which: str,
    question_id: str,
    split: str,
    settings: GenerationSettings,
) -> FeatureExample:
    from glassbox.data.halueval import pair_label
    from glassbox.data.schema import make_example_id

    vectors, hidden_count, _layer, prompt_mode, _prompt = completion_hidden_states(
        backend.ensure_loaded(),
        question,
        answer,
        settings,
    )
    loaded = backend.ensure_loaded()
    return FeatureExample(
        example_id=make_example_id("halueval_qa", f"{question_id}.{which}"),
        question_id=question_id,
        split=split,
        label=pair_label(which),
        answer=answer,
        reference_answer=reference,
        vectors=vectors,
        prompt_mode=prompt_mode,
        position=settings.position,
        hidden_state_count=int(hidden_count),
        model_name=loaded.spec.name,
        model_revision=loaded.resolved_revision,
        dtype=loaded.dtype,
    )


def _fit(
    examples: list[FeatureExample],
    layer: int,
    *,
    seed: int,
) -> tuple[LogisticProbe | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    matrix = _matrix(examples, layer)
    labels = _labels(examples)
    if matrix is None or set(labels.tolist()) != {0, 1}:
        return None, None, None, None
    mean, scale = _standardize_fit(matrix)
    probe = LogisticProbe(ProbeConfig(random_state=seed))
    probe.fit(_standardize_apply(matrix, mean, scale), labels)
    scores = probe.predict_proba(_standardize_apply(matrix, mean, scale))
    return probe, mean, scale, scores


def _scores(
    probe: LogisticProbe,
    mean: np.ndarray,
    scale: np.ndarray,
    examples: list[FeatureExample],
    layer: int,
) -> np.ndarray | None:
    matrix = _matrix(examples, layer)
    if matrix is None or not examples:
        return None
    return probe.predict_proba(_standardize_apply(matrix, mean, scale))


def _validation_layer_table(
    train: list[FeatureExample],
    validation: list[FeatureExample],
    hidden_count: int,
    seed: int,
) -> list[dict[str, object]]:
    """Validation AUROC at every layer. The test split is not read here."""
    rows: list[dict[str, object]] = []
    if not validation or set(_labels(validation).tolist()) != {0, 1}:
        return rows
    present = set(train[0].vectors) if train else set()
    layers = sorted(present) if present and len(present) < hidden_count else range(hidden_count)
    for layer in layers:
        probe, mean, scale, _train_scores = _fit(train, layer, seed=seed)
        if probe is None or mean is None or scale is None:
            rows.append({"layer_index": layer, "validation_auroc": None, "note": "could not fit"})
            continue
        scores = _scores(probe, mean, scale, validation, layer)
        if scores is None:
            rows.append(
                {
                    "layer_index": layer,
                    "validation_auroc": None,
                    "note": "missing validation states",
                }
            )
            continue
        metrics = compute_binary_metrics(_labels(validation), scores, threshold=0.5)
        rows.append(
            {
                "layer_index": layer,
                "validation_auroc": metrics["auroc"],
                "validation_auprc": metrics["auprc"],
                "note": "Validation only. This table does not choose the saved layer and does not read the test split.",
            }
        )
    return rows


def _baselines(
    train: list[FeatureExample],
    validation: list[FeatureExample],
    test: list[FeatureExample],
    layer: int,
    seed: int,
    overlap_note: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "majority": majority_baseline(_labels(train), _labels(test))
        if test
        else {"note": "no test rows"},
    }
    if not test or set(_labels(test).tolist()) != {0, 1}:
        result["shuffled_label"] = {
            "note": "Test split lacks both classes, so the shuffled probe was not scored."
        }
        result["answer_length"] = {"note": "Test split lacks both classes."}
        result["lexical_overlap"] = {"note": "Test split lacks both classes."}
        return result
    result["shuffled_label"] = _shuffled_baseline(train, test, layer, seed)
    result["answer_length"] = _length_baseline(train, validation, test, seed)
    overlap = np.asarray(
        [overlap_score(item.answer, item.reference_answer) for item in test],
        dtype=np.float64,
    )
    try:
        result["lexical_overlap"] = compute_binary_metrics(_labels(test), overlap, threshold=0.5)
        result["lexical_overlap_note"] = overlap_note or (
            "Overlap uses the gold answer string. Under the pair protocol that string is the faithful "
            "answer, so this baseline can recover the label without hidden states."
        )
    except ValueError as exc:
        result["lexical_overlap"] = {"note": str(exc)}
    return result


def _shuffled_baseline(
    train: list[FeatureExample],
    test: list[FeatureExample],
    layer: int,
    seed: int,
) -> dict[str, object]:
    labels = _labels(train).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    matrix = _matrix(train, layer)
    test_matrix = _matrix(test, layer)
    if matrix is None or test_matrix is None or set(labels.tolist()) != {0, 1}:
        return {"note": "Shuffled-label probe was not scored."}
    mean, scale = _standardize_fit(matrix)
    probe = LogisticProbe(ProbeConfig(random_state=seed))
    probe.fit(_standardize_apply(matrix, mean, scale), labels)
    scores = probe.predict_proba(_standardize_apply(test_matrix, mean, scale))
    metrics = compute_binary_metrics(_labels(test), scores, threshold=0.5)
    metrics_out = dict(metrics)
    return {
        "metrics": metrics_out,
        "note": "Training labels were permuted with the run seed. Test labels were left intact.",
    }


def _length_baseline(
    train: list[FeatureExample],
    validation: list[FeatureExample],
    test: list[FeatureExample],
    seed: int,
) -> dict[str, object]:
    """Length logistic regression. The threshold is chosen on validation only."""
    if not validation or set(_labels(validation).tolist()) != {0, 1}:
        return {"note": "Validation split lacks both classes, so the length threshold was not fit."}
    probe = LogisticProbe(ProbeConfig(random_state=seed))
    probe.fit(_length_matrix(train), _labels(train))
    threshold = select_threshold(
        _labels(validation),
        probe.predict_proba(_length_matrix(validation)),
    )
    scores = probe.predict_proba(_length_matrix(test))
    try:
        return {
            "metrics": compute_binary_metrics(_labels(test), scores, threshold=threshold),
            "threshold": threshold,
            "note": (
                "Two features: character length and whitespace-separated word count of the supplied answer. "
                "The threshold was chosen on validation. AUROC does not use that threshold. "
                "halueval-pair-002 scored this baseline at threshold 0.5; do not mix the two F1 numbers."
            ),
        }
    except ValueError as exc:
        return {"note": str(exc)}


def _length_matrix(examples: list[FeatureExample]) -> np.ndarray:
    return np.asarray(
        [[len(item.answer), len(item.answer.split())] for item in examples],
        dtype=np.float64,
    )


def _save_without_test(
    output_dir: Path,
    probe: LogisticProbe,
    mean: np.ndarray,
    scale: np.ndarray,
    examples: list[FeatureExample],
    layer: int,
    report: dict[str, object],
    dataset_record: dict[str, object] | None,
    *,
    reason: str,
    label_protocol: str | None = None,
    metrics_note: str | None = None,
    seed: int = 0,
) -> TrainingOutcome:
    metadata = _metadata(
        examples[0],
        layer_index=layer,
        threshold=None,
        seed=seed,
        report=report,
        dataset_record=dataset_record or {},
        calibration_validated=False,
        examples=examples,
        label_protocol=label_protocol,
        metrics_note=metrics_note,
    )
    directory = save_probe(output_dir, probe, metadata=metadata, mean=mean, scale=scale)
    return TrainingOutcome(
        saved=True,
        reason=f"{reason} Saved an experimental probe to {directory}.",
        directory=directory,
        report=report,
    )


def _metadata(
    first: FeatureExample,
    *,
    layer_index: int,
    threshold: float | None,
    seed: int,
    report: dict[str, object],
    dataset_record: dict[str, object],
    calibration_validated: bool,
    examples: list[FeatureExample],
    label_protocol: str | None = None,
    metrics_note: str | None = None,
) -> dict[str, object]:
    platt = report.get("platt") if calibration_validated else None
    return {
        "schema_version": 1,
        "model_name": first.model_name,
        "model_revision": first.model_revision,
        "dtype": first.dtype,
        "layer_index": layer_index,
        "hidden_state_count": first.hidden_state_count,
        "position": first.position,
        "prompt_mode": first.prompt_mode,
        "label_protocol": label_protocol or LABEL_SOURCE,
        "positive_class": "hallucinated",
        "calibration": "platt" if calibration_validated else "none",
        "calibration_validated": calibration_validated,
        "platt": platt,
        "experimental": not calibration_validated,
        "threshold": threshold,
        "seed": seed,
        "probe_config": {
            "kind": "logistic_regression",
            "class_weight": ProbeConfig().class_weight,
            "max_iter": ProbeConfig().max_iter,
            "random_state": seed,
        },
        "dataset": dataset_record,
        "train_class_counts": report.get("train_class_counts"),
        "validation_class_counts": report.get("validation_class_counts"),
        "test_class_counts": report.get("test_class_counts"),
        "metrics": report.get("test_metrics"),
        "metrics_note": metrics_note
        or (
            "Scores are probabilities of the HaluEval hallucinated-answer class for a supplied answer. "
            "They are not labels of a new model generation. Early detection and intervention were not run."
        ),
        "test_ece": report.get("test_ece"),
        "test_confusion": report.get("test_confusion"),
        "baselines": _jsonable(report.get("baselines")),
        "validation_layer_metrics": _jsonable(report.get("validation_layer_metrics")),
        "split_ids": {
            "train": [item.example_id for item in examples if item.split == "train"],
            "validation": [item.example_id for item in examples if item.split == "validation"],
            "test": [item.example_id for item in examples if item.split == "test"],
        },
    }


def _dataset_record(
    prepared: PreparedPairs, skipped: dict[str, int], args: argparse.Namespace
) -> dict[str, object]:
    return {
        "dataset_id": "halueval_qa",
        "label_source": LABEL_SOURCE,
        "source_commit": PINNED_COMMIT,
        "sha256": PINNED_SHA256,
        "filter": prepared.report.as_dict(),
        "audit_notes": list(prepared.audit_notes),
        "seed": args.seed,
        "max_questions": args.max_questions,
        "used_question_ids": {
            "train": list(prepared.assignment.train),
            "validation": list(prepared.assignment.validation),
            "test": list(prepared.assignment.test),
        },
        "skipped_during_extraction": skipped,
        "prompt": "demo render_prompt(question) plus the supplied answer; knowledge is not in the prompt",
        "position": REGISTERED_POSITION,
    }


def _write_experiment(
    run_id: str, outcome: TrainingOutcome, dataset_record: dict[str, object], model_name: str
) -> None:
    metrics = outcome.metrics or {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "false_positive_rate": 0.0,
    }
    if outcome.metrics is None:
        notes = "Test metrics were not computed. The zeros in this record are the required schema's empty placeholder, not measurements. See probe metadata."
        # Don't write fake zeros as if they were measurements. Skip the run record if we have no metrics.
        print("No test metrics, so no experiment record was written.")
        return
    commit, dirty = _git_state()
    notes = (
        "HaluEval pair protocol on Qwen2.5-0.5B-Instruct. "
        f"Calibration validated={outcome.calibration_validated}. "
        f"Worktree dirty={dirty}. Early detection and intervention were not run. "
        f"Model requested: {model_name}."
    )
    record = build_run_record(
        run_id=run_id,
        git_commit=commit,
        config={
            "protocol": LABEL_SOURCE,
            "model_name": model_name,
            "position": REGISTERED_POSITION,
            "dataset": dataset_record,
            "calibration_validated": outcome.calibration_validated,
            "threshold": outcome.threshold,
        },
        metrics={**metrics, "early_warning_lead": None},
        notes=notes,
    )
    root = repo_root() / "artifacts"
    write_json(root / "experiments" / run_id / "record.json", record, root=root)
    write_json(
        root / "experiments" / run_id / "report.json",
        {"reason": outcome.reason, "report": _jsonable(outcome.report)},
        root=root,
    )


def _print_report(outcome: TrainingOutcome) -> None:
    counts = outcome.report.get("train_class_counts")
    if counts:
        print(f"Train class counts: {counts}")
    print(f"Validation class counts: {outcome.report.get('validation_class_counts')}")
    print(f"Test class counts: {outcome.report.get('test_class_counts')}")
    if outcome.threshold is not None:
        print(f"Frozen threshold: {outcome.threshold:.2f}")
    if outcome.metrics is not None:
        for name, value in outcome.metrics.items():
            print(f"  test {name}: {value:.3f}")
    confusion = outcome.report.get("test_confusion")
    if isinstance(confusion, dict):
        print(
            f"  test confusion tn={confusion.get('tn')} fp={confusion.get('fp')} fn={confusion.get('fn')} tp={confusion.get('tp')}"
        )
    if outcome.report.get("test_ece") is not None:
        print(f"  test ECE: {outcome.report['test_ece']:.3f}")
    baselines = outcome.report.get("baselines")
    if isinstance(baselines, dict):
        for name, payload in baselines.items():
            print(f"  baseline {name}: {_brief(payload)}")


def _brief(payload: object) -> str:
    if isinstance(payload, dict) and "metrics" in payload and isinstance(payload["metrics"], dict):
        auroc = payload["metrics"].get("auroc")
        return f"auroc={auroc}"
    if isinstance(payload, dict) and "auroc" in payload:
        return f"auroc={payload.get('auroc')} f1={payload.get('f1')}"
    return str(payload)[:180]


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
    return commit, dirty


def _snapshot_path(path: Path | None) -> Path:
    if path is not None:
        return path
    return repo_root() / "artifacts" / "datasets" / "halueval" / "qa_data.json"


def _by_split(examples: list[FeatureExample]) -> dict[str, list[FeatureExample]]:
    grouped: dict[str, list[FeatureExample]] = {}
    for item in examples:
        grouped.setdefault(item.split, []).append(item)
    return grouped


def _labeled(examples: list[FeatureExample]) -> list[FeatureExample]:
    return [item for item in examples if item.label in {Label.FAITHFUL, Label.HALLUCINATED}]


def _counts(examples: list[FeatureExample]) -> dict[str, int]:
    return class_counts(_labels(examples)) if examples else {"faithful": 0, "hallucinated": 0}


def _labels(examples: list[FeatureExample]) -> np.ndarray:
    return np.asarray([int(item.label.value) for item in examples], dtype=int)


def _matrix(examples: list[FeatureExample], layer: int) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    for item in examples:
        vector = item.vectors.get(layer)
        if vector is None:
            return None
        rows.append(np.asarray(vector, dtype=np.float64).reshape(-1))
    if not rows:
        return None
    return np.vstack(rows)


def _standardize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return fit_standard_scaler(features)


def _standardize_apply(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return apply_standard_scaler(features, mean, scale)


def _require_same_model(examples: list[FeatureExample]) -> None:
    first = examples[0]
    for item in examples[1:]:
        if (
            item.model_name != first.model_name
            or item.model_revision != first.model_revision
            or item.dtype != first.dtype
            or item.prompt_mode != first.prompt_mode
            or item.position != first.position
            or item.hidden_state_count != first.hidden_state_count
        ):
            raise ValueError(
                "All feature rows must come from the same model, revision, prompt, and position"
            )


def _require_position(examples: list[FeatureExample]) -> None:
    if any(item.position != REGISTERED_POSITION for item in examples):
        raise ValueError(f"Every row must use position {REGISTERED_POSITION}")


def _require_one_split_per_question(examples: list[FeatureExample]) -> None:
    found: dict[str, str] = {}
    for item in examples:
        previous = found.get(item.question_id)
        if previous is not None and previous != item.split:
            raise ValueError(f"Question {item.question_id} is in {previous} and {item.split}")
        found[item.question_id] = item.split


def _enough_for_calibration(labels: np.ndarray) -> bool:
    counts = class_counts(labels)
    return (
        counts["faithful"] >= CALIBRATION_MIN_PER_CLASS
        and counts["hallucinated"] >= CALIBRATION_MIN_PER_CLASS
    )


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
