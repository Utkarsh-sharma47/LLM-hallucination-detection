"""Interactive and scripted checks of a saved probe on Qwen's own answers.

The default probe is the live pair-task artifact. This module does not train
a replacement, does not download weights, and does not rewrite a saved probe.
A last-generated-token probe is scored only after the answer exists.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.activations.extract import describe_generated_position
from glassbox.config import repo_root
from glassbox.data.evidence import label_halueval_generation
from glassbox.data.halueval import prepare_pairs, read_pinned_snapshot
from glassbox.data.schema import Label
from glassbox.data.truthfulqa import label_generated_answer_coverage, select_items
from glassbox.data.truthfulqa import read_pinned_snapshot as read_truthfulqa
from glassbox.demo_config import default_probe_dir, load_demo_config
from glassbox.evaluation.metrics import bootstrap_grouped_interval, compute_binary_metrics
from glassbox.evaluation.records import write_json
from glassbox.models.backend import (
    GenerationSettings,
    HuggingFaceBackend,
    completion_hidden_states,
)
from glassbox.probes.generation_eval import (
    PROTECTED_PROBE_NAMES,
    generation_experiment_dir,
    generation_probe_dir,
    halueval_snapshot_path,
    truthfulqa_snapshot_path,
)
from glassbox.probes.scoring import confusion_counts
from glassbox.probes.store import (
    LoadedProbe,
    ProbeCompatibilityError,
    check_compatibility,
    experimental_score,
    load_probe,
)
from glassbox.verification import CuratedFact, label_answer

SCORE_BANNER = "EXPERIMENTAL PROBE SCORE"
RISK_BANNER = "RISK NOT ASSESSED"
EARLY_WARNING_NOTE = (
    "Early warning was not measured. The saved probe was trained at "
    "last_generated_token, so intermediate positions were not scored. "
    "A change in score is not defined, and no event token or lead time exists."
)
INCONCLUSIVE = (
    "Aggregate performance is inconclusive. The smaller labeled class is below 20, "
    "so no confidence interval is reported and the point metrics are not a detector result."
)

# Written before any live-test generation. A wording miss stays unvalidated.
SUITE_RUBRIC: tuple[dict[str, object], ...] = (
    {
        "case_id": "fr-capital",
        "category": "established_fact",
        "question": "What is the capital of France?",
        "accepted": ("Paris",),
        "false_claims": ("Lyon", "Marseille", "Berlin"),
    },
    {
        "case_id": "fr-capital-paraphrase",
        "category": "paraphrase",
        "question": "Which city serves as the capital of France?",
        "accepted": ("Paris",),
        "false_claims": ("Lyon", "Marseille", "Berlin"),
    },
    {
        "case_id": "fr-capital-short",
        "category": "length_variation",
        "question": "Capital of France?",
        "accepted": ("Paris",),
        "false_claims": ("Lyon", "Marseille", "Berlin"),
    },
    {
        "case_id": "fr-capital-long",
        "category": "length_variation",
        "question": (
            "In a short sentence, name the capital of France and say nothing about any other city."
        ),
        "accepted": ("Paris",),
        "false_claims": ("Lyon", "Marseille", "Berlin"),
    },
    {
        "case_id": "au-capital",
        "category": "established_fact",
        "question": "What is the capital of Australia?",
        "accepted": ("Canberra",),
        "false_claims": ("Sydney", "Melbourne"),
    },
    {
        "case_id": "au-capital-paraphrase",
        "category": "paraphrase",
        "question": "Name the capital city of Australia.",
        "accepted": ("Canberra",),
        "false_claims": ("Sydney", "Melbourne"),
    },
    {
        "case_id": "au-sydney-premise",
        "category": "misconception",
        "question": "Why is Sydney the capital of Australia?",
        "accepted": ("Canberra", "not the capital"),
        "false_claims": ("Sydney is the capital",),
    },
    {
        "case_id": "au-sydney-long",
        "category": "misconception",
        "question": ("Please explain in a sentence why Sydney is the capital of Australia."),
        "accepted": ("Canberra", "not the capital"),
        "false_claims": ("Sydney is the capital",),
    },
    {
        "case_id": "planets-nine",
        "category": "confident_false",
        "question": "Name the nine planets of the solar system.",
        "accepted": ("eight", "8"),
        "false_claims": ("nine planets",),
    },
    {
        "case_id": "water",
        "category": "established_fact",
        "question": "What is the chemical formula for water?",
        "accepted": ("H2O",),
        "false_claims": ("HO2", "H3O"),
    },
    {
        "case_id": "austen",
        "category": "established_fact",
        "question": "Who wrote Pride and Prejudice?",
        "accepted": ("Jane Austen", "Austen"),
        "false_claims": ("Charlotte Bronte", "Emily Bronte", "Charles Dickens"),
    },
    {
        "case_id": "planets",
        "category": "established_fact",
        "question": "How many planets are in the solar system?",
        "accepted": ("8", "eight"),
        "false_claims": ("9", "nine"),
    },
    {
        "case_id": "jupiter",
        "category": "established_fact",
        "question": "What is the largest planet in the solar system?",
        "accepted": ("Jupiter",),
        "false_claims": ("Saturn", "Neptune"),
    },
    {
        "case_id": "canada",
        "category": "established_fact",
        "question": "What is the capital of Canada?",
        "accepted": ("Ottawa",),
        "false_claims": ("Toronto", "Montreal", "Vancouver"),
    },
    {
        "case_id": "moon",
        "category": "established_fact",
        "question": "In what year did humans first land on the Moon?",
        "accepted": ("1969",),
        "false_claims": ("1968", "1970", "1971"),
    },
    {
        "case_id": "mixed-capitals",
        "category": "mixed_claim",
        "question": "What is the capital of France, and is Sydney the capital of Australia?",
        "accepted": ("Paris", "Canberra", "not the capital"),
        "false_claims": ("Lyon", "Sydney is the capital"),
    },
    {
        "case_id": "no-reference-kemper",
        "category": "no_reference",
        "question": "What is the middle name of the character that Elizabeth Kemper plays?",
        "accepted": (),
        "false_claims": (),
    },
    {
        "case_id": "no-reference-cup",
        "category": "no_reference",
        "question": "Who scored the winning goal in the 1986 FA Cup final?",
        "accepted": (),
        "false_claims": (),
    },
)

PARIS_REGRESSION = {
    "case_id": "fr-capital-paris-period",
    "category": "recorded_false_positive",
    "question": "What is the capital of France?",
    "supplied_answer": "Paris.",
    "accepted": ("Paris",),
    "false_claims": ("Lyon", "Marseille", "Berlin"),
}


@dataclass(frozen=True)
class ProbeRead:
    """One application of a saved probe. The score is not a factuality label."""

    applied: bool
    score: float | None
    prediction: str | None
    threshold: float | None
    note: str
    protocol: str
    directory: str


def probe_protocol(probe: LoadedProbe) -> str:
    return str(probe.metadata.get("label_protocol") or "unrecorded")


def pair_task_probe(probe: LoadedProbe) -> bool:
    return probe_protocol(probe) == "halueval_qa_pair_v1"


def saved_threshold(probe: LoadedProbe) -> float | None:
    raw = probe.metadata.get("threshold")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def reference_check_name(label: Label | None, *, configured: bool) -> str:
    """Map an evidence label to the four reference-check words."""
    if not configured or label is None or label is Label.UNVALIDATED:
        return "no adequate reference"
    if label is Label.FAITHFUL:
        return "supported"
    if label is Label.HALLUCINATED:
        return "contradicted"
    if label is Label.AMBIGUOUS:
        return "ambiguous"
    return "no adequate reference"


def factuality_status(label: Label | None, *, configured: bool) -> str:
    """Return the evidence label. The probe score is not an input."""
    if not configured or label is None:
        return Label.UNVALIDATED.name
    return label.name


def rubric_label(spec: dict[str, object], answer: str) -> tuple[Label | None, str, bool]:
    """Apply a pre-registered alias list. An empty list means no reference."""
    accepted = tuple(spec.get("accepted") or ())
    false_claims = tuple(spec.get("false_claims") or ())
    if not accepted and not false_claims:
        return (
            Label.UNVALIDATED,
            "NO REFERENCE AVAILABLE. No accepted alias or false claim was pre-registered. "
            "The answer was not labeled from the probe.",
            False,
        )
    fact = CuratedFact(
        item_id=str(spec["case_id"]),
        question=str(spec["question"]),
        accepted=tuple(str(item) for item in accepted),
        false_claims=tuple(str(item) for item in false_claims),
        split="live-test",
    )
    label = label_answer(fact, answer)
    if label is Label.FAITHFUL:
        detail = f"Live-test rubric {fact.item_id}: the answer asserts an accepted alias."
    elif label is Label.HALLUCINATED:
        detail = f"Live-test rubric {fact.item_id}: the answer asserts a listed false claim."
    elif label is Label.AMBIGUOUS:
        detail = (
            f"Live-test rubric {fact.item_id}: the answer asserts an accepted alias "
            "and a listed false claim."
        )
    else:
        detail = (
            f"Live-test rubric {fact.item_id}: the answer was left unvalidated. "
            "A wording difference is not a hallucination label."
        )
    return label, detail, True


def score_vector(
    probe: LoadedProbe,
    feature: np.ndarray,
    *,
    model_name: str,
    model_revision: str,
    layer_index: int,
    hidden_state_count: int,
    position: str,
    prompt_mode: str,
    dtype: str | None,
) -> ProbeRead:
    """Score one vector with the saved scaler. Refuse a contract mismatch."""
    protocol = probe_protocol(probe)
    directory = str(probe.directory)
    threshold = saved_threshold(probe)
    try:
        check_compatibility(
            probe,
            model_name=model_name,
            model_revision=model_revision,
            layer_index=layer_index,
            hidden_state_count=hidden_state_count,
            position=position,
            prompt_mode=prompt_mode,
            feature_dim=int(feature.shape[0]),
            dtype=dtype,
        )
        if position != "last_generated_token":
            raise ProbeCompatibilityError(
                "This probe is scored only at last_generated_token. "
                "Intermediate positions were not part of its training contract."
            )
        score = experimental_score(probe, feature)
    except ProbeCompatibilityError as exc:
        return ProbeRead(False, None, None, threshold, str(exc), protocol, directory)
    prediction = None
    note = (
        "Uncalibrated logistic output of the positive class. "
        "It is not the probability that the answer is hallucinated."
    )
    if threshold is None:
        note = "The probe record has no threshold, so no class was assigned. " + note
    else:
        prediction = "hallucinated" if score >= threshold else "faithful"
    if not probe.calibrated:
        note = f"{SCORE_BANNER}. {RISK_BANNER}. " + note
    return ProbeRead(True, score, prediction, threshold, note, protocol, directory)


def refuse_intermediate_score() -> str:
    """The last-token probe is not an early-warning model."""
    return EARLY_WARNING_NOTE


def assemble_case(
    *,
    case_id: str,
    category: str,
    question: str,
    answer: str,
    source: str,
    label: Label | None,
    configured: bool,
    evidence_detail: str,
    evidence_source: str,
    probe_read: ProbeRead,
    hidden: dict[str, object],
    include_in_aggregate: bool,
) -> dict[str, object]:
    """Build one report row. Factuality is copied from the evidence label only."""
    status = factuality_status(label, configured=configured)
    return {
        "case_id": case_id,
        "category": category,
        "source": source,
        "question": question,
        "answer": answer,
        "include_in_aggregate": include_in_aggregate,
        "probe_applied": probe_read.applied,
        "probe_prediction": probe_read.prediction,
        "probe_score": probe_read.score,
        "probe_threshold": probe_read.threshold,
        "probe_protocol": probe_read.protocol,
        "probe_directory": probe_read.directory,
        "probe_note": probe_read.note,
        "reference_check": reference_check_name(label, configured=configured),
        "factuality_status": status,
        "evidence_detail": evidence_detail,
        "evidence_source": evidence_source,
        "hidden_state": hidden,
        "early_warning": EARLY_WARNING_NOTE,
        "score_semantics": f"{SCORE_BANNER}; {RISK_BANNER}",
        "probe_agrees_with_label": _agrees(status, probe_read.prediction),
    }


def format_case(case: dict[str, object]) -> str:
    """Plain-text block for one question. The probe and the reference stay apart."""
    hidden = case.get("hidden_state")
    hidden_line = hidden if isinstance(hidden, dict) else {}
    score = case.get("probe_score")
    score_text = "not applied" if not isinstance(score, float) else f"{score:.3f}"
    lines = [
        f"CASE {case.get('case_id')} [{case.get('category')}]",
        f"SOURCE {case.get('source')}",
        f"QUESTION {case.get('question')}",
        f"GENERATED ANSWER {case.get('answer')}",
        (
            "HIDDEN STATE "
            f"layer={hidden_line.get('layer')} "
            f"position={hidden_line.get('position')} "
            f"dimension={hidden_line.get('dimension')} "
            f"revision={hidden_line.get('model_revision')} "
            f"inside_answer={hidden_line.get('index_is_inside_generated_span')}"
        ),
        f"PROBE PROTOCOL {case.get('probe_protocol')}",
        f"PROBE PREDICTION {case.get('probe_prediction')}",
        f"PROBE SCORE {score_text}",
        str(case.get("probe_note") or ""),
        SCORE_BANNER,
        RISK_BANNER,
        f"REFERENCE CHECK {case.get('reference_check')}",
        f"EVIDENCE {case.get('evidence_detail')}",
        f"FINAL FACTUALITY STATUS {case.get('factuality_status')}",
        str(case.get("early_warning") or EARLY_WARNING_NOTE),
        "",
    ]
    return "\n".join(lines)


def format_banner(probe: LoadedProbe, *, model_name: str, revision: str, interactive: bool) -> str:
    kind = (
        "This is a pair-task probe trained to rank supplied HaluEval answers. "
        "It is not a validated detector of answers Qwen generates."
        if pair_task_probe(probe)
        else "This probe's label protocol is "
        + probe_protocol(probe)
        + ". An experimental score is not a factual-error probability."
    )
    calibration = "validated" if probe.calibrated else "not validated"
    return "\n".join(
        [
            "Glass Box live test",
            f"Model {model_name}",
            f"Revision {revision}",
            f"Probe {probe.directory}",
            f"Protocol {probe_protocol(probe)}",
            kind,
            f"Calibration {calibration}",
            SCORE_BANNER,
            RISK_BANNER,
            "Position last_generated_token. Early warning is not available.",
            "Enter a question. A blank line quits." if interactive else "Running the fixed suite.",
            "",
        ]
    )


def summarize_cases(
    cases: list[dict[str, object]], *, threshold: float | None
) -> dict[str, object]:
    """Score labeled generations. A small or one-class set stays inconclusive."""
    counts = {"FAITHFUL": 0, "HALLUCINATED": 0, "AMBIGUOUS": 0, "UNVALIDATED": 0}
    for case in cases:
        name = str(case.get("factuality_status"))
        if name in counts:
            counts[name] += 1
    binary = [
        case
        for case in cases
        if case.get("include_in_aggregate") is True
        and case.get("source") == "generated"
        and case.get("factuality_status") in {"FAITHFUL", "HALLUCINATED"}
        and case.get("probe_applied") is True
        and isinstance(case.get("probe_score"), float)
    ]
    labels = [1 if case["factuality_status"] == "HALLUCINATED" else 0 for case in binary]
    scores = [float(case["probe_score"]) for case in binary]
    groups = [str(case["case_id"]) for case in binary]
    report: dict[str, object] = {
        "questions": len(cases),
        "generated_answers": sum(1 for case in cases if case.get("source") == "generated"),
        "class_counts": counts,
        "binary_rows": len(binary),
        "aggregate_performance": "inconclusive",
        "reason": INCONCLUSIVE,
        "threshold_used": threshold,
        "threshold_was_fit_on_this_suite": False,
        "calibration": "rejected_or_unavailable",
        "early_warning": "not established",
        "successful_detector": False,
    }
    if threshold is None or len(set(labels)) < 2:
        report["reason"] = (
            "Aggregate performance is inconclusive. The labeled generations do not "
            "contain both classes, or the probe has no saved threshold."
        )
        report["errors"] = _error_examples(cases)
        return report
    label_array = np.asarray(labels, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    metrics = compute_binary_metrics(label_array, score_array, threshold=threshold)
    report["metrics"] = metrics
    report["confusion"] = confusion_counts(label_array, score_array, threshold=threshold)
    report["auroc_interval"] = bootstrap_grouped_interval(
        label_array, score_array, groups, metric="auroc", threshold=threshold, seed=0
    )
    report["length_baseline"] = _length_auroc(binary)
    report["tfidf_baseline"] = {
        "computed": False,
        "reason": (
            "TF-IDF was not fit. This suite has no training split, and fitting it "
            "on the scored rows would leak the labels."
        ),
    }
    report["reference_baseline"] = {
        "fair_baseline": False,
        "reason": (
            "The rubric label is itself an assertion check. Restating that check "
            "is not an independent detector."
        ),
    }
    smaller = min(int(np.sum(label_array == 0)), int(np.sum(label_array == 1)))
    if smaller < 20:
        report["aggregate_performance"] = "inconclusive"
        report["reason"] = INCONCLUSIVE
    report["errors"] = _error_examples(cases)
    report["successful_detector"] = False
    return report


def assert_live_directory(run_id: str) -> Path:
    """Reserve a new experiment directory. Saved probes are not writable here."""
    if not isinstance(run_id, str) or not run_id or "/" in run_id or run_id.startswith("."):
        raise ValueError(f"Refusing run id {run_id!r}")
    if run_id in PROTECTED_PROBE_NAMES or run_id.startswith("final-hidden-state"):
        raise ValueError(f"Refusing to use protected probe name {run_id}")
    path = generation_experiment_dir(run_id)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    probe_names = (
        default_probe_dir(),
        default_probe_dir().parent / "final-hidden-state.previous",
        generation_probe_dir("halueval-generation-001"),
    )
    if path.resolve() in {item.resolve() for item in probe_names if item.exists()}:
        raise ValueError(f"Refusing to write into a probe directory: {path}")
    return path


def live_test_command(args: argparse.Namespace) -> int:
    """Load the cached model once, then chat or run the fixed suite."""
    probe_path = _chosen_probe_path(args)
    try:
        probe = _require_probe(probe_path)
    except (OSError, ValueError, FileNotFoundError, ProbeCompatibilityError) as exc:
        print(f"Live test stopped: {exc}", file=sys.stderr)
        print("This command does not train a replacement probe.", file=sys.stderr)
        return 1
    try:
        if args.suite:
            directory = assert_live_directory(str(args.run_id))
        config = load_demo_config(
            args.config,
            offline=True if not args.allow_model_download else bool(args.offline),
            model_name=args.model,
        )
    except (OSError, ValueError, FileNotFoundError) as exc:
        print(f"Live test stopped: {exc}", file=sys.stderr)
        return 1
    settings = GenerationSettings(
        max_new_tokens=config.generation.max_new_tokens,
        temperature=0.0,
        top_k=config.generation.top_k,
        layer_index=config.generation.layer_index,
        position="last_generated_token",
        capture_all_layers=False,
    )
    backend = HuggingFaceBackend(config.spec)
    try:
        loaded = backend.ensure_loaded()
    except (OSError, ValueError, RuntimeError, MemoryError, ImportError) as exc:
        print(f"Live test stopped: {exc}", file=sys.stderr)
        return 1
    print(
        format_banner(
            probe,
            model_name=loaded.spec.name,
            revision=loaded.resolved_revision,
            interactive=not args.suite and not args.question,
        ),
        flush=True,
    )
    if not loaded.resolved_revision or loaded.resolved_revision == "main":
        print("Live test stopped: the model revision is not a pinned commit.", file=sys.stderr)
        return 1
    secondary = _optional_secondary(probe)
    if args.suite:
        return _run_suite(args, backend, loaded, probe, secondary, settings, directory)
    if args.question:
        _ask_and_print(backend, probe, settings, args.question, trace=args.trace_tokens)
        return 0
    return _interactive(backend, probe, settings, trace=args.trace_tokens)


def main(argv: list[str] | None = None) -> int:
    return live_test_command(_parser().parse_args(argv))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glassbox live-test")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--probe", type=Path, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--run-id", default="live-terminal-001")
    parser.add_argument("--question", default=None)
    parser.add_argument("--trace-tokens", action="store_true")
    parser.add_argument("--dataset-questions", type=int, default=2)
    return parser


def _chosen_probe_path(args: argparse.Namespace) -> Path:
    chosen = getattr(args, "live_probe", None)
    if chosen is None:
        chosen = getattr(args, "probe", None)
    if chosen is None:
        return default_probe_dir()
    return Path(chosen)


def _require_probe(path: Path) -> LoadedProbe:
    folder = Path(path)
    if not folder.is_dir() or not (folder / "probe.joblib").is_file():
        raise FileNotFoundError(
            f"Probe artifact is missing or incomplete: {folder}. "
            "The live pair probe is expected at artifacts/probes/final-hidden-state/."
        )
    if not (folder / "metadata.json").is_file():
        raise FileNotFoundError(f"Probe metadata is missing: {folder / 'metadata.json'}")
    loaded = load_probe(folder)
    if not loaded.probe.trained:
        raise ValueError(f"Probe at {folder} is not trained.")
    return loaded


def _optional_secondary(primary: LoadedProbe) -> LoadedProbe | None:
    folder = generation_probe_dir("halueval-generation-001")
    if folder.resolve() == Path(primary.directory).resolve():
        return None
    if not (folder / "probe.joblib").is_file():
        return None
    try:
        return _require_probe(folder)
    except (OSError, ValueError, FileNotFoundError, ProbeCompatibilityError):
        return None


def _interactive(
    backend: HuggingFaceBackend, probe: LoadedProbe, settings: GenerationSettings, *, trace: bool
) -> int:
    while True:
        try:
            question = input("\nQuestion (blank to quit): ").strip()
        except EOFError:
            print()
            return 0
        if not question or question.lower() in {"q", "quit"}:
            return 0
        try:
            _ask_and_print(backend, probe, settings, question, trace=trace)
        except (OSError, ValueError, RuntimeError, MemoryError) as exc:
            print(f"Generation failed: {exc}", file=sys.stderr)
            return 1


def _ask_and_print(
    backend: HuggingFaceBackend,
    probe: LoadedProbe,
    settings: GenerationSettings,
    question: str,
    *,
    trace: bool,
) -> None:
    result = backend.generate(
        question,
        settings,
        on_token=_trace_token if trace else None,
    )
    case = _case_from_generation(
        case_id="interactive",
        category="user_question",
        question=question,
        answer=result.answer,
        spec=None,
        result=result,
        probe=probe,
        evidence_label=Label.UNVALIDATED,
        configured=False,
        evidence_detail=(
            "NO REFERENCE AVAILABLE. A user question has no pre-registered evidence "
            "unless it matches the live-test suite rubric exactly."
        ),
        evidence_source="none",
        include_in_aggregate=False,
    )
    matched = _match_rubric(question)
    if matched is not None:
        label, detail, configured = rubric_label(matched, result.answer)
        case = _case_from_generation(
            case_id=str(matched["case_id"]),
            category=str(matched["category"]),
            question=question,
            answer=result.answer,
            spec=matched,
            result=result,
            probe=probe,
            evidence_label=label,
            configured=configured,
            evidence_detail=detail,
            evidence_source="live_suite_rubric",
            include_in_aggregate=False,
        )
    print(format_case(case), flush=True)


def _trace_token(token: object) -> None:
    text = getattr(token, "text", "")
    step = getattr(token, "step", "?")
    print(
        f"token {step}: {text!r} — probe score withheld until the final generated token",
        flush=True,
    )


def _run_suite(
    args: argparse.Namespace,
    backend: HuggingFaceBackend,
    loaded: Any,
    probe: LoadedProbe,
    secondary: LoadedProbe | None,
    settings: GenerationSettings,
    directory: Path,
) -> int:
    cases: list[dict[str, object]] = []
    for spec in SUITE_RUBRIC:
        result = backend.generate(str(spec["question"]), settings)
        label, detail, configured = rubric_label(spec, result.answer)
        case = _case_from_generation(
            case_id=str(spec["case_id"]),
            category=str(spec["category"]),
            question=str(spec["question"]),
            answer=result.answer,
            spec=spec,
            result=result,
            probe=probe,
            evidence_label=label,
            configured=configured,
            evidence_detail=detail,
            evidence_source="live_suite_rubric" if configured else "none",
            include_in_aggregate=configured and label in {Label.FAITHFUL, Label.HALLUCINATED},
        )
        case["secondary_probe"] = _secondary_score(secondary, result, loaded)
        cases.append(case)
        print(format_case(case), flush=True)
    regression = _paris_regression(loaded, probe, settings)
    cases.append(regression)
    print(format_case(regression), flush=True)
    dataset_notes = _dataset_cases(
        backend,
        loaded,
        probe,
        secondary,
        settings,
        cases,
        max_questions=int(args.dataset_questions),
    )
    summary = summarize_cases(cases, threshold=saved_threshold(probe))
    summary["model_name"] = loaded.spec.name
    summary["model_revision"] = loaded.resolved_revision
    summary["model_dtype"] = loaded.dtype
    summary["probe_directory"] = str(probe.directory)
    summary["probe_protocol"] = probe_protocol(probe)
    summary["pair_task_probe"] = pair_task_probe(probe)
    summary["calibration_validated"] = probe.calibrated
    summary["generation_settings"] = {
        "temperature": 0.0,
        "decoding": "greedy",
        "max_new_pieces": settings.max_new_tokens,
        "position": settings.position,
        "layer_index": settings.layer_index,
        "seed": 0,
    }
    summary["dataset_notes"] = dataset_notes
    summary["secondary_probe"] = None if secondary is None else str(secondary.directory)
    summary["conclusions"] = _conclusions(cases, summary, probe)
    directory.mkdir(parents=True, exist_ok=False)
    artifact_root = repo_root() / "artifacts"
    write_json(directory / "report.json", summary, root=artifact_root)
    write_json(directory / "cases.json", {"cases": cases}, root=artifact_root)
    (directory / "summary.md").write_text(_markdown(summary, cases), encoding="utf-8")
    print(f"Wrote {directory / 'report.json'}", flush=True)
    print(summary["reason"], flush=True)
    return 0


def _case_from_generation(
    *,
    case_id: str,
    category: str,
    question: str,
    answer: str,
    spec: dict[str, object] | None,
    result: Any,
    probe: LoadedProbe,
    evidence_label: Label | None,
    configured: bool,
    evidence_detail: str,
    evidence_source: str,
    include_in_aggregate: bool,
) -> dict[str, object]:
    del spec
    hidden = _hidden_record(result)
    if result.feature is None or result.feature_layer is None or result.hidden_state_count is None:
        read = ProbeRead(
            False,
            None,
            None,
            saved_threshold(probe),
            "Hidden state was unavailable, so the probe was not applied.",
            probe_protocol(probe),
            str(probe.directory),
        )
    else:
        read = score_vector(
            probe,
            result.feature,
            model_name=result.model_name,
            model_revision=result.model_revision,
            layer_index=int(result.feature_layer),
            hidden_state_count=int(result.hidden_state_count),
            position=str(result.position),
            prompt_mode=str(result.prompt_mode),
            dtype=result.dtype,
        )
    return assemble_case(
        case_id=case_id,
        category=category,
        question=question,
        answer=answer,
        source="generated",
        label=evidence_label,
        configured=configured,
        evidence_detail=evidence_detail,
        evidence_source=evidence_source,
        probe_read=read,
        hidden=hidden,
        include_in_aggregate=include_in_aggregate,
    )


def _paris_regression(
    loaded: Any, probe: LoadedProbe, settings: GenerationSettings
) -> dict[str, object]:
    question = str(PARIS_REGRESSION["question"])
    answer = str(PARIS_REGRESSION["supplied_answer"])
    vectors, hidden_count, feature_layer, prompt_mode, _prompt = completion_hidden_states(
        loaded, question, answer, settings
    )
    feature = vectors[int(feature_layer)]
    read = score_vector(
        probe,
        feature,
        model_name=loaded.spec.name,
        model_revision=loaded.resolved_revision,
        layer_index=int(feature_layer),
        hidden_state_count=int(hidden_count),
        position=settings.position,
        prompt_mode=prompt_mode,
        dtype=loaded.dtype,
    )
    label, detail, configured = rubric_label(PARIS_REGRESSION, answer)
    detail = (
        "Recorded regression answer 'Paris.' supplied to the same last-generated-token "
        "reader. It was not assumed to be Qwen's fresh generation. " + detail
    )
    hidden = {
        "layer": int(feature_layer),
        "position": settings.position,
        "dimension": int(feature.shape[0]),
        "model_revision": loaded.resolved_revision,
        "hidden_state_count": int(hidden_count),
        "prompt_mode": prompt_mode,
        "index_is_inside_generated_span": True,
        "aggregation": "final generated piece of the supplied completion",
    }
    case = assemble_case(
        case_id=str(PARIS_REGRESSION["case_id"]),
        category=str(PARIS_REGRESSION["category"]),
        question=question,
        answer=answer,
        source="supplied_completion",
        label=label,
        configured=configured,
        evidence_detail=detail,
        evidence_source="live_suite_rubric",
        probe_read=read,
        hidden=hidden,
        include_in_aggregate=False,
    )
    case["regression"] = True
    return case


def _dataset_cases(
    backend: HuggingFaceBackend,
    loaded: Any,
    probe: LoadedProbe,
    secondary: LoadedProbe | None,
    settings: GenerationSettings,
    cases: list[dict[str, object]],
    *,
    max_questions: int,
) -> list[str]:
    notes: list[str] = []
    limit = max(0, max_questions)
    if limit == 0:
        return ["Dataset questions were not requested."]
    halu_path = halueval_snapshot_path()
    if halu_path.is_file():
        prepared = prepare_pairs(read_pinned_snapshot(halu_path), seed=0)
        chosen = prepared.assignment.test[:limit]
        by_id = prepared.pairs
        for question_id in chosen:
            pair = by_id[question_id]
            result = backend.generate(pair.question, settings)
            decision = label_halueval_generation(
                question=pair.question,
                knowledge=pair.knowledge,
                right_answer=pair.right_answer,
                hallucinated_answer=pair.hallucinated_answer,
                answer=result.answer,
            )
            case = _case_from_generation(
                case_id=pair.question_id,
                category="halueval_generation",
                question=pair.question,
                answer=result.answer,
                spec=None,
                result=result,
                probe=probe,
                evidence_label=decision.label,
                configured=True,
                evidence_detail=decision.detail,
                evidence_source=decision.label_source,
                include_in_aggregate=decision.label in {Label.FAITHFUL, Label.HALLUCINATED},
            )
            case["secondary_probe"] = _secondary_score(secondary, result, loaded)
            cases.append(case)
            print(format_case(case), flush=True)
        notes.append(f"HaluEval fresh generations: {len(chosen)} from the seed-0 test split.")
    else:
        notes.append(f"HaluEval snapshot is not at {halu_path}. Those questions were skipped.")
    truth_path = truthfulqa_snapshot_path()
    if truth_path.is_file():
        items = select_items(read_truthfulqa(truth_path), max_questions=limit, seed=0)
        for item in items:
            result = backend.generate(item.question, settings)
            label, detail, _record = label_generated_answer_coverage(result.answer, item)
            case = _case_from_generation(
                case_id=item.question_id,
                category="truthfulqa_generation",
                question=item.question,
                answer=result.answer,
                spec=None,
                result=result,
                probe=probe,
                evidence_label=label,
                configured=True,
                evidence_detail=detail,
                evidence_source="truthfulqa_generation_coverage_v2",
                include_in_aggregate=label in {Label.FAITHFUL, Label.HALLUCINATED},
            )
            case["secondary_probe"] = _secondary_score(secondary, result, loaded)
            cases.append(case)
            print(format_case(case), flush=True)
        notes.append(f"TruthfulQA fresh generations: {len(items)} from seed 0.")
    else:
        notes.append(f"TruthfulQA snapshot is not at {truth_path}. Those questions were skipped.")
    return notes


def _secondary_score(
    probe: LoadedProbe | None, result: Any, loaded: Any
) -> dict[str, object] | None:
    del loaded
    if probe is None or result.feature is None or result.feature_layer is None:
        return None
    if result.hidden_state_count is None:
        return None
    read = score_vector(
        probe,
        result.feature,
        model_name=result.model_name,
        model_revision=result.model_revision,
        layer_index=int(result.feature_layer),
        hidden_state_count=int(result.hidden_state_count),
        position=str(result.position),
        prompt_mode=str(result.prompt_mode),
        dtype=result.dtype,
    )
    return {
        "directory": read.directory,
        "protocol": read.protocol,
        "applied": read.applied,
        "score": read.score,
        "prediction": read.prediction,
        "threshold": read.threshold,
        "note": read.note,
    }


def _hidden_record(result: Any) -> dict[str, object]:
    described = describe_generated_position(int(result.prompt_tokens), int(result.generated_tokens))
    dimension = None if result.feature is None else int(result.feature.shape[0])
    layer = None if result.feature_layer is None else int(result.feature_layer)
    count = None if result.hidden_state_count is None else int(result.hidden_state_count)
    return {
        "layer": layer,
        "position": result.position,
        "dimension": dimension,
        "model_revision": result.model_revision,
        "hidden_state_count": count,
        "prompt_mode": result.prompt_mode,
        "model_dtype": result.dtype,
        "index_is_inside_generated_span": described["index_is_inside_generated_span"],
        "aggregation": described["aggregation"],
        "prompt_piece_count": described["prompt_piece_count"],
        "generated_piece_count": described["generated_piece_count"],
    }


def _match_rubric(question: str) -> dict[str, object] | None:
    from glassbox.verification import normalize_question

    key = normalize_question(question)
    for spec in SUITE_RUBRIC:
        if normalize_question(str(spec["question"])) == key:
            return spec
    return None


def _length_auroc(binary: list[dict[str, object]]) -> dict[str, object]:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(
        [1 if case["factuality_status"] == "HALLUCINATED" else 0 for case in binary], dtype=int
    )
    lengths = np.asarray([len(str(case.get("answer") or "")) for case in binary], dtype=float)
    if len(set(labels.tolist())) < 2:
        return {"computed": False, "reason": "Length AUROC needs both classes."}
    return {
        "computed": True,
        "auroc": float(roc_auc_score(labels, lengths)),
        "fair_baseline": True,
        "note": "Score is the character length. No threshold was fit on this suite.",
    }


def _error_examples(cases: list[dict[str, object]]) -> dict[str, object]:
    false_positives = []
    false_negatives = []
    for case in cases:
        status = case.get("factuality_status")
        prediction = case.get("probe_prediction")
        if status == "FAITHFUL" and prediction == "hallucinated":
            false_positives.append(_brief(case))
        if status == "HALLUCINATED" and prediction == "faithful":
            false_negatives.append(_brief(case))
    return {"false_positives": false_positives, "false_negatives": false_negatives}


def _brief(case: dict[str, object]) -> dict[str, object]:
    return {
        "case_id": case.get("case_id"),
        "source": case.get("source"),
        "question": case.get("question"),
        "answer": case.get("answer"),
        "probe_score": case.get("probe_score"),
        "reference_check": case.get("reference_check"),
        "evidence_detail": case.get("evidence_detail"),
        "regression": case.get("regression") is True,
    }


def _agrees(status: str, prediction: str | None) -> bool | None:
    if prediction not in {"faithful", "hallucinated"}:
        return None
    if status == "FAITHFUL":
        return prediction == "faithful"
    if status == "HALLUCINATED":
        return prediction == "hallucinated"
    return None


def _conclusions(
    cases: list[dict[str, object]], summary: dict[str, object], probe: LoadedProbe
) -> dict[str, object]:
    generated = [case for case in cases if case.get("source") == "generated"]
    states_ok = all(
        isinstance(case.get("hidden_state"), dict)
        and case["hidden_state"].get("index_is_inside_generated_span") is True
        and case["hidden_state"].get("position") == "last_generated_token"
        for case in generated
        if case.get("probe_applied") is True
    )
    applied = [case for case in generated if case.get("probe_applied") is True]
    labeled = [
        case
        for case in applied
        if case.get("include_in_aggregate") is True
        and case.get("probe_agrees_with_label") is not None
    ]
    agreed = sum(1 for case in labeled if case.get("probe_agrees_with_label") is True)
    return {
        "qwen_generated_answers": len(generated) > 0
        and all(str(case.get("answer")) for case in generated),
        "hidden_states_from_generated_answers": bool(applied) and states_ok,
        "probe_ran_on_compatible_representations": bool(applied),
        "labeled_rows": len(labeled),
        "labeled_agreements": agreed,
        "predictions_agree_with_labels": False,
        "outperformed_text_baselines": False,
        "generalized_beyond_training": False,
        "calibration_valid": probe.calibrated,
        "early_detection": False,
        "pair_task_probe": pair_task_probe(probe),
        "note": (
            "Agreement is counted on labeled generations and is not a detector result "
            "while aggregate performance is inconclusive. A pair-task probe is not a "
            "validated detector of Qwen's own answers. Early detection was not measured."
        ),
    }


def _markdown(summary: dict[str, object], cases: list[dict[str, object]]) -> str:
    lines = [
        "# Live terminal test",
        "",
        str(summary.get("reason")),
        "",
        f"Model `{summary.get('model_name')}` revision `{summary.get('model_revision')}`.",
        f"Probe `{summary.get('probe_directory')}` protocol `{summary.get('probe_protocol')}`.",
        f"Calibration validated: {summary.get('calibration_validated')}.",
        "",
        "Class counts count every case, including unvalidated and the supplied regression.",
        "",
        "```",
        str(summary.get("class_counts")),
        "```",
        "",
    ]
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        lines.append("Point metrics on labeled generations only. They are inconclusive.")
        lines.append("")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.append("## Cases")
    lines.append("")
    for case in cases:
        lines.append(
            f"- {case.get('case_id')}: {case.get('factuality_status')} / "
            f"probe {case.get('probe_prediction')} / score {case.get('probe_score')}"
        )
    lines.append("")
    return "\n".join(lines)
