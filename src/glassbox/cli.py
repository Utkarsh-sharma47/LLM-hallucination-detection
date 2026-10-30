"""Interactive terminal demo.

The model is loaded on the first question and reused. A missing probe is
reported as ``PROBE NOT TRAINED`` rather than replaced with a heuristic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from rich.console import Console

from glassbox.config import repo_root
from glassbox.demo_config import default_probe_dir, load_demo_config
from glassbox.glassbox_report import build_report
from glassbox.models.backend import HuggingFaceBackend
from glassbox.models.resources import ModelMemoryError
from glassbox.present import (
    DEMO_QUESTIONS,
    LiveTokenView,
    make_console,
    render_banner,
    render_epilogue,
    render_external,
    render_samples,
)
from glassbox.probes.logistic import PROBE_NOT_TRAINED
from glassbox.probes.store import load_probe
from glassbox.reporting import (
    format_diagnostics,
    format_export_message,
    format_probe_view,
    format_token_view,
    format_verification_view,
)
from glassbox.session import DemoSession
from glassbox.verification import (
    assess_answer,
    find_fact,
    load_curated_facts,
    reference_for_question,
)


def main(argv: list[str] | None = None) -> int:
    """Launch the terminal demo."""
    parser = argparse.ArgumentParser(prog="glassbox")
    parser.add_argument(
        "--config", type=Path, default=None, help="Demo YAML. Defaults to configs/demo.yaml."
    )
    parser.add_argument("--question", default=None, help="Ask one question without the menu.")
    parser.add_argument("--once", action="store_true", help="Print one answer and exit.")
    parser.add_argument(
        "--offline", action="store_true", help="Use cached weights and do not download."
    )
    parser.add_argument(
        "--model", default=None, help="Override the model id, for example sshleifer/tiny-gpt2."
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=None,
        help="Probe directory. Omit to use the default if it exists.",
    )
    parser.add_argument("--references", type=Path, default=None, help="Curated reference YAML.")
    parser.add_argument(
        "--demo", action="store_true", help="Run the three built-in sample questions and exit."
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="List the sample questions and exit. Does not load a model.",
    )
    parser.add_argument(
        "--analyze-text",
        action="store_true",
        dest="analyze_text",
        help="Check a supplied answer. Does not load a model or probe hidden states.",
    )
    parser.add_argument("--answer", default=None, help="Answer text for --analyze-text.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="Override the generation length."
    )
    parser.add_argument(
        "--distribution-export",
        choices=("topk", "full"),
        default="topk",
        help="topk stores the shown alternatives. full also keeps the vocabulary for export.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Export the analysis report as JSON to the specified path.",
    )
    commands = parser.add_subparsers(dest="command")
    train = commands.add_parser(
        "train-probe",
        help="Train and evaluate the HaluEval pair probe on the demo model.",
    )
    train.add_argument("--max-questions", type=int, default=240)
    train.add_argument("--download-data", action="store_true")
    train.add_argument("--snapshot", type=Path, default=None)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--run-id", default="halueval-pair-001")
    train.add_argument("--output", type=Path, default=None)
    external = commands.add_parser(
        "eval-truthfulqa",
        help="Generate TruthfulQA answers with the saved probe. Does not train.",
    )
    external.add_argument("--max-questions", type=int, default=40)
    external.add_argument("--snapshot", type=Path, default=None)
    external.add_argument("--probe", type=Path, default=None)
    external.add_argument("--seed", type=int, default=0)
    external.add_argument("--run-id", default="truthfulqa-generation-001")
    baselines = commands.add_parser(
        "text-baselines",
        help="Score length, overlap, and answer-text classifiers. Does not load Qwen.",
    )
    baselines.add_argument("--snapshot", type=Path, default=None)
    baselines.add_argument("--max-questions", type=int, default=200)
    baselines.add_argument("--seed", type=int, default=0)
    baselines.add_argument("--run-id", default="halueval-text-baselines-001")
    audit = commands.add_parser(
        "audit-generations",
        help="Compare saved generation labels with the current reference labeler. Does not load Qwen.",
    )
    audit.add_argument("--report", type=Path, default=None)
    audit.add_argument("--snapshot", type=Path, default=None)
    audit.add_argument("--run-id", default="truthfulqa-label-audit-001")
    freeze = commands.add_parser(
        "freeze-generation-split",
        help="Write TruthfulQA question ids before generation. Does not load Qwen or train.",
    )
    freeze.add_argument("--max-questions", type=int, default=40)
    freeze.add_argument("--seed", type=int, default=0)
    freeze.add_argument("--snapshot", type=Path, default=None)
    freeze.add_argument("--run-id", default="truthfulqa-generation-split")
    freeze.add_argument("--output", type=Path, default=None)
    freeze.add_argument("--artifacts-root", type=Path, default=None)
    dataset = commands.add_parser(
        "build-generation-dataset",
        help="Label supplied TruthfulQA answers. Does not load Qwen or train a probe.",
    )
    dataset.add_argument("--split", type=Path, required=True)
    dataset.add_argument("--answers", type=Path, required=True)
    dataset.add_argument("--model-name", required=True)
    dataset.add_argument("--model-revision", required=True)
    dataset.add_argument("--dtype", required=True)
    dataset.add_argument("--snapshot", type=Path, default=None)
    dataset.add_argument("--run-id", default="truthfulqa-generation-dataset")
    dataset.add_argument("--output", type=Path, default=None)
    dataset.add_argument("--artifacts-root", type=Path, default=None)
    evaluation = commands.add_parser(
        "write-generation-evaluation",
        help="Write a generation-evaluation record. Does not load Qwen or train a probe.",
    )
    evaluation.add_argument("--payload", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--artifacts-root", type=Path, required=True)
    transfer = commands.add_parser(
        "eval-saved-probe",
        help="Score cached Qwen generations with the saved HaluEval pair probe. Does not train.",
    )
    transfer.add_argument("--max-questions", type=int, default=None)
    transfer.add_argument("--seed", type=int, default=0)
    transfer.add_argument("--run-id", default=None)
    transfer.add_argument("--snapshot", type=Path, default=None)
    transfer.add_argument("--halueval", type=Path, default=None)
    transfer.add_argument("--probe", type=Path, default=None)
    transfer.add_argument("--artifacts-root", type=Path, default=None)
    generate = commands.add_parser(
        "train-generation-probe",
        help="Train on Qwen's own answers. Does not replace the saved pair probe.",
    )
    generate.add_argument("--max-questions", type=int, default=240)
    generate.add_argument("--external-questions", type=int, default=40)
    generate.add_argument("--snapshot", type=Path, default=None)
    generate.add_argument("--external-snapshot", type=Path, default=None)
    generate.add_argument("--reviews", type=Path, default=None)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--run-id", default="halueval-generation-001")
    generate.add_argument("--output", type=Path, default=None)
    generate.add_argument("--pair-probe", type=Path, default=None)
    generate.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Arm a model download. Omit this to read the local cache only.",
    )
    live = commands.add_parser(
        "live-test",
        help=(
            "Score Qwen's answers with the saved probe. Cache only. "
            "Does not train or replace a probe."
        ),
    )
    live.add_argument("--suite", action="store_true", help="Run the fixed question suite and exit.")
    live.add_argument("--run-id", default="live-terminal-001")
    live.add_argument("--trace-tokens", action="store_true")
    live.add_argument("--dataset-questions", type=int, default=2)
    live.add_argument(
        "--probe",
        type=Path,
        default=None,
        dest="live_probe",
        help="Probe directory. Omit to use the live pair probe when it exists.",
    )
    live.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Arm a model download. Omit this to read the local cache only.",
    )
    review = commands.add_parser(
        "review-generations",
        help="Display the generation review queue and record human reviews. Does not load Qwen.",
    )
    review.add_argument("--queue", type=Path, default=None)
    review.add_argument("--history", type=Path, default=None)
    review.add_argument("--show", type=int, default=0)
    review.add_argument("--append", type=Path, default=None)
    review.add_argument("--export-snapshot", type=Path, default=None)
    review.add_argument("--run-id", default=None)
    length = commands.add_parser(
        "eval-length-shortcut",
        help="Diagnose length and format effects. Does not rewrite the pair probe.",
    )
    length.add_argument("--predictions", type=Path, default=None)
    length.add_argument("--controlled", action="store_true")
    length.add_argument("--scores", type=Path, default=None)
    length.add_argument("--dry-run", action="store_true")
    length.add_argument("--probe", type=Path, default=None)
    length.add_argument("--references", type=Path, default=None)
    length.add_argument("--threshold", type=float, default=0.65)
    length.add_argument("--run-id", default="length-shortcut-001")
    expand = commands.add_parser(
        "plan-generation-expansion",
        help="Plan a larger generation split without rewriting generation-001.",
    )
    expand.add_argument("--max-questions", type=int, default=2400)
    expand.add_argument("--seed", type=int, default=0)
    expand.add_argument("--snapshot", type=Path, default=None)
    expand.add_argument("--predictions", type=Path, default=None)
    expand.add_argument("--run-id", default="halueval-generation-expansion-001")
    variants = commands.add_parser(
        "compare-generation-variants",
        help="Compare last-token, length, and TF-IDF on a frozen generation split.",
    )
    variants.add_argument("--predictions", type=Path, required=True)
    variants.add_argument("--features", type=Path, default=None)
    variants.add_argument("--seed", type=int, default=0)
    variants.add_argument("--threshold", type=float, default=None)
    variants.add_argument("--pair-probe", type=Path, default=None)
    variants.add_argument("--run-id", default="generation-variants-001")
    early = commands.add_parser(
        "eval-early-warning",
        help="Score earlier generated pieces. Does not publish a lead time or load Qwen.",
    )
    early.add_argument("--scores", type=Path, required=True)
    early.add_argument("--run-id", default="early-warning-001")
    layers = commands.add_parser(
        "compare-generation-layers",
        help="Compare pre-registered generation representations. Does not replace the pair probe.",
    )
    layers.add_argument("--predictions", type=Path, required=True)
    layers.add_argument("--features", type=Path, default=None)
    layers.add_argument("--seed", type=int, default=0)
    layers.add_argument("--pair-probe", type=Path, default=None)
    layers.add_argument("--run-id", default="generation-layers-001")
    v2 = commands.add_parser(
        "build-generation-v2",
        help="Assemble generation dataset v2 without rewriting generation-001.",
    )
    v2.add_argument("--max-questions", type=int, default=2400)
    v2.add_argument("--seed", type=int, default=0)
    v2.add_argument("--snapshot", type=Path, default=None)
    v2.add_argument("--predictions", type=Path, default=None)
    v2.add_argument("--generated", type=Path, default=None)
    v2.add_argument("--features", type=Path, default=None)
    v2.add_argument("--threshold", type=float, default=None)
    v2.add_argument("--pair-probe", type=Path, default=None)
    v2.add_argument("--run-id", default="halueval-generation-002")
    view = commands.add_parser(
        "view-session",
        help="Render a session JSON export as HTML. Does not load Qwen or claim calibrated risk.",
    )
    view.add_argument("--input", type=Path, required=True)
    view.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "train-probe":
        from glassbox.probes.training import train_command

        return train_command(args)
    if args.command == "eval-truthfulqa":
        from glassbox.probes.truthfulqa_eval import evaluate_command

        return evaluate_command(args)
    if args.command == "text-baselines":
        from glassbox.probes.text_baseline import text_baseline_command

        return text_baseline_command(args)
    if args.command == "audit-generations":
        from glassbox.probes.generation_audit import audit_command

        return audit_command(args)
    if args.command == "freeze-generation-split":
        from glassbox.data.generation_dataset import freeze_command

        return freeze_command(args)
    if args.command == "build-generation-dataset":
        from glassbox.data.generation_dataset import build_command

        return build_command(args)
    if args.command == "write-generation-evaluation":
        from glassbox.evaluation.generation_record import write_command

        return write_command(args)
    if args.command == "eval-saved-probe":
        from glassbox.evaluation.probe_transfer import transfer_command

        return transfer_command(args)
    if args.command == "train-generation-probe":
        from glassbox.probes.generation_train import train_generation_command

        return train_generation_command(args)
    if args.command == "live-test":
        from glassbox.probes.live_test import live_test_command

        return live_test_command(args)
    if args.command == "review-generations":
        from glassbox.data.review_cli import review_command

        return review_command(args)
    if args.command == "eval-length-shortcut":
        from glassbox.evaluation.length_eval import length_eval_command

        return length_eval_command(args)
    if args.command == "plan-generation-expansion":
        from glassbox.data.generation_expand import expand_command

        return expand_command(args)
    if args.command == "compare-generation-variants":
        from glassbox.probes.generation_variants import variants_command

        return variants_command(args)
    if args.command == "eval-early-warning":
        from glassbox.evaluation.early_warning import early_warning_command

        return early_warning_command(args)
    if args.command == "compare-generation-layers":
        from glassbox.probes.layer_compare import layer_compare_command

        return layer_compare_command(args)
    if args.command == "build-generation-v2":
        from glassbox.data.generation_v2 import v2_command

        return v2_command(args)
    if args.command == "view-session":
        from glassbox.reports.session_view import view_command

        return view_command(args)
    console = make_console()
    if args.samples:
        render_samples(console)
        return 0
    if args.analyze_text and args.demo:
        print("Choose either --demo or --analyze-text.", file=sys.stderr)
        return 2
    try:
        config = load_demo_config(args.config, offline=args.offline, model_name=args.model)
        references_path = args.references or config.references_path
        facts = load_curated_facts(references_path)
    except (OSError, ValueError, FileNotFoundError) as exc:
        print(f"Glass Box could not start: {exc}", file=sys.stderr)
        return 1
    if args.analyze_text:
        if not args.question or args.answer is None:
            print("Pass --question and --answer with --analyze-text.", file=sys.stderr)
            return 2
        return _analyze_text(
            console, args.question, args.answer, facts, args.probe or config.probe_dir
        )
    try:
        probe = _load_optional_probe(args.probe if args.probe is not None else config.probe_dir)
    except (OSError, ValueError, FileNotFoundError) as exc:
        print(f"Glass Box could not start: {exc}", file=sys.stderr)
        return 1
    generation = config.generation
    generation = replace(generation, distribution_export=args.distribution_export)
    if args.max_new_tokens is not None:
        generation = replace(generation, max_new_tokens=args.max_new_tokens)
    backend = HuggingFaceBackend(config.spec)
    session = DemoSession(backend=backend, settings=generation, facts=facts, probe=probe)
    if probe is None:
        detector = PROBE_NOT_TRAINED
        detail = f"{config.selection_reason} No saved probe was loaded. No prediction was invented."
    elif probe.calibrated:
        detector = "calibrated probe loaded"
        detail = config.selection_reason
    else:
        detector = "experimental probe loaded"
        detail = (
            f"{config.selection_reason} The loaded score is an experimental pair-task score. "
            "RISK NOT ASSESSED."
        )
    render_banner(
        console,
        model_name=config.spec.name,
        backend_label="Hugging Face causal LM",
        detector_status=detector,
        detail=detail,
    )
    if args.once and not args.question and not args.demo:
        print("Pass --question with --once.", file=sys.stderr)
        return 2
    if args.demo:
        return _run_demo(session, console, detector, args.output_json)
    if args.question:
        code = _run_question(session, args.question, console, args.output_json)
        if code != 0 or args.once:
            return code
        if not _menu(session):
            return 0
    return _interactive(session, console)


def app() -> None:
    """Console-script entry. Preserves the integer exit code."""
    raise SystemExit(main())


def _run_demo(
    session: DemoSession, console: Console, detector: str, output_json: Path | None = None
) -> int:
    collected = []
    for question in DEMO_QUESTIONS:
        code = _run_question(session, question, console, output_json)
        if code != 0:
            return code
        collected.append(session.turns[-1])
    render_epilogue(console, collected, detector)
    return 0


def _analyze_text(
    console: Console, question: str, answer: str, facts, probe_dir: Path | None
) -> int:
    fact = find_fact(question, facts)
    reference = (
        assess_answer(fact, answer) if fact is not None else reference_for_question(question, facts)
    )
    folder = default_probe_dir() if probe_dir is None else probe_dir
    detail = "Text analysis does not load model weights."
    if (folder / "probe.joblib").is_file():
        detail += " A saved probe is on disk and was not applied to this text."
    render_banner(
        console,
        model_name="not loaded",
        backend_label="supplied text",
        detector_status=PROBE_NOT_TRAINED,
        detail=detail,
    )
    render_external(
        console,
        question=question,
        answer=answer,
        reference_status=reference.status,
        reference_detail=reference.detail,
        probe_status=PROBE_NOT_TRAINED,
    )
    return 0


def _run_question(
    session: DemoSession, question: str, console: Console, output_json: Path | None = None
) -> int:
    view = LiveTokenView(console, session.settings.max_new_tokens) if console.is_terminal else None
    try:
        _ensure_model(session, console)
        if view is None:
            turn = session.ask(question)
        else:
            with view:
                turn = session.ask(question, on_token=view.update)
    except (ModelMemoryError, OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1

    # Use new Glass Box report format
    catalog_path = repo_root() / "configs" / "claim_references.yaml"
    report = build_report(
        question=question,
        generation=turn.generation,
        turn=turn,
        probe=session.probe,
        catalog_path=catalog_path,
    )

    console.print(report.to_text())

    # Export JSON if requested
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(report.to_json(), f, indent=2, default=str)
        console.print(f"\n[dim]JSON report saved to: {output_json}[/dim]")

    return 0


COMMAND_LEGEND = (
    "n = ask another question\n"
    "r = show full report (10-section Glass Box analysis)\n"
    "t = token probability analysis — generated output-token probabilities and alternatives\n"
    "p = probe analysis — hidden-state probe score, separate from token probabilities\n"
    "v = reference verification — external curated check, separate from the probe\n"
    "d = diagnostics — model, prompt length, latency, and hidden-state shape\n"
    "e = export report — write JSON, Markdown, and the distribution table\n"
    "q = quit\n"
    "A blank line quits.\n"
)


def _interactive(session: DemoSession, console: Console) -> int:
    """Ask questions until a blank line or quit. Previous turns stay in the session."""
    console.print(
        "[dim]Ask any question. It does not need a curated reference. A blank line quits.[/dim]"
    )
    while True:
        try:
            question = input("\nQuestion (blank to quit): ").strip()
        except EOFError:
            print()
            return 0
        if not question:
            return 0
        code = _run_question(session, question, console)
        if code != 0:
            return code
        if not _menu(session):
            return 0


def _menu(session: DemoSession) -> bool:
    """Handle inspection commands. Return True when the user wants another question."""
    print(COMMAND_LEGEND, end="")
    while True:
        try:
            command = input("glassbox [n/t/p/v/d/e/q]> ").strip().lower()
        except EOFError:
            print()
            return False
        if command in {"n", "next"}:
            return True
        if command in {"", "q", "quit"}:
            return False
        rendered = render_command(session, command, root=repo_root() / "artifacts")
        if rendered is None:
            print(COMMAND_LEGEND, end="")
            continue
        print(rendered, end="" if rendered.endswith("\n") else "\n")


def render_command(session: DemoSession, command: str, *, root: Path) -> str | None:
    """Return the text for an inspection command, or None when it is unknown."""
    if command not in {"r", "t", "p", "v", "d", "e"}:
        return None
    if not session.turns:
        return "No turn yet.\n"
    turn = session.turns[-1]
    if command == "r":
        catalog_path = repo_root() / "configs" / "claim_references.yaml"
        report = build_report(
            question=turn.generation.question,
            generation=turn.generation,
            turn=turn,
            probe=session.probe,
            catalog_path=catalog_path,
        )
        return report.to_text()
    if command == "t":
        return format_token_view(turn)
    if command == "p":
        return format_probe_view(turn)
    if command == "v":
        return format_verification_view(turn)
    if command == "d":
        return format_diagnostics(turn)
    path = session.export(root)
    extras = sorted(item for item in path.parent.glob("distributions-*") if item.is_file())
    return format_export_message(path, path.with_suffix(".md"), extras)


def _ensure_model(session: DemoSession, console: Console) -> None:
    backend = session.backend
    ensure = getattr(backend, "ensure_loaded", None)
    load_count = getattr(backend, "load_count", None)
    if callable(ensure) and load_count == 0:
        console.print("[dim]Loading model once for this session...[/dim]")
        ensure()
        console.print(
            f"[dim]Loaded {getattr(backend, 'model_name', 'model')} on {getattr(backend, 'device', 'unknown')}.[/dim]"
        )


def _load_optional_probe(path: Path | None):
    """Load a saved probe, or return None when the default directory is absent.

    An explicit path, or a default directory that exists but is incomplete,
    fails before any score is invented.
    """
    explicit = path is not None
    folder = default_probe_dir() if path is None else Path(path)
    if not folder.exists():
        if explicit:
            raise FileNotFoundError(f"Probe directory does not exist: {folder}")
        return None
    complete = (folder / "probe.joblib").is_file() and (folder / "metadata.json").is_file()
    if not complete:
        raise FileNotFoundError(
            f"Probe directory is incomplete: {folder}. No prediction was invented."
        )
    return load_probe(folder)
