"""Terminal report text for one demo turn."""

from __future__ import annotations

from dataclasses import dataclass

from glassbox.models.backend import GenerationResult
from glassbox.probes.logistic import PROBE_NOT_TRAINED
from glassbox.verification import ReferenceResult

INTERVENTION_NOT_RUN = "Intervention: not run"


@dataclass
class TurnRecord:
    """One question, its model trace, and the separate checks around it."""

    generation: GenerationResult
    probe_status: str
    experimental_score: float | None
    score_note: str
    factual_risk_score: float | None
    reference: ReferenceResult
    intervention_status: str = "not run"
    probe_prediction: str | None = None
    probe_threshold: float | None = None
    calibration_status: str = "not assessed"
    probe_task: str = ""


def format_turn_report(turn: TurnRecord) -> str:
    """Render the presentation view for one answer."""
    generation = turn.generation
    answer = generation.answer or "(empty generation)"
    lines = [
        "",
        "Question",
        generation.question,
        "",
        "Answer",
        answer,
        "",
        "Generation",
        f"Model: {generation.model_name}",
        f"Device: {generation.device}",
        f"Dtype: {generation.dtype}",
        f"Latency: {generation.latency_seconds:.2f} s",
        f"Tokens: {generation.generated_tokens} generated, {generation.prompt_tokens} prompt",
        "",
        "Generated-output token probabilities",
        "These are decoding probabilities of tokens the model wrote.",
        "They are not input-token probabilities and not a hallucination score.",
        "Step 1 is conditioned on the prompt. Later steps also use earlier output.",
    ]
    lines.extend(_step_lines(generation))
    lines.extend(
        [
            "",
            "Internal probe",
            f"Hidden states: {'available' if generation.hidden_states_available else 'unavailable'}",
            f"Logits: {'available' if generation.logits_available else 'unavailable'}",
            f"Detector status: {turn.probe_status}",
        ]
    )
    if turn.experimental_score is not None:
        lines.append(f"Experimental probe score: {turn.experimental_score:.3f}")
        lines.append(turn.score_note)
    elif turn.probe_status == PROBE_NOT_TRAINED:
        lines.append(turn.score_note)
    if turn.factual_risk_score is not None:
        lines.append(f"Factual-risk score: {turn.factual_risk_score:.3f}")
    lines.extend(
        [
            "",
            "Reference check",
            turn.reference.detail,
            "",
            "Intervention",
            INTERVENTION_NOT_RUN
            if turn.intervention_status == "not run"
            else turn.intervention_status,
            "",
        ]
    )
    return "\n".join(lines)


def format_token_view(turn: TurnRecord) -> str:
    """Show generated-output probabilities. This is not the probe or the reference check."""
    generation = turn.generation
    lines = [
        "Token probability analysis",
        "Generated-output next-token probabilities and alternatives.",
        "This is not the hidden-state probe and not the reference check.",
        f"Prompt tokens: {generation.prompt_tokens} (input, not scored here).",
        f"Decoding: {generation.decoding}",
        *_step_lines(generation),
        "",
    ]
    return "\n".join(lines)


def format_probe_view(turn: TurnRecord) -> str:
    """Show the hidden-state probe. This is not the token-probability table."""
    generation = turn.generation
    lines = [
        "Probe analysis",
        "Hidden-state probe for the supplied or generated answer.",
        "This is not a token probability and not the reference check.",
        "Probe",
        f"Detector status: {turn.probe_status}",
        turn.probe_task or "Task: not recorded",
        f"Hidden states: {'available' if generation.hidden_states_available else 'unavailable'}",
        f"Layer: {generation.feature_layer}",
        f"Position: {generation.position}",
        "Feature dimension: "
        + ("unavailable" if generation.feature is None else str(int(generation.feature.shape[0]))),
        turn.score_note,
    ]
    if turn.experimental_score is not None:
        lines.append("EXPERIMENTAL PROBE SCORE")
        lines.append(f"Experimental probe score: {turn.experimental_score:.3f}")
        lines.append("RISK NOT ASSESSED")
    if turn.probe_prediction is not None:
        cutoff = "unrecorded" if turn.probe_threshold is None else f"{turn.probe_threshold:.2f}"
        lines.append(f"Predicted class: {turn.probe_prediction} at threshold {cutoff}")
    lines.append(f"Calibration: {turn.calibration_status}")
    if turn.factual_risk_score is not None:
        lines.append(f"Factual-risk score: {turn.factual_risk_score:.3f}")
    lines.append("")
    return "\n".join(lines)


def format_session_markdown(turns: list[TurnRecord]) -> str:
    """Readable report for every turn in the current session."""
    lines = [
        "# Glass Box session",
        "",
        "Chosen values are P(written output token | the decoding context).",
        "Step 1 is conditioned on the prompt. Later steps also use earlier output.",
        "They are not input-token probabilities and not hallucination risk.",
        "",
    ]
    for index, turn in enumerate(turns, start=1):
        generation = turn.generation
        if turn.factual_risk_score is not None:
            probe = "calibrated probe"
            risk = f"{turn.factual_risk_score:.3f}"
        elif turn.experimental_score is not None:
            probe = "EXPERIMENTAL PROBE SCORE"
            risk = "RISK NOT ASSESSED"
        else:
            probe = PROBE_NOT_TRAINED
            risk = "RISK NOT ASSESSED"
        if turn.reference.status == "not_configured":
            reference = "NO REFERENCE AVAILABLE"
        else:
            reference = turn.reference.status
        hidden = "unavailable"
        if generation.hidden_states_available and generation.feature is not None:
            hidden = (
                f"{generation.position}, layer {generation.feature_layer}, "
                f"dim {int(generation.feature.shape[0])}"
            )
        lines.extend(
            [
                f"## {index}. {_md_cell(generation.question)}",
                "",
                _md_cell(generation.answer or "(empty generation)"),
                "",
                f"- Model: {generation.model_name}",
                f"- Revision: {generation.model_revision}",
                f"- Device: {generation.device} ({generation.dtype})",
                f"- Calibration: {turn.calibration_status}",
                f"- Latency: {generation.latency_seconds:.2f} s",
                f"- Probe: {probe}",
                f"- Risk: {risk}",
                f"- Reference: {reference}",
                f"- Hidden states: {hidden}",
                f"- Reference detail: {_md_cell(turn.reference.detail)}",
                "",
                "| # | Token | Chosen | Alternatives | Probe |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        if not generation.tokens:
            lines.append("|  | (none) |  |  |  |")
        for step, token in enumerate(generation.tokens, start=1):
            alternatives = ", ".join(
                f"{_md_cell(item.text)} {item.probability:.3f}" for item in token.top_alternatives
            )
            row = (
                f"| {step} | {_md_cell(token.text)} | {token.probability:.3f} "
                f"| {alternatives} | {probe} |"
            )
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


def format_export_message(json_path, markdown_path, extra_paths=()) -> str:
    """Tell the operator where the reports are and how to open them."""
    lines = [
        "Saved this session.",
        f"JSON report: {json_path}",
        f"Markdown report: {markdown_path}",
        f'Open the Markdown report with: less "{markdown_path}"',
    ]
    for path in extra_paths:
        lines.append(f"Distribution table: {path}")
    if extra_paths:
        lines.append(
            "piece_id in those tables is the vocabulary id. "
            "The terminal does not print the full vocabulary."
        )
    return "\n".join(lines) + "\n"


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def format_verification_view(turn: TurnRecord) -> str:
    """Show the curated reference check on its own."""
    if turn.reference.status == "not_configured":
        status = "NO REFERENCE AVAILABLE"
    else:
        status = turn.reference.status
    return (
        "Reference verification\n"
        "External curated check. This is not the probe and not a token probability.\n"
        f"{status}\n{turn.reference.detail}\n"
    )


def format_diagnostics(turn: TurnRecord) -> str:
    """Show generation bookkeeping for the current turn."""
    generation = turn.generation
    lines = [
        "Diagnostics",
        "Model, prompt length, latency, and hidden-state shape. Not a score.",
        f"Model revision: {generation.model_revision}",
        f"Prompt mode: {generation.prompt_mode}",
        f"Prompt tokens: {generation.prompt_tokens}",
        f"Generated tokens: {generation.generated_tokens}",
        f"Latency seconds: {generation.latency_seconds:.4f}",
        f"Hidden-state count: {generation.hidden_state_count}",
        f"Model loads this session: {generation.load_count}",
        f"Logits available: {generation.logits_available}",
        f"Hidden states available: {generation.hidden_states_available}",
        f"Note: {generation.note or 'none'}",
        "",
    ]
    return "\n".join(lines)


def _step_lines(generation: GenerationResult) -> list[str]:
    if not generation.tokens:
        return ["(no generated output tokens)"]
    lines: list[str] = []
    for index, token in enumerate(generation.tokens, start=1):
        step = token.step or index
        if token.conditioned_on == "prompt_plus_generated":
            context = "conditioned on the prompt plus earlier generated output"
        else:
            context = "conditioned on the prompt"
        others = [item for item in token.top_alternatives if item.token_id != token.token_id]
        alternatives = ", ".join(
            f"{alternative.text} {alternative.probability:.3f}" for alternative in others
        )
        lines.append(
            f"{step:>2}. {token.text!r}  p={token.probability:.3f}  {context}  "
            f"alternatives: {alternatives}"
        )
    return lines
