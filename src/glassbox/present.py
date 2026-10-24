"""Rich presentation for one Glass Box turn.

Token probabilities stay labeled as model preference. A hallucination
percentage is shown only when a calibrated probe score exists.
"""

from __future__ import annotations

import time
from io import StringIO

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from glassbox.models.backend import GeneratedToken, GenerationResult
from glassbox.probes.logistic import PROBE_NOT_TRAINED
from glassbox.reporting import TurnRecord

EXTERNAL_PROBE_NOTE = (
    "Hidden-state probing cannot be performed on this text alone. "
    "Claude Code's CLI returns generated text in text, json, or stream-json. "
    "It does not expose logits, token probabilities, or hidden states. "
    "No Claude-internal signal was read, and none was invented. "
    "The hallucination probe was not applied."
)

DEMO_QUESTIONS = (
    "What is the capital of Australia?",
    "Why is Sydney the capital of Australia?",
    "How many planets are in the solar system?",
)


def make_console() -> Console:
    return Console(highlight=False)


def format_percent(probability: float) -> str:
    if probability <= 0:
        return "0.0%"
    percent = probability * 100
    if percent < 0.05:
        return "<0.1%"
    return f"{percent:.1f}%"


def display_token(text: str, limit: int | None = 28) -> str:
    if text == "":
        shown = "∅"
    else:
        shown = text.replace("\n", "⏎").replace("\t", "⇥")
        if shown.strip() == "":
            shown = shown.replace(" ", "·")
        elif shown.startswith(" "):
            shown = "·" + shown[1:]
    if limit is not None and len(shown) > limit:
        return shown[: limit - 1] + "…"
    return shown


def next_best(token: GeneratedToken):
    for alternative in token.top_alternatives:
        if alternative.token_id != token.token_id:
            return alternative
    return None


def reference_label(status: str) -> str:
    """Show the evidence label. A missing reference stays unlabeled."""
    labels = {
        "faithful": "FAITHFUL",
        "hallucinated": "HALLUCINATED",
        "unvalidated": "UNVALIDATED",
        "ambiguous": "AMBIGUOUS",
        "not_configured": "NO REFERENCE AVAILABLE",
        "configured": "NO ANSWER CHECKED",
    }
    return labels.get(status, status)


def risk_level(turn: TurnRecord) -> str:
    score = turn.factual_risk_score
    if score is None:
        return "RISK NOT ASSESSED"
    if score < 0.33:
        band = "LOW"
    elif score < 0.66:
        band = "MODERATE"
    else:
        band = "HIGH"
    return f"{band} — calibrated probe estimate"


def render_banner(
    console: Console,
    *,
    model_name: str,
    backend_label: str,
    detector_status: str,
    detail: str = "",
) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", width=12)
    grid.add_column()
    grid.add_row("Model", Text(model_name))
    grid.add_row("Backend", Text(backend_label))
    style = "bold yellow" if detector_status == PROBE_NOT_TRAINED else "bold cyan"
    grid.add_row("Detector", Text(detector_status, style=style))
    grid.add_row("Claude", Text("text only, no logits or hidden states"))
    header = Text("GLASS BOX", style="bold")
    header.append("   ·   LLM HALLUCINATION DETECTION", style="cyan")
    body: list = [header, Text(""), grid]
    if detail:
        body.extend([Text(""), Text(detail, style="dim")])
    console.print(Panel(Group(*body), border_style="bright_cyan", padding=(1, 2)))


def render_samples(console: Console) -> None:
    table = Table(
        title="Sample questions", box=box.SIMPLE_HEAD, header_style="bold cyan", expand=True
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("Question")
    for index, question in enumerate(DEMO_QUESTIONS, start=1):
        table.add_row(str(index), question)
    console.print(table)
    console.print("[dim]The showcase runs these with: python -m glassbox --demo[/dim]")


def render_turn(console: Console, turn: TurnRecord) -> None:
    generation = turn.generation
    width = console.width or 80
    console.print(
        Panel(
            Text(generation.question), title="QUESTION", border_style="bright_cyan", padding=(0, 1)
        )
    )
    answer = Text(generation.answer or "The model produced no text.")
    if not generation.answer:
        answer.stylize("dim")
    console.print(
        Panel(answer, title="GENERATED ANSWER", border_style="bright_cyan", padding=(0, 1))
    )
    console.print(_token_view(generation, width, _probe_label(turn)))
    console.print(_risk_panel(turn))
    console.print(_uncertainty_panel(generation))
    console.print(_reference_panel(turn))
    console.print(_summary(turn))


def render_external(
    console: Console,
    *,
    question: str,
    answer: str,
    reference_status: str,
    reference_detail: str,
    probe_status: str,
) -> None:
    console.print(
        Panel(Text(question), title="QUESTION", border_style="bright_cyan", padding=(0, 2))
    )
    console.print(
        Panel(Text(answer), title="SUPPLIED ANSWER", border_style="bright_cyan", padding=(1, 2))
    )
    console.print(
        Panel(
            Text(EXTERNAL_PROBE_NOTE),
            title="HIDDEN-STATE PROBE",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    console.print(
        Text(
            f"Local probe file: {probe_status}. It was not applied to this text.",
            style="bold yellow",
        )
    )
    body = Text()
    body.append(reference_label(reference_status) + "\n", style="bold")
    body.append(reference_detail)
    console.print(Panel(body, title="REFERENCE CHECK", border_style="cyan", padding=(1, 2)))


def render_epilogue(console: Console, turns: list[TurnRecord], detector_status: str) -> None:
    counts: dict[str, int] = {}
    for turn in turns:
        label = reference_label(turn.reference.status)
        counts[label] = counts.get(label, 0) + 1
    body = Text()
    body.append("Showcase finished.\n", style="bold")
    body.append(f"Questions: {len(turns)}\n")
    body.append(f"Detector: {detector_status}\n\n")
    body.append("Curated reference results (external string check, not probe accuracy):\n")
    for label, count in counts.items():
        body.append(f"  {count}  {label}\n")
    console.print(Panel(body, title="DEMO", border_style="bright_cyan", padding=(1, 2)))


def render_turn_text(turn: TurnRecord, width: int = 100) -> str:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=width, color_system=None)
    render_turn(console, turn)
    return buffer.getvalue()


def _clip(text: str, limit: int) -> str:
    if limit < 1:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[: limit - 1] + "…"


def _probe_label(turn: TurnRecord) -> str:
    if turn.factual_risk_score is not None:
        return format_percent(turn.factual_risk_score)
    if turn.experimental_score is not None:
        return "experimental"
    return PROBE_NOT_TRAINED


def _alternatives_text(token: GeneratedToken, limit: int, count: int) -> str:
    others = [item for item in token.top_alternatives if item.token_id != token.token_id][:count]
    if not others:
        return "—"
    rendered = "  ".join(
        f"{display_token(item.text, 12)} {format_percent(item.probability)}" for item in others
    )
    return _clip(rendered, limit)


def _condition_short(token: GeneratedToken) -> str:
    if token.conditioned_on == "prompt_plus_generated":
        return "+output"
    return "prompt"


def _token_view(generation: GenerationResult, width: int, probe_label: str):
    tokens = generation.tokens
    if not tokens:
        return Panel(
            Text("No output tokens were generated."),
            title="OUTPUT TOKENS",
            border_style="yellow",
            padding=(0, 1),
        )
    if width < 88:
        return _token_stack(generation, width, probe_label)
    return _token_table(generation, width, probe_label)


def _token_table(generation: GenerationResult, width: int, probe_label: str) -> Table:
    tokens = generation.tokens
    probe_width = 18
    token_width = 12
    given_width = 8
    alt_width = max(18, width - token_width - probe_width - given_width - 22)
    shown = 5
    table = Table(
        title="OUTPUT TOKEN PROBABILITIES",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        expand=True,
        padding=(0, 1),
        caption_style="dim",
    )
    table.add_column("#", justify="right", width=3, max_width=4, no_wrap=True)
    table.add_column(
        "Token", width=token_width, max_width=token_width, no_wrap=True, overflow="ellipsis"
    )
    table.add_column("P(selected)", justify="right", width=11, max_width=11, no_wrap=True)
    table.add_column(
        "Given", width=given_width, max_width=given_width, no_wrap=True, overflow="ellipsis"
    )
    table.add_column(
        "Alternatives", ratio=1, max_width=alt_width, no_wrap=True, overflow="ellipsis"
    )
    table.add_column(
        "Probe", width=probe_width, max_width=probe_width, no_wrap=True, overflow="ellipsis"
    )
    for token in tokens:
        table.add_row(
            f"{token.step:>2}",
            display_token(token.text, token_width),
            format_percent(token.probability),
            _condition_short(token),
            _alternatives_text(token, alt_width, shown),
            _clip(probe_label, probe_width),
        )
    table.caption = (
        f"Prompt tokens: {generation.prompt_tokens} (input, not scored). "
        f"Output tokens: {generation.generated_tokens}. "
        f"Decoding: {generation.decoding}"
        + ("" if generation.temperature <= 0 else f" {generation.temperature:g}")
        + ". Showing up to "
        f"{shown} alternatives. Full lists are in the export, not this table.\n"
        "Step 1 is conditioned on the prompt. Later steps also use output already written. "
        "These are not input-token probabilities. Probe is the answer-level probe, not P(selected)."
    )
    return table


def _token_stack(generation: GenerationResult, width: int, probe_label: str) -> Panel:
    inner = max(12, width - 4)
    probe = _clip(probe_label, min(18, max(6, inner // 4)))
    lines = [
        _clip(
            f"Prompt tokens {generation.prompt_tokens} are not scored. "
            f"Output decoding: {generation.decoding}.",
            inner,
        )
    ]
    for token in generation.tokens:
        shown = display_token(token.text, max(6, inner // 4))
        chosen = format_percent(token.probability)
        given = _condition_short(token)
        head = _clip(f"{token.step:>2}  {shown}  {chosen}  {given}  {probe}", inner)
        alt = _clip("alt " + _alternatives_text(token, inner, 2), inner)
        lines.append(head)
        lines.append(alt)
    lines.append(
        _clip(
            "Step 1: after the prompt. Later steps: prompt + earlier output. Not input-token probabilities.",
            inner,
        )
    )
    return Panel(
        Text("\n".join(lines)),
        title="OUTPUT TOKENS",
        border_style="cyan",
        padding=(0, 1),
    )


def _risk_panel(turn: TurnRecord) -> Panel:
    body = Text()
    generation = turn.generation
    if turn.factual_risk_score is None:
        body.append(turn.probe_status + "\n", style="bold yellow")
        body.append("RISK NOT ASSESSED\n", style="bold yellow")
        body.append(
            "A calibrated hallucination probability is unavailable. "
            "Token probability and entropy are not a risk score.\n"
        )
        if turn.experimental_score is not None:
            body.append("EXPERIMENTAL PROBE SCORE\n", style="bold yellow")
            body.append(
                f"Raw score: {turn.experimental_score:.3f}. "
                "This is not a calibrated factual-risk percentage.\n"
            )
            body.append("RISK NOT ASSESSED\n", style="bold yellow")
        if turn.probe_prediction is not None:
            cutoff = "unrecorded" if turn.probe_threshold is None else f"{turn.probe_threshold:.2f}"
            body.append(f"Predicted class: {turn.probe_prediction} at threshold {cutoff}.\n")
        body.append(f"Calibration: {turn.calibration_status}.\n")
    else:
        body.append("Calibrated probe\n", style="bold cyan")
        body.append(risk_level(turn) + "\n", style="bold")
        body.append(
            f"Calibrated probe output is {format_percent(turn.factual_risk_score)}. "
            "This percentage passed the held-out calibration check. "
            "It is not a token probability and it is not a measured accuracy.\n"
        )
    if turn.probe_task:
        body.append(turn.probe_task + "\n")
    if turn.score_note:
        body.append(turn.score_note + "\n", style="dim")
    if generation.hidden_states_available and generation.feature is not None:
        body.append(
            f"Hidden states: {generation.position}, layer {generation.feature_layer}, "
            f"dim {int(generation.feature.shape[0])}.\n",
            style="dim",
        )
    else:
        body.append("Hidden states: unavailable.", style="dim")
    return Panel(body, title="FACTUAL RISK", border_style="yellow", padding=(0, 1))


def _uncertainty_panel(generation: GenerationResult) -> Panel:
    body = Text()
    tokens = generation.tokens
    body.append(
        "Uncertainty of the output-token decoding distribution. "
        "Not an input-token probability, and not the probe.\n"
    )
    if not tokens:
        body.append("No tokens were generated.")
    else:
        mean_p = sum(token.probability for token in tokens) / len(tokens)
        mean_h = sum(token.normalized_entropy for token in tokens) / len(tokens)
        mean_gap = sum(token.margin for token in tokens) / len(tokens)
        least_index, least = min(
            enumerate(tokens, start=1), key=lambda item: (item[1].probability, item[0])
        )
        body.append(f"Mean chosen probability   {format_percent(mean_p)}\n")
        body.append(f"Mean normalized entropy   {mean_h:.2f}   (0 peaked, 1 uniform)\n")
        body.append(f"Mean top-1 gap            {format_percent(mean_gap)}\n")
        body.append(
            f"Not the top choice        {sum(1 for token in tokens if not token.is_argmax)} of {len(tokens)}\n"
        )
        body.append(
            f"Lowest chosen probability {format_percent(least.probability)} "
            f"at step {least_index} ({display_token(least.text, 24)}).\n"
        )
    body.append(
        "A low-probability token can be correct, and a high-probability answer can be false.",
        style="dim",
    )
    return Panel(body, title="MODEL UNCERTAINTY", border_style="cyan", padding=(0, 1))


def _reference_panel(turn: TurnRecord) -> Panel:
    body = Text()
    label = reference_label(turn.reference.status)
    detail = turn.reference.detail
    prefix = f"{label}. "
    if detail.startswith(prefix):
        detail = detail[len(prefix) :]
    body.append(label + "\n", style="bold")
    body.append(detail)
    return Panel(body, title="REFERENCE CHECK", border_style="cyan", padding=(0, 1))


def _summary(turn: TurnRecord) -> Table:
    generation = turn.generation
    table = Table(
        title="SUMMARY",
        box=box.SIMPLE_HEAD,
        show_header=False,
        expand=True,
        header_style="bold cyan",
    )
    table.add_column("Field", style="bold", width=16, max_width=16, no_wrap=True)
    table.add_column("Value", ratio=1, no_wrap=True, overflow="ellipsis")
    rate = ""
    if generation.latency_seconds > 0 and generation.generated_tokens:
        rate = f"   ({generation.generated_tokens / generation.latency_seconds:.1f} tok/s)"
    table.add_row("Risk level", risk_level(turn))
    table.add_row("Output tokens", str(generation.generated_tokens))
    table.add_row("Prompt tokens", f"{generation.prompt_tokens} (not scored)")
    table.add_row("Latency", f"{generation.latency_seconds:.2f} s{rate}")
    hidden = "unavailable"
    if generation.hidden_states_available and generation.feature is not None:
        hidden = (
            f"{generation.position}, layer {generation.feature_layer}, "
            f"dim {int(generation.feature.shape[0])}"
        )
    table.add_row("Model", generation.model_name)
    table.add_row("Revision", generation.model_revision)
    table.add_row("Device", f"{generation.device} · {generation.dtype}")
    prediction = turn.probe_prediction or turn.probe_status
    table.add_row("Probe class", prediction)
    table.add_row("Prompt mode", generation.prompt_mode)
    table.add_row("Calibration", turn.calibration_status)
    table.add_row("Hidden states", hidden)
    table.add_row("Reference", reference_label(turn.reference.status))
    return table


class LiveTokenView:
    """Shows each generated token while the model is still decoding."""

    def __init__(self, console: Console, max_new_tokens: int) -> None:
        self.console = console
        self.max_new_tokens = max(1, max_new_tokens)
        self.tokens: list[GeneratedToken] = []
        self.started = time.perf_counter()
        self._live: Live | None = None

    def __enter__(self) -> LiveTokenView:
        self.started = time.perf_counter()
        self._live = Live(self._panel(), console=self.console, refresh_per_second=8, transient=True)
        self._live.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._live is not None:
            self._live.stop()
            self._live = None
        return False

    def update(self, token: GeneratedToken) -> None:
        self.tokens.append(token)
        if self._live is not None:
            self._live.update(self._panel())

    def _panel(self):
        elapsed = time.perf_counter() - self.started
        fraction = min(1.0, len(self.tokens) / self.max_new_tokens)
        filled = int(round(fraction * 24))
        bar = "█" * filled + "░" * (24 - filled)
        header = Text(
            f"{bar}  {len(self.tokens)}/{self.max_new_tokens} tokens    {elapsed:.1f}s",
            style="cyan",
        )
        answer = Text(
            "".join(token.text for token in self.tokens) or "Waiting for the first token…"
        )
        if not self.tokens:
            answer.stylize("dim")
        footer = Text()
        if self.tokens:
            last = self.tokens[-1]
            footer.append(display_token(last.text, 32), style="bold")
            footer.append(f"   P={format_percent(last.probability)}")
            alternative = next_best(last)
            if alternative is not None:
                footer.append(
                    f"    next-best {display_token(alternative.text, 24)} {format_percent(alternative.probability)}",
                    style="dim",
                )
        return Panel(
            Group(header, Text(""), answer, Text(""), footer),
            title="GENERATING",
            border_style="bright_cyan",
            padding=(1, 2),
        )
