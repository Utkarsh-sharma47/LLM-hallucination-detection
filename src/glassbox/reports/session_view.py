"""Render a session export without presenting uncalibrated scores as probabilities.

Issue #1 is a full frontend. This command is the offline, read-only slice:
it turns ``session.json`` into HTML that keeps ``EXPERIMENTAL PROBE SCORE``
and ``RISK NOT ASSESSED`` until calibration is actually validated.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path

from glassbox.evaluation.generation_calibration import RISK_STATUS, SCORE_NAME

CALIBRATED_RISK_LABEL = "factual-risk probability"


def render_session_html(session: Mapping[str, object]) -> str:
    """Return a self-contained HTML report for one exported session."""
    turns = session.get("turns")
    if not isinstance(turns, list):
        raise ValueError("session.json must contain a turns list")
    blocks = [_render_turn(index, turn) for index, turn in enumerate(turns, start=1)]
    body = "\n".join(blocks) if blocks else "<p>No turns in this session.</p>"
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        "<title>Glass Box session report</title>"
        "<style>"
        "body{font-family:sans-serif;max-width:52rem;margin:2rem auto;line-height:1.4}"
        "article{border:1px solid #ccc;padding:1rem;margin:1rem 0}"
        ".warn{color:#7a1212;font-weight:600}"
        ".meta{color:#444;font-size:0.95rem}"
        "</style></head><body>"
        "<h1>Glass Box session report</h1>"
        f'<p class="warn">{html.escape(SCORE_NAME)}. {html.escape(RISK_STATUS)}.</p>'
        "<p>Token probability is the decoding distribution. It is not factual "
        "confidence. A wording difference from a reference is not a hallucination.</p>"
        f"{body}"
        "</body></html>\n"
    )


def _render_turn(index: int, turn: object) -> str:
    if not isinstance(turn, Mapping):
        raise ValueError(f"turn {index} is not an object")
    question = html.escape(str(turn.get("question") or ""))
    answer = html.escape(str(turn.get("answer") or ""))
    evidence = html.escape(
        str(turn.get("evidence_label") or turn.get("verification") or "UNVALIDATED")
    )
    score_line = _score_line(turn)
    risk_line = _risk_line(turn)
    return (
        f"<article><h2>Turn {index}</h2>"
        f"<p><strong>Question</strong> {question}</p>"
        f"<p><strong>Answer</strong> {answer}</p>"
        f"<p><strong>Evidence</strong> {evidence}</p>"
        f'<p class="warn">{html.escape(score_line)}</p>'
        f'<p class="warn">{html.escape(risk_line)}</p>'
        f'<p class="meta">calibration: {html.escape(str(turn.get("calibration_status") or "not validated"))}</p>'
        "</article>"
    )


def _score_line(turn: Mapping[str, object]) -> str:
    if turn.get("factual_risk_score") is not None and turn.get("calibration_validated") is True:
        return f"{CALIBRATED_RISK_LABEL}: {turn['factual_risk_score']}"
    raw = turn.get("experimental_score")
    if raw is None:
        return f"{SCORE_NAME}: unavailable"
    return f"{SCORE_NAME}: {raw}"


def _risk_line(turn: Mapping[str, object]) -> str:
    if turn.get("calibration_validated") is True and turn.get("factual_risk_score") is not None:
        return "Calibrated factual risk is available for this turn."
    return RISK_STATUS


def view_command(args: object) -> int:
    """Write HTML next to a session export. Does not load Qwen."""
    import sys

    try:
        source = Path(args.input)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("session file must contain a JSON object")
        rendered = render_session_html(payload)
        destination = Path(args.output) if args.output else source.with_name("session-view.html")
        destination.write_text(rendered, encoding="utf-8")
        print(f"Wrote {destination}")
        return 0
    except (OSError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Session view stopped: {exc}", file=sys.stderr)
        return 1
