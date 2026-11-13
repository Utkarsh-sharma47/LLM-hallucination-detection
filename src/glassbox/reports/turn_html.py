"""Open one Glass Box turn as a self-contained HTML page.

The page shows only what the terminal report already holds. It keeps
``HALLUCINATION PROBABILITY: NOT AVAILABLE`` unless the report carries a
validated generation-level probability, and it never presents token
confidence as a factuality result. No network access; no external assets.
"""

from __future__ import annotations

import html
import webbrowser
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

LOW_CONFIDENCE = 0.5

_STATUS_CLASS = {
    "FAITHFUL": "ok",
    "HALLUCINATED": "bad",
    "AMBIGUOUS": "warn",
    "UNVALIDATED": "warn",
}

_STATUS_NOTE = {
    "FAITHFUL": "Matches curated evidence in {catalog}.",
    "HALLUCINATED": "Contradicts curated evidence in {catalog}.",
    "AMBIGUOUS": "Curated evidence in {catalog} is mixed for this answer.",
    "UNVALIDATED": "Missing evidence in {catalog}; not a confirmed hallucination.",
}


def render_turn_html(report: Mapping[str, Any], *, run_id: str = "") -> str:
    """Return a self-contained HTML page for one ``GlassBoxReport.to_json()`` payload."""
    config = _mapping(report.get("generation_config"))
    probabilities = _mapping(report.get("token_probabilities"))
    alternatives = _mapping(report.get("token_alternatives"))
    factuality = _mapping(report.get("factuality"))
    probe = _mapping(report.get("probe"))
    conclusion = _mapping(report.get("conclusion"))
    hidden = _mapping(report.get("hidden_state_metadata"))
    layer_analysis = _mapping(report.get("layer_analysis"))
    early_warning = _mapping(report.get("early_warning"))
    tokens = _rows(probabilities.get("tokens"))
    contested = _most_contested(tokens)
    body = [
        _header(str(config.get("model_name") or "unknown model")),
        '<main class="wrap">',
        _run_card(report, config, tokens, contested, run_id),
        _metric_cards(probabilities, factuality, conclusion),
        _token_table(_rows(alternatives.get("tokens")), contested),
        _claims_card(factuality),
        _internal_model_card(config, hidden),
        _probe_card(report, probe),
        _layer_analysis_card(layer_analysis),
        _early_warning_card(early_warning),
        _summary_card(conclusion, factuality, probe, len(tokens)),
        "</main>",
        _footer(config, hidden),
    ]
    title = _e(f"Glass Box - {report.get('question', '')}"[:120])
    return _PAGE.format(title=title, css=_CSS, body="\n".join(body), script=_SCRIPT)


def write_turn_html(
    report: Mapping[str, Any], directory: Path, *, stamp: datetime | None = None
) -> Path:
    """Write the page under ``directory`` with a unique, time-based name."""
    run_id = (stamp or datetime.now()).strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"turn-{run_id}.html"
    suffix = 1
    while path.exists():
        suffix += 1
        path = directory / f"turn-{run_id}-{suffix}.html"
    path.write_text(render_turn_html(report, run_id=path.stem), encoding="utf-8")
    return path


def open_in_browser(path: Path) -> bool:
    """Open a written page; return False when no browser could be started."""
    try:
        return bool(webbrowser.open(Path(path).resolve().as_uri()))
    except (webbrowser.Error, OSError):
        return False


def _header(model: str) -> str:
    return (
        '<header class="top"><div class="brand"><span class="logo">Glass Box</span>'
        '<span class="sub">LLM DIAGNOSTIC OUTPUT</span>'
        f'<span class="chip"><span class="k">MODEL:</span> {_e(model)}</span>'
        '<span class="chip live"><i></i>COMPLETE</span></div>'
        '<button class="theme" type="button" id="theme" aria-label="Toggle theme">'
        "Theme</button></header>"
    )


def _run_card(
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    tokens: list[dict[str, Any]],
    contested: int | None,
    run_id: str,
) -> str:
    chips = [
        _chip("ENGINE", config.get("model_name"), accent=True),
        _chip("LENGTH", f"{len(tokens)} tokens"),
        _chip("PROMPT", f"{config.get('prompt_tokens', '?')} tokens"),
        _chip("LATENCY", _latency(config.get("latency_seconds"))),
    ]
    sequence = "".join(_seq_token(row, row.get("position") == contested) for row in tokens)
    if not sequence:
        sequence = '<span class="muted">No tokens were generated.</span>'
    return (
        '<section class="card run"><div class="run-head">'
        f'<div class="mono small">RUN {_e(run_id or "-")} &middot; Single-step diagnostic</div>'
        f'<div class="chips">{"".join(chips)}</div></div>'
        '<div class="run-grid"><div class="pane">'
        '<div class="label">[01] QUESTION</div>'
        f'<div class="query">&ldquo;{_e(report.get("question", ""))}&rdquo;</div>'
        f'<div class="label gap">[02] GENERATED ANSWER</div><div>{_e(report.get("answer", ""))}</div>'
        '</div><div class="pane"><div class="label row">INFERENCE SEQUENCE'
        '<span class="legend"><b class="sw hi"></b>most contested '
        f'<b class="sw lo"></b>p &lt; {LOW_CONFIDENCE:.0%}</span></div>'
        f'<div class="seq">{sequence}</div></div></div></section>'
    )


def _seq_token(row: Mapping[str, Any], contested: bool) -> str:
    probability = _float(row.get("probability"))
    classes = ["tok"]
    if contested:
        classes.append("hi")
    elif probability is not None and probability < LOW_CONFIDENCE:
        classes.append("lo")
    tip = f"p = {_pct(probability)}"
    text = _e(_visible(row.get("text")))
    return f'<span class="{" ".join(classes)}" title="{tip}">{text}</span>'


def _metric_cards(
    probabilities: Mapping[str, Any],
    factuality: Mapping[str, Any],
    conclusion: Mapping[str, Any],
) -> str:
    if not probabilities.get("available"):
        return '<section class="card"><h2>[03] Token Generation Probabilities</h2><p class="muted">No token probabilities.</p></section>'
    mean = _float(probabilities.get("average_selected_probability"))
    geo = _float(probabilities.get("geometric_mean_probability"))
    entropy = _float(probabilities.get("average_entropy"))
    uncertainty = str(conclusion.get("generation_uncertainty") or "UNKNOWN")
    status = str(factuality.get("overall_status") or "UNVALIDATED")
    catalog = Path(str(factuality.get("catalog_path") or "claim_references.yaml")).name
    note = _STATUS_NOTE.get(status, "{catalog}").format(catalog=f"<code>{_e(catalog)}</code>")
    tone = "ok" if uncertainty == "LOW" else "warn"
    cards = [
        _metric("AVG PROBABILITY", _pct(mean), _fixed(mean, 4), "Mean selected-token softmax."),
        _metric("GEOMETRIC MEAN", _fixed(geo, 4), "&Pi;(p)<sup>1/n</sup>", "Penalises weak steps."),
        _metric(
            "AVG ENTROPY",
            _fixed(entropy, 4),
            "NATS",
            f'<span class="dot {tone}"></span>{_e(uncertainty.title())} uncertainty profile',
        ),
        '<div class="card metric"><div class="label">FACTUALITY VERIFICATION</div>'
        f'<div><span class="badge {_STATUS_CLASS.get(status, "warn")}">{_e(status)}</span></div>'
        f'<p class="note">{note}</p></div>',
    ]
    return (
        '<section class="metrics"><h2 class="metrics-title">[03] Token Generation Probabilities</h2>'
        f"{''.join(cards)}</section>"
    )


def _metric(label: str, value: str, unit: str, note: str) -> str:
    return (
        f'<div class="card metric"><div class="label">{label}</div>'
        f'<div class="value">{value}<span class="unit">{unit}</span></div>'
        f'<p class="note">{note}</p></div>'
    )


def _token_table(rows: list[dict[str, Any]], contested: int | None) -> str:
    if not rows:
        return ""
    body = "".join(_token_row(row, row.get("position") == contested) for row in rows)
    return (
        '<section class="card table"><div class="table-head"><div>'
        "<h2>[04] Token Sequence &amp; Logit Decomposition</h2>"
        "<p>Step-by-step next-token probabilities and runner-up alternatives.</p></div>"
        '<span class="label">TOP-3 CANDIDATES</span></div>'
        '<div class="scroll"><table><thead><tr><th>POS</th><th>SELECTED TOKEN</th>'
        "<th>PROBABILITY</th><th>COMPETING ALTERNATIVES</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div></section>"
    )


def _token_row(row: Mapping[str, Any], contested: bool) -> str:
    probability = _float(row.get("selected_probability")) or 0.0
    alternatives = _rows(row.get("alternatives"))[:3]
    chips = "".join(
        f'<span class="alt">{_e(_visible(item.get("text")))} '
        f'<span class="muted">({_pct(_float(item.get("probability")))})</span></span>'
        for item in alternatives
    )
    note = ""
    if contested and alternatives:
        top = alternatives[0]
        gap = probability - (_float(top.get("probability")) or 0.0)
        note = (
            '<div class="flag">Most contested step: closest alternative '
            f"&ldquo;{_e(_visible(top.get('text')))}&rdquo;, margin {gap * 100:.1f} pts</div>"
        )
    low = probability < LOW_CONFIDENCE
    row_class = "hi" if contested else ("lo" if low else "")
    width = max(0.0, min(1.0, probability)) * 100
    return (
        f'<tr class="{row_class}"><td class="mono muted">{_e(row.get("position", ""))}</td>'
        f'<td><span class="tok">{_e(_visible(row.get("selected_token")))}</span></td>'
        f'<td><div class="bar"><b style="width:{width:.1f}%"></b></div>'
        f'<span class="mono">{_pct(probability)}</span></td>'
        f"<td>{chips or '<span class=muted>none recorded</span>'}{note}</td></tr>"
    )


def _claims_card(factuality: Mapping[str, Any]) -> str:
    claims = _rows(factuality.get("claims"))
    if not claims:
        return ""
    items = []
    for claim in claims:
        status = str(claim.get("status") or "UNVALIDATED")
        details = [
            f"<div><span class='k'>{name}:</span> {_e(claim.get(key))}</div>"
            for key, name in (("support", "Support"), ("contradiction", "Contradiction"))
            if claim.get(key)
        ]
        if claim.get("source"):
            details.append(f"<div class='muted'>{_e(claim.get('source'))}</div>")
        items.append(
            f'<li><span class="badge {_STATUS_CLASS.get(status, "warn")}">{_e(status)}</span>'
            f'<div><div class="claim">{_e(claim.get("text"))}</div>{"".join(details)}</div></li>'
        )
    return (
        '<section class="card"><h2>[05] Factuality Verification &amp; Claim Check</h2>'
        "<p class='muted'>Each sentence is checked against the curated catalog only. "
        "Missing evidence is not a hallucination.</p>"
        f'<ul class="claims">{"".join(items)}</ul></section>'
    )


def _probe_card(report: Mapping[str, Any], probe: Mapping[str, Any]) -> str:
    probability = _float(report.get("calibrated_hallucination_probability"))
    if probability is None:
        headline = '<div class="headline warn">HALLUCINATION PROBABILITY: NOT AVAILABLE</div>'
    else:
        headline = f'<div class="headline">Hallucination probability: {_pct(probability)}</div>'
    if not probe.get("available"):
        rows = "<p class='muted'>No probe loaded. No prediction was invented.</p>"
    elif _float(probe.get("raw_score")) is None:
        reason = probe.get("score_note") or probe.get("status") or "The probe did not score."
        rows = (
            '<p class="note"><span class="badge warn">NO SCORE</span> '
            f"{_e(reason)} No prediction was invented.</p>"
        )
    else:
        pairs = [
            ("Protocol", probe.get("protocol")),
            (
                "Calibrated score"
                if probe.get("calibration_status") == "validated"
                else "Raw experimental score",
                _fixed(_float(probe.get("raw_score")), 3),
            ),
            ("Prediction", probe.get("prediction")),
            ("Threshold", probe.get("threshold")),
            ("Calibration status", probe.get("calibration_status")),
        ]
        cells = "".join(f"<dt>{name}</dt><dd>{_e(value or '-')}</dd>" for name, value in pairs)
        rows = f'<dl class="kv">{cells}</dl><p class="note">{_e(probe.get("note") or "")}</p>'
    return f'<section class="card"><h2>[07] Probe Analysis</h2>{headline}{rows}</section>'


def _internal_model_card(config: Mapping[str, Any], hidden: Mapping[str, Any]) -> str:
    """Show the model and capture details that support this turn's analysis."""
    pairs = [
        ("Model", config.get("model_name")),
        ("Revision", config.get("model_revision")),
        ("Device / dtype", f"{config.get('device') or '-'} / {config.get('dtype') or '-'}"),
        ("Decoding", config.get("decoding")),
        ("Hidden-state layers", hidden.get("hidden_state_count")),
        ("Feature layer", hidden.get("feature_layer")),
        ("Feature width", hidden.get("feature_dimension")),
        ("Feature position", hidden.get("position")),
        ("Capture mode", hidden.get("capture_mode")),
        ("Retained capture", _bytes(hidden.get("capture_bytes"))),
        ("Transient hidden-state peak", _bytes(hidden.get("transient_hidden_state_bytes"))),
        ("Analysis forward passes", hidden.get("analysis_forward_calls")),
        ("Analysis latency", _latency(hidden.get("analysis_latency_seconds"))),
        ("Capture note", hidden.get("capture_note")),
    ]
    return _details_card("[06] Internal Model Analysis", pairs)


def _layer_analysis_card(analysis: Mapping[str, Any]) -> str:
    """Show trained-layer evaluation separately from per-answer representation drift."""
    probe_layer = analysis.get("probe_layer")
    hidden_count = analysis.get("hidden_state_count")
    layer_label = (
        "Calibrated generation probe layer"
        if analysis.get("compatible_generation_probe")
        else "Probe feature layer"
    )
    intro = (
        f"{layer_label}: {probe_layer} (zero-indexed) of {hidden_count} hidden-state layers."
        if probe_layer is not None
        else "No trained probe layer is recorded for this report."
    )
    evaluations = _rows(analysis.get("probe_evaluation"))
    if evaluations:
        rows = "".join(
            "<tr>"
            f"<td>{_e(item.get('dataset') or '-')}</td>"
            f"<td>{_e(item.get('n') if item.get('n') is not None else '-')}</td>"
            f"<td>{_fixed(_float(item.get('auroc')), 3)}</td>"
            f"<td>{_fixed(_float(item.get('ece')), 3)}</td>"
            "</tr>"
            for item in evaluations
        )
        evaluation_table = (
            '<div class="scroll"><table class="compact"><thead><tr>'
            "<th>EVALUATION SPLIT</th><th>N</th><th>AUROC</th><th>ECE</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )
    else:
        evaluation_table = (
            "<p class='muted'>No held-out layer metrics are stored with this probe.</p>"
        )

    changes = _rows(analysis.get("captured_layer_changes"))
    if changes:
        change_rows = "".join(
            "<tr>"
            f"<td>{_e(item.get('layer_index'))}</td>"
            f"<td>{_fixed(_float(item.get('cosine_distance')), 6)}</td>"
            f"<td>{_e(item.get('interpretation') or '')}</td>"
            "</tr>"
            for item in changes
        )
        change_table = (
            '<h3>Captured representation changes</h3><div class="scroll">'
            '<table class="compact"><thead><tr><th>LAYER</th><th>FIRST-TO-LAST COSINE DISTANCE</th>'
            f"<th>INTERPRETATION</th></tr></thead><tbody>{change_rows}</tbody></table></div>"
        )
    else:
        change_table = (
            "<p class='muted'>Per-answer token spans were not captured, so representation change "
            "cannot be calculated for this turn.</p>"
        )
    note = analysis.get("interpretation_note") or (
        "A probe trained at one layer does not compare probe quality across layers."
    )
    return (
        '<section class="card"><h2>[08] Layer-Wise Analysis</h2>'
        f"<p class='muted'>{_e(intro)}</p>"
        f"{evaluation_table}{change_table}"
        f"<p class='note'>{_e(note)}</p></section>"
    )


def _early_warning_card(analysis: Mapping[str, Any]) -> str:
    """Explain whether sequential warning lead time was actually measured."""
    available = analysis.get("available") is True
    status = "Measured for this answer" if available else "Not measured for this answer"
    position = analysis.get("probe_position") or "unavailable"
    reason = analysis.get("reason") or "No sequential warning evidence is stored for this turn."
    required = analysis.get("required_data")
    rows = [
        ("Status", status),
        ("Probe position", position),
        ("Reason", reason),
    ]
    if required:
        rows.append(("Needed to measure this answer", required))
    evaluation = _mapping(analysis.get("held_out_evaluation"))
    if evaluation:
        lead = _mapping(evaluation.get("test_early_warning"))
        prefix = _mapping(evaluation.get("test_prefix_metrics"))
        training = _mapping(evaluation.get("training"))
        threshold = evaluation.get("threshold", training.get("threshold"))
        false_warning_rate = _float(lead.get("faithful_answer_false_warning_rate"))
        flags_everything = false_warning_rate is not None and false_warning_rate >= 0.95
        events = lead.get("hallucinated_answers_with_prefix_event")
        timely = lead.get("timely_warning_count")
        rows.extend(
            [
                ("Held-out Task 7 evaluation", "Aggregate prefix-level result; not this answer"),
                (
                    "Prefix-probe threshold",
                    f"{threshold} (separate prefix probe; not the section 7 threshold)",
                ),
                (
                    "Threshold rule",
                    training.get("threshold_rule")
                    or "validation F1 only, with no limit on false warnings",
                ),
                ("Test prefix AUROC", _fixed(_float(prefix.get("auroc")), 3)),
                (
                    "Hallucinated answers with a defined prefix event",
                    _ratio(events, lead.get("reviewed_hallucinated_answers")),
                ),
                (
                    "Timely warning coverage",
                    "Not meaningful: this threshold flags every answer"
                    if flags_everything
                    else _ratio(timely, events),
                ),
                (
                    "Mean lead among timely warnings",
                    "Not meaningful at this threshold"
                    if flags_everything
                    else f"{_tokens(lead.get('mean_lead_tokens_among_timely_warnings'))} (n = {timely})",
                ),
                ("False-warning rate on faithful answers", _pct(false_warning_rate)),
                (
                    "Prefix labels vs reviewed labels",
                    _disagreement_text(lead),
                ),
                ("Interpretation", _early_warning_interpretation(lead, flags_everything)),
            ]
        )
    return _details_card("[09] Early-Warning Analysis", rows, muted=not available)


def _details_card(title: str, pairs: list[tuple[str, object]], *, muted: bool = False) -> str:
    cells = "".join(
        f"<dt>{_e(name)}</dt><dd>{_e(value if value is not None else '-')}</dd>"
        for name, value in pairs
    )
    tone = " muted" if muted else ""
    return (
        f'<section class="card{tone}"><h2>{_e(title)}</h2>'
        f'<dl class="kv details">{cells}</dl></section>'
    )


def _summary_card(
    conclusion: Mapping[str, Any],
    factuality: Mapping[str, Any],
    probe: Mapping[str, Any],
    token_count: int,
) -> str:
    uncertainty = str(conclusion.get("generation_uncertainty") or "UNKNOWN")
    status = str(factuality.get("overall_status") or "UNVALIDATED")
    reliability = str(conclusion.get("probe_reliability") or "UNAVAILABLE")
    claims = _rows(factuality.get("claims"))
    claim_text = f"{len(claims)} extracted claim(s)"
    if claims:
        claim_text += f": &ldquo;{_e(claims[0].get('text'))}&rdquo;"
    if not probe.get("available"):
        probe_text = "No probe loaded."
    elif reliability == "CALIBRATED":
        probe_text = "Calibrated generation-level detector."
    else:
        probe_text = f"{_e(reliability.title())}. Not a generation-level factuality detector."
    tiles = [
        (
            "ok" if uncertainty == "LOW" else "warn",
            f"Generation Uncertainty: {uncertainty.title()}",
            f"Selected-token probabilities across all {token_count} steps.",
        ),
        (_STATUS_CLASS.get(status, "warn"), f"Factuality Status: {status.title()}", claim_text),
        (
            "bad",
            "Critical Caveat",
            "High token probability measures predictability, not factual correctness.",
        ),
        ("muted", "Probe Status", probe_text),
    ]
    html_tiles = "".join(
        f'<div class="tile"><span class="pip {tone}"></span><div>'
        f"<div class='tile-title'>{_e(title)}</div><div class='note'>{text}</div></div></div>"
        for tone, title, text in tiles
    )
    limits = "".join(f"<li>{_e(item)}</li>" for item in _rows_text(conclusion.get("limitations")))
    return (
        '<section class="card"><h2>[10] Diagnostic Summary &amp; Limitations</h2>'
        f'<div class="tiles">{html_tiles}</div>'
        f'<ul class="limits">{limits}</ul></section>'
    )


def _footer(config: Mapping[str, Any], hidden: Mapping[str, Any]) -> str:
    revision = str(config.get("model_revision") or "")[:12] or "-"
    parts = [
        f"REVISION {revision}",
        f"{config.get('dtype') or '-'} on {config.get('device') or '-'}",
        f"{config.get('decoding') or '-'} decoding",
        f"layer {hidden.get('feature_layer', '-')} / {hidden.get('hidden_state_count', '-')}",
        f"width {hidden.get('feature_dimension', '-')}",
    ]
    text = " &middot; ".join(_e(part) for part in parts)
    return f'<footer class="foot mono small">{text}</footer>'


def _chip(name: str, value: object, *, accent: bool = False) -> str:
    tone = " accent" if accent else ""
    label = f'<span class="k">{name}:</span>'
    return f'<span class="chip">{label}<b class="v{tone}">{_e(value)}</b></span>'


def _most_contested(tokens: list[dict[str, Any]]) -> int | None:
    """Position of the step with the smallest margin over its runner-up."""
    best: tuple[float, int] | None = None
    for row in tokens:
        margin = _float(row.get("margin"))
        position = row.get("position")
        if margin is None or not isinstance(position, int):
            continue
        if best is None or margin < best[0]:
            best = (margin, position)
    return None if best is None else best[1]


def _visible(text: object) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\n", "↵").strip()
    return value or "␣"


def _latency(seconds: object) -> str:
    value = _float(seconds)
    if value is None:
        return "-"
    return f"{value * 1000:.0f} ms" if value < 1 else f"{value:.2f} s"


def _bytes(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return "-"
    if value < 1024:
        return f"{value:.0f} B"
    if value < 1024**2:
        return f"{value / 1024:.1f} KiB"
    return f"{value / 1024**2:.2f} MiB"


def _disagreement_text(lead: Mapping[str, Any]) -> str:
    count = lead.get("reviewed_vs_prefix_final_label_disagreements")
    total = lead.get("test_answers_compared")
    if isinstance(count, int) and isinstance(total, int) and total > 0:
        head = f"{count} of {total} test answers disagree ({count / total:.0%})"
    elif isinstance(count, int):
        head = f"{count} test answers disagree"
    else:
        return "-"
    return (
        f"{head}. Prefix events come from the automatic evidence labeler, not the review, "
        "which is the main reason this result is weak."
    )


def _early_warning_interpretation(lead: Mapping[str, Any], flags_everything: bool) -> str:
    if flags_everything:
        return (
            "Exploratory only: every faithful answer is flagged at this threshold, "
            "so the probe is not useful as an early warning."
        )
    timely = lead.get("timely_warning_count")
    events = lead.get("hallucinated_answers_with_prefix_event")
    rate = _float(lead.get("faithful_answer_false_warning_rate"))
    if isinstance(timely, int) and isinstance(events, int) and events and rate is not None:
        return (
            f"Exploratory only: at a {rate:.0%} false-warning rate it warns on {timely} of {events} "
            "defined events, so it is not a useful early warning."
        )
    return (
        "Exploratory Task 7 result; this aggregate evaluation does not measure the current answer."
    )


def _ratio(numerator: object, denominator: object) -> str:
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        return "-"
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})"


def _tokens(value: object) -> str:
    number = _float(value)
    return "-" if number is None else f"{number:.2f} tokens"


def _pct(value: float | None) -> str:
    if value is None:
        return "-"
    if 0 < value < 0.001:
        return "&lt;0.1%"
    return f"{value * 100:.2f}%"


def _fixed(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _rows_text(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _e(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{css}</style></head>
<body>
{body}
<script>{script}</script>
</body></html>
"""

_CSS = """
:root{--bg:#f6f8fb;--card:#ffffff;--line:#e3e8ef;--ink:#0f172a;--muted:#64748b;
--accent:#0284c7;--accent-soft:#e0f2fe;--ok:#059669;--ok-soft:#d1fae5;--warn:#b45309;
--warn-soft:#fef3c7;--bad:#dc2626;--bad-soft:#fee2e2;--chip:#f1f5f9;--bar:#e2e8f0;
--mono:ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",Menlo,monospace;
--sans:"Segoe UI",system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0b1220;
--card:#131c2e;--line:#22304a;--ink:#e2e8f0;--muted:#8a9bb5;--accent:#22d3ee;
--accent-soft:#0e3a47;--ok:#34d399;--ok-soft:#0f3b2e;--warn:#fbbf24;--warn-soft:#3d2f0c;
--bad:#f87171;--bad-soft:#3f1717;--chip:#1b2740;--bar:#22304a}}
:root[data-theme="dark"]{--bg:#0b1220;--card:#131c2e;--line:#22304a;--ink:#e2e8f0;
--muted:#8a9bb5;--accent:#22d3ee;--accent-soft:#0e3a47;--ok:#34d399;--ok-soft:#0f3b2e;
--warn:#fbbf24;--warn-soft:#3d2f0c;--bad:#f87171;--bad-soft:#3f1717;--chip:#1b2740;
--bar:#22304a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 var(--sans)}code,.mono{font-family:var(--mono)}.small{font-size:12px}
.muted{color:var(--muted)}.top{display:flex;justify-content:space-between;align-items:center;
gap:12px;padding:14px 28px;border-bottom:1px solid var(--line);background:var(--card);
position:sticky;top:0;z-index:2;flex-wrap:wrap}.brand{display:flex;align-items:center;
gap:12px;flex-wrap:wrap}.logo{color:var(--accent);font-weight:700;font-size:19px}
.sub{font:11px var(--mono);letter-spacing:.08em;color:var(--muted)}
.chip{font:12px var(--mono);background:var(--chip);border:1px solid var(--line);
border-radius:6px;padding:4px 9px;max-width:100%;overflow-wrap:anywhere}.chip .k{color:var(--muted)}
.chip .v{font-weight:500;margin-left:4px}.chip .accent{color:var(--accent)}
.chip.live{color:var(--ok);background:var(--ok-soft);border-color:transparent;
border-radius:99px}.chip.live i{display:inline-block;width:7px;height:7px;border-radius:50%;
background:var(--ok);margin-right:6px}.theme{font:12px var(--mono);background:var(--card);
color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 12px;
cursor:pointer}.wrap{max-width:1240px;margin:0 auto;padding:24px 16px 40px;display:grid;
gap:20px;grid-template-columns:minmax(0,1fr)}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:20px}h2{margin:0 0 4px;font-size:19px}.label{font:11px var(--mono);
letter-spacing:.08em;color:var(--muted);margin-bottom:6px}.label.row{display:flex;
justify-content:space-between;gap:8px;flex-wrap:wrap}.gap{margin-top:14px}
.run-head{display:flex;justify-content:space-between;align-items:center;gap:12px;
flex-wrap:wrap;padding-bottom:14px;border-bottom:1px solid var(--line);margin-bottom:16px}
.chips{display:flex;gap:8px;flex-wrap:wrap}.run-grid{display:grid;
grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}.pane{border:1px solid var(--line);border-radius:10px;
padding:14px 16px;background:var(--bg)}.query{font-size:19px}.legend{display:flex;
gap:6px;align-items:center;text-transform:none;letter-spacing:0}.sw{display:inline-block;
width:10px;height:10px;border-radius:3px}.sw.hi{background:var(--accent)}
.sw.lo{background:var(--warn)}.seq{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.tok{font:14px var(--mono);border:1px solid var(--line);background:var(--card);
border-radius:6px;padding:3px 9px}.tok.hi,tr.hi .tok{background:var(--accent);
color:var(--card);border-color:var(--accent);font-weight:600}.tok.lo,tr.lo .tok{
border-color:var(--warn);color:var(--warn)}.metrics{display:grid;
grid-template-columns:repeat(4,1fr);gap:16px}.metrics-title{grid-column:1/-1;margin-bottom:-8px}
.metric .value{font:600 30px var(--mono)}
.unit{font:11px var(--mono);color:var(--muted);margin-left:6px;letter-spacing:.05em}
.note{color:var(--muted);margin:8px 0 0;font-size:14px}.dot{display:inline-block;width:8px;
height:8px;border-radius:50%;margin-right:6px}.dot.ok{background:var(--ok)}
.dot.warn{background:var(--warn)}.badge{font:600 12px var(--mono);letter-spacing:.06em;
padding:3px 9px;border-radius:6px;display:inline-block}.badge.ok{color:var(--ok);
background:var(--ok-soft)}.badge.warn{color:var(--warn);background:var(--warn-soft)}
.badge.bad{color:var(--bad);background:var(--bad-soft)}.table{padding:0}
.table-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;
padding:20px}.table-head p{margin:0;color:var(--muted)}.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;min-width:640px}th{text-align:left;
font:11px var(--mono);letter-spacing:.08em;color:var(--muted);font-weight:400;
padding:10px 16px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
td{padding:11px 16px;border-bottom:1px solid var(--line);vertical-align:middle}
tr:last-child td{border-bottom:0}tr.hi td{background:var(--accent-soft)}
.compact{margin:12px 0;min-width:540px}.compact th,.compact td{padding:8px 12px}
.compact td{font-size:13px}.card h3{font-size:15px;margin:16px 0 4px}
.details{margin-top:14px}.details dd{line-height:1.5}.card.muted{border-style:dashed}
.bar{display:inline-block;width:140px;height:7px;border-radius:9px;background:var(--bar);
margin-right:10px;vertical-align:middle;overflow:hidden}.bar b{display:block;height:100%;
background:var(--accent);border-radius:9px}tr.lo .bar b{background:var(--warn)}
.alt{display:inline-block;font:12px var(--mono);border:1px solid var(--line);
border-radius:6px;padding:2px 8px;margin:2px 6px 2px 0;background:var(--card)}
.flag{font:11px var(--mono);color:var(--accent);margin-top:6px;letter-spacing:.03em}
.claims{list-style:none;padding:0;margin:12px 0 0;display:grid;gap:10px}.claims li{
display:flex;gap:12px;align-items:flex-start;border:1px solid var(--line);border-radius:10px;
padding:12px}.claim{font-weight:500}.k{color:var(--muted)}.headline{font:600 15px var(--mono);
margin:10px 0;letter-spacing:.03em}.headline.warn{color:var(--warn)}.kv{display:grid;
grid-template-columns:max-content minmax(0,1fr);gap:6px 18px;margin:0}.kv dt{color:var(--muted)}
.kv dd{margin:0;font-family:var(--mono);overflow-wrap:anywhere}.tiles{display:grid;grid-template-columns:1fr 1fr;
gap:14px;margin-top:14px}.tile{display:flex;gap:12px;border:1px solid var(--line);
border-radius:10px;padding:14px;background:var(--bg)}.tile-title{font:600 14px var(--mono)}
.pip{flex:none;width:30px;height:30px;border-radius:8px;background:var(--chip)}
.pip.ok{background:var(--ok-soft);box-shadow:inset 0 0 0 2px var(--ok)}
.pip.warn{background:var(--warn-soft);box-shadow:inset 0 0 0 2px var(--warn)}
.pip.bad{background:var(--bad-soft);box-shadow:inset 0 0 0 2px var(--bad)}
.limits{color:var(--muted);font-size:14px;margin:14px 0 0;padding-left:18px}
.foot{max-width:1240px;margin:0 auto;padding:0 16px 28px;color:var(--muted)}
@media (max-width:900px){.metrics{grid-template-columns:1fr 1fr}.run-grid,.tiles{
grid-template-columns:1fr}}@media (max-width:520px){.metrics{grid-template-columns:1fr}
.top{padding:12px 16px}}
"""

_SCRIPT = """
(function(){var root=document.documentElement,b=document.getElementById("theme");
function dark(){var t=root.getAttribute("data-theme");if(t)return t==="dark";
return window.matchMedia("(prefers-color-scheme: dark)").matches}
function label(){b.textContent=dark()?"Light theme":"Dark theme"}
b.addEventListener("click",function(){root.setAttribute("data-theme",dark()?"light":"dark");
label()});label()})();
"""
