"""Structured generation report. Probabilities stay unavailable until calibration."""

from __future__ import annotations

from typing import Any

import numpy as np

from glassbox.analysis.capture import representation_change
from glassbox.analysis.protocol import (
    PROBABILITY_UNAVAILABLE,
    ProbeAccount,
    probability_lines,
)

_RULE = "=" * 50


def build_report(
    *,
    question: str,
    answer: str,
    claims: dict[str, object],
    pair_account: ProbeAccount,
    pair_score: float | None,
    generation_account: ProbeAccount,
    generation_score: float | None,
    generation_metadata: dict[str, Any] | None,
    model_name: str,
    model_revision: str,
    hidden_count: int | None,
    feature_layer: int | None,
    feature_dim: int | None,
    capture_mode: str,
    capture_bytes: int,
    capture_note: str,
    generated_count: int,
    latency_seconds: float,
    token_texts: tuple[str, ...],
    spans: dict[int, np.ndarray],
) -> dict[str, object]:
    """Assemble one analysis. Claim status does not read the probe score."""
    probabilities = probability_lines(generation_account, generation_metadata)
    layers = _layers(
        spans, feature_layer, pair_account, pair_score, generation_account, generation_score
    )
    tokens = _tokens(
        token_texts, spans.get(feature_layer) if feature_layer is not None else None, claims
    )
    return {
        "question": question,
        "answer": answer,
        "factuality": claims,
        "probabilities": probabilities,
        "pair_probe": {
            "role": pair_account.role,
            "protocol": pair_account.protocol,
            "status": pair_account.generation_status,
            "note": pair_account.note,
            "raw_score": pair_score,
            "threshold": pair_account.threshold,
        },
        "generation_probe": {
            "role": generation_account.role,
            "protocol": generation_account.protocol,
            "status": generation_account.generation_status,
            "note": generation_account.note,
            "raw_score": generation_score,
            "threshold": generation_account.threshold,
            "compatible_detector": generation_account.compatible_generation_detector,
        },
        "capture": {
            "model_name": model_name,
            "model_revision": model_revision,
            "hidden_state_count": hidden_count,
            "feature_layer": feature_layer,
            "feature_dimension": feature_dim,
            "capture_mode": capture_mode,
            "capture_bytes": capture_bytes,
            "capture_note": capture_note,
            "generated_count": generated_count,
            "latency_seconds": latency_seconds,
            "extra_forward": "The final-token read is the same forward that supplies these rows.",
        },
        "layers": layers,
        "pieces": tokens,
        "early_warning": {
            "available": False,
            "reason": (
                "No token-level probe is trained and no hallucination event is defined. "
                "A cosine change is not a lead time."
            ),
        },
        "conclusion": _conclusion(claims, pair_account, generation_account),
    }


def format_report(report: dict[str, object]) -> str:
    """Plain-text report for a terminal."""
    claims = _mapping(report.get("factuality"))
    probabilities = _mapping(report.get("probabilities"))
    pair = _mapping(report.get("pair_probe"))
    generation = _mapping(report.get("generation_probe"))
    capture = _mapping(report.get("capture"))
    early = _mapping(report.get("early_warning"))
    conclusion = _mapping(report.get("conclusion"))
    counts = _mapping(claims.get("counts"))
    lines = [
        _RULE,
        "GLASS BOX | GENERATION ANALYSIS",
        _RULE,
        "",
        "[01] QUESTION",
        str(report.get("question") or ""),
        "",
        "[02] GENERATED ANSWER",
        str(report.get("answer") or ""),
        "",
        "[03] FACTUALITY VERIFICATION",
        f"Overall status: {claims.get('overall')}",
        f"Claim count: {claims.get('claim_count')}",
        f"Verified claims: {counts.get('FAITHFUL', 0)}",
        f"Contradicted claims: {counts.get('HALLUCINATED', 0)}",
        f"Ambiguous claims: {counts.get('AMBIGUOUS', 0)}",
        f"Unvalidated claims: {counts.get('UNVALIDATED', 0)}",
        f"Note: {claims.get('note')}",
    ]
    for index, claim in enumerate(_rows(claims.get("claims")), start=1):
        lines.append(
            f"  {index}. {claim.get('status')} | {claim.get('text')} | "
            f"source={claim.get('source') or 'none'}"
        )
        if claim.get("support"):
            lines.append(f"     support: {claim.get('support')}")
        if claim.get("contradiction"):
            lines.append(f"     contradiction: {claim.get('contradiction')}")
    lines.extend(
        [
            "",
            "[04] HALLUCINATION PROBABILITY",
            f"Estimated probability: {probabilities.get('hallucination_probability', PROBABILITY_UNAVAILABLE)}",
            f"Factuality probability: {probabilities.get('factuality_probability', PROBABILITY_UNAVAILABLE)}",
            f"Calibration status: {probabilities.get('calibration_status')}",
            f"Generation probe: {generation.get('status')}",
            f"Pairwise experimental score: {_score(pair.get('raw_score'))}",
            f"Generation raw score: {_score(generation.get('raw_score'))}",
            str(pair.get("note") or ""),
            str(generation.get("note") or ""),
            "",
            "[05] INTERNAL MODEL ANALYSIS",
            f"Model: {capture.get('model_name')}",
            f"Model revision: {capture.get('model_revision')}",
            f"Transformer layers captured: {capture.get('hidden_state_count')}",
            f"Hidden-state dimension: {capture.get('feature_dimension')}",
            f"Feature layer: {capture.get('feature_layer')}",
            f"Token capture mode: {capture.get('capture_mode')}",
            f"Generation length: {capture.get('generated_count')}",
            f"Captured rows: {capture.get('capture_bytes')} bytes",
            f"Generation latency: {capture.get('latency_seconds')}",
            str(capture.get("extra_forward") or ""),
            str(capture.get("capture_note") or ""),
            "",
            "[06] LAYER-WISE ANALYSIS",
            "Layer | Probe | Final-token score | First-to-last cosine distance | Interpretation",
        ]
    )
    for layer in _rows(report.get("layers")):
        lines.append(
            f"{layer.get('layer')} | {layer.get('probe')} | {_score(layer.get('score'))} | "
            f"{_score(layer.get('cosine_distance'))} | {layer.get('interpretation')}"
        )
    lines.extend(
        [
            "",
            "[07] TOKEN-LEVEL ANALYSIS",
            "Index | Piece | Drift from first piece | Evidence overlap",
        ]
    )
    pieces = _rows(report.get("pieces"))
    shown = pieces[:12]
    for piece in shown:
        lines.append(
            f"{piece.get('index')} | {piece.get('text')!s} | {_score(piece.get('drift'))} | "
            f"{piece.get('evidence')}"
        )
    if len(pieces) > len(shown):
        lines.append(f"... {len(pieces) - len(shown)} more pieces. Use tokens or export.")
    lines.extend(
        [
            "",
            "[08] EARLY-WARNING ANALYSIS",
            "Available: no" if early.get("available") is not True else "Available: yes",
            str(early.get("reason") or ""),
            "",
            "[09] FINAL CONCLUSION",
            f"Factuality status: {conclusion.get('factuality_status')}",
            f"Probe status: {conclusion.get('probe_status')}",
            f"Calibration status: {conclusion.get('calibration_status')}",
            str(conclusion.get("explanation") or ""),
            _RULE,
            "",
        ]
    )
    return "\n".join(lines)


def format_section(report: dict[str, object], section: str) -> str:
    """Return one interactive view of a stored report."""
    if section == "summary":
        return format_report(report)
    if section == "claims":
        return format_report(report).split("[04] HALLUCINATION PROBABILITY")[0]
    if section == "evidence":
        claims = _mapping(report.get("factuality"))
        lines = ["[EVIDENCE]"]
        for claim in _rows(claims.get("claims")):
            lines.append(f"{claim.get('status')} | {claim.get('text')}")
            lines.append(f"  source: {claim.get('source') or 'none'}")
            lines.append(f"  support: {claim.get('support') or 'none'}")
            lines.append(f"  contradiction: {claim.get('contradiction') or 'none'}")
        return "\n".join(lines) + "\n"
    if section == "probe":
        pair = _mapping(report.get("pair_probe"))
        generation = _mapping(report.get("generation_probe"))
        probabilities = _mapping(report.get("probabilities"))
        return "\n".join(
            [
                "[PROBE]",
                f"Hallucination probability: {probabilities.get('hallucination_probability')}",
                f"Factuality probability: {probabilities.get('factuality_probability')}",
                f"Pair protocol: {pair.get('protocol') or 'none'}",
                f"Pair raw score: {_score(pair.get('raw_score'))}",
                str(pair.get("note") or ""),
                f"Generation status: {generation.get('status')}",
                f"Generation raw score: {_score(generation.get('raw_score'))}",
                str(generation.get("note") or ""),
                "",
            ]
        )
    if section == "layers":
        lines = ["[LAYERS]", "Layer | Probe | Score | Cosine distance | Interpretation"]
        for layer in _rows(report.get("layers")):
            lines.append(
                f"{layer.get('layer')} | {layer.get('probe')} | {_score(layer.get('score'))} | "
                f"{_score(layer.get('cosine_distance'))} | {layer.get('interpretation')}"
            )
        return "\n".join(lines) + "\n"
    if section == "tokens":
        lines = ["[TOKENS]", "Index | Piece | Drift | Evidence overlap"]
        for piece in _rows(report.get("pieces")):
            lines.append(
                f"{piece.get('index')} | {piece.get('text')!s} | {_score(piece.get('drift'))} | "
                f"{piece.get('evidence')}"
            )
        return "\n".join(lines) + "\n"
    raise ValueError(f"Unknown section {section}")


def _layers(
    spans: dict[int, np.ndarray],
    feature_layer: int | None,
    pair_account: ProbeAccount,
    pair_score: float | None,
    generation_account: ProbeAccount,
    generation_score: float | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer in sorted(spans):
        change = representation_change(spans[layer])
        probe = "no probe trained for this layer"
        score = None
        if feature_layer is not None and layer == feature_layer and pair_score is not None:
            probe = "pairwise experimental, final piece only"
            score = pair_score
        elif feature_layer is not None and layer == feature_layer and generation_score is not None:
            probe = generation_account.generation_status
            score = generation_score
        interpretation = str(change.get("interpretation") or change.get("reason") or "")
        if score is not None and pair_account.role == "pairwise_experimental":
            interpretation = (
                "Pairwise experimental score at the final generated piece. " + interpretation
            )
        rows.append(
            {
                "layer": layer,
                "probe": probe,
                "score": score,
                "cosine_distance": change.get("cosine_distance_first_to_last"),
                "interpretation": interpretation,
            }
        )
    if not rows:
        rows.append(
            {
                "layer": feature_layer,
                "probe": "span not captured",
                "score": pair_score,
                "cosine_distance": None,
                "interpretation": "Final-token capture does not record a cross-token change.",
            }
        )
    return rows


def _tokens(
    texts: tuple[str, ...],
    span: np.ndarray | None,
    claims: dict[str, object],
) -> list[dict[str, object]]:
    claim_rows = _rows(claims.get("claims"))
    rows: list[dict[str, object]] = []
    first = None if span is None or span.ndim != 2 or span.shape[0] < 1 else span[0]
    for index, text in enumerate(texts):
        drift = None
        if first is not None and span is not None and index < span.shape[0]:
            drift = _cosine_distance(first, span[index])
        rows.append(
            {
                "index": index,
                "text": text,
                "drift": drift,
                "evidence": _overlap(text, claim_rows),
            }
        )
    return rows


def _overlap(text: str, claims: list[dict[str, object]]) -> str:
    stripped = text.strip()
    if len(stripped) < 3:
        return "not aligned"
    for claim in claims:
        claim_text = str(claim.get("text") or "")
        if stripped.lower() in claim_text.lower():
            status = claim.get("status")
            claim_id = claim.get("claim_id") or "unlisted"
            return f"overlaps {claim_id} ({status})"
    return "not aligned"


def _cosine_distance(first: np.ndarray, current: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(first) * np.linalg.norm(current))
    if denom == 0.0:
        return None
    return float(1.0 - np.dot(first, current) / denom)


def _conclusion(
    claims: dict[str, object],
    pair_account: ProbeAccount,
    generation_account: ProbeAccount,
) -> dict[str, str]:
    factuality = str(claims.get("overall") or "UNVALIDATED")
    if (
        generation_account.compatible_generation_detector
        or generation_account.role == "generation_experimental"
    ):
        probe_status = generation_account.generation_status
    else:
        probe_status = generation_account.generation_status
    return {
        "factuality_status": factuality,
        "probe_status": probe_status,
        "calibration_status": "not validated"
        if not generation_account.calibration_validated
        else "validated",
        "explanation": (
            f"Factuality is {factuality} from the claim catalog, not from the probe. "
            f"{pair_account.note} {generation_account.note} "
            "An uncalibrated logistic output is not a hallucination probability."
        ),
    }


def _score(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.3f}"


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
