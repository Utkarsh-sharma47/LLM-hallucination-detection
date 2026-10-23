"""Next-token probabilities from logits.

The softmax is taken over the whole vocabulary. Top-k tokens are a view of that
distribution, not a separate score, and they are not a hallucination detector.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TokenAlternative:
    """One vocabulary entry from a next-token distribution."""

    token_id: int
    text: str
    probability: float


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    """Return a 1-D softmax distribution that sums to one."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("logits must be a non-empty 1-D vector")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = float(exp_values.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("softmax is undefined for these logits")
    return exp_values / total


def top_k_ids(probabilities: np.ndarray, k: int) -> np.ndarray:
    """Return the ``k`` most probable ids, highest probability first."""
    if k < 1:
        raise ValueError("k must be at least 1")
    values = np.asarray(probabilities)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities must be a non-empty 1-D vector")
    width = min(k, int(values.size))
    chosen = np.argpartition(values, -width)[-width:]
    return chosen[np.argsort(values[chosen])[::-1]]


def entropy_nats(probabilities: np.ndarray) -> float:
    """Shannon entropy in nats. This is uncertainty, not a truth score."""
    usable = probabilities[probabilities > 0]
    if usable.size == 0:
        return 0.0
    return float(-np.sum(usable * np.log(usable)))


def normalized_entropy(probabilities: np.ndarray) -> float:
    """Entropy divided by log(vocab size). Uniform is 1 and a one-hot is 0."""
    if probabilities.size <= 1:
        return 0.0
    return float(entropy_nats(probabilities) / np.log(probabilities.size))


def scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return the logits that the decoder turns into a distribution.

    Temperature 0 leaves the logits unchanged and the caller takes the argmax.
    A positive temperature divides the logits before the softmax, which is the
    distribution sampling draws from.
    """
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("logits must be a non-empty 1-D vector")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a number")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0:
        return values
    return values / float(temperature)


def other_candidates(
    probabilities: np.ndarray,
    token_id: int,
    *,
    count: int,
    text_for_id,
) -> tuple[TokenAlternative, ...]:
    """Return ``count`` high-probability tokens other than the one that was written.

    Probabilities are the full-vocabulary values. They are not renormalized
    over this short list.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be an integer >= 1")
    values = np.asarray(probabilities)
    if token_id < 0 or token_id >= values.shape[0]:
        raise ValueError("token_id is outside the distribution")
    width = min(values.shape[0], count + 1)
    found: list[TokenAlternative] = []
    for index in top_k_ids(values, width):
        piece = int(index)
        if piece == int(token_id):
            continue
        found.append(
            TokenAlternative(
                token_id=piece,
                text=str(text_for_id(piece)),
                probability=float(values[piece]),
            )
        )
        if len(found) == count:
            break
    return tuple(found)


def alternatives_from_logits(
    logits: np.ndarray,
    token_id: int,
    *,
    top_k: int,
    text_for_id,
) -> tuple[float, tuple[TokenAlternative, ...]]:
    """Return the chosen token's probability and the top-k alternatives."""
    probabilities = probabilities_from_logits(logits)
    if token_id < 0 or token_id >= probabilities.shape[0]:
        raise ValueError("token_id is outside the logit vector")
    ranking = top_k_ids(probabilities, top_k)
    alternatives = tuple(
        TokenAlternative(
            token_id=int(index),
            text=str(text_for_id(int(index))),
            probability=float(probabilities[index]),
        )
        for index in ranking
    )
    return float(probabilities[token_id]), alternatives


_DISTRIBUTION_FIELDS = (
    "step",
    "piece_id",
    "decoded",
    "probability",
    "selected",
    "conditioned_on",
    "decoding",
    "temperature",
)


def distribution_rows(token, *, mode: str, text_for_id=None) -> list[dict[str, object]]:
    """Rows for one generated step. ``piece_id`` is the vocabulary id.

    ``topk`` writes the selected token and the stored alternatives.
    ``full`` writes every vocabulary id from ``vocabulary_probabilities``.
    Neither view is a probability of an input token.
    """
    if mode not in {"topk", "full"}:
        raise ValueError("mode must be topk or full")
    if mode == "full":
        return _full_rows(token, text_for_id)
    rows = [_row(token, token.token_id, token.text, token.probability, selected=1)]
    for alternative in token.top_alternatives:
        if alternative.token_id == token.token_id:
            continue
        rows.append(
            _row(token, alternative.token_id, alternative.text, alternative.probability, selected=0)
        )
    return rows


def write_distribution_csv(path: Path, tokens, *, mode: str, text_for_id=None) -> Path:
    """Write a distribution table. The terminal view does not use this file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_DISTRIBUTION_FIELDS)
        writer.writeheader()
        for token in tokens:
            for row in distribution_rows(token, mode=mode, text_for_id=text_for_id):
                writer.writerow(row)
    return destination


def write_distribution_jsonl(path: Path, tokens, *, mode: str, text_for_id=None) -> Path:
    """Write one JSON object per distribution row."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for token in tokens:
            for row in distribution_rows(token, mode=mode, text_for_id=text_for_id):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return destination


def _full_rows(token, text_for_id) -> list[dict[str, object]]:
    values = getattr(token, "vocabulary_probabilities", None)
    if values is None:
        raise ValueError("full export needs vocabulary_probabilities on each generated step")
    rows: list[dict[str, object]] = []
    for piece_id, probability in enumerate(np.asarray(values)):
        decoded = str(piece_id) if text_for_id is None else str(text_for_id(int(piece_id)))
        selected = 1 if int(piece_id) == int(token.token_id) else 0
        rows.append(_row(token, piece_id, decoded, float(probability), selected=selected))
    return rows


def _row(
    token, piece_id: int, decoded: str, probability: float, *, selected: int
) -> dict[str, object]:
    return {
        "step": int(getattr(token, "step", 1)),
        "piece_id": int(piece_id),
        "decoded": decoded.replace("\n", "\\n").replace("\r", "\\r"),
        "probability": float(probability),
        "selected": int(selected),
        "conditioned_on": str(getattr(token, "conditioned_on", "prompt")),
        "decoding": str(getattr(token, "decoding", "greedy")),
        "temperature": float(getattr(token, "temperature", 0.0)),
    }
