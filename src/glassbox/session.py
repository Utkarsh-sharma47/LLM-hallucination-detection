"""A demo session that keeps every turn from the current process."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from glassbox.evaluation.records import write_json
from glassbox.models.backend import GenerationResult, GenerationSettings
from glassbox.models.distributions import write_distribution_csv, write_distribution_jsonl
from glassbox.probes.logistic import PROBE_NOT_TRAINED
from glassbox.probes.store import (
    LoadedProbe,
    ProbeCompatibilityError,
    check_compatibility,
    experimental_score,
)
from glassbox.reporting import TurnRecord, format_session_markdown
from glassbox.verification import (
    CuratedFact,
    ReferenceResult,
    assess_answer,
    find_fact,
    reference_for_question,
)

_SCORE_NOTE_UNTRAINED = (
    "No trained probe is loaded. Token probabilities, entropy, and top alternatives "
    "are not a hallucination score."
)
_SCORE_NOTE_EXPERIMENTAL = (
    "Experimental probe score from logistic regression on one frozen hidden state. "
    "It is not a calibrated hallucination probability."
)


@dataclass
class DemoSession:
    """Backend, optional probe, and the turns asked so far."""

    backend: object
    settings: GenerationSettings
    facts: tuple[CuratedFact, ...]
    probe: LoadedProbe | None = None
    turns: list[TurnRecord] = field(default_factory=list)
    export_count: int = 0

    def ask(self, question: str, on_token: object | None = None) -> TurnRecord:
        """Generate one answer and append it to the session."""
        result = self.backend.generate(question, self.settings, on_token=on_token)  # type: ignore[attr-defined]
        turn = TurnRecord(
            generation=result,
            probe_status=PROBE_NOT_TRAINED,
            experimental_score=None,
            score_note=_SCORE_NOTE_UNTRAINED,
            factual_risk_score=None,
            reference=_reference(question, result.answer, self.facts),
        )
        _apply_probe(turn, self.probe)
        self.turns.append(turn)
        return turn

    def export(self, root: Path) -> Path:
        """Write every turn in this session under ``root``."""
        if not self.turns:
            raise ValueError("There is no turn to export")
        self.export_count += 1
        directory = Path(root) / "sessions" / f"session-{self.export_count}"
        while directory.exists():
            self.export_count += 1
            directory = Path(root) / "sessions" / f"session-{self.export_count}"
        json_path = write_json(
            directory / "session.json",
            {"schema_version": 1, "turns": [_payload(turn) for turn in self.turns]},
            root=root,
        )
        markdown_path = json_path.with_suffix(".md")
        markdown_path.write_text(format_session_markdown(self.turns), encoding="utf-8")
        tokens = [token for turn in self.turns for token in turn.generation.tokens]
        write_distribution_csv(directory / "distributions-topk.csv", tokens, mode="topk")
        write_distribution_jsonl(directory / "distributions-topk.jsonl", tokens, mode="topk")
        if any(token.vocabulary_probabilities is not None for token in tokens):
            text_for_id = _vocabulary_text(self.backend)
            write_distribution_csv(
                directory / "distributions-full.csv",
                tokens,
                mode="full",
                text_for_id=text_for_id,
            )
            write_distribution_jsonl(
                directory / "distributions-full.jsonl",
                tokens,
                mode="full",
                text_for_id=text_for_id,
            )
        return json_path


def _vocabulary_text(backend: object):
    """Return a decoder for full-vocabulary export, or None when no tokenizer is loaded."""
    loaded = getattr(backend, "_loaded", None)
    tokenizer = getattr(loaded, "tokenizer", None)
    if tokenizer is None:
        return None
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert):
        return None

    def text_for_id(index: int) -> str:
        return str(convert(int(index)))

    return text_for_id


def _reference(question: str, answer: str, facts: tuple[CuratedFact, ...]) -> ReferenceResult:
    fact = find_fact(question, facts)
    if fact is None:
        return reference_for_question(question, facts)
    return assess_answer(fact, answer)


def _apply_probe(turn: TurnRecord, probe: LoadedProbe | None) -> None:
    if probe is None:
        return
    generation: GenerationResult = turn.generation
    if (
        generation.feature is None
        or generation.feature_layer is None
        or generation.hidden_state_count is None
    ):
        turn.probe_status = PROBE_NOT_TRAINED
        turn.score_note = "Hidden state was unavailable, so the saved probe was not applied."
        return
    try:
        check_compatibility(
            probe,
            model_name=generation.model_name,
            model_revision=generation.model_revision,
            layer_index=generation.feature_layer,
            hidden_state_count=generation.hidden_state_count,
            position=generation.position,
            prompt_mode=generation.prompt_mode,
            feature_dim=int(generation.feature.shape[0]),
            dtype=generation.dtype,
        )
        turn.experimental_score = experimental_score(probe, generation.feature)
        _record_probe_decision(turn, probe)
    except ProbeCompatibilityError as exc:
        turn.probe_status = "PROBE INCOMPATIBLE"
        turn.score_note = f"Saved probe was not applied: {exc}"
        turn.experimental_score = None
        turn.probe_prediction = None
        turn.factual_risk_score = None
        return
    extra = probe.metadata.get("metrics_note")
    note = _SCORE_NOTE_EXPERIMENTAL
    if isinstance(extra, str) and extra:
        note = f"{note} {extra}"
    if probe.calibrated:
        turn.probe_status = "calibrated probe loaded"
        turn.factual_risk_score = turn.experimental_score
        ece = probe.metadata.get("test_ece")
        ece_text = "not recorded" if not isinstance(ece, (int, float)) else f"{float(ece):.3f}"
        turn.probe_task = _probe_task(probe)
        turn.score_note = (
            "Calibrated probe output from Platt scaling fit on the validation split only. "
            f"Held-out ECE was {ece_text}. "
            "The positive class is the HaluEval hallucinated-answer label for a supplied answer, "
            "not a new judgment that this demo answer is false, and not a measured detector accuracy. "
            f"{note}"
        )
        return
    turn.probe_status = "experimental probe loaded"
    turn.calibration_status = "not validated"
    turn.probe_task = _probe_task(probe)
    turn.score_note = note


def _probe_task(probe: LoadedProbe) -> str:
    protocol = str(probe.metadata.get("label_protocol") or "")
    if protocol == "halueval_qa_pair_v1":
        return (
            "Task: paired-answer discrimination on supplied HaluEval answers. "
            "This is not a measurement of answers generated by Qwen."
        )
    if protocol:
        return f"Task: {protocol}."
    return "Task: the probe record does not name a label protocol."


def _record_probe_decision(turn: TurnRecord, probe: LoadedProbe) -> None:
    raw = probe.metadata.get("threshold")
    turn.probe_threshold = (
        float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
    )
    cutoff = 0.5 if turn.probe_threshold is None else turn.probe_threshold
    if turn.experimental_score is not None:
        turn.probe_prediction = "hallucinated" if turn.experimental_score >= cutoff else "faithful"
    turn.calibration_status = "validated" if probe.calibrated else "not validated"


def _payload(turn: TurnRecord) -> dict[str, object]:
    generation = turn.generation
    steps = [
        {
            "step": token.step,
            "text": token.text,
            "piece_id": token.token_id,
            "probability": token.probability,
            "conditioned_on": token.conditioned_on,
            "decoding": token.decoding,
            "temperature": token.temperature,
            "alternatives": [
                {
                    "text": alternative.text,
                    "piece_id": alternative.token_id,
                    "probability": alternative.probability,
                }
                for alternative in token.top_alternatives
            ],
        }
        for token in generation.tokens
    ]
    return {
        "question": generation.question,
        "answer": generation.answer,
        "model_name": generation.model_name,
        "model_revision": generation.model_revision,
        "device": generation.device,
        "dtype": generation.dtype,
        "latency_seconds": generation.latency_seconds,
        "prompt_count": generation.prompt_tokens,
        "generated_count": generation.generated_tokens,
        "output_steps": steps,
        "detector_status": turn.probe_status,
        "experimental_score": turn.experimental_score,
        "probe_prediction": turn.probe_prediction,
        "probe_threshold": turn.probe_threshold,
        "calibration_status": turn.calibration_status,
        "score_note": turn.score_note,
        "factual_risk_score": turn.factual_risk_score,
        "reference_status": turn.reference.status,
        "reference_detail": turn.reference.detail,
        "intervention": turn.intervention_status,
        "prompt_mode": generation.prompt_mode,
        "feature_layer": generation.feature_layer,
        "position": generation.position,
        "hidden_state_count": generation.hidden_state_count,
        "load_count": generation.load_count,
        "logits_available": generation.logits_available,
        "hidden_states_available": generation.hidden_states_available,
        "note": generation.note,
    }
