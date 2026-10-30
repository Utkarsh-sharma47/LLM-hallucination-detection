"""Glass Box generation analysis report with 10-section format.

This module provides the structured report output required by the task,
integrating token probabilities, hidden-state analysis, probe compatibility,
and factuality verification into a clean terminal presentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.analysis.capture import representation_change
from glassbox.analysis.claims import load_catalog, verify_answer
from glassbox.analysis.protocol import account_probe
from glassbox.models.backend import GenerationResult
from glassbox.probes.store import LoadedProbe
from glassbox.reporting import TurnRecord

_RULE = "=" * 60


@dataclass
class GlassBoxReport:
    """Complete generation analysis report."""

    question: str
    answer: str
    generation: GenerationResult
    turn: TurnRecord
    probe: LoadedProbe | None
    catalog_path: Path | None

    def to_text(self) -> str:
        """Return the 10-section terminal report."""
        lines = [
            _RULE,
            "GLASS BOX | LLM GENERATION ANALYSIS",
            _RULE,
            "",
            "[01] QUESTION",
            self.question,
            "",
            "[02] GENERATED ANSWER",
            self.answer,
            "",
            "[03] TOKEN GENERATION PROBABILITIES",
            self._token_probabilities_section(),
            "",
            "[04] FINAL TOKEN ALTERNATIVES",
            self._token_alternatives_section(),
            "",
            "[05] FACTUALITY VERIFICATION",
            self._factuality_section(),
            "",
            "[06] INTERNAL MODEL ANALYSIS",
            self._internal_model_section(),
            "",
            "[07] PROBE ANALYSIS",
            self._probe_section(),
            "",
            "[08] LAYER-WISE ANALYSIS",
            self._layer_section(),
            "",
            "[09] EARLY-WARNING ANALYSIS",
            self._early_warning_section(),
            "",
            "[10] FINAL CONCLUSION",
            self._conclusion_section(),
            _RULE,
            "",
        ]
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Return machine-readable report."""
        return {
            "question": self.question,
            "answer": self.answer,
            "token_probabilities": self._token_probabilities_json(),
            "token_alternatives": self._token_alternatives_json(),
            "generation_config": self._generation_config_json(),
            "hidden_state_metadata": self._hidden_state_metadata_json(),
            "probe": self._probe_json(),
            "factuality": self._factuality_json(),
            "calibration": self._calibration_json(),
            "conclusion": self._conclusion_json(),
        }

    def _token_probabilities_section(self) -> str:
        tokens = self.generation.tokens
        if not tokens:
            return "No tokens were generated."

        mean_prob = sum(t.probability for t in tokens) / len(tokens)
        mean_entropy = sum(t.entropy_nats for t in tokens) / len(tokens)
        mean_norm_entropy = sum(t.normalized_entropy for t in tokens) / len(tokens)

        # Geometric mean
        log_probs = [np.log(max(t.probability, 1e-10)) for t in tokens]
        geometric_mean = np.exp(np.mean(log_probs))

        lines = [
            f"Generated tokens: {len(tokens)}",
            f"Average selected-token probability: {mean_prob:.4f}",
            f"Geometric mean probability: {geometric_mean:.4f}",
            f"Average entropy (nats): {mean_entropy:.4f}",
            f"Average normalized entropy: {mean_norm_entropy:.4f}",
        ]

        # Show low-probability tokens
        low_prob = [t for t in tokens if t.probability < 0.5]
        if low_prob:
            lines.append("")
            lines.append("Low-probability tokens (< 50%):")
            for t in low_prob[:5]:
                lines.append(f"  Position {t.step}: '{t.text}' (p={t.probability:.4f})")

        return "\n".join(lines)

    def _token_alternatives_section(self) -> str:
        tokens = self.generation.tokens
        if not tokens:
            return "No tokens were generated."

        lines = ["Position | Selected Token | Probability | Top Alternatives"]
        lines.append("-" * 80)

        for token in tokens[:15]:  # Show first 15
            alt_str = ", ".join(
                f"{a.text} ({a.probability:.3f})" for a in token.top_alternatives[:3]
            )
            lines.append(f"{token.step:8} | {token.text:15} | {token.probability:.4f} | {alt_str}")

        if len(tokens) > 15:
            lines.append(f"... ({len(tokens) - 15} more tokens)")

        return "\n".join(lines)

    def _factuality_section(self) -> str:
        catalog = load_catalog(self.catalog_path)
        claims = verify_answer(self.answer, catalog)

        counts = claims.get("counts", {})
        overall = claims.get("overall", "UNVALIDATED")
        note = claims.get("note", "")

        lines = [
            f"Overall status: {overall}",
            f"Number of claims: {claims.get('claim_count', 0)}",
            f"Verified claims: {counts.get('FAITHFUL', 0)}",
            f"Hallucinated claims: {counts.get('HALLUCINATED', 0)}",
            f"Ambiguous claims: {counts.get('AMBIGUOUS', 0)}",
            f"Unvalidated claims: {counts.get('UNVALIDATED', 0)}",
            f"Reference: {self.catalog_path or 'NOT AVAILABLE'}",
            "",
            note,
        ]

        # Show individual claims
        claim_list = claims.get("claims", [])
        if claim_list:
            lines.append("")
            lines.append("Claim details:")
            for claim in claim_list:
                status = claim.get("status", "UNVALIDATED")
                text = claim.get("text", "")[:60]
                lines.append(f"  [{status}] {text}")

        return "\n".join(lines)

    def _internal_model_section(self) -> str:
        gen = self.generation
        lines = [
            f"Model: {gen.model_name}",
            f"Model revision: {gen.model_revision}",
            f"Captured layers: {gen.hidden_state_count}",
            f"Hidden-state dimension: {int(gen.feature.shape[0]) if gen.feature is not None else 'unavailable'}",
            f"Feature layer: {gen.feature_layer}",
            f"Token capture configuration: {gen.capture_mode}",
            f"Capture bytes: {gen.capture_bytes}",
            f"Capture note: {gen.capture_note}",
        ]
        return "\n".join(lines)

    def _probe_section(self) -> str:
        account = account_probe(self.probe)

        lines = [
            f"Probe protocol: {account.protocol or 'none'}",
            f"Raw experimental score: {self.turn.experimental_score if self.turn.experimental_score is not None else 'NOT AVAILABLE'}",
            f"Prediction: {self.turn.probe_prediction or 'NOT AVAILABLE'}",
            f"Calibration status: {self.turn.calibration_status}",
        ]

        if account.role == "pairwise_experimental":
            lines.append("")
            lines.append("PAIRWISE EXPERIMENTAL PROBE - NOT VALIDATED FOR GENERATION FACTUALITY")
            lines.append(account.note)
        elif account.role == "generation_experimental":
            lines.append("")
            lines.append("GENERATION PROBE EXPERIMENTAL - NOT CALIBRATED")
            lines.append(account.note)
        elif account.compatible_generation_detector:
            lines.append("")
            lines.append("GENERATION PROBE CALIBRATED")
            lines.append(account.note)
        else:
            lines.append("")
            lines.append(account.note)

        if not self.turn.factual_risk_score:
            lines.append("")
            lines.append("HALLUCINATION PROBABILITY: NOT AVAILABLE")

        return "\n".join(lines)

    def _layer_section(self) -> str:
        spans = self.generation.token_spans or {}
        if not spans:
            return "No layer-wise analysis available. Token spans not captured."

        lines = ["Layer | Cosine Distance | Interpretation"]
        lines.append("-" * 60)

        for layer in sorted(spans.keys()):
            change = representation_change(spans[layer])
            distance = change.get("cosine_distance_first_to_last")
            interpretation = change.get("interpretation", "")
            lines.append(
                f"{layer:5} | {distance if distance is not None else 'n/a':15} | {interpretation[:40]}"
            )

        return "\n".join(lines)

    def _early_warning_section(self) -> str:
        return (
            "Available: no\n"
            "Reason: No token-level probe is trained and no hallucination event is defined. "
            "A cosine change is not a lead time."
        )

    def _conclusion_section(self) -> str:
        catalog = load_catalog(self.catalog_path)
        claims = verify_answer(self.answer, catalog)
        overall = claims.get("overall", "UNVALIDATED")

        account = account_probe(self.probe)

        lines = [
            f"Factuality status: {overall}",
            f"Generation uncertainty: {'LOW' if self.generation.tokens and all(t.probability > 0.7 for t in self.generation.tokens) else 'MIXED'}",
            f"Probe reliability: {'CALIBRATED' if account.compatible_generation_detector else 'EXPERIMENTAL' if self.probe else 'UNAVAILABLE'}",
            f"Evidence status: {'AVAILABLE' if catalog else 'UNAVAILABLE'}",
            "",
            "Main limitations:",
            "  - Token probabilities are generation confidence, not factuality confidence",
            "  - High probability does not guarantee correctness",
            "  - Pairwise probe is not validated for generation factuality",
        ]

        return "\n".join(lines)

    def _token_probabilities_json(self) -> dict[str, Any]:
        tokens = self.generation.tokens
        if not tokens:
            return {"available": False}

        mean_prob = sum(t.probability for t in tokens) / len(tokens)
        log_probs = [np.log(max(t.probability, 1e-10)) for t in tokens]
        geometric_mean = np.exp(np.mean(log_probs))

        return {
            "available": True,
            "token_count": len(tokens),
            "average_selected_probability": mean_prob,
            "geometric_mean_probability": geometric_mean,
            "average_entropy": sum(t.entropy_nats for t in tokens) / len(tokens),
            "average_normalized_entropy": sum(t.normalized_entropy for t in tokens) / len(tokens),
            "tokens": [
                {
                    "position": t.step,
                    "text": t.text,
                    "probability": t.probability,
                    "entropy": t.entropy_nats,
                    "normalized_entropy": t.normalized_entropy,
                    "margin": t.margin,
                }
                for t in tokens
            ],
        }

    def _token_alternatives_json(self) -> dict[str, Any]:
        tokens = self.generation.tokens
        return {
            "available": bool(tokens),
            "tokens": [
                {
                    "position": t.step,
                    "selected_token": t.text,
                    "selected_probability": t.probability,
                    "alternatives": [
                        {"text": a.text, "probability": a.probability}
                        for a in t.top_alternatives[:5]
                    ],
                }
                for t in tokens
            ],
        }

    def _generation_config_json(self) -> dict[str, Any]:
        gen = self.generation
        return {
            "model_name": gen.model_name,
            "model_revision": gen.model_revision,
            "device": gen.device,
            "dtype": gen.dtype,
            "decoding": gen.decoding,
            "temperature": gen.temperature,
            "max_new_tokens": gen.generated_tokens,
            "prompt_tokens": gen.prompt_tokens,
            "latency_seconds": gen.latency_seconds,
        }

    def _hidden_state_metadata_json(self) -> dict[str, Any]:
        gen = self.generation
        return {
            "available": gen.hidden_states_available,
            "hidden_state_count": gen.hidden_state_count,
            "feature_layer": gen.feature_layer,
            "feature_dimension": int(gen.feature.shape[0]) if gen.feature is not None else None,
            "position": gen.position,
            "capture_mode": gen.capture_mode,
            "capture_bytes": gen.capture_bytes,
            "capture_note": gen.capture_note,
        }

    def _probe_json(self) -> dict[str, Any]:
        account = account_probe(self.probe)
        return {
            "available": self.probe is not None,
            "protocol": account.protocol,
            "role": account.role,
            "compatible_generation_detector": account.compatible_generation_detector,
            "calibration_validated": account.calibration_validated,
            "raw_score": self.turn.experimental_score,
            "prediction": self.turn.probe_prediction,
            "threshold": self.turn.probe_threshold,
            "calibration_status": self.turn.calibration_status,
            "note": account.note,
        }

    def _factuality_json(self) -> dict[str, Any]:
        catalog = load_catalog(self.catalog_path)
        claims = verify_answer(self.answer, catalog)
        return {
            "catalog_available": bool(catalog),
            "catalog_path": str(self.catalog_path) if self.catalog_path else None,
            "overall_status": claims.get("overall"),
            "claim_count": claims.get("claim_count"),
            "counts": claims.get("counts"),
            "claims": claims.get("claims"),
        }

    def _calibration_json(self) -> dict[str, Any]:
        account = account_probe(self.probe)
        if not self.probe:
            return {"available": False}

        metadata = self.probe.metadata if self.probe else {}
        return {
            "available": True,
            "calibrated": account.calibration_validated,
            "method": metadata.get("calibration"),
            "ece": metadata.get("test_ece"),
            "brier": metadata.get("brier"),
            "threshold": account.threshold,
        }

    def _conclusion_json(self) -> dict[str, Any]:
        catalog = load_catalog(self.catalog_path)
        claims = verify_answer(self.answer, catalog)
        account = account_probe(self.probe)

        return {
            "factuality_status": claims.get("overall"),
            "generation_uncertainty": "LOW"
            if self.generation.tokens and all(t.probability > 0.7 for t in self.generation.tokens)
            else "MIXED",
            "probe_reliability": "CALIBRATED"
            if account.compatible_generation_detector
            else "EXPERIMENTAL"
            if self.probe
            else "UNAVAILABLE",
            "evidence_status": "AVAILABLE" if catalog else "UNAVAILABLE",
            "limitations": [
                "Token probabilities are generation confidence, not factuality confidence",
                "High probability does not guarantee correctness",
                "Pairwise probe is not validated for generation factuality",
            ],
        }


def build_report(
    question: str,
    generation: GenerationResult,
    turn: TurnRecord,
    probe: LoadedProbe | None = None,
    catalog_path: Path | None = None,
) -> GlassBoxReport:
    """Build a Glass Box report from generation results."""
    return GlassBoxReport(
        question=question,
        answer=generation.answer,
        generation=generation,
        turn=turn,
        probe=probe,
        catalog_path=catalog_path,
    )
