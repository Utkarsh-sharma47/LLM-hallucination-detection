"""Registered representation comparisons. None of these are promoted detectors.

The question split is the generation split already frozen in
``halueval-generation-001``. A variant without a saved probe stays unrun.
This module does not train one.
"""

from __future__ import annotations

VARIANTS = (
    {
        "variant_id": "final-token",
        "question": "Does the final generated piece detect factual errors?",
        "status": "RUN_NOT_VALIDATED",
        "evidence": "halueval-generation-001 test AUROC 0.646, below TF-IDF 0.657.",
    },
    {
        "variant_id": "mean-pooled-generated",
        "question": "Does the mean of generated pieces do better than the final piece?",
        "status": "AUDIT_ONLY",
        "evidence": "halueval-generation-001 validation mean-pool AUROC 0.622. It did not replace the saved probe.",
    },
    {
        "variant_id": "layer-specific",
        "question": "Does another layer beat the pre-registered last layer?",
        "status": "NOT_RUN",
        "evidence": "No layer-specific probe has been trained. Untrained layers are not scored.",
    },
    {
        "variant_id": "multi-layer",
        "question": "Does concatenating layers beat one layer?",
        "status": "NOT_RUN",
        "evidence": "No multi-layer probe artifact exists.",
    },
    {
        "variant_id": "token-signal",
        "question": "Does a score change before the answer ends?",
        "status": "NOT_RUN",
        "evidence": "No token-level probe and no event token. Lead time stays unset.",
    },
    {
        "variant_id": "length-controlled",
        "question": "Does a probe beat length when answer length is matched?",
        "status": "NOT_RUN",
        "evidence": "The live pair probe tracks sentence length. A length-controlled retraining has not been run.",
    },
    {
        "variant_id": "paraphrase-robustness",
        "question": "Does the same question wording change the score without changing the claim?",
        "status": "OBSERVED_FAILURE",
        "evidence": "live-terminal-001: 'Paris' scored near 0 and 'The capital of France is Paris.' scored 1.0.",
    },
    {
        "variant_id": "matched-answers",
        "question": "On the same question, is the false answer scored above the true answer?",
        "status": "NOT_RUN",
        "evidence": "No new matched-pair generation set has been scored since the pair protocol.",
    },
)


def variant_report() -> dict[str, object]:
    """Return the registry. Nothing here is a promoted detector."""
    return {
        "split": "halueval-generation-001 knowledge-group-v1 seed 0",
        "promoted": False,
        "variants": list(VARIANTS),
        "note": (
            "Identical question groups are required before any future comparison. "
            "This registry does not train or overwrite a probe."
        ),
    }
