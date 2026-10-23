"""Scientific constraints that stay attached to the code, not only the docs."""

from __future__ import annotations

RESEARCH_CAVEATS: tuple[str, ...] = (
    "Dataset labels require validation before they are used as ground truth.",
    "A wording mismatch with a reference answer is not a hallucination label.",
    (
        "Probe predictability does not establish causality and does not prove "
        "that the model knows an answer is wrong."
    ),
)
