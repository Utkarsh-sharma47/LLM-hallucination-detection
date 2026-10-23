"""Softmax and top-k probabilities come from the full vocabulary."""

import numpy as np

from glassbox.models.distributions import (
    TokenAlternative,
    alternatives_from_logits,
    entropy_nats,
    normalized_entropy,
    probabilities_from_logits,
)


def test_softmax_is_stable_and_sums_to_one():
    probabilities = probabilities_from_logits(np.array([1000.0, 1000.0, 0.0]))
    assert probabilities.shape == (3,)
    assert np.allclose(probabilities[:2], [0.5, 0.5])
    assert probabilities.sum() == np.float64(1.0) or np.isclose(probabilities.sum(), 1.0)


def test_top_k_keeps_full_vocabulary_probabilities():
    logits = np.zeros(4)
    probability, alternatives = alternatives_from_logits(
        logits,
        token_id=0,
        top_k=2,
        text_for_id=lambda index: f"t{index}",
    )
    assert probability == np.float64(0.25) or np.isclose(probability, 0.25)
    assert len(alternatives) == 2
    assert sum(item.probability for item in alternatives) == np.float64(0.5) or np.isclose(
        sum(item.probability for item in alternatives), 0.5
    )
    assert {item.token_id for item in alternatives} <= {0, 1, 2, 3}


def test_chosen_probability_is_not_renormalized_over_top_k():
    logits = np.array([2.0, 1.0, 0.0, -1.0])
    full = probabilities_from_logits(logits)
    probability, alternatives = alternatives_from_logits(
        logits, token_id=1, top_k=2, text_for_id=str
    )
    assert np.isclose(probability, full[1])
    assert np.isclose(alternatives[0].probability, full[int(np.argmax(full))])
    assert not np.isclose(sum(item.probability for item in alternatives), 1.0)


def test_entropy_of_a_one_hot_distribution_is_zero():
    peaked = np.array([1.0, 0.0, 0.0])
    assert entropy_nats(peaked) == 0.0
    assert normalized_entropy(peaked) == 0.0


def test_alternative_has_no_hallucination_field():
    alternative = TokenAlternative(token_id=1, text="a", probability=0.2)
    assert set(alternative.__dataclass_fields__) == {"token_id", "text", "probability"}
