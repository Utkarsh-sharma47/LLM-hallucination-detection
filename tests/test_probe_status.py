"""An untrained probe must refuse to invent a score."""

import numpy as np
import pytest

from glassbox.probes.logistic import PROBE_NOT_TRAINED, LogisticProbe, ProbeNotTrainedError


def test_untrained_probe_refuses_a_score():
    probe = LogisticProbe()
    with pytest.raises(ProbeNotTrainedError) as caught:
        probe.predict_proba(np.zeros(4))
    assert str(caught.value) == PROBE_NOT_TRAINED
    assert PROBE_NOT_TRAINED == "PROBE NOT TRAINED"


def test_trained_probe_returns_a_probability_in_unit_interval():
    features = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    probe = LogisticProbe().fit(features, labels)
    scores = probe.predict_proba(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert scores.shape == (2,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
