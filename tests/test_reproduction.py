"""Dirty worktrees and a high ECE are not a finished, calibrated reproduction.

These tests do not load Qwen and do not read the gitignored probe artifact.
"""

from __future__ import annotations

import pytest

from glassbox.probes.scoring import CALIBRATION_ECE_LIMIT, calibration_is_validated
from glassbox.reproducibility import mark_reproduction, pair_record_notes


def test_dirty_status_is_not_a_finished_reproduction() -> None:
    with pytest.raises(ValueError, match="finished reproduction"):
        mark_reproduction(dirty=True, finished_reproduction=True)
    claim = mark_reproduction(dirty=True, finished_reproduction=False)
    assert claim["worktree_dirty"] is True
    assert claim["reproducible"] is False
    assert claim["finished_reproduction"] is False
    notes = pair_record_notes(
        dirty=True,
        calibration_validated=False,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
    )
    assert "not a finished reproduction" in notes
    assert "lexical overlap" in notes


def test_clean_status_can_be_marked_finished() -> None:
    claim = mark_reproduction(dirty=False, finished_reproduction=True)
    assert claim["reproducible"] is True
    assert claim["finished_reproduction"] is True
    assert claim["worktree_dirty"] is False


def test_ece_above_the_limit_keeps_calibration_unvalidated() -> None:
    assert CALIBRATION_ECE_LIMIT == 0.10
    assert calibration_is_validated(0.114, validation_ready=True, test_ready=True) is False
    assert calibration_is_validated(0.10, validation_ready=True, test_ready=True) is True
    assert calibration_is_validated(0.0, validation_ready=False, test_ready=True) is False
