"""Generation calibration gate. These tests do not load Qwen."""

from __future__ import annotations

from pathlib import Path

from glassbox.data.schema import Label
from glassbox.evaluation.generation_calibration import (
    GENERATION_CALIBRATION_ECE_LIMIT,
    GENERATION_CALIBRATION_MIN_PER_CLASS,
    PAIR_THRESHOLD,
    RISK_STATUS,
    SCORE_NAME,
    evaluate_generation_calibration_gate,
)
from glassbox.probes.scoring import CALIBRATION_ECE_LIMIT


def test_pre_registered_constants_are_not_lowered() -> None:
    assert PAIR_THRESHOLD == 0.65
    assert GENERATION_CALIBRATION_MIN_PER_CLASS == 20
    assert GENERATION_CALIBRATION_ECE_LIMIT == 0.10
    assert CALIBRATION_ECE_LIMIT == 0.10


def test_five_label_pilot_is_ineligible() -> None:
    rows = _rows(
        faithful=3, hallucinated=2, unvalidated=35, score_faithful=0.1, score_hallucinated=0.9
    )
    decision = evaluate_generation_calibration_gate(rows, is_five_label_pilot=True)
    assert decision["status"] == "rejected"
    assert decision["calibration_validated"] is False
    assert decision["factual_risk_available"] is False
    assert decision["score_name"] == SCORE_NAME
    assert decision["risk_status"] == RISK_STATUS
    assert "pilot" in str(decision["reason"]).lower()
    assert decision["pair_probe_rewritten"] is False


def test_below_twenty_per_class_leaves_risk_unassessed(tmp_path: Path) -> None:
    rows = _rows(
        faithful=19, hallucinated=20, unvalidated=4, score_faithful=0.1, score_hallucinated=0.9
    )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        '{"calibration_validated": false, "threshold": 0.65}', encoding="utf-8"
    )
    before = metadata_path.read_text(encoding="utf-8")
    decision = evaluate_generation_calibration_gate(
        rows,
        pair_probe_metadata={"calibration_validated": False, "threshold": 0.65},
        pair_probe_path=metadata_path,
        score_source="pair_probe",
    )
    assert decision["calibration_validated"] is False
    assert decision["factual_risk_available"] is False
    assert decision["score_name"] == SCORE_NAME
    assert decision["risk_status"] == RISK_STATUS
    assert "below 20" in str(decision["reason"])
    assert metadata_path.read_text(encoding="utf-8") == before
    assert decision["pair_probe_rewritten"] is False
    assert decision["pair_probe_calibration_validated"] is False


def test_ece_above_limit_leaves_risk_unassessed(tmp_path: Path) -> None:
    rows = _rows(
        faithful=20, hallucinated=20, unvalidated=5, score_faithful=0.95, score_hallucinated=0.95
    )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"calibration_validated": false}', encoding="utf-8")
    before = metadata_path.read_text(encoding="utf-8")
    decision = evaluate_generation_calibration_gate(
        rows,
        pair_probe_metadata={"calibration_validated": False},
        pair_probe_path=metadata_path,
        score_source="pair_probe",
    )
    assert decision["ece"] is not None
    assert decision["ece"] > GENERATION_CALIBRATION_ECE_LIMIT
    assert decision["calibration_validated"] is False
    assert decision["factual_risk_available"] is False
    assert decision["risk_status"] == RISK_STATUS
    assert metadata_path.read_text(encoding="utf-8") == before
    assert decision["sample_counts"]["unvalidated"] == 5
    assert decision["sample_counts"]["eligible_faithful"] == 20
    assert decision["sample_counts"]["eligible_hallucinated"] == 20


def test_unvalidated_and_ambiguous_stay_out_of_the_denominator() -> None:
    rows = _rows(
        faithful=20, hallucinated=20, unvalidated=10, score_faithful=0.0, score_hallucinated=1.0
    )
    rows.extend(_row("ambiguous", Label.AMBIGUOUS, 0.99) for _ in range(8))
    rows.append(_row("missing", Label.FAITHFUL, None))
    decision = evaluate_generation_calibration_gate(rows, score_source="pair_probe")
    assert decision["sample_counts"]["unvalidated"] == 10
    assert decision["sample_counts"]["ambiguous"] == 8
    assert decision["sample_counts"]["eligible_faithful"] == 20
    assert decision["sample_counts"]["eligible_hallucinated"] == 20
    assert decision["ece"] == 0.0
    assert decision["threshold"] == 0.65


def test_a_refit_threshold_is_refused() -> None:
    rows = _rows(
        faithful=20, hallucinated=20, unvalidated=0, score_faithful=0.0, score_hallucinated=1.0
    )
    decision = evaluate_generation_calibration_gate(rows, threshold=0.25)
    assert decision["status"] == "rejected"
    assert "0.65" in str(decision["reason"])
    assert decision["calibration_validated"] is False


def test_generation_probe_scores_cannot_open_the_gate() -> None:
    rows = _rows(
        faithful=20, hallucinated=20, unvalidated=0, score_faithful=0.02, score_hallucinated=0.98
    )
    decision = evaluate_generation_calibration_gate(rows, threshold=PAIR_THRESHOLD)
    assert decision["status"] == "rejected"
    assert decision["calibration_validated"] is False
    assert decision["factual_risk_available"] is False
    assert decision["ece"] is None
    assert decision["score_name"] == SCORE_NAME
    assert "pair-probe" in str(decision["reason"])
    accepted = evaluate_generation_calibration_gate(
        rows, threshold=PAIR_THRESHOLD, score_source="pair_probe"
    )
    assert accepted["status"] == "accepted"
    assert accepted["calibration_validated"] is True
    assert accepted["ece"] < GENERATION_CALIBRATION_ECE_LIMIT


def test_held_out_rows_without_a_validation_split_are_ineligible() -> None:
    rows = _rows(
        faithful=20, hallucinated=20, unvalidated=0, score_faithful=0.0, score_hallucinated=1.0
    )
    for row in rows:
        row["split"] = "test"
    decision = evaluate_generation_calibration_gate(rows)
    assert decision["status"] == "rejected"
    assert decision["factual_risk_available"] is False
    assert "validation split" in str(decision["reason"])


def _rows(
    *,
    faithful: int,
    hallucinated: int,
    unvalidated: int,
    score_faithful: float,
    score_hallucinated: float,
) -> list[dict[str, object]]:
    rows = [_row(f"f{index}", Label.FAITHFUL, score_faithful) for index in range(faithful)]
    rows.extend(
        _row(f"h{index}", Label.HALLUCINATED, score_hallucinated) for index in range(hallucinated)
    )
    rows.extend(_row(f"u{index}", Label.UNVALIDATED, 0.8) for index in range(unvalidated))
    return rows


def _row(question_id: str, label: Label, score: float | None) -> dict[str, object]:
    return {
        "question_id": question_id,
        "split": "validation",
        "label": label.name,
        "probe_score": score,
        "split_overlap": False,
    }
