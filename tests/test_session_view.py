"""Session HTML view. These tests do not load Qwen or claim calibrated risk."""

from __future__ import annotations

from pathlib import Path

from glassbox.cli import main
from glassbox.evaluation.generation_calibration import RISK_STATUS, SCORE_NAME
from glassbox.reports.session_view import render_session_html


def test_html_keeps_experimental_banners_and_does_not_invent_a_probability() -> None:
    html = render_session_html(
        {
            "turns": [
                {
                    "question": "What is the capital of France?",
                    "answer": "The capital of France is Paris.",
                    "evidence_label": "FAITHFUL",
                    "experimental_score": 0.999,
                    "factual_risk_score": None,
                    "calibration_validated": False,
                    "calibration_status": "not validated",
                }
            ]
        }
    )
    assert SCORE_NAME in html
    assert RISK_STATUS in html
    assert "factual-risk probability" not in html
    assert "99.9%" not in html
    assert "Paris" in html
    assert "FAITHFUL" in html


def test_calibrated_turn_is_the_only_path_that_names_a_probability() -> None:
    html = render_session_html(
        {
            "turns": [
                {
                    "question": "Q",
                    "answer": "A",
                    "evidence_label": "HALLUCINATED",
                    "experimental_score": 0.8,
                    "factual_risk_score": 0.42,
                    "calibration_validated": True,
                }
            ]
        }
    )
    assert "factual-risk probability" in html
    assert "0.42" in html


def test_view_cli_writes_html(tmp_path: Path) -> None:
    source = tmp_path / "session.json"
    source.write_text(
        '{"turns":[{"question":"Q","answer":"A","experimental_score":0.2,'
        '"factual_risk_score":null,"calibration_validated":false}]}',
        encoding="utf-8",
    )
    code = main(["view-session", "--input", str(source)])
    assert code == 0
    rendered = (tmp_path / "session-view.html").read_text(encoding="utf-8")
    assert SCORE_NAME in rendered
    assert RISK_STATUS in rendered
