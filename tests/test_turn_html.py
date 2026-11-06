"""HTML turn page: escaping, honest probability line, markers, and file writing."""

from __future__ import annotations

from datetime import datetime

from glassbox.reports import turn_html
from glassbox.reports.turn_html import render_turn_html, write_turn_html


def _report(**overrides):
    report = {
        "question": "Is <b>this</b> escaped?",
        "answer": "Yes & it is.",
        "token_probabilities": {
            "available": True,
            "average_selected_probability": 0.9,
            "geometric_mean_probability": 0.88,
            "average_entropy": 0.2,
            "tokens": [
                {"position": 1, "text": "Yes", "probability": 0.95, "margin": 0.9},
                {"position": 2, "text": " so", "probability": 0.62, "margin": 0.3},
                {"position": 3, "text": " maybe", "probability": 0.4, "margin": 0.35},
            ],
        },
        "token_alternatives": {
            "available": True,
            "tokens": [
                {
                    "position": 1,
                    "selected_token": "Yes",
                    "selected_probability": 0.95,
                    "alternatives": [{"text": "No", "probability": 0.05}],
                },
                {
                    "position": 2,
                    "selected_token": " so",
                    "selected_probability": 0.62,
                    "alternatives": [{"text": " and", "probability": 0.32}],
                },
            ],
        },
        "generation_config": {"model_name": "example-model", "latency_seconds": 0.5},
        "probe": {"available": False},
        "calibrated_hallucination_probability": None,
        "factuality": {"overall_status": "UNVALIDATED", "claims": []},
        "conclusion": {"generation_uncertainty": "MIXED", "limitations": ["Limit one"]},
    }
    report.update(overrides)
    return report


def test_page_escapes_user_text_and_keeps_probability_unavailable():
    page = render_turn_html(_report())
    assert "<b>this</b>" not in page
    assert "Is &lt;b&gt;this&lt;/b&gt; escaped?" in page
    assert "Yes &amp; it is." in page
    assert "HALLUCINATION PROBABILITY: NOT AVAILABLE" in page
    assert "No probe loaded. No prediction was invented." in page
    assert "not a confirmed hallucination" in page
    assert "Limit one" in page


def test_contested_and_low_confidence_steps_are_marked():
    page = render_turn_html(_report())
    assert 'class="tok hi" title="p = 62.00%">so</span>' in page
    assert 'class="tok lo" title="p = 40.00%">maybe</span>' in page
    assert "Most contested step: closest alternative &ldquo;and&rdquo;, margin 30.0 pts" in page


def test_validated_probability_is_shown_only_when_present():
    page = render_turn_html(_report(calibrated_hallucination_probability=0.25))
    assert "Hallucination probability: 25.00%" in page
    assert "NOT AVAILABLE" not in page


def test_empty_generation_still_renders():
    page = render_turn_html({"question": "Q", "answer": ""})
    assert "No tokens were generated." in page
    assert "No token probabilities." in page
    assert "HALLUCINATION PROBABILITY: NOT AVAILABLE" in page


def test_write_turn_html_never_overwrites(tmp_path):
    stamp = datetime(2026, 11, 3, 10, 15, 0)
    first = write_turn_html(_report(), tmp_path, stamp=stamp)
    second = write_turn_html(_report(), tmp_path, stamp=stamp)
    assert first.name == "turn-20261103-101500.html"
    assert second.name == "turn-20261103-101500-2.html"
    assert "example-model" in second.read_text(encoding="utf-8")


def test_open_in_browser_reports_failure(monkeypatch, tmp_path):
    def broken(_url):
        raise turn_html.webbrowser.Error("no browser")

    monkeypatch.setattr(turn_html.webbrowser, "open", broken)
    assert turn_html.open_in_browser(tmp_path / "missing.html") is False


def test_probe_without_a_score_explains_why():
    probe = {
        "available": True,
        "raw_score": None,
        "score_note": "Probe expects width 896; this model gives 1536.",
    }
    page = render_turn_html(_report(probe=probe))
    assert "NO SCORE" in page
    assert "Probe expects width 896; this model gives 1536." in page
    assert "Raw experimental score" not in page
