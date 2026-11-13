"""Turn results.jsonl from ask_generation_probe.py into a report.

Optional grades file (JSON): {"<question text>": "correct" | "wrong" | "garbled"}.
With grades, the report compares scores for correct vs wrong answers and gives AUROC
(wrong answers should score higher). Without grades, it only lists answers and scores.

Usage
  python scripts/probe_report.py --results results.jsonl --grades grades.json --out probe-report.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

BASE_RATE = 0.83


def auroc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results.jsonl"))
    ap.add_argument("--grades", type=Path)
    ap.add_argument("--out", type=Path, default=Path("probe-report.html"))
    a = ap.parse_args()

    rows = [
        json.loads(line)
        for line in a.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grades = json.loads(a.grades.read_text(encoding="utf-8")) if a.grades else {}
    for r in rows:
        r["grade"] = grades.get(r["question"], "ungraded")

    ok = [r["p_hallucinated"] for r in rows if r["grade"] == "correct" and "p_hallucinated" in r]
    bad = [
        r["p_hallucinated"]
        for r in rows
        if r["grade"] in ("wrong", "garbled") and "p_hallucinated" in r
    ]

    def mean(v):
        return sum(v) / len(v) if v else None

    def fmt(v):
        return "n/a" if v is None else f"{v:.2f}"

    auc = auroc(bad, ok)

    def row_html(r):
        g = r["grade"]
        color = {"correct": "#1a7f37", "wrong": "#cf222e", "garbled": "#bf8700"}.get(g, "#6e7781")
        score = r.get("p_hallucinated")
        bar = (
            ""
            if score is None
            else (
                f'<div style="background:#eee;width:120px;height:10px"><div style="background:{color};'
                f'width:{int(score * 120)}px;height:10px"></div></div>'
            )
        )
        return (
            f"<tr><td>{html.escape(r['question'])}</td><td>{html.escape(r['answer'])}</td>"
            f"<td style='color:{color};font-weight:600'>{g}</td>"
            f"<td>{fmt(score)}</td><td>{bar}</td><td>{html.escape(r.get('band', r.get('error', '')))}</td></tr>"
        )

    summary = ""
    if ok or bad:
        summary = (
            f"<p><b>Graded answers:</b> {len(ok)} correct, {len(bad)} wrong or garbled.<br>"
            f"Mean score, correct answers: <b>{fmt(mean(ok))}</b> &nbsp; wrong answers: <b>{fmt(mean(bad))}</b><br>"
            f"AUROC (wrong ranked above correct): <b>{fmt(auc)}</b> "
            f"(0.5 is chance; with {len(ok) + len(bad)} answers this is only a rough check).</p>"
        )
    page = f"""<!doctype html><meta charset="utf-8"><title>Glass Box probe report</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1100px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}}</style>
<h1>Glass Box: generation probe report</h1>
<p>Qwen2.5-0.5B-Instruct answers scored by the generation probe (trained on AI-reviewed task-7 labels).
Score = estimated chance the answer is hallucinated; the average answer in training data scored {BASE_RATE:.2f}.
Experimental: HaluEval-style questions, single model, AI-reviewed labels.</p>
{summary}
<table><tr><th>Question</th><th>Answer</th><th>Graded</th><th>P(halluc.)</th><th></th><th>Band</th></tr>
{"".join(row_html(r) for r in rows)}</table>"""
    a.out.write_text(page, encoding="utf-8")
    print(
        f"Wrote {a.out}  |  correct mean {fmt(mean(ok))}, wrong mean {fmt(mean(bad))}, AUROC {fmt(auc)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
