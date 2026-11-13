"""Re-select the Task 7 early-warning threshold with a false-warning cap.

The first evaluation (run 002) picked its threshold by validation F1 alone. That drove the
threshold to 0.05, which flagged every faithful answer (false-warning rate 1.0) and made the
timely-coverage number meaningless.

This script reuses the per-prefix probe scores saved in run 002 (no model reload, no
retraining). It picks a new threshold on VALIDATION ONLY:

  maximise timely event coverage on validation hallucinated answers,
  subject to an answer-level false-warning rate on validation faithful answers <= cap.

Then it scores TEST once and writes a new run folder. Run 002 is not modified.

Caveat: this rule was chosen after run 002's test result was seen, so the test set has now
been looked at twice. Treat the result as exploratory.

Usage
  python scripts/rethreshold_task7_early_warning.py                 # cap 0.20
  python scripts/rethreshold_task7_early_warning.py --cap 0.10
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "artifacts" / "probes"
SOURCE = "early-warning-task7-ai-v1-002"
BINARY = {"FAITHFUL", "HALLUCINATED"}


def lead(flags: list[bool], event_index: int) -> int | None:
    """Same rule as glassbox.evaluation.metrics.early_warning_lead."""
    for index in range(event_index + 1):
        if flags[index]:
            return event_index - index
    return None


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    values = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(values):  # average ranks for ties
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def answer_stats(answers: list[dict], threshold: float) -> dict[str, object]:
    faithful_alerts, examples, missing, disagreements = [], [], 0, 0
    for a in answers:
        prefixes = a["prefixes"]
        if not prefixes:
            continue
        flags = [p["probe_score"] >= threshold for p in prefixes]
        if prefixes[-1]["evidence_label"] != a["reviewed_label"]:
            disagreements += 1
        if a["reviewed_label"] == "FAITHFUL":
            faithful_alerts.append(any(flags))
            continue
        if a["reviewed_label"] != "HALLUCINATED":
            continue
        event = a["event_index"]
        if event is None:
            missing += 1
            continue
        lt = lead(flags, int(event))
        examples.append(
            {
                "question_id": a["question_id"],
                "event_index": event,
                "first_flag_index": None if lt is None else event - lt,
                "lead_tokens": lt,
                "timely_warning": lt is not None,
            }
        )
    leads = [e["lead_tokens"] for e in examples if e["lead_tokens"] is not None]
    hallucinated = sum(a["reviewed_label"] == "HALLUCINATED" for a in answers if a["prefixes"])
    compared = sum(1 for a in answers if a["prefixes"])
    return {
        "reviewed_hallucinated_answers": hallucinated,
        "reviewed_faithful_answers": len(faithful_alerts),
        "test_answers_compared": compared,
        "hallucinated_answers_with_prefix_event": len(examples),
        "hallucinated_answers_without_prefix_event": missing,
        "events_flagged_at_first_prefix": sum(e["first_flag_index"] == 0 for e in examples),
        "timely_warning_count": len(leads),
        "timely_warning_coverage": len(leads) / len(examples) if examples else None,
        "mean_lead_tokens_among_timely_warnings": float(np.mean(leads)) if leads else None,
        "median_lead_tokens_among_timely_warnings": float(np.median(leads)) if leads else None,
        "faithful_answer_false_warning_rate": float(np.mean(faithful_alerts))
        if faithful_alerts
        else None,
        "reviewed_vs_prefix_final_label_disagreements": disagreements,
        "event_examples": examples,
    }


def prefix_metrics(answers: list[dict], threshold: float) -> dict[str, object]:
    y, s = [], []
    for a in answers:
        for p in a["prefixes"]:
            if p["evidence_label"] in BINARY:
                y.append(int(p["evidence_label"] == "HALLUCINATED"))
                s.append(p["probe_score"])
    y, s = np.array(y), np.array(s)
    pred = s >= threshold
    tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "n_binary_prefixes": int(len(y)),
        "faithful_prefixes": int((y == 0).sum()),
        "hallucinated_prefixes": int((y == 1).sum()),
        "auroc": auroc(y, s),
        "f1_at_validation_threshold": float(f1),
        "accuracy_at_validation_threshold": float((pred == (y == 1)).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cap", type=float, default=0.20, help="max answer-level false-warning rate on validation"
    )
    ap.add_argument("--run-id", default="early-warning-task7-ai-v1-003")
    a = ap.parse_args()
    src = PROBES / SOURCE
    out = PROBES / a.run_id
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    answers = [
        json.loads(x)
        for x in (src / "prefix_evidence.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    base = json.loads((src / "early_warning_evaluation.json").read_text(encoding="utf-8"))
    val = [x for x in answers if x["split"] == "validation"]
    test = [x for x in answers if x["split"] == "test"]

    grid = []
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        st = answer_stats(val, float(t))
        grid.append(
            {
                "threshold": float(t),
                "fwr": st["faithful_answer_false_warning_rate"],
                "coverage": st["timely_warning_coverage"] or 0.0,
            }
        )
    ok = [g for g in grid if g["fwr"] is not None and g["fwr"] <= a.cap]
    if not ok:
        raise SystemExit(f"No threshold keeps validation false warnings at or below {a.cap}")
    best = max(ok, key=lambda g: (g["coverage"], g["threshold"]))
    threshold = best["threshold"]

    payload = dict(base)
    payload.update(
        {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "run_id": a.run_id,
            "derived_from": SOURCE,
            "test_prefix_metrics": prefix_metrics(test, threshold),
            "test_early_warning": answer_stats(test, threshold),
            "note": (
                f"Threshold re-selected on validation with an answer-level false-warning cap of {a.cap:.2f}, "
                f"using the per-prefix scores saved in {SOURCE}; the probe was not retrained. The rule was "
                "chosen after the first test result was seen, so this is exploratory. Prefix event labels "
                "come from the automatic evidence labeler, not the review. Single AI reviewer."
            ),
        }
    )
    training = dict(base.get("training", {}))
    training.update(
        {
            "threshold": threshold,
            "threshold_selected_on": "validation answers only",
            "threshold_rule": f"max validation timely coverage s.t. validation false-warning rate <= {a.cap:.2f}",
            "previous_threshold": base.get("training", {}).get("threshold"),
            "validation_coverage_at_threshold": best["coverage"],
            "validation_false_warning_rate_at_threshold": best["fwr"],
        }
    )
    payload["training"] = training
    out.mkdir(parents=True)
    (out / "early_warning_evaluation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out / "threshold_grid_validation.json").write_text(
        json.dumps(grid, indent=2) + "\n", encoding="utf-8"
    )
    t = payload["test_early_warning"]
    print(
        json.dumps(
            {
                "threshold": threshold,
                "validation": best,
                "test_prefix_auroc": payload["test_prefix_metrics"]["auroc"],
                "test_coverage": f"{t['timely_warning_count']}/{t['hallucinated_answers_with_prefix_event']}",
                "test_false_warning_rate": t["faithful_answer_false_warning_rate"],
                "test_mean_lead": t["mean_lead_tokens_among_timely_warnings"],
                "events_defined": f"{t['hallucinated_answers_with_prefix_event']}/{t['reviewed_hallucinated_answers']}",
                "label_disagreements": f"{t['reviewed_vs_prefix_final_label_disagreements']}/{t['test_answers_compared']}",
            },
            indent=2,
        )
    )
    print(f"Wrote {out / 'early_warning_evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
