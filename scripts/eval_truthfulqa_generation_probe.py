"""Run Qwen on TruthfulQA questions and score each answer with the new generation probe.

The probe was trained on HaluEval-style questions, never on TruthfulQA, so this is an
out-of-distribution test. Nothing here retrains or recalibrates the probe.

Labels are AUTOMATIC: an answer is FAITHFUL / HALLUCINATED only when it asserts a listed
TruthfulQA correct / incorrect reference. Most answers stay UNVALIDATED and are left out of
AUROC and ECE. Treat the numbers as a rough transfer check, not as a validated result.

Outputs (new folder, never overwritten): answers.jsonl, summary.json, answers.csv

Usage
  python scripts/eval_truthfulqa_generation_probe.py --max-questions 200
  python scripts/eval_truthfulqa_generation_probe.py --all --run-id truthfulqa-newprobe-full
  python scripts/eval_truthfulqa_generation_probe.py --max-questions 200 --run-id r1 --resume
Add --allow-model-download only if the Qwen weights are not cached.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from glassbox.config import repo_root
from glassbox.data.schema import Label
from glassbox.data.truthfulqa import (
    label_generated_answer_coverage,
    read_pinned_snapshot,
    select_items,
)
from glassbox.demo_config import load_demo_config
from glassbox.models.backend import GenerationSettings, HuggingFaceBackend
from glassbox.probes.scoring import expected_calibration_error
from glassbox.probes.store import check_compatibility, experimental_score, load_probe

ROOT = repo_root()
PROBE_DIR = ROOT / "artifacts/probes/generation-reviewed-task7-v1-glassbox"
CSV_PATH = ROOT / "artifacts/datasets/truthfulqa/TruthfulQA.csv"
MIN_PER_CLASS = 20
BINARY = {Label.FAITHFUL.name, Label.HALLUCINATED.name}


def bootstrap_ci(fn, y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            vals.append(fn(y[idx], s[idx]))
    if not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def summarise(rows, threshold):
    counts = {
        name: sum(r["label"] == name for r in rows)
        for name in ("FAITHFUL", "HALLUCINATED", "AMBIGUOUS", "UNVALIDATED")
    }
    scored = [r for r in rows if r["probe_score"] is not None]
    out = {
        "answers": len(rows),
        "scored": len(scored),
        "label_counts": counts,
        "threshold": threshold,
    }
    for name in counts:
        vals = [r["probe_score"] for r in scored if r["label"] == name]
        out.setdefault("mean_score_by_label", {})[name] = (
            round(float(np.mean(vals)), 3) if vals else None
        )
    binary = [r for r in scored if r["label"] in BINARY]
    y = np.array([int(r["label"] == "HALLUCINATED") for r in binary])
    s = np.array([r["probe_score"] for r in binary], dtype=float)
    words = np.array([len(r["answer"].split()) for r in binary], dtype=float)
    nf, nh = int((y == 0).sum()), int((y == 1).sum())
    out["binary_rows"] = {"faithful": nf, "hallucinated": nh}
    if nf >= MIN_PER_CLASS and nh >= MIN_PER_CLASS:
        pred = s >= threshold
        out["auroc"] = round(float(roc_auc_score(y, s)), 3)
        out["auroc_95ci"] = bootstrap_ci(roc_auc_score, y, s)
        out["auroc_answer_length_baseline"] = round(float(roc_auc_score(y, words)), 3)
        out["ece_calibrated"] = round(float(expected_calibration_error(y, s)), 3)
        out["ece_95ci"] = bootstrap_ci(expected_calibration_error, y, s)
        out["brier"] = round(float(brier_score_loss(y, s)), 3)
        out["at_threshold"] = {
            "flagged_share_of_hallucinated": round(float(pred[y == 1].mean()), 3),
            "flagged_share_of_faithful (false alarms)": round(float(pred[y == 0].mean()), 3),
        }
        out["note"] = (
            "Out-of-distribution transfer check with automatic labels. TruthfulQA's class mix "
            "differs from the training data, so ECE is expected to be worse than on HaluEval."
        )
    else:
        out["auroc"] = None
        out["note"] = (
            f"Not enough labelled rows for AUROC/ECE (need {MIN_PER_CLASS} per class, have "
            f"{nf} faithful / {nh} hallucinated). Most answers are UNVALIDATED because they do not "
            "match a listed reference. Use more questions (--max-questions or --all)."
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--max-questions", type=int, default=200)
    ap.add_argument("--all", action="store_true", help="use all questions in the file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default="truthfulqa-newprobe-001")
    ap.add_argument("--resume", action="store_true", help="continue an existing run folder")
    ap.add_argument("--probe", type=Path, default=PROBE_DIR)
    ap.add_argument("--allow-model-download", action="store_true")
    a = ap.parse_args()

    out_dir = ROOT / "artifacts/experiments" / a.run_id
    jsonl = out_dir / "answers.jsonl"
    if out_dir.exists() and not a.resume:
        raise SystemExit(f"{out_dir} already exists. Use another --run-id or add --resume.")
    out_dir.mkdir(parents=True, exist_ok=True)

    items = read_pinned_snapshot(CSV_PATH)
    chosen = select_items(
        items, max_questions=len(items) if a.all else a.max_questions, seed=a.seed
    )
    done = {}
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["question_id"]] = r

    probe = load_probe(a.probe)
    threshold = probe.metadata.get("threshold")
    config = load_demo_config(
        ROOT / "configs/demo.yaml", offline=not a.allow_model_download, model_name=None
    )
    settings = GenerationSettings(
        max_new_tokens=config.generation.max_new_tokens,
        temperature=0.0,
        top_k=config.generation.top_k,
        layer_index=config.generation.layer_index,
        position="last_generated_token",
        capture_all_layers=False,
    )
    backend = HuggingFaceBackend(config.spec)
    print(f"Loading {config.spec.name} ...", flush=True)
    loaded = backend.ensure_loaded()
    todo = [it for it in chosen if it.question_id not in done]
    print(
        f"TruthfulQA questions: {len(chosen)} ({len(done)} already done, {len(todo)} to run). "
        f"Probe: {a.probe.name}, calibrated={probe.calibrated}, threshold={threshold}",
        flush=True,
    )

    with jsonl.open("a", encoding="utf-8") as handle:
        for n, item in enumerate(todo, start=1):
            result = backend.generate(item.question, settings)
            label, detail, _ = label_generated_answer_coverage(result.answer, item)
            score, note = None, "hidden state unavailable"
            if result.feature is not None:
                try:
                    check_compatibility(
                        probe,
                        model_name=result.model_name,
                        model_revision=result.model_revision,
                        layer_index=int(result.feature_layer),
                        hidden_state_count=int(result.hidden_state_count),
                        position=str(result.position),
                        prompt_mode=str(result.prompt_mode),
                        feature_dim=int(result.feature.shape[0]),
                        dtype=result.dtype,
                    )
                    score, note = float(experimental_score(probe, result.feature)), "ok"
                except ValueError as exc:
                    note = f"not scored: {exc}"
            row = {
                "question_id": item.question_id,
                "category": item.category,
                "question": item.question,
                "answer": result.answer.strip(),
                "label": label.name,
                "label_detail": detail,
                "probe_score": score,
                "score_note": note,
                "predicted": None
                if score is None or threshold is None
                else ("hallucinated" if score >= threshold else "faithful"),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            done[item.question_id] = row
            shown = "n/a" if score is None else f"{score:.2f}"
            print(
                f"[{n}/{len(todo)}] {label.name:12s} score={shown}  {row['answer'][:70]}",
                flush=True,
            )

    rows = [done[it.question_id] for it in chosen if it.question_id in done]
    summary = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": a.run_id,
        "dataset": "TruthfulQA (pinned CSV)",
        "seed": a.seed,
        "model": loaded.spec.name,
        "model_revision": loaded.resolved_revision,
        "probe": str(a.probe.relative_to(ROOT)) if a.probe.is_relative_to(ROOT) else str(a.probe),
        "labels": "automatic reference-phrase matching (not human, not the HaluEval review)",
        **summarise(rows, threshold),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "answers.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "category",
                "question",
                "answer",
                "label",
                "probe_score",
                "predicted",
                "label_detail",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
