"""Ask Qwen questions and score each answer with the new generation probe.

Uses the same settings the task-7 training data was generated with
(configs/demo.yaml: Qwen2.5-0.5B-Instruct, pinned revision, greedy decoding,
32 new tokens, last layer, last generated token). Refuses to score if the
hidden state does not match that contract.

Usage
  python scripts/ask_generation_probe.py                       # interactive
  python scripts/ask_generation_probe.py --question "Who wrote Hamlet?"
  python scripts/ask_generation_probe.py --file questions.txt  # one question per line
  add --allow-model-download if the weights are not cached yet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from glassbox.config import repo_root
from glassbox.demo_config import load_demo_config
from glassbox.models.backend import GenerationSettings, HuggingFaceBackend

ROOT = repo_root()
PROBE = ROOT / "artifacts/probes/generation-reviewed-task7-v1/probe.joblib"
CONTRACT = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "layer": 24,
    "position": "last_generated_token",
    "dim": 896,
}
BASE_RATE = 0.83  # share of hallucinated answers in the reviewed validation data


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def band(p):
    if p >= 0.90:
        return "HIGH risk"
    if p >= 0.75:
        return "typical risk"
    return "LOWER risk"


def ask(backend, settings, saved, question):
    r = backend.generate(question, settings)
    out = {"question": question, "answer": r.answer.strip()}
    problems = []
    if r.feature is None:
        problems.append("no hidden state")
    else:
        if r.model_name != CONTRACT["model"]:
            problems.append(f"model {r.model_name}")
        if int(r.feature_layer) != CONTRACT["layer"]:
            problems.append(f"layer {r.feature_layer}")
        if str(r.position) != CONTRACT["position"]:
            problems.append(f"position {r.position}")
        if r.feature.shape[0] != CONTRACT["dim"]:
            problems.append(f"dim {r.feature.shape[0]}")
    if problems:
        out["error"] = "Not scored, contract mismatch: " + ", ".join(problems)
        return out
    x = np.asarray(r.feature, dtype=np.float64).reshape(1, -1)
    raw = float(saved["probe"].predict_proba(x)[0, 1])
    cal = float(saved["platt"].predict_proba(logit(np.array([raw])))[0, 1])
    out.update({"p_hallucinated": round(cal, 3), "raw_score": round(raw, 3), "band": band(cal)})
    return out


def show(o):
    print(f"\nQ: {o['question']}\nA: {o['answer']}")
    if "error" in o:
        print(f"   {o['error']}")
    else:
        print(
            f"   P(hallucinated) = {o['p_hallucinated']:.2f}  [{o['band']}]  (average answer: {BASE_RATE:.2f})"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question")
    ap.add_argument("--file", type=Path)
    ap.add_argument("--out", type=Path, help="save results as JSON lines")
    ap.add_argument("--allow-model-download", action="store_true")
    a = ap.parse_args()

    saved = joblib.load(PROBE)
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
    backend.ensure_loaded()
    print("Experimental probe: AI-reviewed labels, HaluEval-style questions, Qwen2.5-0.5B only.")

    if a.question:
        questions = [a.question]
    elif a.file:
        questions = [
            line.strip() for line in a.file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    else:
        questions = None

    results = []
    if questions is not None:
        for q in questions:
            o = ask(backend, settings, saved, q)
            show(o)
            results.append(o)
    else:
        while True:
            try:
                q = input("\nQuestion (blank to quit): ").strip()
            except EOFError:
                break
            if not q:
                break
            o = ask(backend, settings, saved, q)
            show(o)
            results.append(o)
    if a.out and results:
        a.out.write_text("".join(json.dumps(o) + "\n" for o in results), encoding="utf-8")
        print(f"\nSaved {len(results)} results to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
