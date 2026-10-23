# Project overview

Glass Box inspects a local causal language model from the terminal. The research question is whether one hidden state from frozen Qwen can flag a factual error. That question is not settled. The pair-protocol score is high and is explained in large part by the answer text. The generation-protocol score does not beat TF-IDF. Neither probe is calibrated. Hallucination detection is not validated.

## Pipeline

1. Load `Qwen/Qwen2.5-0.5B-Instruct` from the local Hugging Face cache when a command needs it. The revision in [`configs/demo.yaml`](../configs/demo.yaml) is `7ae557604adf67be50417f59c2c2f167def9a775`, dtype `bfloat16`. Weights are not in Git and are not updated.
2. Generate an answer, or score a supplied answer. The demo uses the chat template. Temperature 0 is greedy.
3. Read the hidden state at the last generated piece, layer `-1` (layer 24 of 25 tensors on this checkpoint, dimension 896).
4. Optionally apply a saved logistic probe. The probe is a classifier trained on those vectors. Its parameters are the logistic weights and a scaler fit on its own training split. They are not Qwen's weights.
5. Separately, compare the answer text with [`configs/curated_facts.yaml`](../configs/curated_facts.yaml). This check never calls the probe.

Token probability is the decoding distribution. It is not the probe score. The probe score is not a factual-risk percentage unless `calibration_validated` is true. No saved probe has that flag.

## Two probes

| Probe | Where it lives | What the label is | Demo default |
| --- | --- | --- | --- |
| `halueval-pair-002` | `artifacts/probes/final-hidden-state/` | Benchmark right answer versus benchmark false answer, both supplied to the model | Yes. Uncalibrated. |
| `halueval-generation-001` | `artifacts/probes/generation-runs/halueval-generation-001/` | Evidence rule on answers Qwen generated | No. Not a successful detector. |

The scaffold config [`configs/default.yaml`](../configs/default.yaml) still names `Qwen/Qwen2.5-3B-Instruct` with downloads off. Smoke checks that file. It is not the model the demo or the probes load.

The three constraints in `src/glassbox/caveats.py` are unchanged: labels need validation, a wording mismatch is not a hallucination, and probe predictability is not causality. Early detection and activation intervention are not implemented. `early_warning_lead` is a unit-tested helper only.

Run index: [EXPERIMENTS.md](EXPERIMENTS.md). Method: [RESEARCH.md](RESEARCH.md). Components: [ARCHITECTURE.md](ARCHITECTURE.md).
