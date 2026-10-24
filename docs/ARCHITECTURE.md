# Architecture

Glass Box is one installable package, `src/glassbox`. Importing it does not download a model or a dataset. Two configuration files describe two different jobs. They are not interchangeable.

## Two sources of configuration

| Job | Config | Code | Model in the file |
| --- | --- | --- | --- |
| Research scaffold and smoke test | [`configs/default.yaml`](../configs/default.yaml) | `glassbox.config.ProjectConfig` | `Qwen/Qwen2.5-3B-Instruct`, downloads off, `phase: scaffold` |
| Terminal demo and probe training | [`configs/demo.yaml`](../configs/demo.yaml) | `glassbox.demo_config.DemoConfig` | `Qwen/Qwen2.5-0.5B-Instruct` |

`python scripts/smoke_test.py` fails if `default.yaml` and `ProjectConfig()` disagree, if downloads are enabled, or if `phase` is not `scaffold`. Changing the demo model does not change that check. A recorded experiment must replace `revision: main` with a 40-character commit. `require_commit_hash` rejects `main`. The demo config pins revision `7ae557604adf67be50417f59c2c2f167def9a775` and dtype `bfloat16`. Training refuses a revision that is not a 40-character commit.

## Components

```text
cli.py
  ├─ demo_config.py          demo YAML
  ├─ models/backend.py       one loaded causal LM, generation, hidden states
  ├─ models/distributions.py softmax, top alternatives, distribution export
  ├─ models/resources.py     RAM estimate and dtype choice
  ├─ verification.py         curated string check
  ├─ probes/store.py         save/load probe and compatibility check
  ├─ session.py              turns and export
  ├─ present.py              Rich report
  └─ reporting.py            plain-text menu views and Markdown

probes/training.py
  ├─ data/halueval.py        pinned HaluEval QA pairs and splits
  ├─ models/backend.py       hidden state at the registered position
  ├─ probes/scoring.py       threshold, Platt scaling, baselines
  └─ evaluation/             metrics and run records

config.py, guards.py, smoke.py, data/loading.py
  scaffold only; smoke and research tests
```

`models/loading.py` is a thin wrapper around `backend.load_hf_model`. Callers should use the backend. The wrapper exists so a loader can be imported without pulling the generation loop into every test.

## Data flow for one question

1. `cli.main` loads `DemoConfig`, curated facts, and a probe directory if `artifacts/probes/final-hidden-state/probe.joblib` exists or `--probe` is set.
2. The first `HuggingFaceBackend.ensure_loaded` loads the tokenizer and model. Later questions reuse it (`load_count` stays 1).
3. `render_prompt` applies the chat template when the tokenizer has one. Qwen uses that path.
4. Each new token is chosen from `scale_logits` plus a full-vocabulary softmax. Temperature 0 is argmax. A positive temperature samples `softmax(logits / temperature)` and reports that same distribution.
5. Step 1 is conditioned on the prompt. Later steps are conditioned on the prompt plus output tokens already written. Prompt tokens are counted and not scored.
6. A second forward pass reads hidden states. The demo position is `last_generated_token`, layer `-1` (the last tensor the model returns, including the embedding stream at index 0).
7. `verification.py` checks the answer against `configs/curated_facts.yaml`. An unmatched question is `NO REFERENCE AVAILABLE`. A match is faithful only for an accepted alias, hallucinated only for a listed false claim, and otherwise unvalidated. This check never calls the probe.
8. `session._apply_probe` runs only when the saved metadata matches the live model name, revision, layer, hidden-state count, position, prompt mode, feature dimension, and dtype (dtype is checked only when the artifact recorded one). A calibrated score requires `calibration_validated: true`. Otherwise the UI says `PROBE NOT TRAINED` or shows an experimental score with `RISK NOT ASSESSED`.

Menu commands `t`, `p`, `v`, `d`, and `e` read the last `TurnRecord`. They do not recompute a different analysis. Rich (`present.py`) and plain text (`reporting.py`) are two views of that record.

## Probe training flow

`python -m glassbox train-probe` reads HaluEval through `data/halueval.py`. It does not call `data.loading.load_examples`. That loader reads a pinned file already on disk: HaluEval rows become pair-protocol `QAExample` records, and TruthfulQA rows become reference questions labeled `UNVALIDATED` (`truthfulqa_reference_only_v1`). A missing file with downloads disabled raises `DownloadDisabledError`. A missing file with a download requested still raises `NotImplementedError`, because this function is not a download client.

1. Read `artifacts/datasets/halueval/qa_data.json` and refuse a file whose SHA-256 is not the pin in `halueval.py`.
2. Filter pairs. Dropped rows are counted and are not moved into training.
3. Split by question id with the run seed before any forward pass. Both answers for a question stay in one split.
4. Encode the demo prompt plus the supplied answer. The knowledge passage is stored and is not fed to the model.
5. Save the pre-registered final layer (`layer_index: -1` resolved to the last hidden state). A validation table of other layers is recorded and does not choose the saved layer.
6. Fit logistic regression on training rows only after both classes have at least 20 rows. Choose the threshold on validation. Fit Platt scaling on validation only. Accept calibration only when validation and test each have at least 15 rows per class and test ECE is at most 0.10.
7. Write the probe under `artifacts/probes/final-hidden-state/` and, when test metrics exist, `artifacts/experiments/<run_id>/record.json`.

A local `halueval-pair-001` directory is already on disk and gitignored. Its calibration flag is false, so the demo must not treat the score as a factual-risk percentage. Numbers are in [RESEARCH.md](RESEARCH.md).

The positive class is the HaluEval `hallucinated_answer` string. It is not a new judgment of a Qwen generation.

## Generation dataset flow

`python -m glassbox freeze-generation-split` reads the pinned TruthfulQA CSV and writes question ids before any answer is labeled. `python -m glassbox build-generation-dataset` then labels supplied answer text with `label_generated_answer`. Neither command loads Qwen or calls `publish_probe`. The output is `artifacts/experiments/<run-id>/dataset.json` unless `--output` is set. A path under `artifacts/probes/final-hidden-state/`, including the `.previous`, `.staging`, and `.incoming` directories, is refused. Ambiguous and unvalidated rows stay in the file. The record states whether each binary class meets the interval minimum of 20 and records the 5-of-40 pilot as below that minimum. It does not store pair-protocol metrics.

## Shared behavior, and where it is duplicated

Use these modules instead of copying the rule:

| Behavior | Source of truth |
| --- | --- |
| Offline download refusal | `guards.reject_download` |
| Softmax and decoding logits | `models/distributions.py` |
| Curated reference labels | `verification.py` plus `configs/curated_facts.yaml` |
| Classification metrics | `evaluation/metrics.py` |
| Run JSON and secret-like keys | `evaluation/records.py` |
| Probe file format | `probes/store.py` |
| Scientific limits | `caveats.py` |

Known duplication, left in place:

- Hidden-state standardization lives in `probes/preprocess.py`. Both training and `probes/store.py` call `apply_standard_scaler`. The saved `preprocessing` record is what inference applies.
- `data/schema.py` `QAExample` is the loader record. `data/halueval.py` `HaluPair` is the record training actually uses. They are not the same schema. `load_examples("halueval_qa")` builds `QAExample` rows from the same filtered pairs.
- `PROJECT_OVERVIEW.md` used to name the last prompt token as the planned probe position. The training constant `REGISTERED_POSITION` is `last_generated_token`, and `demo.yaml` matches it. Do not train a probe at `last_prompt_token` and load it into this demo.

## Dependencies

Runtime: NumPy, Rich, scikit-learn, Transformers, Hugging Face Hub, PyYAML, and PyTorch. PyTorch is imported only when a model loads. Dev: pytest and Ruff. There is no notebook extra, no MLflow, and no database.

## Errors and logging

There is no `logging` configuration. The CLI prints failures to stderr and returns `0` on success, `2` for a bad flag combination, and `1` for a config, generation, memory, or training failure. `DownloadDisabledError`, `ModelMemoryError`, and `ProbeCompatibilityError` are the typed failures. `write_json` refuses paths outside its root and mapping keys that contain `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `API_KEY`, or `PRIVATE_KEY`. Distribution exports therefore use `piece_id` for the vocabulary id.

## Interfaces that must stay stable for a saved probe

A probe directory is `probe.joblib` plus `metadata.json`. Inference rejects a directory when `model_name`, `model_revision`, `layer_index`, `hidden_state_count`, `position`, `prompt_mode`, or `feature_dim` differ. `dtype` is compared when both sides have it. Do not rename those keys without a new metadata schema version.
