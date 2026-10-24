# Datasets

Moved to [RESEARCH.md](RESEARCH.md). That page has the HaluEval pin, the filter counts, the curated demo file, and the TruthfulQA reference rules.

`glassbox.data.loading.load_examples` reads a local pinned file and does not download. HaluEval rows use the pair-protocol labels from `glassbox.data.halueval`. TruthfulQA rows are reference questions with `UNVALIDATED` labels. A missing snapshot stays offline: `DownloadDisabledError` when downloads are disabled, and `NotImplementedError` when a download is requested.

## Generation-labeled TruthfulQA evaluation

`glassbox.data.generation_dataset` writes an evaluation dataset. The rubric id is `truthfulqa_generation_v1`, the same rule as `label_generated_answer`:

- `HALLUCINATED` only when the answer asserts a listed incorrect reference.
- `FAITHFUL` only when the answer asserts a listed correct reference and no incorrect one.
- `AMBIGUOUS` when the answer asserts both a listed correct reference and a listed incorrect reference.
- `UNVALIDATED` otherwise, including a paraphrase that does not assert a listed phrase.

A wording difference is not a hallucination label. `AMBIGUOUS` and `UNVALIDATED` stay in the dataset and stay out of both binary classes. The dataset is evaluation-only and is not a training source for the pair probe.

`freeze-generation-split` writes question ids before any answer is labeled. `build-generation-dataset` stores those ids with the seed, dataset SHA-256, model name, revision, and dtype on every row. Class counts name the `bootstrap_interval` minimum of 20 rows per binary class. The existing pilot `truthfulqa-generation-001` is recorded in that file as 3 faithful, 2 hallucinated, 0 ambiguous, and 35 unvalidated out of 40, which is below that minimum. The file does not contain pair-protocol metrics. The command does not train a probe and refuses to write under `artifacts/probes/final-hidden-state/`.

The pilot counts and the pair-protocol results stay in [RESEARCH.md](RESEARCH.md). They are different experiments.

`write-generation-evaluation` is the scored record for that dataset. It keeps one row per generated answer, including the question id, answer, label, justifying reference phrase, and either a probe score or a missing-score reason. `AMBIGUOUS` and `UNVALIDATED` stay in the file and stay out of both binary classes. A missing provenance field, a missing row, a dirty worktree, or a missing probe when a score was requested exits non-zero and leaves no file. The record does not copy pair-protocol metrics and does not set `calibration_validated`.

`eval-saved-probe` is the generation transfer of the saved HaluEval pair probe. It freezes a TruthfulQA split, generates with the cached demo model, scores the saved probe, and labels each answer with `label_with_phrases`. The probe prediction is not the reference label. `AMBIGUOUS` and `UNVALIDATED` stay in `record.json` and stay out of binary metrics. A question whose normalized text appears in the probe's HaluEval train, validation, or test ids also stays in the record and stays out of those metrics. The classification threshold remains the saved value 0.65. Length, reference-overlap, and reference TF-IDF baselines use predeclared maps and are not fit on the generation labels. Aggregate metrics go to `metrics.json` because the evaluation record refuses metric keys. A generation ECE does not set `calibration_validated`. The score name stays `EXPERIMENTAL PROBE SCORE` and the risk line stays `RISK NOT ASSESSED`. The planned sample is seed 0 and 200 questions, run id `truthfulqa-probe-transfer-001`. The command requires `--offline`, a clean worktree, and the local CSV, HaluEval snapshot, and probe. It does not download Qwen and it is not part of CI. The saved probe was trained on paired HaluEval strings, not on Qwen generations.
