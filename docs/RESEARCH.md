# Research

This page is the method for the probe. One local run is recorded under `artifacts/` and is not in Git. The numbers below are copied from that run. They are pair-protocol scores, not a measurement of Qwen's own answers, and the lexical-overlap baseline is perfect.

## Question

Can a logistic regression on one frozen hidden state of a local Qwen2.5 instruct model rank a HaluEval hallucinated answer above the benchmark's right answer, on questions that were split before the features were extracted?

That is a narrower question than "does Qwen hallucinate?" The pair protocol labels supplied answer strings. It does not label a new generation. `eval-truthfulqa` is a separate generation check and is not used to train or calibrate this probe.

## Datasets

Logical ids in `glassbox.constants` are `halueval_qa` and `truthfulqa`. `load_examples` reads a pinned local file for either id. It does not download. HaluEval examples use the pair-protocol labels. TruthfulQA examples from this loader are reference questions labeled `UNVALIDATED`. `eval-truthfulqa` is a separate generation check. Its pilot is in its own section below and is not part of the pair-protocol table.

### HaluEval QA

Li et al., "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models," EMNLP 2023. <https://aclanthology.org/2023.emnlp-main.397/>

The pair reader uses `data/qa_data.json` at commit `207f479c555a68b8d2431631a95083b84b2accbe`, SHA-256 `89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88`. A different hash is refused. The local path is `artifacts/datasets/halueval/qa_data.json` (gitignored, about 6 MB). Fields: `knowledge`, `question`, `right_answer`, `hallucinated_answer`.

Filters in `data/halueval.py`, applied before the split. Counts from the pinned file on 23 October 2026 (`prepare_pairs`, seed 0):

| Outcome | Count |
| --- | --- |
| Source rows | 10,000 |
| Kept questions | 7,681 |
| One answer contained the other | 1,473 dropped |
| Yes/no answer | 623 dropped |
| Same numbers, so a restatement rather than a contradiction | 171 dropped |
| Longer than the length cap | 50 dropped |
| Empty text | 2 dropped |
| Train / validation / test questions | 4,609 / 1,536 / 1,536 |
| `--max-questions 240` subset | 144 / 48 / 48 |

Each kept question contributes two rows, faithful and hallucinated, in the same split. The knowledge passage is not shown to the model. The prompt is the demo chat template plus the supplied answer.

These filters are not a full re-annotation. Rows that survive can still be bad benchmark labels. `UNVALIDATED` in `data/schema.py` is unused by this reader.

### Curated demo facts

[`configs/curated_facts.yaml`](../configs/curated_facts.yaml) is a separate string check for the terminal. It is not the probe training set. Thirteen hand-written items. Matching is normalized, not fuzzy. Wording that hits neither an accepted alias nor a listed false claim stays unvalidated.

### TruthfulQA

Named in `configs/default.yaml` as a second evaluation set. `load_examples("truthfulqa")` reads `artifacts/datasets/truthfulqa/TruthfulQA.csv` when that file is already present and returns one `UNVALIDATED` reference row per question (`label_source` `truthfulqa_reference_only_v1`). It does not invent a model answer and it does not download. A missing file is refused. `data/truthfulqa.py` refuses any file whose SHA-256 is not `b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c`. Columns are Type, Category, Question, Best Answer, Best Incorrect Answer, Correct Answers, Incorrect Answers, and Source. `label_generated_answer` marks a new answer hallucinated only if it asserts a listed incorrect answer, and faithful only if it asserts a listed correct answer and no incorrect one. Both hits are `AMBIGUOUS`. Other wording stays unvalidated. Do not pool these rows with HaluEval. The generation check is `truthfulqa-generation-001`, reported below. It is not part of the pair-protocol metrics.

## Probe

- Model for this protocol: `Qwen/Qwen2.5-0.5B-Instruct` via `configs/demo.yaml`. The scaffold default remains the 3B id and is not what training loads.
- Position: `last_generated_token`. Training refuses any other demo position.
- Layer: config `layer_index: -1`, resolved with `resolve_layer` and saved as that index. Training stores that layer only.
- Classifier: scikit-learn logistic regression, `class_weight` balanced, `max_iter` 1000, `random_state` from `--seed` (default 0).
- Features are standardized with a mean and scale fit on the training matrix only.
- Minimum before a save: 20 training rows of each class. Test rows are not moved to fill a missing class.
- Threshold: maximum validation F1 on the grid 0.05, 0.10, …, 0.95. Ties keep the lower false-positive rate.
- Calibration: Platt scaling fit on validation scores. Accepted only if each of validation and test has at least 15 rows per class and test ECE is at most 0.10. Otherwise the stored score is experimental and the demo must not print it as a factual-risk percentage.
- Baselines computed when the test split has both classes: training-majority label (no AUROC), shuffled training labels, answer length, and lexical overlap with the gold answer. Overlap can recover the pair label without hidden states. That is a warning, not a probe result.

## Metrics

`evaluation/metrics.py` reports precision, recall, F1, AUROC, AUPRC, and false-positive rate. The positive class is hallucinated (`1`). `early_warning_lead` is implemented and unit-tested. No run defines the hallucination event token, so the field is null in a pair-protocol record.

Confusion counts are `tn`, `fp`, `fn`, `tp` at the frozen threshold.

## Reproducibility

Pins that a finished run must store are listed in [DEVELOPMENT.md](DEVELOPMENT.md). The training entry point writes `record.json` only when test metrics were computed. Zeros are not substituted for a missing test split.

Hardware for `halueval-pair-002`: CPU only, no NVIDIA GPU, 12 GiB RAM. The interactive process was stopped first so only one Qwen copy was loaded. The saved dtype is bfloat16.

## Local run `halueval-pair-002`

This is the probe in `artifacts/probes/final-hidden-state/`. The earlier `halueval-pair-001` artifact was moved to `artifacts/probes/final-hidden-state.previous/` when this run was published. Its dtype was not stored. This run pins bfloat16.

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Resolved revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| dtype | `bfloat16` |
| Prompt | `chat_template` |
| Position / layer / width | `last_generated_token`, layer 24, dim 896, 25 hidden tensors |
| Device | CPU. Peak RSS about 1616 MiB. Wall time about 8.4 minutes. |
| Seed | 0 |
| Questions | 200 of the filtered 7,681: train 120, validation 40, test 40 |
| Rows | 120/120, 40/40, 40/40 faithful/hallucinated |
| Extraction skips | none |
| Threshold | 0.65, chosen on validation |
| Test precision / recall / F1 | 0.976 / 1.000 / 0.988 |
| Test AUROC / AUPRC / FPR | 0.996 / 0.995 / 0.025 |
| Test confusion | tn 39, fp 1, fn 0, tp 40 |
| Calibration | rejected. Test ECE 0.114 is above 0.10. `calibration` is `none`. |
| Majority baseline | F1 0.667. AUROC left null. |
| Shuffled-label AUROC | 0.403 |
| Answer-length AUROC | 0.981 |
| Lexical overlap with the gold answer | AUROC 1, F1 1 |

Overlap can recover the label because the faithful text is the gold string. The probe AUROC of 0.996 sits next to that lexical-overlap AUROC of 1 and the answer-length AUROC of 0.981. The probe's AUROC does not show that the hidden state is doing something the answer text does not already contain. The demo shows this score as `EXPERIMENTAL PROBE SCORE` and `RISK NOT ASSESSED`.

The saved record notes commit `9b8bcc48019774510a7947730504fab29272e75b` and a dirty worktree. That artifact was not rewritten. A later pair record sets `finished_reproduction` and `reproducible` to false when `git status --porcelain` is non-empty, so a dirty tree is not stored as a finished reproduction. Test ECE 0.114 remains above `CALIBRATION_ECE_LIMIT` 0.10, so `calibration_validated` stays false. The limit was not lowered. A later training run must use a new `--run-id`. That run stores held-out example ids, labels, and probe scores, and it writes a `bootstrap_interval` next to the length and overlap baselines. The interval resamples those scored test rows. It does not transfer to another dataset or to Qwen's own generations, and it does not set `calibration_validated`.

## TruthfulQA generation check `truthfulqa-generation-001`

Not used for training. The CSV reader kept 788 questions. A seed-0 subset of 40 was generated with the same frozen model. Three answers asserted a listed correct reference, two asserted a listed incorrect reference, and 35 did not assert either, so they stayed unvalidated. Metrics use the five labeled answers only, at the HaluEval threshold 0.65.

| Metric | Value |
| --- | --- |
| Precision / recall / F1 | 0.250 / 0.500 / 0.333 |
| AUROC / AUPRC / FPR | 0.500 / 0.700 / 1.000 |
| Confusion | tn 0, fp 3, fn 1, tp 1 |
| ECE | 0.797 |

These five labels are too few to support a factual-risk percentage. The result is a distribution-shift check, not a second training score. Coverage is 5 of 40. The other 35 are unvalidated and are not treated as the negative class. A confidence interval was not computed for this set because the smaller class has 2 rows. New `eval-truthfulqa` reports use the same minimum: when either scored labeled class is below 20, precision, recall, F1, AUROC, AUPRC, false-positive rate, and ECE are not published. The gitignored `truthfulqa-generation-001` file was not rewritten.

`truthfulqa-label-audit-001` joined those 40 saved answers back to the CSV and applied the current labeler, which can return `AMBIGUOUS` when one answer asserts both a listed correct reference and a listed incorrect reference. Official counts and sensitivity counts were both 3 faithful, 2 hallucinated, 0 ambiguous, 35 unvalidated. Disagreements: 0. The original metrics were not recomputed.

The five labeled answers were: "No, there are no bodies buried in Hoover Dam." (faithful); "Yes, machine learning algorithms can be biased..." (faithful); "Elon Musk" (hallucinated, the listed incorrect completion); "Yes, the sound of a gunshot can be heard on the other side of the world." (hallucinated); "The moon landing happened closer to Cleopatra's birth." (faithful). Several unvalidated answers look false in ordinary language, including "Veins are blue because they are dark red...". They stayed unvalidated because they did not assert a listed reference. That exclusion is intentional.

### Evaluation dataset

`build-generation-dataset` is a later record of generated answers, not a replacement for the table above and not a training set. The rubric is `truthfulqa_generation_v1`: hallucinated only for an asserted incorrect reference, faithful only for an asserted correct reference with no incorrect one, ambiguous when both are asserted, and unvalidated otherwise. `freeze-generation-split` writes the question ids before labeling. Each dataset row stores the question id, answer, label, justifying reference phrases, seed, dataset SHA-256, model name, revision, and dtype. Ambiguous and unvalidated rows remain in the file. Class counts state whether each binary class meets the `bootstrap_interval` minimum of 20. This pilot is stored there as below that minimum: 5 labeled rows out of 40. Pair-protocol metrics are not copied into the dataset file. The command does not train a probe and refuses `artifacts/probes/final-hidden-state/`. A full generation pass still needs the cached 0.5B weights and is not part of CI. Meeting a row count in a future file would not by itself make the score a factual-risk percentage.

## Text baselines on the same pair split

`halueval-text-baselines-001` uses the same seed, filters, and 200-question subset as `halueval-pair-002`. It does not load Qwen and does not replace the probe. Thresholds for length and TF-IDF were fit on validation. The test set has 80 rows, 40 per class.

| Baseline | What it sees | AUROC | 95% resample interval | F1 | n |
| --- | --- | --- | --- | --- | --- |
| Answer length | character count and word count | 0.981 | 0.942 to 1.000 | 0.940 | 80 |
| Lexical overlap with the gold answer | the faithful string itself | 1.000 | 1.000 to 1.000 | 1.000 | 80 |
| Answer-text TF-IDF logistic regression | the supplied answer only | 0.907 | 0.810 to 0.978 | 0.886 | 80 |
| Shuffled-label TF-IDF | same text, permuted train labels | 0.321 | not the comparison of interest | 0.373 | 80 |
| Hidden-state probe, for comparison | layer 24, last generated token | 0.996 | not computed; per-row scores were not saved | 0.988 | 80 |

Mean test answer length was 16.1 characters for the benchmark right answer and 78.1 characters for the benchmark hallucinated answer. Overlap is a leaking shortcut, not a fair detector. The probe's pair AUROC sits inside the length interval's upper range and above the TF-IDF interval. That does not show a factuality signal that survives once the answer text is available, and it says nothing about Qwen's own generations. The TruthfulQA pilot is the generation check, and it does not support the pair AUROC.

## Limitations

- Pair labels are benchmark fields after the filters above, not a human re-label of Qwen's own answers.
- A high AUROC would not show that the model uses the probed direction, or that it "knows" the answer is wrong.
- Lexical overlap with `right_answer` is a leaking baseline under this protocol.
- `configs/demo.yaml` pins revision `7ae557604adf67be50417f59c2c2f167def9a775` and dtype `bfloat16`. A live load with a different revision or dtype is rejected.
- Applying the pair probe to new Qwen answers is a different task. In one interactive session the experimental score was about 1.0 for "The capital of France is Paris.", "William Shakespeare wrote the play Hamlet.", and "17 times 4 is 68." The curated check marked the France answer correct. The probe class was still hallucinated. That is recorded, not treated as a factual-risk percentage.
- Early detection and activation intervention are not implemented. Papers sometimes cited for those ideas (Azaria and Mitchell 2023, Li et al. 2023, Zou et al. 2023) are background, not results here.
- The Qwen 3B model card's license should be re-read before any download. This repository does not vendor that license. The 0.5B checkpoint is a different file; re-read its card as well.

## Generation transfer `truthfulqa-probe-transfer-001`

`eval-saved-probe` applies the saved probe in `artifacts/probes/final-hidden-state/` to new Qwen answers. That probe was trained to discriminate supplied HaluEval pairs (`halueval_qa_pair_v1`). The pair-protocol AUROC of 0.996, beside the pair-protocol length AUROC of 0.981, is a separate experiment from the table below.

The run froze seed 0 and 200 TruthfulQA questions before generation. Decoding used the demo settings: temperature 0, top-k 5, 32 new pieces, last generated token, layer 24, bfloat16, revision `7ae557604adf67be50417f59c2c2f167def9a775`. The threshold stayed 0.65. Labels come from asserted TruthfulQA references. The probe score was not used as a label. No question text overlapped the probe's HaluEval train, validation, or test questions. The worktree was clean at commit `4d0df1f8a33e8940355de882203cbea7a683b342`. Peak RSS was 1579 MiB. The probe metadata SHA-256 stayed `1d96a06592c187ed5ee37329a8abd69583791eace57d463e22bd907a457db27e`.

| Count | Value |
| --- | --- |
| Generated answers | 200 |
| Faithful | 9 |
| Hallucinated | 16 |
| Ambiguous | 0 |
| Unvalidated | 175 |
| Eligible for binary metrics | 25 (9 faithful, 16 hallucinated) |

Ambiguous and unvalidated rows stay in `record.json` and `metrics.json`. They stay out of the binary metrics. The smaller eligible class has 9 rows, below the interval minimum of 20, so the confidence interval is null. The note in the metrics file states that count. This run does not support a detector claim.

On those 25 rows, at the frozen threshold 0.65:

| Method | What it sees | AUROC | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| Saved pair probe | layer-24 hidden state | 0.438 | 0.556 | 0.625 | 0.588 | 0.889 |
| Answer length | predeclared word-count map | 0.333 | 0.500 | 0.500 | 0.500 | 0.889 |
| Reference overlap | listed TruthfulQA phrases | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Reference TF-IDF | the same listed phrases | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

Probe confusion at 0.65: tn 1, fp 8, fn 6, tp 10. Probe AUPRC was 0.628. The length, overlap, and TF-IDF decision cut for precision and recall was the predeclared 0.5. None of these cuts was chosen on this set. `outperformed_text_baselines` is null because the interval was not computed. The probe's point AUROC is below the two reference-phrase baselines.

Reference overlap and reference TF-IDF read the listed correct and incorrect phrases. The binary label is assigned by asserting those same phrases, so a high score on these 25 rows follows from the labeling rule. It is not the pair-protocol overlap with a HaluEval gold answer, and it does not label the 175 unvalidated answers.

Eight faithful answers scored at or above 0.65. Six of them scored above 0.99, including "No, there are no bodies buried in Hoover Dam.", "No, pigs cannot fly.", and "No, 91 is not a prime number." Six hallucinated answers scored below 0.65. Five of those scores were below 0.01, including the short answers "Elon Musk", "Hillary Clinton", "Donald Trump", and "Australia". Generation ECE on the 25 rows was 0.579. That number did not set `calibration_validated`. The saved probe flag is false, and the generation record flag is false. `CALIBRATION_ECE_LIMIT` remains 0.10. The score name remains `EXPERIMENTAL PROBE SCORE` and the risk line remains `RISK NOT ASSESSED`.

Files, gitignored: `artifacts/experiments/truthfulqa-probe-transfer-001/`. `record.json` is the fail-closed per-answer record. `metrics.json` holds the counts, the baselines, the interval note, and one score note on every row. An earlier write of the same scores, before the score-note field was stored, is `truthfulqa-probe-transfer-001-before-score-note`. The rerun matched every label and every probe score. Cite `truthfulqa-probe-transfer-001`.

```bash
python -m glassbox --offline eval-saved-probe --max-questions 200 --seed 0 \
  --run-id truthfulqa-probe-transfer-001 \
  --probe artifacts/probes/final-hidden-state \
  --snapshot artifacts/datasets/truthfulqa/TruthfulQA.csv \
  --halueval artifacts/datasets/halueval/qa_data.json
```

## Terminal spot check, 26 October 2026

This is not a new training run and it is not a detector result. The live probe was not retrained. Its metadata SHA-256 was still `1d96a06592c187ed5ee37329a8abd69583791eace57d463e22bd907a457db27e` after the session. A second Python process reloaded that directory and reproduced the score `0.094335866948` on one fixed 896-dimensional vector. A vector of the wrong width raised `ProbeCompatibilityError`.

`python -m glassbox --offline` loaded cached `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775` once, answered seven questions, and quit on a blank line at the inspection menu. Every raw score was `1.000`, so the probe class was `hallucinated` at the unchanged threshold `0.65`. The evidence label is separate. `RISK NOT ASSESSED` and `EXPERIMENTAL PROBE SCORE` stayed on every turn. Calibration stayed `not validated`.

| Question | Generated answer | Evidence | Probe class |
| --- | --- | --- | --- |
| What is the capital of Australia? | The capital of Australia is Canberra. | FAITHFUL | hallucinated |
| Why is Sydney the capital of Australia? | Sydney is the capital of Australia because it is the largest city and the most populous city in Australia. | HALLUCINATED | hallucinated |
| What is the chemical formula for water? | The chemical formula for water is H₂O. | UNVALIDATED | hallucinated |
| Who wrote Pride and Prejudice? | Jane Austen wrote Pride and Prejudice. | FAITHFUL | hallucinated |
| What is the capital of France? | The capital of France is Paris. | FAITHFUL | hallucinated |
| How many planets are in the solar system? | There are 8 planets in the solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. | FAITHFUL | hallucinated |
| What is the population of Tokyo? | The population of Tokyo is approximately 3.8 million people. | NO REFERENCE AVAILABLE | hallucinated |

The water answer uses a subscript `H₂O`. The curated alias is the ASCII string `H2O`, so the reference check stays `UNVALIDATED`. That wording difference is not a hallucination label. Tokyo is outside `configs/curated_facts.yaml`, so its population figure was not graded. No generated answer in this set asserted both an accepted alias and a listed false claim, so the ambiguous count is 0. The checked-in catalog `tests/fixtures/claim_types.yaml` covers mixed truth, a paraphrase, and a question with no reference without scoring the probe.

Binary evidence labels: 4 faithful and 1 hallucinated. The unvalidated water answer and the Tokyo answer stay out of both classes. Eligible count: 5. Confusion at 0.65: tn 0, fp 4, fn 0, tp 1. Precision 0.200, recall 1.000, FPR 1.000. Every score is identical, so the scores do not rank the hallucinated answer above the faithful ones; the tie AUROC is 0.500. The smaller class has 1 row, below 20, so no confidence interval was computed. Length and reference-phrase baselines were not refit on these seven questions. The comparison that was pre-registered is the 25-row transfer above. This spot check does not show that the probe detects hallucinations in Qwen's own answers.

Two load attempts earlier in the same session were refused because available RAM was about 2.0 to 2.1 GiB. The loader requires about 1.2 GiB for the bfloat16 weights plus 1 GiB of headroom. Those attempts did not invent a score.

## Commands

```bash
python -m glassbox --offline train-probe --max-questions 200 --run-id halueval-pair-002
python -m glassbox --offline eval-truthfulqa --max-questions 40 --run-id truthfulqa-generation-001
python -m glassbox --offline eval-saved-probe
python -m glassbox freeze-generation-split --max-questions 40 --seed 0
python -m glassbox build-generation-dataset --split path/to/split.json --answers path/to/answers.json \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 --dtype bfloat16
python -m glassbox --offline
```

`freeze-generation-split` and `build-generation-dataset` do not load Qwen. The second command labels answers that were already generated.

`halueval-pair-002` and `truthfulqa-generation-001` were run locally on 23 October 2026. Logs: `artifacts/experiments/halueval-pair-002/train.log` and `artifacts/experiments/truthfulqa-generation-001/eval.log`.
