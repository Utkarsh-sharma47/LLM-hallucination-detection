# Experiments

Index of local runs. Full methods and tables are in [RESEARCH.md](RESEARCH.md). How a record is stored is in [DEVELOPMENT.md](DEVELOPMENT.md). Nothing under `artifacts/` is in Git. A number on this page was copied from the report named in the same row.

Status words used here and in [ROADMAP.md](ROADMAP.md):

| Status | Meaning |
| --- | --- |
| `COMPLETED` | The acceptance bar for that task is met in this tree. |
| `PLANNED` | Not started. No result is claimed. |
| `IN PROGRESS` | Code or a run exists, and the acceptance bar is not met yet. |
| `BLOCKED` | Waiting on a decision or a missing input this tree cannot supply. |
| `NOT VALIDATED` | A run or a check exists, and it does not support the scientific claim. |

## Verified results

These numbers match a saved report. They do not validate hallucination detection.

| Run | Protocol | What was scored | Model | Result | Artifacts |
| --- | --- | --- | --- | --- | --- |
| `halueval-pair-002` | `halueval_qa_pair_v1` | Supplied HaluEval answers. 200 questions, seed 0. | `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`, `bfloat16` | Test precision 0.976, recall 1.000, F1 0.988, AUROC 0.996, AUPRC 0.995, FPR 0.025. Confusion tn 39, fp 1, fn 0, tp 40. Threshold 0.65. | Live probe: `artifacts/probes/final-hidden-state/`. Record: `artifacts/experiments/halueval-pair-002/record.json`. |
| `halueval-text-baselines-001` | Same pair split | Answer text only. No Qwen load. | none | Length AUROC 0.981. Gold-answer overlap AUROC 1.000. TF-IDF AUROC 0.907. | `artifacts/experiments/halueval-text-baselines-001/` |

The pair probe's AUROC sits next to the length baseline, and overlap recovers the label because the faithful text is the gold string. Dataset SHA-256 `89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88`, HaluEval commit `207f479c555a68b8d2431631a95083b84b2accbe`.

Command that wrote the live probe. A later run with the same destination replaces it:

```bash
python -m glassbox --offline train-probe --max-questions 200 --run-id halueval-pair-002
python -m glassbox text-baselines --max-questions 200 --run-id halueval-text-baselines-001
```

## Known failures

| Run or check | Failure |
| --- | --- |
| `halueval-pair-002` calibration | Test ECE 0.114, above 0.10. `calibration_validated` is false. The demo shows `EXPERIMENTAL PROBE SCORE` and `RISK NOT ASSESSED`. |
| `halueval-pair-002` provenance | Recorded commit `9b8bcc48019774510a7947730504fab29272e75b` with a dirty worktree. Not a clean-checkout reproduction. Related: issue #8. |
| Interactive pair-probe scores | Correct answers, including "The capital of France is Paris.", received an experimental score near 1.0 while the curated check marked France correct. |
| `halueval-pair-001` | Previous live probe, now `artifacts/probes/final-hidden-state.previous/`. dtype was not stored. Do not quote it as the current result. |
| `halueval-generation-001` | Qwen generated 720 HaluEval answers and 40 TruthfulQA answers. Hidden-state test AUROC 0.646 on 20 labeled rows. TF-IDF 0.657. Length 0.626. False-span similarity 1.000. Right-answer dissimilarity 0.949. Confusion tn 6, fp 3, fn 4, tp 7. Threshold 0.25, chosen on the generation validation split, not the pair threshold 0.65. |
| Generation calibration | Rejected. Validation had 14 faithful and 7 hallucinated rows. Issue #18 now requires 20 of each class on the validation split and ECE ≤ 0.10 before a generation score can be called a calibrated probability. The five-label TruthfulQA pilot is ineligible. The pair probe is not rewritten. Test ECE was not computed. |
| Generation intervals | Not computed. The smaller labeled test class has 9 rows, below 20. |
| Generation coverage | 590 of 720 development answers stayed `UNVALIDATED`. 12 were `AMBIGUOUS`. Those rows were excluded from training. No human review file was applied. |
| Generation provenance | Recorded commit `9692d1ff8f4c205274a53d399894f632e4d7a3fd` with a dirty worktree. `publishable_as_reproducible` is false. That commit does not contain the training code. |
| Pair probe on the generation test | At threshold 0.65: AUROC 0.333, tn 0, fp 9, fn 1, tp 10. The pair AUROC of 0.996 is a different task. |
| TruthfulQA external slice of `halueval-generation-001` | 3 faithful, 3 hallucinated, 34 unvalidated, 1 coverage upgrade. AUROC 0.444 on 6 rows. Threshold was frozen at 0.25 and was not fit here. |

Generation command. It refuses an existing run directory and refuses `final-hidden-state`:

```bash
python -m glassbox --offline train-generation-probe \
  --max-questions 720 --external-questions 40 \
  --run-id halueval-generation-001
```

Probe: `artifacts/probes/generation-runs/halueval-generation-001/`. Report: `artifacts/experiments/halueval-generation-001/report.json`. Also in that directory: `predictions.json`, `review_queue.json`, `splits.json`, `features.npz`, `record.json`.

False positives on the labeled generation test include "Löwenherz was created first..." (score 0.970, right answer `Löwenherz`). False negatives include "Sydney" where the right answer is Melbourne. One false negative is a weak label: the Elvin Hayes answer mentions the Houston Rockets because the question does. Unvalidated misses include "Vincenzo Bellini" where the right answer is Niccolò Paganini.

## Live check `live-terminal-001`

`NOT VALIDATED`. This is a fresh terminal run of the live pair probe on answers Qwen wrote. It is not a second training run. Command:

```bash
python -m glassbox live-test --suite --run-id live-terminal-001
```

Interactive use is `python -m glassbox live-test`. The model loads once from the cache. A blank line quits. `--trace-tokens` prints each piece and withholds the probe until the final generated token.

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`, greedy, 32 new pieces |
| Probe | `artifacts/probes/final-hidden-state/`, protocol `halueval_qa_pair_v1`, threshold 0.65, not refit |
| Representation | last generated token, layer 24, dimension 896, inside the answer span |
| Cases | 22 generations plus the supplied completion `Paris.` |
| Labels | 14 faithful, 4 hallucinated, 0 ambiguous, 5 unvalidated. The faithful count includes `Paris.` |
| Binary rows | 17 generated labeled rows. The smaller class has 4, so the interval was withheld and the result is inconclusive |
| Point metrics | precision 0.231, recall 0.750, F1 0.353, AUROC 0.558, AUPRC 0.329, FPR 0.769. Confusion tn 3, fp 10, fn 1, tp 3 |
| Length | AUROC 0.731. No threshold was fit. TF-IDF was not fit, because this suite has no training split |
| Calibration | not validated. The report keeps `EXPERIMENTAL PROBE SCORE` and `RISK NOT ASSESSED` |
| Early warning | not measured |

The supplied answer `Paris.` scored 1.000 and was predicted hallucinated. The reference check is supported, and the factuality status stays `FAITHFUL`. The same pattern appears on Qwen's own sentence "The capital of France is Paris." Short answers such as `Paris` and `1969` scored near 0. The false answer `Sydney` scored 0.002 and was the false negative. `H₂O` stayed unvalidated because the subscript does not match the registered alias `H2O`.

The generation probe was scored on the same vectors and was not used to choose the threshold or the label. Its scores are in `cases.json`. They do not turn this run into a validated detector.

Report: `artifacts/experiments/live-terminal-001/report.json`. Cases: `cases.json`. Summary: `summary.md`.

## Preliminary findings

Not large enough to support a detector claim. Intervals were withheld.

| Run | What it is | Counts | Metrics on the labeled rows only |
| --- | --- | --- | --- |
| `truthfulqa-generation-001` | `eval-truthfulqa`, seed 0, threshold frozen at the pair value 0.65. Dataset SHA-256 `b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c`. No upstream commit is pinned, only that hash. | 40 generations, 3 faithful, 2 hallucinated, 35 unvalidated | Precision 0.250, recall 0.500, F1 0.333, AUROC 0.500, FPR 1.000, ECE 0.797. Confusion tn 0, fp 3, fn 1, tp 1. |
| `truthfulqa-label-audit-001` | Same 40 answers, current assertion labeler, no model load | 3 faithful, 2 hallucinated, 0 ambiguous, 35 unvalidated. Disagreements: 0. | Metrics were not recomputed. |
| `halueval-generation-pilot-001` | 24-question timing check before the span filter was corrected | Did not train a probe. | Do not cite its labels. The 720-question run used the corrected labeler. |

```bash
python -m glassbox --offline eval-truthfulqa --max-questions 40 --run-id truthfulqa-generation-001
python -m glassbox audit-generations --run-id truthfulqa-label-audit-001
```

## Hypotheses

Not tested. No number belongs in this section.

- An earlier generated token can flag a false claim before the answer is finished. `early_warning_lead` is a helper. No event token is defined. Issue #7.
- Editing the probed direction changes what Qwen writes. No activation edit has been run.
- A layer other than the pre-registered last tensor would be a better generation feature. The generation run confirmed layer 24, dim 896, 25 tensors, and `last_generated_token` on the loaded model. It did not sweep layers. A mean over generated pieces scored 0.622 AUROC on 21 validation rows and was not allowed to replace the saved vector. Issue #13.

## Future work

- A reviewed label set for the unvalidated generation queue, with contradiction defined against the passage rather than against the benchmark false span. The queue is `artifacts/experiments/halueval-generation-001/review_queue.json`. Issues #11 and #12. The generation run does not close them: it trained a new probe, it refit the threshold to 0.25, and the labeled test class is below 20.
- A generation calibration gate that is pre-registered before ECE is computed. Issue #18. The five-label TruthfulQA pilot is ineligible. `halueval-generation-001` also missed the existing 15-per-class rule.
- A generation record that fails closed when provenance or a per-example row is missing. Issue #17. The generation run stores provenance and sets `publishable_as_reproducible` false for a dirty tree. That is not the same as the fail-closed test those issues ask for.
- A clean-checkout rerun of `halueval-pair-002`. Issue #8. Do not lower the ECE limit.

## What is not an experiment

`python -m glassbox --samples` and `--analyze-text` do not load Qwen and do not score a probe. Smoke uses `configs/default.yaml` (`Qwen/Qwen2.5-3B-Instruct`, downloads off) and is not a probe result.
