# Research report

## Question

Does a logistic probe on one frozen Qwen2.5-0.5B hidden state detect factual errors in answers the model wrote?

## Method

The model is not trained. The live probe was fit on supplied HaluEval pairs: layer 24, last generated token, width 896, chat template, threshold 0.65 chosen on the pair validation split. A second probe was fit on Qwen generations labeled by passage evidence (`halueval_generation_evidence_v1`). Ambiguous and unvalidated rows stay out of binary metrics. Preprocessing for the pair probe is fit on its training split. Baselines are answer length, TF-IDF, and lexical or reference overlap. Overlap baselines that read the same phrases as the label are shortcuts.

## Results

| Check | n | Result |
| --- | --- | --- |
| Pair test, supplied answers | 80 examples, 40 per class | AUROC 0.996. Length AUROC 0.981. Overlap AUROC 1. ECE 0.114. Calibration rejected. |
| Pair probe, controlled format | 52 completions, 13 questions | AUROC 0.494. Sentence faithful FPR 1.00. Faithful length–score r 0.938. Length-matched AUROC 0.438. |
| Pair probe on TruthfulQA generations | 200 generated, 25 labeled | AUROC 0.438. Smaller class 9. Interval null. |
| Generation-001 hidden state | 20 labeled test rows | AUROC 0.646. TF-IDF 0.657. Length 0.626. |
| Generation-001 coverage | 720 development answers | 590 unvalidated, 12 ambiguous. |
| Issue #18 gate | validation 14 faithful / 7 hallucinated | Rejected. ECE limit remains 0.10. |

Mean-pooled hidden states were not stored for generation-001, so that comparison was skipped. Early-warning lead time stays null because per-prefix evidence labels are not stored. No layer was selected on the test set.

## Conclusion

The hidden state at the last generated token separates supplied HaluEval strings largely by length and wording. It does not separate factual errors in Qwen's own answers on the labeled sets above. The honest status is experimental. `EXPERIMENTAL PROBE SCORE` and `RISK NOT ASSESSED` stay on the terminal.

## Next measurement

A human review of the unvalidated queue, with evidence notes, and a validation split that reaches 20 faithful and 20 hallucinated rows without reusing the generation-001 test ids. Until that exists, do not retune 0.65 and do not treat a new AUROC as a detector claim. Details and the shortcut analysis are in [SCIENTIFIC_LIMITATIONS.md](SCIENTIFIC_LIMITATIONS.md) and [RESEARCH.md](RESEARCH.md).
