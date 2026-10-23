# Research plan

Recorded numbers and reproduction commands are in [EXPERIMENTS.md](EXPERIMENTS.md). The method is in [RESEARCH.md](RESEARCH.md). Open tasks are in [ROADMAP.md](ROADMAP.md). This page only keeps the categories apart.

## Verified results

- `halueval-pair-002` discriminates supplied HaluEval answers. Test AUROC 0.996, next to an answer-length AUROC of 0.981. Gold-answer overlap AUROC is 1. Calibration was rejected (test ECE 0.114).
- `halueval-generation-001` trained a separate probe on Qwen's own answers. Test AUROC 0.646 on 20 labeled rows, below TF-IDF at 0.657. False-span similarity AUROC is 1. Calibration was rejected. This is not a validated detector.

## Preliminary findings

- `truthfulqa-generation-001` labeled 5 of 40 generations. Metrics on those five rows are not a detector result.
- The generation run's TruthfulQA slice labeled 6 of 40. AUROC 0.444. Not used for training or threshold selection.
- `halueval-generation-pilot-001` was a timing check with an older span filter. Do not cite it.

## Hypotheses

Untested. Do not attach a metric.

- Earlier tokens can warn before the answer is finished.
- An activation edit along the probe direction would change factuality.
- Another layer would detect generation errors better than the pre-registered last tensor.

## Known failures

- The live pair probe assigns experimental scores near 1 to correct answers such as "The capital of France is Paris."
- Transferred to the generation test at its own threshold, that probe marked every faithful answer hallucinated.
- Generation labels abstain on 590 of 720 answers, including answers that look false, and the positive class is recoverable from the contradicted span.
- `halueval-pair-002` and `halueval-generation-001` both record a dirty worktree. Neither is publishable as a clean-checkout reproduction.

## Future work

Review the unvalidated generation queue before training another classifier. Issue [#11](https://github.com/Utkarsh-sharma47/LLM-hallucination-detection/issues/11) asks for that dataset without training a probe. Issue [#12](https://github.com/Utkarsh-sharma47/LLM-hallucination-detection/issues/12) asks to score the saved pair probe at threshold 0.65, which the generation run did as a transfer check and then went beyond by fitting a new threshold. Issue [#18](https://github.com/Utkarsh-sharma47/LLM-hallucination-detection/issues/18) is the later calibration gate. None of those issues is closed by the current runs.
