# Scientific limitations

This project does not have a validated hallucination detector. The limits below are the recorded results. They are not placeholders.

## Pairwise discrimination is not factuality detection

`halueval-pair-002` ranks a supplied HaluEval false answer against the benchmark's right answer. Its pair-test AUROC is about 0.996, next to an answer-length AUROC of about 0.981 and a lexical-overlap AUROC of 1. The threshold 0.65 was chosen on that validation split. Calibration was rejected (test ECE 0.114, limit 0.10).

On 52 controlled completions of the same questions, short versus sentence form, the same probe has AUROC 0.494. Sentence-form faithful answers have a false-positive rate of 1.00 at 0.65. The faithful length–score correlation is 0.938. After length matching, AUROC is 0.438. The probe is tracking answer format.

## A generation probe is a different task, and it did not win

`halueval-generation-001` trained on Qwen's own answers. The held-out test has 20 labeled rows. Last-piece AUROC is 0.646, below TF-IDF at 0.657. Length is 0.626. False-span similarity and right-answer overlap are shortcuts because they see the phrases that define the label. 590 of 720 development answers stayed unvalidated. The generation threshold 0.25 was fit on that validation split. It is not the live pair threshold.

## Calibration

Issue #18 requires 20 faithful and 20 hallucinated validation rows and ECE at most 0.10 at threshold 0.65, using pair-probe scores. Generation-001 validation has 14 faithful and 7 hallucinated rows. The gate stays rejected. `calibration_validated` is false. The terminal says `EXPERIMENTAL PROBE SCORE` and `RISK NOT ASSESSED`.

Closing the issue requires a disjoint validation split that meets those counts and the ECE limit. A larger generation set does not close it by itself. Generation-probe scores are not accepted as pair-probe calibration.

## What is not evidence

- A high pair-task AUROC.
- A green CLI session.
- An automatic evidence label. Those labels are proposed by phrase or passage rules. `human_confirmed` stays false until a person appends a review with an evidence note.
- A wording mismatch. That stays `UNVALIDATED`.
- A question outside `configs/curated_facts.yaml`. That stays `NO REFERENCE AVAILABLE`.

## What would support a detector claim

A pre-registered split with at least 20 examples in the smaller class, labels that the baselines cannot read directly, a threshold frozen before the test split, ECE at most 0.10 on a validation split that was not used to pick the model, and a result that beats the length and TF-IDF baselines by more than a tie. None of the saved runs meet that bar.
