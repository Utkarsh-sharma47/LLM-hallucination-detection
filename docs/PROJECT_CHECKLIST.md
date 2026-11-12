# Project checklist

Status reviewed on 13 November 2026 against the source tree, documentation, and local artifacts. Artifact folders are gitignored, so these experiment files exist only in this checkout unless copied or archived separately.

## Completed

- [x] Local Glass Box CLI generates answers, captures token distributions and hidden states, checks curated references, and exports a report. Pair-protocol and generation-protocol scores are kept distinct in the UI.
- [x] Offline model loading, pinned revisions, download guards, split isolation, artifact compatibility checks, run records, and routine CI/test scaffolding are implemented.
- [x] The Task 7 answer set and its frozen knowledge-group train/validation/test splits are present: 2,400 generated answers.
- [x] The Task 7 review bundle is complete under `artifacts/reviews/task7-halueval-generation-2400-001-ai-v1/`. Its single AI-assisted review labeled 198 answers FAITHFUL, 1,073 HALLUCINATED, 39 AMBIGUOUS, and 1,090 UNVALIDATED. Ambiguous and unvalidated items are excluded from binary fitting.
- [x] A generation probe was trained on the reviewed binary subset with train-only fitting, validation-only Platt calibration, and a held-out test. The standalone report records test AUROC 0.785, calibrated ECE 0.040, and Brier 0.108 on 188 test items. A separate 609-item calibration batch records AUROC 0.763 and ECE 0.024.
- [x] A Glass Box-compatible export exists at `artifacts/probes/generation-reviewed-task7-v1-glassbox/`. It is for Qwen2.5-0.5B-Instruct at the pinned revision, layer 24, final generated token, and is marked calibrated in that generation-probe metadata.
- [x] Report sections 8 and 9 are implemented in both `src/glassbox/glassbox_report.py` and the diagnostics HTML: section 8 describes available layer evidence; section 9 explains why lead time is not measured.
- [x] The layer-wise extraction and probing run completed from `scripts/layerwise_reviewed_generation_probe.py`. It uses the reviewed Task 7 index and preserves the existing split assignments.
- [x] A Task 7 prefix-level early-warning evaluation pipeline is implemented in `scripts/train_task7_early_warning_probe.py`, including all-prefix layer-24 features and evidence labels, a validation-selected threshold, held-out lead-time metrics, and an HTML-report sidecar. The updated script persists `prefix_records.npz` and `prefix_evidence.jsonl`. Run 002 picked its threshold (0.05) by validation F1 alone and flagged every faithful answer (false-warning rate 100%), so its 22/22 coverage was meaningless. Run 003 (`scripts/rethreshold_task7_early_warning.py`, no retraining) re-selected the threshold on validation with a 20% false-warning cap: threshold 0.82, test false-warning rate 13.8%, timely warnings on 3 of 22 defined events. Only 22 of 159 hallucinated test answers have a prefix event at all, and the automatic prefix labels disagree with the review on 161 of 188 test answers. Exploratory and not a useful early warning. The run-003 rule was chosen after run 002's test result was seen.

## In progress

- [x] The layer-wise run replayed 1,271 binary-reviewed answers, recovered all 25 hidden-state streams, and fit one grouped-CV-selected probe per layer. Its report, features, and probes are in `artifacts/probes/layerwise-task7-ai-v1-001/`.
- [ ] Review the completed layer-wise report before naming a preferred layer. The existing script evaluates every layer on the held-out test set for comparison; choosing the best layer from those test scores would make the chosen score optimistically biased. Treat the sweep as exploratory, then confirm any chosen layer on a new untouched dataset.
- [x] Checked replay alignment: 1,257 rows had matching answer token counts; 14 did not. On the checked rows, layer-24 cosine alignment was 1.0 median and 0.9999999 at the 5th percentile. Original generated token IDs were not retained, so the 14 mismatched rows cannot be exactly aligned to their cached vectors.

## Still needed for a stronger detector claim

- [ ] Obtain independent human passage-grounded reviews for at least a meaningful subset, with blind review, adjudication of disagreements, agreement statistics, and preserved provenance. Current labels are a single AI pass, not human-verified ground truth.
- [x] ECE difference explained: the export refits Platt scaling on validation in the app's form (applied to the raw score, not its logit), so its ECE (0.043 test, 0.022 batch) differs slightly from the standalone report (0.040, 0.024). The export metadata is canonical for the app because it matches what the app computes.
- [ ] Update `README.md`, `docs/RESEARCH.md`, `docs/ROADMAP.md`, and `docs/EXPERIMENTS.md`. Those docs still contain earlier results (for example, AUROC 0.522 / ECE 0.114) and older pair-probe gate status in places.
- [ ] Run a clean-checkout reproduction of the reviewed generation-probe training and calibration, with the code commit, model revision, dtype, dataset hashes, seed, and dependencies recorded. Current artifacts were created from a dirty worktree and are local outputs.
- [ ] Evaluate beyond this one model and HaluEval-style distribution, with independently reviewed data and answer-text/length baselines. Current performance is for one Qwen2.5-0.5B checkpoint and this reviewed dataset; calibration does not establish broad factuality reliability.
- [ ] Keep the pair-probe generation calibration gate separate. The pair-probe artifact itself is marked calibrated for its supplied-answer protocol (test ECE 0.030, threshold 0.95), but the pre-registered generation gate at fixed threshold 0.65 remains rejected. The first attempt had only 16 eligible faithful answers. The preregistered calibration batch 001 fixed the sample size (76 faithful, 533 hallucinated, 609 eligible) and the gate rejected again: ECE 0.259, AUROC 0.514. The pair probe does not transfer to Qwen's own answers. A calibrated generation-probe score does not make that gate pass.
- [ ] Add a reproducible artifact export or archival process for reports, feature provenance, and probe metadata. Current `artifacts/` outputs are gitignored and are not part of a checkout or CI run.

## Not implemented yet

- [ ] Activation intervention: implement an intervention with random-direction and shuffled-label controls plus a fluency check. No causal intervention has been run.
- [ ] Engineering follow-ups from `docs/ROADMAP.md`: remove Ruff's E501 ignore, configure a type checker, and resolve the maintainer's open license decision.

## Recommended order

1. Compare the completed layer metrics as exploratory results; do not select a preferred layer from test scores.
2. Independently review/adjudicate labels, then retrain and recalibrate under the frozen protocol.
3. Reproduce from a clean commit and update the research docs to match the final artifacts.
4. Define an alarm criterion on validation, improve review/protocol agreement with independent human annotation, then evaluate a revised early-warning probe on a new untouched set. Keep activation intervention as separate follow-up work.
