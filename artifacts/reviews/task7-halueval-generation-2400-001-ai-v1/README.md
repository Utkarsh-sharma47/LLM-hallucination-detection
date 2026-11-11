# AI-assisted review of 2,400 frozen Qwen answers

Related to issue #41. This is one passage-grounded AI annotation per generated answer, with a versioned rubric, source hashes and a reproducible handoff index. The team authorized AI-assisted annotation. These are not independent human reviews or adjudicated labels, and this PR does not assert that every original acceptance criterion of #41 is met.

## Contents

~~~text
task7-halueval-generation-2400-001-ai-v1/
├── annotations.jsonl        # immutable first-pass evidence, decisions and timestamps
├── decisions-*.json         # 25 original batches checked against annotations
├── RUBRIC.md                # rules used and interpretation limits
├── source-manifest.json     # source/feature hashes and original run provenance
├── manage_review.py         # stdlib-only validator and derivative builder
└── generated/
    ├── summary.json         # counts by split and review status
    └── training-index.jsonl # binary candidates; IDs and targets, no hidden states
~~~

Each annotation includes the full question, generated answer and knowledge passage. Original automatic labels, predictions and weights are not overwritten. The large local CSV and per-row Markdown convenience views are omitted to avoid duplicating the text.

## Results

| Split | FAITHFUL | HALLUCINATED | AMBIGUOUS | UNVALIDATED | Binary candidates |
|---|---:|---:|---:|---:|---:|
| train | 125 | 695 | 23 | 597 | 820 |
| validation | 44 | 219 | 9 | 208 | 263 |
| test | 29 | 159 | 7 | 285 | 188 |
| Total | 198 | 1,073 | 39 | 1,090 | 1,271 |

The 1,129 uncertain/mixed rows stay visible and excluded from the binary index. These are annotation counts, not detector performance. Training candidates are imbalanced toward HALLUCINATED. Validation and test exceed twenty candidates per binary class, but this does not establish power, calibration or label reliability.

Review used one continuing Codex pass in 25 batches, not multiple independent reviewers. Individual displays omitted automatic labels and scores; aggregate results and some previous examples were already visible, so full blinding is not claimed. There is no inter-rater agreement estimate or adjudication result. Original timestamps are preserved. A precise model version and per-call configuration were not recorded.

## Validate and rebuild

Run from the repository root with Python 3.10 or later; no project dependencies:

~~~powershell
python artifacts/reviews/task7-halueval-generation-2400-001-ai-v1/manage_review.py
python artifacts/reviews/task7-halueval-generation-2400-001-ai-v1/manage_review.py --rebuild
~~~

Checks cover the committed snapshot hash, exact quote containment, matching batch decisions, unique complete coverage, AI provenance, timestamps, and split isolation by recorded group and normalized passage. Rebuilding writes only the two generated files; it never relabels an answer or rewrites review history.

For verification against the original experiment:

~~~powershell
python artifacts/reviews/task7-halueval-generation-2400-001-ai-v1/manage_review.py --source "PATH/TO/predictions.json" --features "PATH/TO/features.npz" --rebuild
~~~

Use the files from the team-provided task7-halueval-generation-2400-001 archive. Ask the teammate who generated that archive for access; it is not bundled in this PR. The manifest records checksums for predictions, features, splits, report and record. A newly generated run with the same questions is not interchangeable.

The source mode checks full original row hashes and unchanged question, answer, passage, group, ID and split. The feature mode checks the complete cached file hash. Source verification found 1,223 differences from the original automatic four-way labels. This is not independent reviewer disagreement or proof that AI labels are superior.

## Handoff for the generation-probe contributor

1. Audit semantic consistency before accepting these labels as supervision. See RUBRIC.md, especially missing names/pronouns, incomplete answers, mixed claims, non-exhaustive lists and dates. Structural checks do not prove the judgments correct.
2. Obtain and verify the matching frozen predictions and features. The NPZ contains question_id, split, label_name and vectors. Vectors have shape (2400, 896), dtype float32. Its label_name array contains OLD automatic labels; do not use those as reviewed targets.
3. Join generated/training-index.jsonl to vectors by exact question_id. Check uniqueness, complete candidate coverage, matching split and feature hash. Do not rely on row order or use pair-probe features for dataset reference answers. Targets are FAITHFUL=0 and HALLUCINATED=1, matching glassbox.data.schema.Label.
4. Fit preprocessing and the classifier on the 820 training candidates only. Tune hyperparameters and thresholds on the 263 validation candidates. Evaluate the 188 test candidates after choices are fixed. Preserve group splits; document weighting or resampling rather than moving held-out rows.
5. Write a fresh experimental run. Preserve existing probes and original answers. Regenerated answers require new review unless exact answer text and provenance are verified. No model is trained by this PR.
6. Retain reviewer_type=AI, human_reviewed=false, adjudicated=false, policy version and annotation/source hashes in run provenance. Report experiments under this rubric; compare text/length baselines and do not call an unvalidated score a hallucination probability.

~~~text
verify annotation and cached-feature hashes
feature_by_id = unique question_id -> (split, vector)
for candidate in binary index
    require matching ID and split
    add vector and reviewed target to that split
fit train; tune validation; evaluate test after freezing choices
~~~

The index is a handoff format, not a new training CLI. The existing train-generation-probe command generates answers and imports --reviews through read_human_reviews/apply_human_review in src/glassbox/data/evidence.py. That path discards AI-specific provenance and sets human_reviewed=True. Do not import this AI bundle through it unchanged. A cached-feature adapter or provenance-aware import is follow-up work. Feature loading in src/glassbox/data/generation_v2.py and training/finalization in src/glassbox/probes/generation_train.py are starting points, not an end-to-end tested command for this bundle.

## Remaining work and provenance

Original #41 additionally asks for independent overlapping judgments, adjudication, atomic claim spans, agreement statistics and justified validation/test coverage. This pass has answer-level rationales and passage quotes, not systematic atomic answer spans, confidence ratings or passage offsets. Splits are preserved and isolated, but this PR cannot retroactively establish when they were frozen.

If the team adopts an AI-only scope, record that scope on #41. Otherwise independent review and adjudication remain follow-up work. Corrections should be append-only events or a separately versioned snapshot recording previous label, reason, reviewer, timestamp and hashes. Preserve v1; do not edit batches to erase prior judgments or bypass integrity checks.

The original generation record reports commit 1efec914a462b00c4491ace549ab21eb7e352cba, a dirty worktree and publishable_as_reproducible=false. These remain in the source manifest. Adding annotations does not make that model run cleanly reproducible, and its old metrics/rejected calibration do not apply to a newly trained probe using these targets.
