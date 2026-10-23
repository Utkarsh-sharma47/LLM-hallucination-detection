# Roadmap

Owners are unassigned unless noted. The GitHub repository owner is the default reviewer in [CODEOWNERS](../.github/CODEOWNERS), not the assignee of these tasks. Status was checked against the tree on 24 October 2026.

## P0 — blocked or required before a result

### 1. Make the saved HaluEval run reproducible

- **Status:** `NOT VALIDATED`. The live probe is `halueval-pair-002` in `artifacts/probes/final-hidden-state/`. dtype is `bfloat16`, revision is pinned, and calibration was rejected (test ECE 0.114). `halueval-pair-001` is kept in `final-hidden-state.previous/`. The record's git commit is `9b8bcc48019774510a7947730504fab29272e75b` and it notes a dirty worktree, so the artifact is not a clean-checkout reproduction. Issue #8. Do not lower the ECE limit.
- **Depends on:** the pinned QA file and a machine that can load `Qwen/Qwen2.5-0.5B-Instruct`. Quit other Glass Box processes first. An earlier attempt the same day failed the RAM check.
- **Acceptance:** a new run from a clean commit exits 0; metadata includes dtype; the demo process started after that run shows an experimental score and `RISK NOT ASSESSED`; the write-up keeps the lexical-overlap baseline next to the probe AUROC. Do not lower the ECE limit after seeing 0.104.
- **Owner:** unassigned.

### 2. Keep the scaffold and the demo from being mistaken for one config

- **Status:** `COMPLETED`. `configs/default.yaml` remains the 3B scaffold with downloads off. `configs/demo.yaml` is the 0.5B CPU demo. Smoke still checks the scaffold. Do not retarget smoke to 0.5B to make the files match.
- **Depends on:** none.
- **Acceptance:** smoke still requires `configs/default.yaml` to name the 3B id with downloads off. The demo and `train-probe` still read `configs/demo.yaml`. A new contributor can point to [ARCHITECTURE.md](ARCHITECTURE.md) for which file a change belongs in. Do not retarget the smoke model to 0.5B just to make the files match.
- **Owner:** unassigned.

## P1 — correctness and maintenance

### 3. Remove the Ruff E501 ignore

- **Status:** `PLANNED`. `pyproject.toml` ignores E501 because `training.py`, `present.py`, and `cli.py` exceed 100 columns.
- **Depends on:** none.
- **Acceptance:** `ignore = ["E501"]` is gone and `ruff check src tests research_tests scripts` exits 0 with no behavior change in tests.
- **Owner:** unassigned.

### 4. One hidden-state scaler

- **Status:** `COMPLETED`. `probes/preprocess.py` is the only scaler. The saved `halueval-pair-002` probe was not rewritten. The arithmetic is unchanged, so that artifact still loads.
- **Depends on:** task 1 if you want a byte-identical probe before and after the refactor. Do not change the arithmetic while claiming an old artifact still matches.
- **Acceptance:** one function is used by fit and by inference; `tests/test_probe_training.py` still passes; a saved probe from task 1 still loads or the schema version is incremented and the old file is rejected with `ProbeCompatibilityError`.
- **Owner:** unassigned.

### 5. Decide the fate of `QAExample`

- **Status:** `COMPLETED` for local reads. `load_examples` reads a pinned file and does not download. HaluEval rows keep the pair-protocol labels. TruthfulQA rows are `UNVALIDATED` reference questions, not training labels. A missing file still raises, and the function does not import `datasets`. A requested download of a missing file is still `NotImplementedError` on purpose.
- **Depends on:** none.
- **Acceptance met for local files:** `load_examples("halueval_qa")` returns pair-protocol `QAExample` rows from the pinned snapshot without a network call. `load_examples("truthfulqa")` returns `UNVALIDATED` reference rows and does not invent a model answer. A requested download of a missing file is still `NotImplementedError`.
- **Owner:** unassigned.

### 6. Choose a code license

- **Status:** `BLOCKED` on the maintainer. No `LICENSE` file.
- **Depends on:** none.
- **Acceptance:** a license file the maintainer selected, and README no longer says the license is unselected.
- **Owner:** repository maintainer. Not chosen in this tree.

## P2 — after a saved probe

### 7. Generation-protocol labels

- **Status:** `NOT VALIDATED`. `halueval-generation-001` generated 720 HaluEval answers and 40 TruthfulQA answers with frozen Qwen2.5-0.5B. The evidence rubric is `halueval_generation_evidence_v1`. The split was frozen before generation. Test AUROC on 20 labeled rows was 0.646, below TF-IDF at 0.657 and far below false-span similarity at 1.000. Calibration was rejected. 590 of 720 development generations stayed unvalidated. The earlier TruthfulQA pilot remains a coverage check, not this result. Do not quote the pair AUROC of 0.996 as this experiment. Related issues: #5, #11, #12. This run does not close them. Issue #12 keeps the pair threshold at 0.65, and this run fit a new threshold of 0.25 for the new probe. The transfer of `halueval-pair-002` at 0.65 is reported separately: AUROC 0.333 on the 20 labeled rows, with the interval withheld because the smaller class has 9 rows.
- **Depends on:** task 1, so the pair protocol is not confused with this one.
- **Acceptance:** a written rubric, labels that are not "the strings differ," a split frozen before generation, and a report that does not reuse pair-protocol metrics as if they were the same experiment. The report exists. The sample-size bar for calibration and confidence intervals is still unmet, and the detector claim is rejected.
- **Owner:** unassigned.

### 8. Early detection

- **Status:** `PLANNED`. `early_warning_lead` is only a helper. No event token is defined, so no lead time has been computed and none will be filled in after the fact. Issue #7.
- **Depends on:** a pre-registered definition of the event token, and a probe that scores more than one position.
- **Acceptance:** the event definition is in this file before any test numbers are computed; the metric is null when no timely flag exists; no threshold is chosen on the test split.
- **Owner:** unassigned.

### 9. Activation intervention

- **Status:** `PLANNED`. No activation edit has been run. Correlation in a probe is not a reason to call a direction causal.
- **Depends on:** task 8 and the controls in the old plan (random direction, shuffled-label direction, a fluency check).
- **Acceptance:** an edit runs in code, the controls run with it, and the write-up does not call the direction a truth direction.
- **Owner:** unassigned.

### 10. Type checker

- **Status:** `PLANNED`. `py.typed` is present. Mypy is not installed as a gate.
- **Depends on:** task 3 is independent; do this after the public probe metadata is stable.
- **Acceptance:** a config checked in CI that exits 0 on `src/glassbox` without disabling the run wholesale.
- **Owner:** unassigned.

### 11. Review the unvalidated generation queue

- **Status:** `PLANNED`. `artifacts/experiments/halueval-generation-001/review_queue.json` lists the abstained rows. No review file was applied. The next experiment defines contradiction against the passage, with a second annotator, before another classifier is trained.
- **Depends on:** task 7. Issues #11, #12, and #18 describe a dataset, a frozen-threshold evaluation, and a later calibration gate. The current generation probe does not meet those bars.
- **Acceptance:** a review file that `data/evidence.py` can read, disagreement counts, and no new probe trained on labels that were filled in after seeing the score.
- **Owner:** unassigned.

## Done

- Offline package import, download guard, and smoke test against `configs/default.yaml`.
- Terminal demo: arbitrary questions, one loaded model per process, token probabilities, curated reference check, export, command legend.
- Pair-protocol training code, filters, disjoint splits, and unit tests that do not download Qwen.
- Pinned HaluEval file downloaded locally and checked against the SHA-256. Not committed.
- One experimental pair probe and held-out record on disk (`halueval-pair-002`, with `halueval-pair-001` kept as `.previous`). Not calibrated. Not committed. Reproducibility is `NOT VALIDATED` (task 1).
- Text-only baselines on the same 200-question pair split (`halueval-text-baselines-001`). No Qwen load. Overlap recovers the pair label. Length is close to the probe.
- TruthfulQA generation pilot and a label audit that did not change the saved metrics. Preliminary only.
- Generation-protocol code and one local run (`halueval-generation-001`). The detector claim is `NOT VALIDATED` (task 7). The live probe was not replaced.
- CI workflow for Ruff, pytest, and smoke. Manual workflow for a tiny model, plus an opt-in cached-Qwen job that skips instead of downloading. No routine Qwen download.

## Explicitly out of scope until the tasks above move

Hosted experiment tracking, a notebook extra, pooling HaluEval with TruthfulQA, and shipping model weights.
