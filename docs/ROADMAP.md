# Roadmap

Owners are unassigned unless noted. The GitHub repository owner is the default reviewer in [CODEOWNERS](../.github/CODEOWNERS), not the assignee of these tasks. Status was checked against the tree on 23 October 2026.

## P0 — blocked or required before a result

### 1. Make the saved HaluEval run reproducible

- **Status:** partial. The live probe is `halueval-pair-002` in `artifacts/probes/final-hidden-state/`. dtype is `bfloat16`, revision is pinned, and calibration was rejected (test ECE 0.114). `halueval-pair-001` is kept in `final-hidden-state.previous/`. The record's git commit is `9b8bcc48019774510a7947730504fab29272e75b` and it notes a dirty worktree, so the artifact is not a clean-checkout reproduction and was not rewritten. New records set `finished_reproduction` false when the worktree is dirty. The ECE limit remains 0.10. A clean rerun has not been executed.
- **Depends on:** the pinned QA file and a machine that can load `Qwen/Qwen2.5-0.5B-Instruct`. Quit other Glass Box processes first. An earlier attempt the same day failed the RAM check.
- **Acceptance:** a new run from a clean commit exits 0; metadata includes dtype; the demo process started after that run shows an experimental score and `RISK NOT ASSESSED`; the write-up keeps the lexical-overlap baseline next to the probe AUROC. Do not lower the ECE limit after seeing 0.104.
- **Owner:** unassigned.

### 2. Keep the scaffold and the demo from being mistaken for one config

- **Status:** explicit. `configs/default.yaml` remains the 3B scaffold with downloads off. `configs/demo.yaml` is the 0.5B CPU demo. Smoke still checks the scaffold. Do not retarget smoke to 0.5B to make the files match.
- **Depends on:** none.
- **Acceptance:** smoke still requires `configs/default.yaml` to name the 3B id with downloads off. The demo and `train-probe` still read `configs/demo.yaml`. A new contributor can point to [ARCHITECTURE.md](ARCHITECTURE.md) for which file a change belongs in. Do not retarget the smoke model to 0.5B just to make the files match.
- **Owner:** unassigned.

## P1 — correctness and maintenance

### 3. Remove the Ruff E501 ignore

- **Status:** open. `pyproject.toml` ignores E501 because `training.py`, `present.py`, and `cli.py` exceed 100 columns.
- **Depends on:** none.
- **Acceptance:** `ignore = ["E501"]` is gone and `ruff check src tests research_tests scripts` exits 0 with no behavior change in tests.
- **Owner:** unassigned.

### 4. One hidden-state scaler

- **Status:** done in code. `probes/preprocess.py` is the only scaler. The saved `halueval-pair-002` probe was not rewritten. The arithmetic is unchanged, so that artifact still loads.
- **Depends on:** task 1 if you want a byte-identical probe before and after the refactor. Do not change the arithmetic while claiming an old artifact still matches.
- **Acceptance:** one function is used by fit and by inference; `tests/test_probe_training.py` still passes; a saved probe from task 1 still loads or the schema version is incremented and the old file is rejected with `ProbeCompatibilityError`.
- **Owner:** unassigned.

### 5. Decide the fate of `QAExample`

- **Status:** local reads are implemented. `load_examples` reads a pinned file and does not download. HaluEval rows keep the pair-protocol labels. TruthfulQA rows are `UNVALIDATED` reference questions, not training labels. A missing file still raises, and the function does not import `datasets`.
- **Depends on:** none.
- **Acceptance met for local files:** `load_examples("halueval_qa")` returns pair-protocol `QAExample` rows from the pinned snapshot without a network call. `load_examples("truthfulqa")` returns `UNVALIDATED` reference rows and does not invent a model answer. A requested download of a missing file is still `NotImplementedError`.
- **Owner:** unassigned.

### 6. Choose a code license

- **Status:** blocked on the maintainer. No `LICENSE` file.
- **Depends on:** none.
- **Acceptance:** a license file the maintainer selected, and README no longer says the license is unselected.
- **Owner:** repository maintainer. Not chosen in this tree.

## P2 — after a saved probe

### 7. Generation-protocol labels

- **Status:** dataset contract is in code. `freeze-generation-split` writes TruthfulQA question ids before labeling. `build-generation-dataset` stores the seed, dataset SHA-256, model name, revision, and dtype with each row, and keeps ambiguous and unvalidated answers in the file. It does not train a probe and does not write `artifacts/probes/final-hidden-state/`. `truthfulqa-generation-001` remains the 40-answer pilot: 3 faithful, 2 hallucinated, 0 ambiguous, 35 unvalidated. Both binary classes are below the interval minimum of 20, so that pilot is not a detector result. Pair-protocol metrics are not copied into the dataset record. `truthfulqa-label-audit-001` reapplied the current labeler and found no disagreements. The original metrics were not recomputed.
- **Depends on:** task 1, so the pair protocol is not confused with this one. A new generation pass still needs the cached 0.5B weights and is not part of CI.
- **Acceptance:** a written rubric, labels that are not "the strings differ," a split frozen before generation, and a report that does not reuse pair-protocol metrics as if they were the same experiment. The rubric and the dataset writer meet that contract. The pilot does not meet the sample-size part of that bar.
- **Owner:** unassigned.

### 8. Early detection

- **Status:** not started. `early_warning_lead` is only a helper. No event token is defined, so no lead time has been computed and none will be filled in after the fact.
- **Depends on:** a pre-registered definition of the event token, and a probe that scores more than one position.
- **Acceptance:** the event definition is in this file before any test numbers are computed; the metric is null when no timely flag exists; no threshold is chosen on the test split.
- **Owner:** unassigned.

### 9. Activation intervention

- **Status:** not started. No activation edit has been run. Correlation in a probe is not a reason to call a direction causal.
- **Depends on:** task 8 and the controls in the old plan (random direction, shuffled-label direction, a fluency check).
- **Acceptance:** an edit runs in code, the controls run with it, and the write-up does not call the direction a truth direction.
- **Owner:** unassigned.

### 10. Type checker

- **Status:** not started. `py.typed` is present. Mypy is not installed as a gate.
- **Depends on:** task 3 is independent; do this after the public probe metadata is stable.
- **Acceptance:** a config checked in CI that exits 0 on `src/glassbox` without disabling the run wholesale.
- **Owner:** unassigned.

## Done

- Offline package import, download guard, and smoke test against `configs/default.yaml`.
- Terminal demo: arbitrary questions, one loaded model per process, token probabilities, curated reference check, export, command legend.
- Pair-protocol training code, filters, disjoint splits, and unit tests that do not download Qwen.
- Pinned HaluEval file downloaded locally and checked against the SHA-256. Not committed.
- One experimental probe and held-out record on disk (`halueval-pair-002`, with `halueval-pair-001` kept as `.previous`). Not calibrated. Not committed.
- Text-only baselines on the same 200-question pair split (`halueval-text-baselines-001`). No Qwen load. Overlap recovers the pair label. Length is close to the probe.
- TruthfulQA generation pilot and a label audit that did not change the saved metrics.
- CI workflow for Ruff, pytest, and smoke. Manual workflow for a tiny model, plus an opt-in cached-Qwen job that skips instead of downloading. No routine Qwen download.

## Explicitly out of scope until the tasks above move

Hosted experiment tracking, a notebook extra, pooling HaluEval with TruthfulQA, and shipping model weights.
