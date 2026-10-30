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

- **Status:** a generation-evidence trainer exists (`train-generation-probe`) and does not replace the live pair probe. `halueval-generation-001` trained on Qwen answers with passage-grounded labels. Held-out AUROC 0.646 on 20 labeled rows, below TF-IDF 0.657. Calibration rejected. A controlled pair-probe check (`length-shortcut-pair-controlled-001`) ranks chance (AUROC 0.494) and assigns hallucination to every sentence-form faithful completion. The review queue and a 2400-question expansion plan exist; no human labels were invented, and the 20-per-class val/test target is not met. Issue #18 remains rejected. Details are in [RESEARCH.md](RESEARCH.md) and [EXPERIMENTS.md](EXPERIMENTS.md).
- **Depends on:** task 1, so the pair protocol is not confused with this one. A reviewed label set for the unvalidated queue, and enough labeled rows for 20 per class on validation and test. `build-generation-v2` assembles cached 001 rows onto a 2400-question plan without rewriting 001. Generating the 336 new validation questions is the remaining compute step; it has not been run.
- **Acceptance:** a written rubric, labels that are not "the strings differ," a split frozen before generation, and a report that does not reuse pair-protocol metrics as if they were the same experiment. The rubric and trainer meet that contract. The recorded run does not meet the sample-size, baseline, or calibration parts of that bar and is not a successful detector.
- **Owner:** unassigned.

### 8. Early detection

- **Status:** event defined; lead unpublished. Issue #7. The event definition below was written before any lead-time number. `eval-early-warning` scores two or more generated pieces from stored vectors and does not load Qwen. `early_warning_lead` stays null until a run stores per-prefix evidence labels. No activation edit is performed.
- **Event definition (pre-registered):** The hallucination event index is the smallest generated-piece index *i* such that the answer prefix through piece *i* is labeled HALLUCINATED by the generation evidence protocol (`halueval_generation_evidence_v1` or `truthfulqa_generation_v1`) or by a documented human review that cites passage evidence. Prefixes that remain UNVALIDATED or AMBIGUOUS are not events. The empty prefix is not an event. `early_warning_lead` is the existing helper and stays null when this event is not stored on the run, when no prefix is an event, or when no probe flag occurs at or before the event. Thresholds are chosen on validation only. This measurement is not a calibrated risk, is not an activation edit, and is not evidence for pair-protocol AUROC.
- **Depends on:** per-prefix evidence labels on a generation run, which do not exist yet.
- **Acceptance:** the event definition is in this file before any test numbers are computed; the metric is null when no timely flag exists; no threshold is chosen on the test split. Unit tests cover two-position scoring, a missing event, and a missing timely flag. No lead-time number is published.
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
