# Testing

Routine tests do not download weights or datasets, do not read `.env`, and do not need a GPU. They use in-memory arrays or a stand-in backend.

## Commands

```bash
pytest tests research_tests
python scripts/smoke_test.py
python -m glassbox --samples
python -m glassbox --analyze-text \
  --question "What is the capital of Australia?" \
  --answer "Canberra is the capital of Australia."
```

On 23 October 2026, `pytest tests research_tests` was run again after the text-baseline and loader changes. The smoke test printed:

```text
glassbox smoke test: ok (version=0.1.0, phase=scaffold, model=Qwen/Qwen2.5-3B-Instruct, downloads=disabled)
```

`--samples` and `--analyze-text` were run again while writing this page. Both exited 0 and did not construct a model backend. `--samples` prints the three built-in questions. The Australia answer above prints `FAITHFUL` and the sentence that hidden-state probing cannot be performed on text alone.

`pytest` alone does not collect `research_tests/`. CI runs both trees.

## What each tree covers

`tests/` is the demo and the pair-protocol trainer:

| File | Covers |
| --- | --- |
| `test_cli_modes.py` | `--samples` and `--analyze-text` do not construct a backend; Sydney false claim; offline config |
| `test_workflow.py` | arbitrary questions, model kept loaded, blank-line exit, narrow report, export, command legend |
| `test_probe_training.py` | one-class training does not borrow test rows; separable features score on held-out ids; the scaler is fit on training rows only |
| `test_decoding.py` | greedy and temperature probabilities, full-vocabulary export stays out of the terminal |
| `test_present.py` | token probability is not shown as factual risk |
| `test_ask_terminal.py` | last-answer hidden-state index, incompatible probe, missing reference, ambiguous evidence, incomplete probe directory. A probe score on those rows stays `RISK NOT ASSESSED`. No Qwen load. |
| `test_claim_catalog.py` | checked-in claim types: faithful, hallucinated, ambiguous, unvalidated, and a question with no reference. Ambiguous and unvalidated rows stay out of both binary classes. No probe score and no Qwen load. |
| `test_artifact_compatibility.py` | seeded in-memory split, metadata mismatches, missing or corrupted artifacts, and a fresh-process reload of a tiny saved probe. No Qwen load. |
| `test_distributions.py` | softmax over the full vocabulary |
| `test_probe_status.py` | missing or uncalibrated probe labels |
| `test_reproduction.py` | a dirty worktree is not a finished reproduction; ECE above 0.10 keeps calibration unvalidated. No Qwen load. |
| `test_halueval_splits.py` | filters and disjoint question splits |
| `test_truthfulqa_labels.py` | reference labels, paraphrase stays unvalidated, probe publish, lock message |
| `test_generation_dataset.py` | frozen generation split, reference phrases, unvalidated and ambiguous rows kept out of both classes, pilot below the interval minimum, no pair-protocol metrics, refusal to write the saved probe directory. No Qwen load. |
| `test_generation_record.py` | a generation-evaluation record fails without provenance, a per-example row, a clean worktree, or a probe when a score was requested. Excluded labels stay out of both classes. No Qwen load. |
| `test_probe_transfer.py` | saved threshold 0.65 is required; an incompatible feature is not scored; ambiguous, unvalidated, and question-text overlap stay out of binary metrics; a missing provenance field or a failed record write leaves no file; generation ECE does not set `calibration_validated`; the interval stays null below 20 per class. No Qwen load. |
| `test_research_baselines.py` | shared scaler, text baseline, tiny-sample interval refusal, ambiguous label, saturated experimental score still says risk is not assessed. No Qwen load. The saved-probe metadata check skips when the gitignored artifact is absent. |

`research_tests/` is the scaffold:

| File | Covers |
| --- | --- |
| `test_config.py` | shipped YAML matches `ProjectConfig` |
| `test_data.py` | example ids, disjoint splits, and an offline `load_examples` (missing file does not download) |
| `test_metrics.py` | precision, recall, F1, AUROC, AUPRC, FPR, early-warning lead |
| `test_records.py` | run records, path jail, secret-like keys |
| `test_pipeline_guards.py` | download guard |
| `test_smoke.py` | smoke entry point |

`research_tests/conftest.py` puts `src/` first on `sys.path`. Do not add a conftest that removes `src` while collecting `tests/`.

## Layers

| Layer | Where it runs | Weight download |
| --- | --- | --- |
| Unit | both test trees | No |
| CLI integration | `test_workflow.py`, `test_cli_modes.py`, with a fake backend or `--analyze-text` | No |
| Smoke | `scripts/smoke_test.py` | No |
| Model validation | [`.github/workflows/model-validation.yml`](../.github/workflows/model-validation.yml), manual | Yes, the chosen small model |
| Probe evaluation | `python -m glassbox train-probe` | Local Qwen cache or an explicit download; not in CI |

There is no separate regression suite. A bug fix adds a unit test next to the file that failed. Generated-answer quality is not a unit test: the fake backend returns a fixed string so the suite does not depend on sampling.

## Lint and types

```bash
ruff check src tests research_tests scripts
ruff format --check src tests research_tests scripts
```

Ruff selects `E`, `F`, `I`, `UP`, `B`, and `SIM`, and ignores `E501`. The package has `py.typed`. Mypy and Pyright are not configured and are not in CI. Do not report the package as type-clean.

## Model and evaluation tests that are not automated

Routine pytest does not load Qwen. A test that reads `artifacts/probes/final-hidden-state/metadata.json` skips when that gitignored file is absent, which is the CI case.

On 23 October 2026 the local machine also ran:

- `python -m glassbox text-baselines --max-questions 200 --run-id halueval-text-baselines-001` (no model load)
- `python -m glassbox audit-generations --run-id truthfulqa-label-audit-001` (no model load)
- a scripted `python -m glassbox --offline` session, recorded under `artifacts/experiments/interactive-spotcheck.log` when that log exists

`train-probe` was not repeated. It would replace `artifacts/probes/final-hidden-state/`. The live probe remains `halueval-pair-002`, threshold 0.65, `calibration_validated` false. On 26 October 2026 a second process reloaded that directory and reproduced one score, and `python -m glassbox --offline` answered seven questions in one session. The answers and the reason that session is not a detector result are in [RESEARCH.md](RESEARCH.md). Two earlier load attempts the same day were refused by the RAM check and did not invent a score.

A future probe run is accepted only when the saved directory loads in a new `python -m glassbox --offline` process and the report distinguishes an experimental score from `RISK NOT ASSESSED`.
