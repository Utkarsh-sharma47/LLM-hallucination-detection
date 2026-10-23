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

`--samples` and `--analyze-text` were run again while writing this page. Both exited 0 and did not construct a model backend. `--samples` prints the three built-in questions. The Australia answer above prints `CORRECT` and the sentence that hidden-state probing cannot be performed on text alone.

`pytest` alone does not collect `research_tests/`. CI runs both trees.

## What each tree covers

`tests/` is the demo and the pair-protocol trainer:

| File | Covers |
| --- | --- |
| `test_cli_modes.py` | `--samples` and `--analyze-text` do not construct a backend; Sydney false claim; offline config |
| `test_workflow.py` | arbitrary questions, model kept loaded, narrow report, export, command legend |
| `test_decoding.py` | greedy and temperature probabilities, full-vocabulary export stays out of the terminal |
| `test_present.py` | token probability is not shown as factual risk |
| `test_distributions.py` | softmax over the full vocabulary |
| `test_probe_status.py` | missing or uncalibrated probe labels |
| `test_probe_training.py` | one-class training does not borrow test rows; separable features score on held-out ids |
| `test_halueval_splits.py` | filters and disjoint question splits |
| `test_truthfulqa_labels.py` | reference labels, paraphrase stays unvalidated, probe publish, lock message |
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

`train-probe` was not repeated in this pass. The live probe remains `halueval-pair-002`.

A future probe run is accepted only when the saved directory loads in a new `python -m glassbox --offline` process and the report distinguishes an experimental score from `RISK NOT ASSESSED`.
