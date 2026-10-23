# Development

## Environment

Python 3.10 or newer. This checkout was exercised with CPython 3.12 and the CPU wheel of PyTorch. Create the virtualenv from the [README](../README.md).

Verified direct dependency versions in that environment on 23 October 2026:

| Package | Version |
| --- | --- |
| torch | 2.14.1+cpu |
| numpy | 2.5.4 |
| rich | 15.0.0 |
| scikit-learn | 1.9.1 |
| transformers | 5.20.2 |
| huggingface_hub | 2.2.0 |
| pyyaml | 6.0.3 |
| pytest | 9.1.1 |
| ruff | 0.17.2 |

`pyproject.toml` records lower bounds, not these exact pins. CI installs `torch==2.14.1` from the CPU index and then `.[dev]`, so a later resolver can still move the other packages. Re-run the test commands after a dependency bump.

Optional local tools, not required by CI:

```bash
python -m pip install pre-commit
pre-commit install
```

The hook config is [`.pre-commit-config.yaml`](../.pre-commit-config.yaml): whitespace, YAML, TOML, large files, merge conflicts, private keys, and Ruff.

## Configuration

| File | Who reads it |
| --- | --- |
| `configs/default.yaml` | `load_config()`, smoke test, research tests |
| `configs/demo.yaml` | `python -m glassbox` and `train-probe` |
| `configs/curated_facts.yaml` | reference check only |
| `.env` | not read by the demo or the smoke test |

`.env.example` documents `HF_TOKEN` and `HF_HOME`. `glassbox.env.load_env_file` can load a file, and nothing in the CLI calls it today. The smoke test does not read `.env` and does not print variable values.

Demo flags that change a run: `--offline`, `--model`, `--probe`, `--max-new-tokens`, `--distribution-export {topk,full}`, `--config`, `--references`. Training flags: `--max-questions` (default 240), `--seed` (default 0), `--download-data`, `--snapshot`, `--run-id`, `--output`.

The demo config pins dtype `bfloat16` so training and interactive inference use the same representation. `resolve_dtype("auto")` still picks float32 when the estimated weights plus 1 GiB headroom fit, otherwise bfloat16. A probe trained in one dtype is rejected for the other when the artifact stored a dtype.

## Debugging

The process does not write a log file. Useful surfaces:

- stderr from a non-zero exit
- `d` in the interactive menu (revision, prompt length, latency, hidden-state shape, load count)
- `e` for the JSON report under `artifacts/sessions/`
- `ModelMemoryError` before a download when the RAM estimate fails

`HF_HUB_OFFLINE=1` is set by the loader when `local_files_only` is true. `--offline` is the supported way to get that behavior.

Do not point `HF_HOME` at a tracked directory. The example uses `artifacts/hf_home`, which Git ignores.

## Reproducible experiments

A run that claims a measurement needs all of the following. Missing any one of them means the number is not a result of this repository.

- Glass Box git commit (`git rev-parse HEAD`), 40 hex characters
- Model revision actually loaded, 40 hex characters, not `main`
- Dataset SHA-256 and the upstream commit recorded in `data/halueval.py`
- `artifacts/experiments/<run_id>/record.json` from `build_run_record`
- dtype, device, seed, and the threshold frozen on validation
- `python --version` and `pip freeze` copied into the run directory by the operator

`write_json` only writes under the `artifacts/` root it is given. Git ignores everything there except `artifacts/.gitkeep`.

There is no hosted experiment tracker.

## Checks before a pull request

```bash
source .venv/bin/activate
ruff check src tests research_tests scripts
ruff format --check src tests research_tests scripts
pytest tests research_tests
python scripts/smoke_test.py
python -m glassbox --samples
```

`pytest` with no arguments runs `tests/` only, because that is `testpaths` in `pyproject.toml`. Include `research_tests` explicitly.

The model validation workflow is manual. Do not add a Qwen download or `train-probe` to the commands above.

## Branches in this integration

`main` contains `7e5d70f`, the merge of the CI pull request. The published history still starts at `9b8bcc4` (`Initial commit`). Do not rewrite that commit.

| Branch | Role |
| --- | --- |
| `feat/glassbox` | Tooling, offline scaffold, terminal demo, and probe training |
| `docs/project-documentation` | README and `docs/` |
| `ci/github-actions` | GitHub Actions, issue templates, and `CODEOWNERS` |

Commits on `feat/glassbox`:

| Commit | Subject |
| --- | --- |
| `f5ca982` | chore: add project tooling and ignore rules |
| `177d848` | feat: add offline scaffold and pinned dataset readers |
| `237af6a` | fix: ignore downloaded weights only at the repository root |
| `724029c` | feat: add the local terminal demo |
| `46a1194` | feat: add probe training and TruthfulQA reference checks |

`177d848` by itself does not collect `research_tests/test_pipeline_guards.py`, because that file imports the model and probe modules added in `724029c`. The branch tip is the tree that was tested.

On 23 October 2026, at `46a1194`:

- `ruff check src tests research_tests scripts` passed.
- `ruff format --check src tests research_tests scripts` passed.
- `pytest tests research_tests` reported 68 passed.
- `python scripts/smoke_test.py` printed `glassbox smoke test: ok` for the 3B scaffold with downloads disabled.
- `python -m glassbox --samples` exited 0.
- `python -m glassbox --analyze-text --question "What is the capital of Australia?" --answer "The capital of Australia is Canberra."` exited 0. The detector line was `PROBE NOT TRAINED` because this mode does not apply the saved probe. The curated check marked the answer correct.

Pull requests merged into `main`, in order:

- https://github.com/Utkarsh-sharma47/LLM-hallucination-detection/pull/2 (`feat/glassbox`) merged as `50dcc1a`
- https://github.com/Utkarsh-sharma47/LLM-hallucination-detection/pull/3 (`docs/project-documentation`) merged as `008a827`
- https://github.com/Utkarsh-sharma47/LLM-hallucination-detection/pull/4 (`ci/github-actions`) merged as `7e5d70f`

The `checks` job on pull request 4 completed with success before that merge. Routine CI does not download Qwen. The manual model workflow has not been run.
