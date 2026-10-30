# Reproducibility

## Environment

Python 3.12 in this checkout. CPU PyTorch. No GPU is required for the 0.5B demo. The loader refuses a model that does not fit in available RAM with 1 GiB of headroom.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

## Model

`Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`, dtype `bfloat16`. The demo reads `configs/demo.yaml`. `--offline` uses the Hugging Face cache and does not download weights.

## Data hashes

| File | SHA-256 |
| --- | --- |
| `artifacts/datasets/halueval/qa_data.json` | `89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88` |
| `artifacts/datasets/truthfulqa/TruthfulQA.csv` | `b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c` |
| `artifacts/probes/final-hidden-state/metadata.json` | `1d96a06592c187ed5ee37329a8abd69583791eace57d463e22bd907a457db27e` |

The pair probe threshold is 0.65. `calibration_validated` is false. `train-probe` with the default publish path replaces that directory. Do not rerun it to reproduce the saved artifact.

## Commands that do not load Qwen

```bash
ruff check src tests research_tests scripts
ruff format --check src tests research_tests scripts
pytest tests research_tests
python scripts/smoke_test.py
python -m glassbox --samples
python -m glassbox review-generations --show 5
```

`pytest` skips the gitignored snapshot, the saved probe, and the TruthfulQA CSV when those files are absent. That skip is expected in CI.

## Commands that load the cached 0.5B model

```bash
python -m glassbox --offline
python -m glassbox --offline generate-v2-answers --max-questions 2400 \
  --run-id halueval-generation-v2-answers
```

The second command appends under `artifacts/experiments/halueval-generation-v2-answers/`. It does not rewrite generation-001. Seed 0. Temperature 0.

## What is not a clean reproduction

`halueval-pair-002` and `halueval-generation-001` were recorded from dirty worktrees. Their numbers stay in [EXPERIMENTS.md](EXPERIMENTS.md). They are not rewritten to look cleaner.
