# Glass Box: LLM Hallucination Detection

Local inspection of a causal language model. The terminal demo generates an answer, shows the real next-token distribution, reads hidden states, and reports a hallucination probability only when a compatible calibrated probe is loaded.

Token probability is the decoding distribution. It is not factual confidence. Entropy is not a risk score. A wording difference from a reference answer is not a hallucination label. Probe predictability is not evidence that the model knows an answer is wrong. Those limits are enforced in [`src/glassbox/caveats.py`](src/glassbox/caveats.py).

Version `0.1.0`. No GitHub release has been cut. The code license is unselected: there is no `LICENSE` file.

## Status

| Area | State |
| --- | --- |
| Interactive demo on a cached causal LM | Working. Shipped demo model is `Qwen/Qwen2.5-0.5B-Instruct`. |
| Next-token probabilities and top alternatives | Working. Full-vocabulary export is opt-in. |
| Curated string reference check | Working for [`configs/curated_facts.yaml`](configs/curated_facts.yaml) only. |
| Hidden-state logistic probe | Pair-protocol probe in `artifacts/probes/final-hidden-state/` (gitignored). It discriminates supplied HaluEval answers. Calibration was rejected. The demo shows `EXPERIMENTAL PROBE SCORE` and `RISK NOT ASSESSED`. |
| Text baselines | Length and gold-answer overlap explain most of the pair-task score. See [docs/RESEARCH.md](docs/RESEARCH.md). This is not a generation detector. |
| TruthfulQA generation check | 40 generated answers, 5 labeled by a listed reference, 35 unvalidated. Not evidence for the pair AUROC. Early detection and activation intervention are not implemented. |

Details, conflicts, and the task list: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/ROADMAP.md](docs/ROADMAP.md), [docs/RESEARCH.md](docs/RESEARCH.md).

## Setup

From the repository root, with Python 3.10 or newer (this checkout was verified with 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

Install the CPU wheel before the package so `pip` does not replace it with a CUDA build. On a CUDA machine, install the matching PyTorch wheel instead, then the same editable install.

Copy [`.env.example`](.env.example) to `.env` only if you need a Hugging Face token. `.env` is gitignored. Do not commit it.

Versions verified in this environment are listed in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Usage

`python -m glassbox` asks questions until you quit. The model loads on the first question and stays loaded. A question does not have to be in the curated list.

After an answer:

```text
n = ask another question
t = token probability analysis
p = probe analysis
v = reference verification
d = diagnostics
e = export report
q = quit
```

`e` writes `artifacts/sessions/session-N/session.json`, `session.md`, and `distributions-topk.csv` / `.jsonl`. Open the Markdown report with `less`. `piece_id` in the distribution files is the vocabulary id. The terminal does not print the full vocabulary. `--distribution-export full` also writes the full-vocabulary CSV and JSONL for that session.

```bash
python -m glassbox --demo
python -m glassbox --once --question "What is the capital of Australia?"
python -m glassbox --offline --once --question "What is the capital of Australia?"
python -m glassbox --samples
python -m glassbox --analyze-text --question "What is the capital of Australia?" --answer "Canberra is the capital of Australia."
python scripts/demo.py --samples
```

`--offline` reads the Hugging Face cache and does not download weights. `--samples` and `--analyze-text` do not load a model. `--analyze-text` cannot see logits or hidden states.

Without a calibrated probe the report says `PROBE NOT TRAINED` and `RISK NOT ASSESSED`. A question outside the curated file says `NO REFERENCE AVAILABLE`.

## Demo model and probe command

[`configs/demo.yaml`](configs/demo.yaml) selects `Qwen/Qwen2.5-0.5B-Instruct` because the research default, `Qwen/Qwen2.5-3B-Instruct` in [`configs/default.yaml`](configs/default.yaml), does not fit the RAM this project was developed on. The loader refuses a model that does not fit before download.

```bash
python -m glassbox --offline train-probe --max-questions 200 --run-id halueval-pair-002
python -m glassbox text-baselines --max-questions 200 --run-id halueval-text-baselines-001
python -m glassbox --offline eval-truthfulqa --max-questions 40
```

`text-baselines` does not load Qwen. Training is not part of routine tests. The live probe is `halueval-pair-002`, described in [docs/RESEARCH.md](docs/RESEARCH.md). It is a paired-answer experiment. Platt scaling failed the held-out ECE check, so the demo does not show a factual-risk percentage. `halueval-pair-001` is kept as `artifacts/probes/final-hidden-state.previous/`.

A tiny checkpoint for a pipeline check, not a scientific run:

```bash
python -m glassbox --model sshleifer/tiny-gpt2 --max-new-tokens 8 --once --question "What is the capital of Australia?"
```

## Layout

```text
src/glassbox/     installable package
configs/          scaffold defaults and demo settings
tests/            demo and probe unit tests
research_tests/   scaffold config, metrics, and guards
scripts/          smoke test and demo entry point
docs/             architecture, development, testing, research, roadmap
artifacts/        gitignored outputs
```

`pytest` runs `tests/` only. The full local check is in [docs/TESTING.md](docs/TESTING.md).

## More

- [Architecture](docs/ARCHITECTURE.md)
- [Development](docs/DEVELOPMENT.md)
- [Testing](docs/TESTING.md)
- [Research](docs/RESEARCH.md)
- [Roadmap](docs/ROADMAP.md)
- [Contributing](docs/CONTRIBUTING.md)
