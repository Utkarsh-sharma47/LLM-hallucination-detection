# Dataset

The pinned files and label rules are in [DATASETS.md](DATASETS.md). This page is the generation and review protocol.

## Sources

| File | Role | SHA-256 |
| --- | --- | --- |
| `artifacts/datasets/halueval/qa_data.json` | Pair training and generation questions | `89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88` |
| `artifacts/datasets/truthfulqa/TruthfulQA.csv` | Held-out generation references | `b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c` |
| `configs/curated_facts.yaml` | Terminal reference list | Checked in. Not a training set. |

Both dataset files are gitignored. A different hash is refused. TruthfulQA labels are not pooled with HaluEval labels.

## Splits

Generation questions are grouped by passage (`knowledge-group-v1`) before any new answer is labeled. A question that shares a passage with another stays in the same split. Generation-001 membership is kept when the 2400-question plan is built. New questions fill the remaining plan slots. They do not move an existing 001 question.

## Generation

`halueval-generation-001` already has 720 answers. `generate-v2-answers` appends only questions that are in the 2400-question plan and absent from that file. The run id is `halueval-generation-v2-answers`. It refuses to write into `halueval-generation-001` or `final-hidden-state`.

Each accepted line records the question id, split, passage, benchmark answers, model name, revision `7ae557604adf67be50417f59c2c2f167def9a775`, seed, temperature 0, and the automatic evidence label. `human_confirmed` is false. `label_origin` is `halueval_generation_evidence_v1`. Empty answers, a wrong revision, and a missing hidden state go to `failures.jsonl` and are not labeled examples. A crash leaves at most one torn last line, which resume ignores. An earlier broken line stops the run.

Count the files after a run. Do not quote a finished total from this page until `answers.jsonl` and `failures.jsonl` have been read.

## Annotation

Automatic labels are proposals. A human review is an append-only event in `artifacts/experiments/generation-reviews/` with a question id, a label in `FAITHFUL`, `HALLUCINATED`, `AMBIGUOUS`, or `UNVALIDATED`, an evidence note of at least 20 characters, a reviewer id, and a timestamp. An unvalidated proposal is not converted to a binary label without that event. No human reviews have been invented for the v2 answers.

```bash
python -m glassbox review-generations --show 5
```

That command reads the generation-001 queue. It does not load Qwen and it does not mark the queue as finished.
