# Datasets

Readers refuse a file whose SHA-256 is not the pin. They do not download. A missing file raises `DownloadDisabledError` when downloads are disabled, and `NotImplementedError` when a download is requested. `load_examples` is not a download client. Hashes and label rules below match the code. Results that use these files are in [EXPERIMENTS.md](EXPERIMENTS.md) and [RESEARCH.md](RESEARCH.md).

## HaluEval QA

Li et al., EMNLP 2023. Upstream file `data/qa_data.json` at commit `207f479c555a68b8d2431631a95083b84b2accbe`. Local path `artifacts/datasets/halueval/qa_data.json` (gitignored). SHA-256 `89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88`.

JSONL fields: `knowledge`, `question`, `right_answer`, `hallucinated_answer`. 10,000 rows. Filters in `data/halueval.py` keep 7,681 questions. The knowledge passage is stored and is not shown to the model.

Two label uses, which must stay separate:

| Use | Source | Semantics |
| --- | --- | --- |
| Pair protocol `halueval_qa_pair_v1` | Benchmark fields after the filters | `right_answer` is faithful and `hallucinated_answer` is hallucinated. Both rows of a question stay in one split. This is not a judgment of a new Qwen answer. |
| Generation protocol `halueval_generation_evidence_v1` | `data/evidence.py`, after Qwen generates | `FAITHFUL` only when the passage contains the right answer and the generation asserts it, or its content tokens in order, without a contradicted substitute. `HALLUCINATED` only when the generation asserts a substitute taken from the benchmark false answer. `AMBIGUOUS` when both. `UNVALIDATED` when the passage does not ground the right answer, or the generation asserts neither side. A missing string match is not a hallucination. |

The generation labeler is automatic. No human review was applied to `halueval-generation-001`. A review file is JSONL with `question_id`, `label`, `reviewer`, and an `evidence_note` of at least 20 characters. Missing reviews are not invented. Evidence limits, including the false span `The Rockets` on a question that already names the Houston Rockets, are in [RESEARCH.md](RESEARCH.md).

The pair split and the generation split are both seed 0 and both group related rows, but they are not the same assignment. The generation split hashes the knowledge passage (`knowledge-group-v1`) before generation. The 200-question pair subset is not the 720-question generation subset.

## TruthfulQA

Lin, Hilton, and Evans, ACL 2022. The code records the URL `https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv`. That URL points at `main`, so the commit is not pinned. The local file is pinned by hash only: SHA-256 `b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c`. Path `artifacts/datasets/truthfulqa/TruthfulQA.csv` (gitignored).

Columns: Type, Category, Question, Best Answer, Best Incorrect Answer, Correct Answers, Incorrect Answers, Source.

`load_examples("truthfulqa")` returns one `UNVALIDATED` reference row per question (`truthfulqa_reference_only_v1`). It does not invent a model answer.

`label_generated_answer` (`truthfulqa_generation_v1`) marks a new answer hallucinated only when it asserts a listed incorrect answer, and faithful only when it asserts a listed correct answer and no incorrect one. Both hits are `AMBIGUOUS`. Other wording, including a paraphrase that does not assert a listed phrase, stays `UNVALIDATED`.

`label_generated_answer_coverage` (`truthfulqa_generation_coverage_v2`) keeps a decisive assertion label. Only after `UNVALIDATED` may in-order content coverage upgrade a short reference. It is not an entailment model. `halueval-generation-001` used this coverage pass on 40 external generations and recorded 1 upgrade. Those rows were not training data.

Do not pool TruthfulQA labels with HaluEval labels.

## Curated demo facts

[`configs/curated_facts.yaml`](../configs/curated_facts.yaml) is a terminal string check. It is not a training set and it does not call the probe. Thirteen hand-written items. An answer is faithful only for an accepted alias, hallucinated only for a listed false claim, and otherwise unvalidated. "Paris." asserts "Paris" because the period is a non-word boundary. That check can mark France correct while the pair probe still emits an experimental hallucinated score.
