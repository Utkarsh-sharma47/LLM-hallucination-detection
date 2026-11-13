# Evidence-based claim verification (issue #40, stage 1)

This is the offline foundation for issue #40. It adds a package, `glassbox.evidence`, and changes nothing the live report shows. The curated catalog check in `analysis/claims.py` keeps its behavior. No network code exists in this stage.

## What exists

| Piece | File | Job |
| --- | --- | --- |
| Records | `evidence/models.py` | `EvidenceItem`, `Claim`, `ClaimVerdict`, and four verdicts: SUPPORTED, CONTRADICTED, CONFLICTING, UNVERIFIED |
| Retrievers | `evidence/retrievers.py` | `Retriever` interface; `CatalogRetriever` (the curated catalog) and `LocalSnapshotRetriever` (a frozen local JSONL of sources) |
| Verification | `evidence/verify.py` | Claim decomposition with stable ids, named verdict rules, answer-level summary |
| Text helpers | `evidence/textmatch.py` | Tokens, a crude stem, negation window. No model and no embeddings |

## What a verdict records

Each evidence item carries the source id, the locator, the retrieval time, the content hash of the whole source text, the query, the quoted span, and the character offsets of that span (`text[span_start:span_end] == quote`). A source date is kept when the source has one. Each verdict carries the rule names that decided it and the evidence it used.

Curated catalog evidence has `evidence_kind = curated_catalog` and a locator that says it was not retrieved at runtime. It is not presented as retrieval.

## Rules

| Rule | Decides | When |
| --- | --- | --- |
| `asserted_value_v1` | SUPPORTED | A sentence about the claim's topic contains every value of the claim and none is negated |
| `negated_value_v1` | CONTRADICTED | The same kind of sentence contains the values and one is negated ("Sydney is not the capital") |
| `rival_value_v1` | CONTRADICTED | The claim lists a rival value, and a relevant sentence asserts it without asserting the claim's values |
| `curated_catalog_v1` | either | The catalog lists an accepted or false phrase for the claim |

Two sources that disagree give CONFLICTING. Anything else is UNVERIFIED, with a reason: `no_evidence`, `retrieval_failed`, `stale_or_undated_evidence`, or `no_rule_applied`. A passage that only mentions the topic gives no verdict. Matching alone never decides a label without a named rule and a recorded span.

Rules are intentionally narrow. A different value in the evidence does not contradict a claim unless the claim names it as a rival, because telling "different" from "also true" needs more than string matching. Expect UNVERIFIED often.

## Staleness

A claim marked `time_sensitive` ignores evidence older than `max_age_days` (default 365) and evidence with no date. Other claims ignore age. If every item is excluded the reason is `stale_or_undated_evidence`.

## Probe separation

Probe scores, token probabilities and entropy are not parameters of any function in this package. `tests/test_evidence.py` checks the signatures.

## Not done yet

- The live report does not use this package. Wiring it into section [05] is a separate step.
- No network retrieval. A network retriever must be opt-in and report its failures through `RetrievalError` or a `failed` result.
- No evaluation. Issue #40 asks for a small independently reviewed set before any aggregate number is published. None exists, so no accuracy figure is claimed.
- Claim decomposition is sentence and "and" splitting. Values and anchors come from capitalization and numbers, which is crude. Pronouns, relative dates and compound claims are not handled.

## Tests

`python -m pytest tests/test_evidence.py` runs offline against `tests/fixtures/evidence_snapshot.jsonl`. It covers supported, contradicted, paraphrased, missing, stale, undated, conflicting, failed-retrieval and mixed-answer cases.
