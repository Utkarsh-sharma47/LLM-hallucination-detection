# Evaluation protocols (issues #13, #37, #7, #40, #41)

Pre-registration notes. No new scores are claimed in this document.

## Layer / position comparison (#13)

Pre-registered representations (named before any new score is computed):

| Name | Layer | Position |
| --- | --- | --- |
| `saved_probe_representation` | -1 (resolved 24 on 0.5B) | last generated piece |
| `mean_pool_final_layer` | -1 | mean of generated pieces |
| `middle_layer_last_piece` | 12 | last generated piece |
| `embedding_last_piece` | 0 | last generated piece |

Scores are computed on the generation-labeled split only, beside the same
text baselines (answer length, lexical overlap, TF-IDF). Pair-protocol AUROC
never ranks or selects a representation. `early_warning_lead` stays null.
A higher score is a comparison result, not a "truth direction".

## Shortcut-controlled cross-domain validation (#37)

- Sources (train / validation / in-domain test / independent-domain test)
  are identified before any new scores are inspected.
- No question, passage/group, or paraphrase family crosses a split boundary;
  assignments are saved and leakage-tested.
- Labels are assigned without probe scores; ambiguous/unvalidated are
  counted but excluded from binary metrics.
- Hidden-state probe, length, overlap, and TF-IDF use the same eligible rows.
- ≥2 meaning-preserving styles are evaluated without changing the factual
  label; subgroup results include counts and intervals.
- Thresholds/preprocessing are chosen on validation only. The report must
  state explicitly whether the probe beats baselines; pair AUROC alone
  never supports a detector claim.

Requires the #41 labeled set; blocked until validation/test targets are met.

## Early warning (#7)

Event definition (pre-registered in ROADMAP task 8): the smallest generated
piece index *i* such that the prefix through *i* is labeled HALLUCINATED by
the generation evidence protocol or a documented human review citing
passage evidence. `early_warning_lead` stays null until a run stores
per-prefix evidence labels, and stays null when no flag occurs at or before
the event. Thresholds are chosen on validation only. This measurement is
not a calibrated risk and is not evidence for pair-protocol AUROC.
`vectors_at_generated_positions` scores multiple positions without loading Qwen.

## Provenance retrieval (#40)

`glassbox.analysis.retrieval` is a library-only foundation: curated catalog
always wins, retrieval fills only UNVALIDATED claims, every evidence item
carries source_id/URL, retrieval time, source date, 64-hex SHA-256 content
hash of the explicit content, query, and quote span. Failed fetch and empty
results yield UNVERIFIED, never CONTRADICTED. No network client is wired
in; fixture tests cover missing/contradictory/paraphrased evidence.
Probe scores are never inputs to the evidence decision.

## Human review (#41)

`glassbox.data.review_agreement` computes observed agreement, Cohen's kappa,
disagreement and unresolved counts from already-collected independent
judgments. It invents no labels: questions without independent review stay
UNVALIDATED. Agreement describes consistency, not truth; adjudicated labels
must still cite passage evidence. The acceptance criteria requiring actual
human annotations, blinding, and adjudication remain open until a reviewed
set exists.
