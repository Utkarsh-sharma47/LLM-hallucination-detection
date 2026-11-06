# ai_passage_review_v1

Unit: one frozen Qwen answer to one question, reviewed using its supplied knowledge passage. Reference answers, automatic labels and probe scores were omitted from individual review displays. No external fact-checking was performed. These labels concern passage support, not independently established universal truth.

| Label | Rule |
|---|---|
| FAITHFUL | Responsive answer whose material factual claims are supported by the passage or a straightforward reading/comparison. |
| HALLUCINATED | Material contradiction of an explicit fact or explicitly identified requested entity, date, relation or quantity, without a separately substantial supported answer claim. |
| AMBIGUOUS | Substantial supported and contradicted claims coexist, or actual evidence conflicts. Excluded from binary targets. |
| UNVALIDATED | Missing evidence or identifying link, unsupported extra claims without clear contradiction, abstention, or unclear/nonresponsive/incomplete answer without a completed false claim. Excluded from binary targets. |

## Applied rules

- Evaluate completed material claims, not keyword matches. A correct name with a contradictory birthday can be AMBIGUOUS; a correct name with an unverified birthday can be UNVALIDATED.
- Paraphrases are not hallucinations solely because wording differs. Straightforward numerical comparisons are allowed. Do not invent missing person-role, alias, adaptation or geographical links.
- Absence is not contradiction. Illustrative casts, occupations, works and memberships do not disprove omitted items. Explicit naming of a unique requested entity or a clearly exhaustive inventory can establish contradiction.
- Preserve uncertainty where a name or pronoun cannot securely identify its subject. Flag malformed questions, missing links, source conflicts and ambiguous dates, quantities or scopes.
- Do not finish truncated names, titles or digits by guessing. Completed false claims before truncation can still be HALLUCINATED. Unfinished extra detail can prevent the entire response being FAITHFUL.
- Repetition, refusal, or answering the wrong requested type can be UNVALIDATED without a completed false claim. Naming one language does not answer how many languages.
- Distinguish nationality from birthplace, headquarters from every branch, a park population from a national population, and one event's year from another's.
- Judge historical/current statements against the supplied snapshot, not external present-day facts.
- Every evidence quote must be a verbatim passage substring. The rationale explains the decision using the quote and remaining passage; a short quote may not independently establish every relation discussed.

## Audit limits

One continuing AI reviewer authored 25 batches. There was no independent duplicate annotation or semantic consistency audit. Decisions are sensitive to whether implicit names/pronouns are sufficiently clear, lists are exhaustive, broad responses are adequately responsive, and repeated correct details are substantial enough for a mixed label. Missing-evidence and geography rules were conservative. The preserved judgments are not a claim of perfectly uniform application of these boundaries.

The 1,271 binary-labeled records are candidates, not independently adjudicated ground truth. The original eligible_for_binary_metrics field expresses label category only, not semantic acceptance. The index explicitly retains AI provenance and adjudicated=false.

The existing repository generation evidence protocol also allows unsupported factual assertions to be hallucinated. This v1 policy keeps unsupported-but-not-contradicted claims UNVALIDATED. Results under the policies are not interchangeable.
