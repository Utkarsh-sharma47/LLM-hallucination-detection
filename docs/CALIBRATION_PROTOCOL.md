# Generation calibration protocol

This protocol is registered before any new generation calibration calculation. It does not revise
the historical results or authorize a probability claim from the existing artifacts.

## Eligible score and labels

Only scores produced by the generation-specific probe are eligible. The calibration input must
identify each score as `generation_probe` and each label as one of the registered independent
evidence sources: `halueval_generation_evidence_v1`, `truthfulqa_generation_coverage_v2`, or
`human_review_v1`. Pair-task scores (`score_source == "pair_probe"`) are explicitly ineligible,
including on generated answers. Labels must be assigned without access to scores; `AMBIGUOUS` and
`UNVALIDATED` rows remain counted and are excluded from binary calibration metrics.

The operating threshold remains 0.65. It is not selected on validation or test data in this
protocol. Validation and test must remain disjoint by question/knowledge group. Validation alone
can decide whether the calibration gate passes. Test reliability metrics are reporting-only.

## Registered metrics and uncertainty

Use ten equal-width bins over `[0, 1]`, left-closed and right-open except for the final bin, which
includes 1.0. Report expected calibration error (ECE) and Brier score from the same eligible rows.
The existing ECE acceptance limit remains 0.10.

For uncertainty, use 1,000 deterministic percentile-bootstrap resamples (`seed=0`) of the
independent question/knowledge group, keeping all observations from a sampled group together.
Report 95% intervals only with at least 20 eligible examples in each class, at least 20 distinct
groups, and at least 100 valid resamples containing both classes. Otherwise report null interval
bounds and the specific reason. Point estimates in an under-supported split are descriptive only.

The 20-per-class floor is an operational minimum inherited from the repository's existing interval
policy; it is not a power or precision justification. No adequate sample-size target has been
established. Consequently, meeting that floor alone is not evidence of adequate power or a
generalizable detector. Before making a probability claim, a larger independently reviewed set and
a sample-size rationale are still required.

## Gate and reporting boundary

A validation split must contain at least 20 eligible examples of each class and have ECE at most
0.10. The gate never reads test ECE or Brier to decide acceptance. Test ECE/Brier and their
uncertainty are reported separately and cannot change the gate, threshold, bins, or resampling
settings. If accepted, the generation-specific logistic probability is used directly; this gate
does not imply a Platt transform. A rejected or unsupported gate keeps `calibration_validated`
false and the user-facing score as `EXPERIMENTAL PROBE SCORE` / `RISK NOT ASSESSED`.

No new calibration result is claimed in this document. The previously recorded validation counts
(14 faithful, 7 hallucinated) remain below the registered class floor, and no new experiment was
run while adding this protocol and its synthetic tests.
