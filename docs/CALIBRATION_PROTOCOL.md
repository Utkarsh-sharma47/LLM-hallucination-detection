# Generation calibration protocol (issues #18, #38)

Pre-registered before any new generation calibration calculation. This does
not revise historical results or authorize a probability claim from existing
artifacts.

## Gate (unchanged)

- Operating threshold stays the frozen HaluEval value **0.65**. It is not
  selected on validation or test data.
- Validation must be disjoint by question/knowledge group from test and
  from HaluEval pair splits.
- Each of faithful and hallucinated needs **≥20 validation rows** whose
  labels were assigned without the probe score. `UNVALIDATED` and
  `AMBIGUOUS` stay out of the ECE denominator and are counted.
- Validation ECE must be **≤0.10**. The test split is never used to choose
  the threshold, bins, or ECE limit, and never to accept calibration.
- The gate reads **pair-probe scores only** (`score_source="pair_probe"`).
  Generation-probe scores are ineligible for this gate.
- A passing gate is the only path that may describe the score as a
  calibrated probability. Pair-protocol AUROC is not evidence for it.
- The five-label TruthfulQA pilot is ineligible.

## Uncertainty reporting (issue #38)

- ECE: 10 equal-width bins on [0,1], left-closed/right-open except final bin.
- Brier score: mean squared error of binary probabilities (proper rule).
- Intervals: 95% percentile bootstrap, 1000 resamples, seed 0, resampling
  **question groups** (rows sharing a group stay together).
- Intervals are reported only with ≥20 per class, ≥20 distinct groups, and
  ≥100 valid resamples containing both classes; otherwise null with an
  explicit reason. Point estimates in under-supported splits are descriptive
  only.

## Sample-size rationale

The 20-per-class floor is an operational minimum, not a power/precision
justification. No adequate sample-size target has been established, so
meeting the floor alone is not evidence of adequate power or a generalizable
detector. Recorded validation counts (14 faithful / 7 hallucinated) remain
below the floor and rejected. No new calibration result is claimed here.
