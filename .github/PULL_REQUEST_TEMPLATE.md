## Summary

-

## Tests

Commands actually run, and the result. Delete any line you did not run.

- [ ] `ruff check src tests research_tests scripts`
- [ ] `ruff format --check src tests research_tests scripts`
- [ ] `pytest tests research_tests`
- [ ] `python scripts/smoke_test.py`

## Claims

- [ ] No new metric, accuracy, or calibration number was added unless a command in this PR produced it.
- [ ] No weights, datasets, `.env`, or `artifacts/` outputs are included.
- [ ] Token probability is not described as hallucination risk.
