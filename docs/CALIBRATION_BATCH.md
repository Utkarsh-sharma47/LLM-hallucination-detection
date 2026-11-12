# Preselected calibration validation batch (issue #18)

The generation calibration gate needs at least 20 faithful and 20 hallucinated reviewed Qwen answers whose questions the pair probe never saw. The reviewed task-7 validation rows left only 16 faithful ones after the overlap filter, and the diagnostic ECE on them was 0.256, above the 0.10 limit. This batch gives the gate fresh data. It does not tune anything.

## Rules fixed before any answer exists

- Questions come from the pinned HaluEval snapshot. Every question id found in a recorded probe, an experiment split or prediction file, or an earlier plan is excluded, and so is every question that shares a knowledge passage with one of them.
- Whole passages are chosen in sha256 order, with no model and no probe score involved. The plan holds the selected ids, a reserve of the remaining eligible ids, and its own hash.
- Shards of 100 are reviewed in order. The analysis set is the longest prefix of fully reviewed shards. Reviewing past a gap is refused.
- The gate runs at the fixed threshold 0.65 with the pre-registered ECE limit 0.10 and the 20-per-class minimum. The command has no threshold or tuning option.
- If the gate fails, this batch is not used to refit the pair probe or choose a threshold. Recalibration would need the reserve as a fit set and a fresh untouched evaluation set.

## Steps

```
python -m glassbox.probes.calibration_batch plan
```

Commit `artifacts/calibration/pair-calibration-validation-001/plan.json` before going further. It is the pre-registration. Then:

```
python -m glassbox.probes.calibration_batch verify
python -m glassbox.probes.calibration_batch generate --run-id calibration-batch-001-qwen0.5b --offline
python -m glassbox.probes.calibration_batch queue --run-id calibration-batch-001-qwen0.5b
```

`generate` writes one shard at a time and skips shards that exist, so a stopped run resumes. Use `--shards 0-1` to do part of the work. The queue lists question, answer and passage only. It withholds the probe score, the automatic label and the benchmark right answer.

Review with the task-7 rubric. Put one `decisions-SSS.json` per shard in `artifacts/reviews/<run-id>-ai-v1/`, each row `[source_index, label, quote, note, flags]`, and fill in `review-meta.json` with the reviewer and whether the reviewer was human. A quote must appear in the passage. Then:

```
python -m glassbox.probes.calibration_batch gate --run-id calibration-batch-001-qwen0.5b --probe-dir artifacts/probes/final-hidden-state
```

For the 1.5B probe use a separate run id, `--config configs/demo-1.5b.yaml` on `generate`, and `--probe-dir artifacts/probes/qwen2.5-1.5b`. The two models answer differently, so each needs its own review.

The gate report names the plan hash, the shards analyzed, label counts, dropped rows, reviewer type, the unchanged pair-probe file hashes and a reliability table. It writes a new file for each run and never overwrites.

## What a pass would and would not mean

A pass would say the saved pair probe's score is calibrated on this model's HaluEval generations, labeled by one reviewer under one rubric. It would not make the score a verified factuality probability on other questions, and AI-assisted labels stay provisional until independently reviewed.
