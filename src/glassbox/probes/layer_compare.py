"""Issue #13: compare pre-registered generation representations.

Layers and positions are named before any score is computed. Pair-protocol
AUROC is not used to rank them. The live pair probe is not overwritten.
``early_warning_lead`` stays null: issue #7 owns that metric.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from glassbox.config import repo_root
from glassbox.evaluation.early_warning import EVENT_MEASUREMENT_ENABLED
from glassbox.evaluation.generation_calibration import (
    PAIR_THRESHOLD,
    evaluate_generation_calibration_gate,
)
from glassbox.evaluation.metrics import BOOTSTRAP_MIN_PER_CLASS
from glassbox.probes.generation_eval import (
    PROTECTED_PROBE_NAMES,
    generation_experiment_dir,
    generation_text_baselines,
    interpret_detector,
)
from glassbox.probes.generation_variants import (
    _baseline_rows,
    _binary_row,
    _counts,
    _fit_vector_variant,
    _is_binary,
)

# Named before any generation score is computed for this comparison.
REGISTERED_REPRESENTATIONS = (
    {
        "name": "saved_probe_representation",
        "layer_index": -1,
        "position": "last_generated_token",
        "field": "vector",
        "included_because": "This is the saved generation-probe representation.",
    },
    {
        "name": "mean_pool_final_layer",
        "layer_index": -1,
        "position": "mean_generated_pieces",
        "field": "answer_mean",
        "included_because": "Mean over generated pieces at the same layer.",
    },
    {
        "name": "middle_layer_last_piece",
        "layer_index": 12,
        "position": "last_generated_token",
        "field": "layer_12_last_piece",
        "included_because": "Mid-depth last piece, pre-registered for a later sweep.",
    },
    {
        "name": "embedding_last_piece",
        "layer_index": 0,
        "position": "last_generated_token",
        "field": "layer_0_last_piece",
        "included_because": "Embedding-stream last piece, pre-registered for a later sweep.",
    },
)


def evaluate_layer_comparison(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int = 0,
    pair_probe_path: Path | None = None,
) -> dict[str, object]:
    """Score named representations that are present. Do not invent missing layers."""
    binary = [_binary_row(row) for row in rows if _is_binary(row)]
    grouped = {
        name: [row for row in binary if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    counts = {name: _counts(group) for name, group in grouped.items()}
    report: dict[str, object] = {
        "task": "generation_layer_comparison",
        "registered_before_scoring": list(REGISTERED_REPRESENTATIONS),
        "live_probe_untouched": True,
        "pair_threshold_unchanged": PAIR_THRESHOLD,
        "early_warning_lead": None,
        "event_measurement_enabled": EVENT_MEASUREMENT_ENABLED,
        "class_counts": counts,
        "representations": {},
        "successful_detector": False,
        "pair_probe_rewritten": False,
    }
    baselines = None
    if all(set(int(row["label"]) for row in grouped[name]) == {0, 1} for name in grouped):
        baselines = generation_text_baselines(_baseline_rows(binary), seed=seed)
        report["text_baselines"] = baselines
    else:
        report["text_baselines"] = {
            "note": "Text baselines were not fit. A binary split does not contain both classes."
        }
    for spec in REGISTERED_REPRESENTATIONS:
        field = str(spec["field"])
        present = [row for row in binary if row.get(field) is not None]
        if not present:
            report["representations"][spec["name"]] = {
                "available": False,
                "layer_index": spec["layer_index"],
                "position": spec["position"],
                "note": "This representation was named before scoring and was not stored.",
            }
            continue
        split_rows = {
            name: [row for row in present if row["split"] == name]
            for name in ("train", "validation", "test")
        }
        scored = _fit_vector_variant(
            split_rows["train"],
            split_rows["validation"],
            split_rows["test"],
            field=field,
            seed=seed,
        )
        scored["layer_index"] = spec["layer_index"]
        scored["position"] = spec["position"]
        scored["saved_as_live_probe"] = False
        if isinstance(scored.get("auroc_interval"), dict):
            interval = scored["auroc_interval"]
            smaller = min(counts["test"]["faithful"], counts["test"]["hallucinated"])
            if smaller < BOOTSTRAP_MIN_PER_CLASS:
                assert interval.get("point") is None
        report["representations"][spec["name"]] = scored
    probe_metrics = None
    saved = report["representations"].get("saved_probe_representation")
    if isinstance(saved, dict) and isinstance(saved.get("metrics"), dict):
        probe_metrics = saved["metrics"]
    conclusion = interpret_detector(
        probe_metrics if isinstance(probe_metrics, dict) else None,
        baselines if isinstance(baselines, dict) else {},
    )
    conclusion["successful_detector"] = False
    conclusion["ranked_by_pair_protocol_auroc"] = False
    report["detector_conclusion"] = conclusion
    gate_rows = [
        {
            "question_id": row["question_id"],
            "split": row["split"],
            "label": "HALLUCINATED" if int(row["label"]) == 1 else "FAITHFUL",
            "probe_score": None,
            "split_overlap": False,
        }
        for row in binary
    ]
    report["generation_calibration_gate"] = evaluate_generation_calibration_gate(
        gate_rows,
        threshold=PAIR_THRESHOLD,
        pair_probe_path=pair_probe_path,
    )
    report["note"] = (
        "A higher score is a comparison result. It does not name a truth direction "
        "and does not describe the probe as a validated hallucination detector."
    )
    return report


def layer_compare_command(args: object) -> int:
    """Compare pre-registered representations. Does not replace saved probes."""
    import json
    import sys

    from glassbox.evaluation.records import write_json

    try:
        run_id = str(args.run_id)
        if run_id in PROTECTED_PROBE_NAMES or run_id.startswith("final-hidden-state"):
            raise ValueError(f"Refusing protected run id {run_id}")
        if run_id == "halueval-generation-001":
            raise ValueError("Refusing to overwrite the historical generation run")
        artifacts = repo_root() / "artifacts"
        directory = generation_experiment_dir(run_id)
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite {directory}")
        predictions = Path(args.predictions)
        payload = json.loads(predictions.read_text(encoding="utf-8"))
        rows = payload.get("scored_rows") or payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("predictions file has no rows")
        joined = [dict(row) for row in rows if isinstance(row, dict)]
        if args.features:
            from glassbox.probes.generation_variants import _attach_vectors

            joined = _attach_vectors(joined, Path(args.features))
        pair_before = None
        pair_path = Path(args.pair_probe) if args.pair_probe else None
        if pair_path is not None and (pair_path / "metadata.json").is_file():
            pair_before = (pair_path / "metadata.json").read_bytes()
        report = evaluate_layer_comparison(joined, seed=int(args.seed), pair_probe_path=pair_path)
        write_json(directory / "report.json", report, root=artifacts)
        if pair_before is not None and (pair_path / "metadata.json").read_bytes() != pair_before:
            raise RuntimeError("Layer comparison rewrote the pair-probe metadata")
        print(report["detector_conclusion"]["reason"])
        print(f"Wrote {directory / 'report.json'}")
        return 0
    except (OSError, ValueError, FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Layer comparison stopped: {exc}", file=sys.stderr)
        return 1
