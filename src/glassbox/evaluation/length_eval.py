"""CLI for the length-shortcut diagnostic. The pair probe is never rewritten."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from glassbox.config import repo_root
from glassbox.demo_config import default_probe_dir, load_demo_config
from glassbox.evaluation.generation_calibration import PAIR_THRESHOLD
from glassbox.evaluation.length_shortcut import controlled_variants, summarize_length_effects
from glassbox.evaluation.records import write_json
from glassbox.models.backend import GenerationSettings, HuggingFaceBackend, completion_hidden_states
from glassbox.probes.generation_eval import (
    PROTECTED_PROBE_NAMES,
    generation_experiment_dir,
)
from glassbox.probes.store import (
    ProbeCompatibilityError,
    check_compatibility,
    experimental_score,
    load_probe,
)
from glassbox.verification import load_curated_facts


def length_eval_command(args: argparse.Namespace) -> int:
    try:
        run_id = str(args.run_id)
        if run_id in PROTECTED_PROBE_NAMES or run_id.startswith("final-hidden-state"):
            raise ValueError(f"Refusing protected run id {run_id}")
        artifacts = repo_root() / "artifacts"
        directory = generation_experiment_dir(run_id)
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite {directory}")
        pair_dir = Path(args.probe) if args.probe else default_probe_dir()
        pair_before = _metadata_bytes(pair_dir)
        if args.predictions:
            rows = _rows_from_predictions(Path(args.predictions))
            report = summarize_length_effects(rows, threshold=float(args.threshold))
            source = "historical_generation_scores"
        elif args.scores:
            scored = json.loads(Path(args.scores).read_text(encoding="utf-8"))
            if not isinstance(scored, list):
                raise ValueError("Controlled scores must be a list")
            rows = [dict(item) for item in scored if isinstance(item, dict)]
            report = summarize_length_effects(rows, threshold=float(args.threshold))
            source = "supplied_scores"
        elif args.controlled:
            if abs(float(args.threshold) - PAIR_THRESHOLD) > 1e-12:
                raise ValueError(
                    f"Refusing threshold {args.threshold} for controlled pair-probe completions. "
                    f"The frozen threshold is {PAIR_THRESHOLD}."
                )
            rows, source = _score_controlled(args, pair_dir)
            report = summarize_length_effects(rows, threshold=PAIR_THRESHOLD)
        else:
            raise ValueError("Choose --predictions, --scores, or --controlled")
        report["source"] = source
        report["pair_probe_directory"] = str(pair_dir)
        report["pair_probe_rewritten"] = False
        report["ui_semantics"] = "EXPERIMENTAL PROBE SCORE; RISK NOT ASSESSED"
        write_json(directory / "report.json", report, root=artifacts)
        pair_after = _metadata_bytes(pair_dir)
        if pair_before is not None and pair_after != pair_before:
            raise RuntimeError("Length evaluation rewrote the pair-probe metadata")
        print(report["interpretation"]["reason"])
        print(f"Wrote {directory / 'report.json'}")
        return 0
    except (
        OSError,
        ValueError,
        FileNotFoundError,
        RuntimeError,
        ProbeCompatibilityError,
        json.JSONDecodeError,
        ImportError,
        MemoryError,
    ) as exc:
        print(f"Length evaluation stopped: {exc}", file=sys.stderr)
        return 1


def _rows_from_predictions(path: Path) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("scored_rows") or payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no scored rows")
    converted = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("label_name") or "")
        if name == "FAITHFUL":
            label = 0
        elif name == "HALLUCINATED":
            label = 1
        elif row.get("label") in {0, 1}:
            label = int(row["label"])
        else:
            continue
        if row.get("probe_score") is None:
            continue
        converted.append(
            {
                "question_id": row.get("question_id"),
                "answer": row.get("answer"),
                "label": label,
                "probe_score": float(row["probe_score"]),
                "split": row.get("split"),
            }
        )
    return converted


def _score_controlled(
    args: argparse.Namespace, pair_dir: Path
) -> tuple[list[dict[str, object]], str]:
    facts = load_curated_facts(
        Path(args.references) if args.references else repo_root() / "configs" / "curated_facts.yaml"
    )
    variants = [item for fact in facts for item in controlled_variants(fact)]
    if args.dry_run or args.scores:
        if args.scores:
            scored = json.loads(Path(args.scores).read_text(encoding="utf-8"))
            if not isinstance(scored, list):
                raise ValueError("Controlled scores must be a list")
            return [dict(item) for item in scored if isinstance(item, dict)], "supplied_scores"
        for item in variants:
            item["probe_score"] = 0.0
        return variants, "dry_run_unscored"
    probe = load_probe(pair_dir)
    config = load_demo_config(args.config, offline=True, model_name=getattr(args, "model", None))
    settings = GenerationSettings(
        max_new_tokens=config.generation.max_new_tokens,
        temperature=0.0,
        top_k=config.generation.top_k,
        layer_index=config.generation.layer_index,
        position="last_generated_token",
        capture_all_layers=False,
    )
    backend = HuggingFaceBackend(config.spec)
    loaded = backend.ensure_loaded()
    for item in variants:
        vectors, hidden_count, feature_layer, prompt_mode, _prompt = completion_hidden_states(
            loaded,
            str(item["question"]),
            str(item["answer"]),
            settings,
        )
        feature = vectors[int(feature_layer)]
        check_compatibility(
            probe,
            model_name=loaded.spec.name,
            model_revision=loaded.resolved_revision,
            layer_index=int(feature_layer),
            hidden_state_count=int(hidden_count),
            position=settings.position,
            prompt_mode=prompt_mode,
            feature_dim=int(feature.shape[0]),
            dtype=loaded.dtype,
        )
        item["probe_score"] = experimental_score(probe, feature)
    return variants, "controlled_completions"


def _metadata_bytes(directory: Path) -> bytes | None:
    path = Path(directory) / "metadata.json"
    if not path.is_file():
        return None
    return path.read_bytes()
