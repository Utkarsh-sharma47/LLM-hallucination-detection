"""CLI for the generation review queue. This command does not train or load Qwen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from glassbox.config import repo_root
from glassbox.data.review_log import (
    REVIEW_PROTOCOL,
    append_reviews,
    apply_current_labels,
    display_queue_item,
    export_snapshot,
    load_history,
    unreviewed_rows,
)
from glassbox.evaluation.records import write_json
from glassbox.probes.generation_eval import generation_experiment_dir

DEFAULT_QUEUE = "artifacts/experiments/halueval-generation-001/review_queue.json"


def review_command(args: argparse.Namespace) -> int:
    try:
        queue_path = Path(args.queue) if args.queue else repo_root() / DEFAULT_QUEUE
        if not queue_path.is_file():
            raise FileNotFoundError(f"Review queue is not at {queue_path}")
        payload = json.loads(queue_path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("Review queue does not contain a row list")
        history_path = (
            Path(args.history)
            if args.history
            else repo_root()
            / "artifacts"
            / "experiments"
            / "generation-reviews"
            / "halueval-generation-001.jsonl"
        )
        events = load_history(history_path)
        if args.append:
            added = append_reviews(
                history_path,
                _read_payloads(Path(args.append)),
                existing=events,
            )
            events = load_history(history_path)
            print(f"Appended {len(added)} review event(s) to {history_path}")
        labeled = apply_current_labels(rows, events)
        pending = unreviewed_rows(rows, events)
        if args.show:
            print(REVIEW_PROTOCOL)
            print(f"Queue {queue_path}: {len(pending)} unreviewed of {len(rows)}")
            for row in pending[: max(0, int(args.show))]:
                print()
                print(display_queue_item(row))
        if args.export_snapshot:
            snapshot = export_snapshot(events)
            destination = Path(args.export_snapshot)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in snapshot),
                encoding="utf-8",
            )
            print(f"Wrote {len(snapshot)} current reviews to {destination}")
        if args.run_id:
            artifacts = repo_root() / "artifacts"
            directory = generation_experiment_dir(str(args.run_id))
            if directory.exists():
                raise FileExistsError(f"Refusing to overwrite {directory}")
            summary = {
                "protocol": REVIEW_PROTOCOL,
                "queue": str(queue_path),
                "history": str(history_path),
                "queue_rows": len(rows),
                "reviewed": sum(1 for row in labeled if row.get("human_reviewed")),
                "unreviewed": len(pending),
                "invented_binary_labels": False,
                "note": (
                    "This export is a review workspace. Unreviewed UNVALIDATED rows "
                    "were not converted into FAITHFUL or HALLUCINATED."
                ),
            }
            write_json(directory / "review_summary.json", summary, root=artifacts)
            print(f"Wrote {directory / 'review_summary.json'}")
        if not args.show and not args.append and not args.export_snapshot and not args.run_id:
            print(REVIEW_PROTOCOL)
            print(f"Queue {queue_path}: {len(pending)} unreviewed of {len(rows)}")
            print("Use --show N, --append events.jsonl, or --export-snapshot path.")
        return 0
    except (OSError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Review stopped: {exc}", file=sys.stderr)
        return 1


def _read_payloads(path: Path) -> list[dict[str, object]]:
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("reviews") or payload.get("rows") or [payload]
        if not isinstance(payload, list):
            raise ValueError("Review payload must be a list of objects")
        return [item for item in payload if isinstance(item, dict)]
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Review line {line_number} is not an object")
        rows.append(item)
    return rows
