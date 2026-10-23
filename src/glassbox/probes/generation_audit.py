"""Re-read a saved generation report. This does not load Qwen or change metrics.

The original report stays the official result. A sensitivity column applies the
current reference labeler, including the ambiguous class, and records disagreements.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from glassbox.config import repo_root
from glassbox.data.schema import Label
from glassbox.data.truthfulqa import read_pinned_snapshot
from glassbox.evaluation.records import write_json
from glassbox.probes.logistic import PROBE_NOT_TRAINED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glassbox audit-generations")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--run-id", default="truthfulqa-label-audit-001")
    args = parser.parse_args(argv)
    return audit_command(args)


def audit_command(args: argparse.Namespace) -> int:
    try:
        report_path = args.report or (
            repo_root() / "artifacts" / "experiments" / "truthfulqa-generation-001" / "report.json"
        )
        snapshot = args.snapshot or (
            repo_root() / "artifacts" / "datasets" / "truthfulqa" / "TruthfulQA.csv"
        )
        if not Path(report_path).is_file():
            raise FileNotFoundError(f"Generation report is not at {report_path}")
        if not Path(snapshot).is_file():
            raise FileNotFoundError(f"TruthfulQA CSV is not at {snapshot}")
        payload = audit_report(Path(report_path), Path(snapshot))
        destination = repo_root() / "artifacts" / "experiments" / args.run_id / "audit.json"
        write_json(destination, payload, root=repo_root() / "artifacts")
        print(
            f"Official labels: {payload['official_counts']}. "
            f"Sensitivity labels: {payload['sensitivity_counts']}. "
            f"Disagreements: {payload['disagreement_count']}."
        )
        print(f"Wrote {destination}")
        print("The original report was not modified and its metrics were not recomputed.")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"{PROBE_NOT_TRAINED}: {exc}", file=sys.stderr)
        return 1
    return 0


def audit_report(report_path: Path, snapshot: Path) -> dict[str, object]:
    """Join saved answers to reference lists and compare labelers."""
    from glassbox.data.truthfulqa import label_generated_answer

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Generation report has no rows")
    items = {item.question_id: item for item in read_pinned_snapshot(snapshot)}
    compared: list[dict[str, object]] = []
    missing = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        question_id = str(row.get("question_id", ""))
        item = items.get(question_id)
        if item is None:
            missing += 1
            continue
        label, detail = label_generated_answer(str(row.get("answer", "")), item)
        official = str(row.get("label", ""))
        compared.append(
            {
                "question_id": question_id,
                "question": row.get("question"),
                "answer": row.get("answer"),
                "official_label": official,
                "sensitivity_label": label.name,
                "sensitivity_detail": detail,
                "changed": official != label.name,
            }
        )
    official_counts = _counts(row["official_label"] for row in compared)
    sensitivity_counts = _counts(row["sensitivity_label"] for row in compared)
    disagreements = [row for row in compared if row["changed"]]
    labeled = [row for row in compared if row["official_label"] != Label.UNVALIDATED.name]
    unvalidated = [row for row in compared if row["official_label"] == Label.UNVALIDATED.name][:8]
    return {
        "source_report": str(report_path),
        "official_metrics_unchanged": True,
        "rows_joined": len(compared),
        "rows_missing_from_csv": missing,
        "official_counts": official_counts,
        "sensitivity_counts": sensitivity_counts,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "official_labeled_sample": labeled,
        "unvalidated_sample": unvalidated,
        "note": (
            "Unvalidated answers stay out of the official metrics. "
            "They are not treated as faithful or hallucinated. "
            "Sensitivity labels can mark an answer ambiguous when it asserts both a correct "
            "and an incorrect reference. That does not replace the saved metrics."
        ),
    }


def _counts(labels) -> dict[str, int]:
    found = {name: 0 for name in ("FAITHFUL", "HALLUCINATED", "AMBIGUOUS", "UNVALIDATED")}
    for label in labels:
        if label not in found:
            found[label] = 0
        found[label] += 1
    return found


if __name__ == "__main__":
    raise SystemExit(main())
