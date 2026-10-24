"""Generation-evaluation records fail closed without a model download."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glassbox.cli import main
from glassbox.data.schema import Label
from glassbox.evaluation.generation_record import (
    build_generation_evaluation_record,
    write_generation_evaluation,
)

REVISION = "b" * 40


def _provenance(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": REVISION,
        "dtype": "bfloat16",
        "dataset_id": "truthfulqa",
        "dataset_sha256": "c" * 64,
        "split_id": "truthfulqa-generation-split-seed0",
        "seed": 0,
        "probe_directory": "probes/not-the-live-probe",
        "probe_model_revision": REVISION,
        "layer": 24,
        "position": "last_generated_token",
        "prompt_mode": "chat_template",
        "calibration_validated": False,
        "config_path": "configs/demo.yaml",
        "git_commit": REVISION,
        "worktree_clean": True,
    }
    payload.update(overrides)
    return payload


def _rows() -> list[dict[str, object]]:
    return [
        {
            "question_id": "row0",
            "answer": "The capital of France is Paris.",
            "label": Label.FAITHFUL.name,
            "reference_phrases": ["Paris"],
            "probe_score": 0.2,
        },
        {
            "question_id": "row1",
            "answer": "The capital of France is Lyon.",
            "label": Label.HALLUCINATED.name,
            "reference_phrases": ["Lyon"],
            "probe_score": 0.8,
        },
        {
            "question_id": "row2",
            "answer": "Paris and also Lyon.",
            "label": Label.AMBIGUOUS.name,
            "reference_phrases": ["Paris", "Lyon"],
            "probe_score": 0.6,
        },
        {
            "question_id": "row3",
            "answer": "A different wording that asserts neither phrase.",
            "label": Label.UNVALIDATED.name,
            "reference_phrases": [],
            "probe_score": None,
            "missing_score_reason": "row excluded from the requested score set",
        },
    ]


def test_complete_record_keeps_excluded_labels_out_of_both_classes() -> None:
    record = build_generation_evaluation_record(
        _provenance(),
        _rows(),
        score_requested=True,
        probe_available=True,
    )
    by_label = {row["label"]: row for row in record["rows"]}
    assert by_label[Label.UNVALIDATED.name]["in_binary_metric_set"] is False
    assert by_label[Label.AMBIGUOUS.name]["in_binary_metric_set"] is False
    assert by_label[Label.UNVALIDATED.name]["label"] == Label.UNVALIDATED.name
    binary = [row for row in record["rows"] if row["in_binary_metric_set"]]
    assert {row["label"] for row in binary} == {Label.FAITHFUL.name, Label.HALLUCINATED.name}
    assert record["calibration_validated"] is False
    assert record["pair_protocol_metrics_included"] is False
    assert record["provenance"]["split_id"] == "truthfulqa-generation-split-seed0"
    assert record["provenance"]["git_commit"] == REVISION
    encoded = json.dumps(record)
    assert "halueval-pair-002" not in encoded
    assert "0.996" not in encoded
    assert "auroc" not in encoded
    assert "Pair-protocol metrics are not this result" in record["note"]


@pytest.mark.parametrize("missing", ["git_commit", "split_id"])
def test_missing_provenance_is_not_a_finished_record(tmp_path: Path, missing: str) -> None:
    provenance = _provenance()
    del provenance[missing]
    destination = tmp_path / "experiments" / "generation-eval" / "record.json"
    with pytest.raises(ValueError, match="Missing provenance"):
        write_generation_evaluation(
            provenance,
            _rows(),
            destination,
            artifacts_root=tmp_path,
            score_requested=True,
            probe_available=True,
        )
    assert not destination.exists()


def test_missing_example_row_is_not_a_finished_record(tmp_path: Path) -> None:
    destination = tmp_path / "record.json"
    with pytest.raises(ValueError, match="one row per generated example"):
        write_generation_evaluation(
            _provenance(),
            [],
            destination,
            artifacts_root=tmp_path,
            score_requested=True,
            probe_available=True,
        )
    assert not destination.exists()


def test_dirty_worktree_does_not_leave_a_finished_record(tmp_path: Path) -> None:
    destination = tmp_path / "record.json"
    with pytest.raises(ValueError, match="cannot be published as a finished reproduction"):
        write_generation_evaluation(
            _provenance(worktree_clean=False),
            _rows(),
            destination,
            artifacts_root=tmp_path,
            score_requested=True,
            probe_available=True,
        )
    assert not destination.exists()


def test_missing_probe_when_a_score_was_requested(tmp_path: Path) -> None:
    destination = tmp_path / "record.json"
    with pytest.raises(ValueError, match="probe is missing"):
        write_generation_evaluation(
            _provenance(),
            _rows(),
            destination,
            artifacts_root=tmp_path,
            score_requested=True,
            probe_available=False,
        )
    assert not destination.exists()


def test_record_does_not_set_calibration_validated() -> None:
    with pytest.raises(ValueError, match="does not set calibration_validated"):
        build_generation_evaluation_record(
            _provenance(calibration_validated=True),
            _rows(),
            score_requested=True,
            probe_available=True,
        )


def test_cli_refuses_a_partial_payload_without_loading_qwen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("model backend was constructed")

    monkeypatch.setattr("glassbox.cli.HuggingFaceBackend", forbidden)
    payload = tmp_path / "payload.json"
    provenance = _provenance()
    del provenance["git_commit"]
    payload.write_text(
        json.dumps(
            {
                "provenance": provenance,
                "rows": _rows(),
                "score_requested": True,
                "probe_available": True,
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "record.json"
    code = main(
        [
            "write-generation-evaluation",
            "--payload",
            str(payload),
            "--output",
            str(destination),
            "--artifacts-root",
            str(tmp_path),
        ]
    )
    assert code == 1
    assert not destination.exists()
