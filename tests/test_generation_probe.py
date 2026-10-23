"""Generation-task probe checks. These tests do not load Qwen or download data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from glassbox.activations.extract import (
    describe_generated_position,
    feature_index,
    mean_answer_vector,
    vectors_at_position,
)
from glassbox.cli import main
from glassbox.data.evidence import (
    apply_human_review,
    label_halueval_generation,
    read_human_reviews,
    separate_probe_score,
)
from glassbox.data.generation_splits import (
    assert_group_isolation,
    assign_generation_splits,
    subset_preserving_groups,
)
from glassbox.data.halueval import HaluPair
from glassbox.data.schema import Label
from glassbox.data.truthfulqa import TruthfulItem, label_generated_answer_coverage
from glassbox.evaluation.metrics import bootstrap_grouped_interval
from glassbox.guards import DownloadDisabledError
from glassbox.models.backend import ModelSpec
from glassbox.probes.generation_eval import (
    assert_output_allowed,
    calibration_decision,
    score_binary,
    validate_provenance,
)
from glassbox.probes.generation_train import finalize_generation_run
from glassbox.probes.preprocess import apply_standard_scaler
from glassbox.probes.store import experimental_score, load_probe
from glassbox.verification import assess_answer, load_curated_facts


def test_generated_piece_is_not_the_prompt_piece() -> None:
    assert feature_index(10, 4, "last_generated_token") == 13
    assert feature_index(10, 4, "last_prompt_token") == 9
    described = describe_generated_position(10, 4)
    assert described["sequence_index"] == 13
    assert described["index_is_inside_generated_span"] is True
    assert described["aggregation"] == "final generated piece only"
    hidden = torch.zeros(1, 6, 3)
    hidden[0, 4] = 1
    hidden[0, 5] = 3
    states = (hidden,)
    last = vectors_at_position(states, feature_index(4, 2, "last_generated_token"))[0]
    pooled = mean_answer_vector(states, 4, 2, 0)
    assert last.tolist() == pytest.approx([3, 3, 3])
    assert pooled.tolist() == pytest.approx([2, 2, 2])
    assert not np.allclose(pooled, hidden[0, 0].numpy())


def test_shared_knowledge_cannot_cross_splits() -> None:
    pairs = []
    for index in range(8):
        shared = index < 2
        pairs.append(
            HaluPair(
                question_id=f"q{index}",
                source_row=index,
                question=f"Question {index}?",
                knowledge="same passage about Delhi" if shared else f"passage {index} about a city",
                right_answer=f"Answer {index}",
                hallucinated_answer=f"Other {index}",
            )
        )
    assignment, groups = assign_generation_splits(pairs, seed=0)
    assert_group_isolation(assignment, groups)
    located = {}
    for split in ("train", "validation", "test"):
        for question_id in assignment.ids(split):
            located[question_id] = split
    assert located["q0"] == located["q1"]
    selected = subset_preserving_groups(assignment, groups, 6)
    assert_group_isolation(selected, groups)
    selected_ids = set(selected.train + selected.validation + selected.test)
    assert ("q0" in selected_ids) == ("q1" in selected_ids)


def test_evidence_labels_support_contradiction_conflict_and_abstention() -> None:
    knowledge = "The Oberoi Group is a hotel company with its head office in Delhi."
    question = "The Oberoi family hotel company has a head office in Delhi or Mumbai?"
    right = "Delhi"
    false = "The hotel company is based in Mumbai."
    faithful = label_halueval_generation(
        question=question,
        knowledge=knowledge,
        right_answer=right,
        hallucinated_answer=false,
        answer="The head office is in Delhi.",
    )
    assert faithful.label is Label.FAITHFUL
    contradicted = label_halueval_generation(
        question=question,
        knowledge=knowledge,
        right_answer=right,
        hallucinated_answer=false,
        answer="It is based in Mumbai.",
    )
    assert contradicted.label is Label.HALLUCINATED
    mixed = label_halueval_generation(
        question=question,
        knowledge=knowledge,
        right_answer=right,
        hallucinated_answer=false,
        answer="The office is in Delhi, but also in Mumbai.",
    )
    assert mixed.label is Label.AMBIGUOUS
    missing = label_halueval_generation(
        question=question,
        knowledge=knowledge,
        right_answer=right,
        hallucinated_answer=false,
        answer="I do not know the city.",
    )
    assert missing.label is Label.UNVALIDATED
    paraphrase = label_halueval_generation(
        question="Who was Milhouse named after?",
        knowledge="Matt Groening named the character after President Richard Nixon.",
        right_answer="Richard Nixon",
        hallucinated_answer="The character was named after Milhouse senior.",
        answer="Richard Milhous Nixon was the namesake.",
    )
    assert paraphrase.label is Label.FAITHFUL
    assert paraphrase.support_kind == "paraphrase"
    ungrounded = label_halueval_generation(
        question=question,
        knowledge="This passage does not name a city.",
        right_answer=right,
        hallucinated_answer=false,
        answer="Delhi",
    )
    assert ungrounded.label is Label.UNVALIDATED
    discourse = label_halueval_generation(
        question="What do the two universities have in common?",
        knowledge=(
            "The University of Arizona is a public research university in the United States. "
            "The University of Massachusetts Amherst is a public research university in the United States."
        ),
        right_answer="public research university",
        hallucinated_answer="Both universities are located in the United States.",
        answer="Both universities are located in the state of Massachusetts, United States.",
    )
    assert discourse.label is Label.UNVALIDATED
    series = label_halueval_generation(
        question="Which league did the club play in?",
        knowledge="The club played in National League A.",
        right_answer="National League A.",
        hallucinated_answer="Swiss Ice Hockey League.",
        answer="The club competes in the Swiss National League.",
    )
    assert series.label is not Label.FAITHFUL


def test_human_review_requires_evidence_and_is_not_invented(tmp_path: Path) -> None:
    decision = label_halueval_generation(
        question="Where?",
        knowledge="The office is in Delhi.",
        right_answer="Delhi",
        hallucinated_answer="The office is in Mumbai.",
        answer="Somewhere in India.",
    )
    assert decision.human_reviewed is False
    with pytest.raises(ValueError, match="evidence_note"):
        read_human_reviews_text = tmp_path / "reviews.jsonl"
        read_human_reviews_text.write_text(
            json.dumps({"question_id": "q1", "label": "FAITHFUL", "reviewer": "ada"}) + "\n",
            encoding="utf-8",
        )
        read_human_reviews(read_human_reviews_text)
    path = tmp_path / "good.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "label": "FAITHFUL",
                "reviewer": "ada",
                "evidence_note": "The passage says the office is in Delhi and the answer restates that.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    review = read_human_reviews(path)["q1"]
    reviewed = apply_human_review(decision, review)
    assert reviewed.label is Label.FAITHFUL
    assert reviewed.automatic_label is Label.UNVALIDATED
    assert reviewed.label_source == "human_review_v1"


def test_paris_period_stays_a_reference_label_not_a_probe_score() -> None:
    facts = load_curated_facts(Path("configs/curated_facts.yaml"))
    france = next(fact for fact in facts if fact.item_id == "fr-capital")
    reference = assess_answer(france, "Paris.")
    assert reference.label is Label.FAITHFUL
    unrelated = label_halueval_generation(
        question="Which magazine was started first, Arthur's Magazine or First for Women?",
        knowledge="Arthur's Magazine (1844-1846) was a periodical. First for Women is a magazine.",
        right_answer="Arthur's Magazine",
        hallucinated_answer="First for Women was started first.",
        answer="Paris.",
    )
    assert unrelated.label is Label.UNVALIDATED
    separated = separate_probe_score(unrelated, 0.99)
    assert separated["factuality_label"] == "UNVALIDATED"
    assert separated["probe_score"] == 0.99
    assert separated["probe_score_is_hallucination_probability"] is False
    assert separated["reference_check_is_separate_from_probe"] is True


def test_truthfulqa_coverage_accepts_a_paraphrase_and_abstains_otherwise() -> None:
    item = TruthfulItem(
        question_id="row0",
        category="Misconceptions",
        question="What happens if you eat watermelon seeds?",
        correct=("The watermelon seeds pass through your digestive system",),
        incorrect=("You grow watermelons in your stomach",),
    )
    label, _detail, record = label_generated_answer_coverage(
        "Watermelon seeds pass through the digestive system.",
        item,
    )
    assert label is Label.FAITHFUL
    assert record["assertion_label"] == "UNVALIDATED"
    assert record["coverage_upgraded"] is True
    abstained, _, abstained_record = label_generated_answer_coverage(
        "The seeds leave the body without growing.",
        item,
    )
    assert abstained is Label.UNVALIDATED
    assert abstained_record["coverage_upgraded"] is False
    conflict, _, _conflict_record = label_generated_answer_coverage(
        "The watermelon seeds pass through your digestive system and you grow watermelons in your stomach.",
        item,
    )
    assert conflict is Label.AMBIGUOUS


def test_single_class_and_small_samples_do_not_invent_metrics() -> None:
    scored = score_binary([0, 0, 0], [0.2, 0.3, 0.4], ["a", "b", "c"], threshold=0.5)
    assert scored["metrics"] is None
    assert "one class" in str(scored["note"])
    interval = bootstrap_grouped_interval(
        [0, 1, 0], [0.1, 0.9, 0.2], ["q", "q", "r"], min_per_class=20
    )
    assert interval["low"] is None
    decision = calibration_decision(
        {"faithful": 5, "hallucinated": 5},
        {"faithful": 20, "hallucinated": 20},
        0.01,
    )
    assert decision["status"] == "rejected"
    assert decision["calibration_validated"] is False
    high_ece = calibration_decision(
        {"faithful": 20, "hallucinated": 20},
        {"faithful": 20, "hallucinated": 20},
        0.2,
    )
    assert high_ece["status"] == "rejected"


def test_protected_and_existing_run_directories_are_refused(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    protected = artifacts / "probes" / "final-hidden-state"
    protected.mkdir(parents=True)
    with pytest.raises(ValueError, match="protected"):
        assert_output_allowed(artifacts, "halueval-generation-001", protected)
    previous = artifacts / "probes" / "final-hidden-state.previous"
    previous.mkdir()
    with pytest.raises(ValueError, match="protected"):
        assert_output_allowed(artifacts, "halueval-generation-001", previous)
    experiment = artifacts / "experiments" / "halueval-pair-002"
    experiment.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        assert_output_allowed(
            artifacts,
            "halueval-pair-002",
            artifacts / "probes" / "generation-runs" / "halueval-pair-002",
        )


def test_provenance_rejects_a_dirty_tree_marked_reproducible() -> None:
    metadata = _provenance_metadata()
    metadata["publishable_as_reproducible"] = True
    metadata["worktree_dirty"] = True
    with pytest.raises(ValueError, match="dirty"):
        validate_provenance(metadata)
    incomplete = dict(metadata)
    incomplete["publishable_as_reproducible"] = False
    incomplete.pop("encoder_revision")
    with pytest.raises(ValueError, match="encoder_revision"):
        validate_provenance(incomplete)
    metadata["publishable_as_reproducible"] = False
    validate_provenance(metadata)


def test_missing_snapshot_names_the_path_and_does_not_download(monkeypatch, capsys) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("dataset download was attempted")

    monkeypatch.setattr("glassbox.data.halueval.download_snapshot", forbidden)
    monkeypatch.setattr("urllib.request.urlretrieve", forbidden)
    missing = Path("artifacts/datasets/halueval/qa_data.json.missing")
    code = main(
        [
            "train-generation-probe",
            "--snapshot",
            str(missing),
            "--run-id",
            "generation-missing-snapshot",
            "--max-questions",
            "30",
            "--external-questions",
            "0",
        ]
    )
    assert code == 1
    assert str(missing) in capsys.readouterr().err


def test_model_cache_download_stays_disabled() -> None:
    with pytest.raises(DownloadDisabledError):
        ModelSpec(
            name="Qwen/Qwen2.5-0.5B-Instruct",
            revision="a" * 40,
            allow_download=False,
            local_files_only=False,
        )


def test_cached_generation_entry_does_not_download_weights(monkeypatch, capsys) -> None:
    snapshot = Path("artifacts/datasets/halueval/qa_data.json")
    if not snapshot.is_file():
        pytest.skip("HaluEval snapshot is not cached")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("download was attempted")

    monkeypatch.setattr("glassbox.data.halueval.download_snapshot", forbidden)
    monkeypatch.setattr("urllib.request.urlretrieve", forbidden)

    class MissingCache:
        def __init__(self, _spec):
            raise RuntimeError("cached model weights were not opened")

    monkeypatch.setattr("glassbox.probes.generation_train.HuggingFaceBackend", MissingCache)
    code = main(
        [
            "--offline",
            "train-generation-probe",
            "--snapshot",
            str(snapshot),
            "--run-id",
            "generation-cache-guard",
            "--max-questions",
            "12",
            "--external-questions",
            "0",
        ]
    )
    assert code == 1
    assert "cached model weights" in capsys.readouterr().err
    assert not Path("artifacts/experiments/generation-cache-guard").exists()
    assert not Path("artifacts/probes/generation-runs/generation-cache-guard").exists()


def test_finalize_rejects_a_one_class_training_pool_without_moving_rows(tmp_path: Path) -> None:
    rows = _population(train_faithful=4, train_hallucinated=0, validation=4, test=4)
    code = finalize_generation_run(
        rows,
        artifacts=tmp_path,
        probe_dir=tmp_path / "probes" / "generation-runs" / "small",
        experiment_dir=tmp_path / "experiments" / "small",
        run_id="small",
        seed=0,
        dataset_record={"sha256": "c" * 64, "label_source": "halueval_generation_evidence_v1"},
        provenance=_provenance_payload(),
        pair_probe_dir=tmp_path / "missing-pair",
        external_rows=None,
        external_note="not run",
    )
    assert code == 1
    report = json.loads((tmp_path / "experiments" / "small" / "report.json").read_text())
    assert report["probe_saved"] is False
    assert report["detector_conclusion"]["successful_detector"] is False
    assert report["calibration"]["status"] == "unavailable"
    assert not (tmp_path / "probes" / "generation-runs" / "small").exists()


def test_finalize_keeps_scores_experimental_and_separate_from_the_live_probe(
    tmp_path: Path,
) -> None:
    rows = _population(train_faithful=20, train_hallucinated=20, validation=8, test=8)
    rows.append(_row("ambiguous", "validation", "AMBIGUOUS", 2, 0.5, "g-ambiguous"))
    probe_dir = tmp_path / "probes" / "generation-runs" / "synthetic"
    code = finalize_generation_run(
        rows,
        artifacts=tmp_path,
        probe_dir=probe_dir,
        experiment_dir=tmp_path / "experiments" / "synthetic",
        run_id="synthetic",
        seed=0,
        dataset_record={"sha256": "c" * 64, "label_source": "halueval_generation_evidence_v1"},
        provenance=_provenance_payload(),
        pair_probe_dir=tmp_path / "missing-pair",
        external_rows=None,
        external_note="TruthfulQA file was not used in this unit test.",
    )
    assert code == 0
    report = json.loads((tmp_path / "experiments" / "synthetic" / "report.json").read_text())
    assert report["protocol"] == "halueval_generation_evidence_v1"
    assert report["held_out"]["metrics"]["auroc"] == pytest.approx(1.0)
    assert report["calibration"]["status"] == "rejected"
    assert report["calibration"]["calibration_validated"] is False
    assert (
        report["detector_conclusion"]["score_semantics"]
        == "EXPERIMENTAL PROBE SCORE; RISK NOT ASSESSED"
    )
    assert report["live_probe_untouched"] is True
    assert report["pair_probe_comparison"]["available"] is False
    assert report["representation"]["saved_probe_uses_answer_mean"] is False
    loaded = load_probe(probe_dir)
    assert loaded.metadata["label_protocol"] == "halueval_generation_evidence_v1"
    assert loaded.metadata["calibration_validated"] is False
    assert loaded.calibrated is False
    vector = np.zeros(4, dtype=np.float64)
    manual = apply_standard_scaler(
        vector,
        np.asarray(loaded.metadata["preprocessing"]["mean"]),
        np.asarray(loaded.metadata["preprocessing"]["scale"]),
    )
    assert experimental_score(loaded, vector) == pytest.approx(
        float(loaded.probe.predict_proba(manual)[0])
    )
    predictions = json.loads(
        (tmp_path / "experiments" / "synthetic" / "predictions.json").read_text()
    )
    assert all("vector" not in row for row in predictions["rows"])
    assert (tmp_path / "experiments" / "synthetic" / "features.npz").is_file()
    assert not (tmp_path / "probes" / "final-hidden-state").exists()


def _population(
    *, train_faithful: int, train_hallucinated: int, validation: int, test: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cursor = 0

    def add(split: str, label_name: str, label: int, value: float, count: int) -> None:
        nonlocal cursor
        for _ in range(count):
            rows.append(_row(f"q{cursor}", split, label_name, label, value, f"g{cursor}"))
            cursor += 1

    add("train", "FAITHFUL", 0, 0.0, train_faithful)
    add("train", "HALLUCINATED", 1, 1.0, train_hallucinated)
    add("validation", "FAITHFUL", 0, 0.0, validation)
    add("validation", "HALLUCINATED", 1, 1.0, validation)
    add("test", "FAITHFUL", 0, 0.0, test)
    add("test", "HALLUCINATED", 1, 1.0, test)
    return rows


def _row(
    question_id: str,
    split: str,
    label_name: str,
    label: int,
    value: float,
    group_id: str,
) -> dict[str, object]:
    vector = np.full(4, value, dtype=np.float64)
    answer = "alpha answer" if label == 0 else "beta contradicted answer"
    if label_name == "AMBIGUOUS":
        answer = "alpha answer and beta contradicted answer"
    return {
        "question_id": question_id,
        "group_id": group_id,
        "split": split,
        "question": f"Where is {question_id}?",
        "answer": answer,
        "knowledge": f"Passage for {question_id}",
        "right_answer": "alpha answer",
        "hallucinated_answer": "beta contradicted answer",
        "label": label,
        "label_name": label_name,
        "exclusion": None,
        "decision": {"detail": "unit evidence"},
        "vector": vector,
        "answer_mean": vector.copy(),
        "hidden_state_count": 2,
        "feature_layer": 1,
        "feature_dim": 4,
        "prompt_mode": "chat_template",
        "position": "last_generated_token",
        "model_name": "example",
        "model_revision": "a" * 40,
        "encoder_revision": "a" * 40,
        "encoder_class": "ExampleEncoder",
        "dtype": "bfloat16",
        "false_spans": ["beta"],
        "support_kind": "exact" if label == 0 else "none",
        "grounding": "phrase",
        "evidence_kind": "support" if label == 0 else "contradiction",
    }


def _provenance_payload() -> dict[str, object]:
    return {
        "git_commit": "b" * 40,
        "worktree_dirty": True,
        "publishable_as_reproducible": False,
        "reproducibility_blockers": ["worktree has uncommitted changes"],
        "encoder_revision": "a" * 40,
        "encoder_class": "ExampleEncoder",
        "model_revision": "a" * 40,
    }


def _provenance_metadata() -> dict[str, object]:
    return {
        "model_revision": "a" * 40,
        "encoder_revision": "a" * 40,
        "dataset_sha256": "c" * 64,
        "seed": 0,
        "position": "last_generated_token",
        "preprocessing": {"type": "standard_scaler", "mean": [0.0], "scale": [1.0]},
        "label_protocol": "halueval_generation_evidence_v1",
        "publishable_as_reproducible": False,
        "worktree_dirty": True,
    }
