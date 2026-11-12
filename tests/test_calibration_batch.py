"""Preselected calibration batch: planning, resumable generation, blind queue, and the gate.

Everything here is synthetic and offline. No model, no network, no real probe.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from glassbox.data.generation_splits import knowledge_group
from glassbox.data.halueval import HaluPair
from glassbox.probes import calibration_batch as cb
from glassbox.probes.logistic import LogisticProbe
from glassbox.probes.store import save_probe


def _pairs(groups: int = 20, per_group: int = 2) -> dict[str, HaluPair]:
    pairs: dict[str, HaluPair] = {}
    for group in range(groups):
        for member in range(per_group):
            question_id = f"row{group * 10 + member}"
            pairs[question_id] = HaluPair(
                question_id=question_id,
                source_row=group * 10 + member,
                question=f"Question {group}-{member}?",
                knowledge=f"Passage number {group} says something specific about topic {group}.",
                right_answer="right",
                hallucinated_answer="wrong",
            )
    return pairs


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    _write(
        root / "probes" / "p" / "metadata.json",
        {"dataset": {"used_question_ids": {"train": ["row0", "row10"]}}},
    )
    _write(root / "experiments" / "e" / "splits.json", {"question_ids": {"test": ["row20"]}})
    _write(root / "experiments" / "e" / "predictions.json", {"rows": [{"question_id": "row30"}]})
    return root


def test_selection_is_deterministic_and_keeps_groups_whole() -> None:
    pairs = _pairs()
    excluded = {"row0", "row20"}
    first = cb.select_batch(pairs, excluded, size=10)
    shuffled = dict(random.Random(1).sample(sorted(pairs.items()), len(pairs)))
    second = cb.select_batch(shuffled, excluded, size=10)
    assert first["selected"] == second["selected"]
    assert first["reserve"] == second["reserve"]
    selected, reserve = set(first["selected"]), set(first["reserve"])
    assert len(selected) >= 10
    group_of = {item: knowledge_group(pairs[item].knowledge) for item in pairs}
    banned = {group_of[item] for item in excluded}
    assert not selected & excluded
    assert not {group_of[item] for item in selected} & banned
    assert not {group_of[item] for item in reserve} & banned
    assert not {group_of[item] for item in selected} & {group_of[item] for item in reserve}
    assert not selected & reserve


def test_questions_sharing_a_passage_with_an_excluded_question_are_removed() -> None:
    pairs = _pairs()
    chosen = cb.select_batch(pairs, {"row0"}, size=4)
    assert "row1" not in chosen["selected"] and "row1" not in chosen["reserve"]


def test_too_small_a_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="eligible"):
        cb.select_batch(_pairs(groups=3), set(), size=50)


def test_scan_finds_ids_in_probe_experiment_and_other_plan_files(tmp_path) -> None:
    root = _artifacts(tmp_path)
    _write(
        root / "calibration" / "older" / "plan.json",
        {"plan_sha256": "x", "selected_question_ids": ["row40"]},
    )
    _write(root / "calibration" / "mine" / "plan.json", {"selected_question_ids": ["row50"]})
    records, ids = cb.scan_exclusions(root, skip_dir=root / "calibration" / "mine")
    assert ids == {"row0", "row10", "row20", "row30", "row40"}
    assert {item["path"] for item in records} >= {
        "probes/p/metadata.json",
        "calibration/older/plan.json",
    }


def _plan(tmp_path, size: int = 10, shard_size: int = 4):
    root = _artifacts(tmp_path)
    pairs = _pairs()
    plan = cb.build_plan(
        pairs,
        batch_id="b",
        size=size,
        dataset_sha256="d" * 64,
        artifacts=root,
        shard_size=shard_size,
    )
    return root, pairs, plan


def test_plan_hash_detects_edits_and_verify_catches_later_contamination(tmp_path) -> None:
    root, pairs, plan = _plan(tmp_path)
    assert cb.verify_plan(plan, pairs, root) == []
    tampered = dict(plan)
    tampered["selected_question_ids"] = list(reversed(plan["selected_question_ids"]))
    assert "plan_sha256" in " ".join(cb.verify_plan(tampered, pairs, root))
    used = plan["selected_question_ids"][0]
    _write(root / "experiments" / "later" / "predictions.json", {"rows": [{"question_id": used}]})
    assert any("recorded probe or run files" in item for item in cb.verify_plan(plan, pairs, root))


def test_plan_records_rules_and_no_probe_score_in_selection(tmp_path) -> None:
    _, _, plan = _plan(tmp_path)
    assert plan["score_of_any_probe_used_in_selection"] is False
    assert plan["created_before_generation"] is True
    assert any("fixed threshold 0.65" in rule for rule in plan["decision_rules"])
    assert plan["shard_count"] == -(-plan["selected_count"] // 4)


class _Rows:
    """Stand-in for the model-backed row generator."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, backend, prepared, group_of, settings, reviews, loaded):
        self.calls += 1
        rows = []
        for question_id in prepared.assignment.ids("validation"):
            pair = prepared.pairs[question_id]
            excluded = int(question_id[3:]) % 9 == 1
            rows.append(
                {
                    "question_id": question_id,
                    "group_id": group_of[question_id],
                    "split": "validation",
                    "question": pair.question,
                    "answer": "" if excluded else f"Answer to {pair.question}",
                    "knowledge": pair.knowledge,
                    "right_answer": pair.right_answer,
                    "hallucinated_answer": pair.hallucinated_answer,
                    "label_name": "EXCLUDED" if excluded else "HALLUCINATED",
                    "vector": None if excluded else np.asarray([0.1, 0.2]),
                    "answer_mean": None,
                    "model_name": "m",
                    "model_revision": "a" * 40,
                    "dtype": "bfloat16",
                    "feature_layer": 24,
                    "hidden_state_count": 25,
                    "position": "last_generated_token",
                    "prompt_mode": "chat_template",
                }
            )
        return rows


def _generate(tmp_path, plan, pairs, rows, shards=None, meta=None):
    run_dir = tmp_path / "run"
    done = cb.generate_shards(
        plan,
        pairs,
        run_dir=run_dir,
        shards=range(plan["shard_count"]) if shards is None else shards,
        backend=None,
        loaded=None,
        settings=None,
        rows_fn=rows,
        run_meta=meta or {"model_name": "m", "model_revision": "a" * 40, "dtype": "bfloat16"},
    )
    return run_dir, done


def test_generation_resumes_and_writes_the_json_last(tmp_path) -> None:
    _, pairs, plan = _plan(tmp_path)
    rows = _Rows()
    run_dir, done = _generate(tmp_path, plan, pairs, rows)
    assert done == list(range(plan["shard_count"]))
    calls = rows.calls
    _, again = _generate(tmp_path, plan, pairs, rows)
    assert again == [] and rows.calls == calls
    assert not list(run_dir.glob("*.tmp"))
    assert [row["plan_index"] for row in cb.read_shard(run_dir, 0)] == [0, 1, 2, 3]
    assert set(cb.read_vectors(run_dir, 0)) <= set(plan["selected_question_ids"][:4])


def test_generation_refuses_to_mix_models_or_plans(tmp_path) -> None:
    _, pairs, plan = _plan(tmp_path)
    run_dir, _ = _generate(tmp_path, plan, pairs, _Rows(), shards=[0])
    with pytest.raises(ValueError, match="mix runs"):
        cb.generate_shards(
            plan,
            pairs,
            run_dir=run_dir,
            shards=[1],
            backend=None,
            loaded=None,
            settings=None,
            rows_fn=_Rows(),
            run_meta={"model_name": "other", "model_revision": "a" * 40, "dtype": "bfloat16"},
        )


def test_queue_is_blind_and_cannot_rewrite_reviewed_rows(tmp_path) -> None:
    _, pairs, plan = _plan(tmp_path)
    run_dir, _ = _generate(tmp_path, plan, pairs, _Rows(), shards=[0])
    review_dir = tmp_path / "review"
    path, count, dropped = cb.write_queue(plan, run_dir, review_dir)
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert set(row) == {
        "source_index",
        "shard",
        "question_id",
        "group_id",
        "question",
        "answer",
        "knowledge",
    }
    text = path.read_text(encoding="utf-8").lower()
    assert "probe" not in text and "automatic" not in text and "right_answer" not in text
    assert count + len(dropped) == 4
    cb.write_queue(plan, run_dir, review_dir)
    path.write_text(
        path.read_text(encoding="utf-8").replace("Answer to", "Edited"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Refusing to change"):
        cb.write_queue(plan, run_dir, review_dir)


def _queue_for_decisions(tmp_path):
    _, pairs, plan = _plan(tmp_path)
    run_dir, _ = _generate(tmp_path, plan, pairs, _Rows())
    review_dir = tmp_path / "review"
    cb.write_queue(plan, run_dir, review_dir)
    return plan, run_dir, review_dir, cb.read_queue(review_dir)


def _decide(review_dir: Path, queue, shard: int, label="HALLUCINATED", quote_of=None):
    rows = []
    for index, item in queue.items():
        if item["shard"] != shard:
            continue
        quote = quote_of(item) if quote_of else item["knowledge"].split(" says ")[0]
        rows.append(
            [index, label, quote, "The passage states a different value than the answer.", []]
        )
    (review_dir / f"decisions-{shard:03d}.json").write_text(json.dumps(rows), encoding="utf-8")


def test_decisions_need_real_quotes_and_unique_rows(tmp_path) -> None:
    _, _, review_dir, queue = _queue_for_decisions(tmp_path)
    _decide(review_dir, queue, 0, quote_of=lambda item: "text that is not in the passage")
    with pytest.raises(ValueError, match="not in the passage"):
        cb.read_decisions(review_dir, queue)
    _decide(review_dir, queue, 0, quote_of=lambda item: "")
    with pytest.raises(ValueError, match="needs a quote"):
        cb.read_decisions(review_dir, queue)
    _decide(review_dir, queue, 0)
    assert len(cb.read_decisions(review_dir, queue)) == sum(
        1 for i in queue.values() if i["shard"] == 0
    )
    wrong = json.loads((review_dir / "decisions-000.json").read_text(encoding="utf-8"))
    (review_dir / "decisions-001.json").write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong shard|Duplicate"):
        cb.read_decisions(review_dir, queue)


def test_analysis_uses_a_reviewed_prefix_and_refuses_gaps(tmp_path) -> None:
    plan, _, review_dir, queue = _queue_for_decisions(tmp_path)
    _decide(review_dir, queue, 0)
    decisions = cb.read_decisions(review_dir, queue)
    assert cb.analysis_shards(queue, decisions, plan["shard_count"]) == [0]
    (review_dir / "decisions-000.json").unlink()
    _decide(review_dir, queue, 1)
    decisions = cb.read_decisions(review_dir, queue)
    with pytest.raises(ValueError, match="after an unreviewed shard"):
        cb.analysis_shards(queue, decisions, plan["shard_count"])


def _probe(directory: Path) -> None:
    rng = np.random.default_rng(0)
    features = np.vstack([rng.normal(-1, 1, (40, 2)), rng.normal(1, 1, (40, 2))])
    labels = np.array([0] * 40 + [1] * 40)
    probe = LogisticProbe()
    probe.fit(features, labels)
    save_probe(
        directory,
        probe,
        metadata={
            "model_name": "m",
            "model_revision": "a" * 40,
            "layer_index": 24,
            "hidden_state_count": 25,
            "position": "last_generated_token",
            "prompt_mode": "chat_template",
            "dtype": "bfloat16",
            "threshold": 0.95,
            "calibration_validated": True,
            "label_protocol": "halueval_qa_pair_v1",
        },
        mean=np.zeros(2),
        scale=np.ones(2),
    )


def _full_batch(tmp_path, per_class: int):
    root = tmp_path / "artifacts"
    pairs = _pairs(groups=60)
    total = per_class * 2 + 6
    plan = cb.build_plan(
        pairs, batch_id="b", size=total, dataset_sha256="d" * 64, artifacts=root, shard_size=total
    )

    def rows_fn(backend, prepared, group_of, settings, reviews, loaded):
        rows = _Rows()(backend, prepared, group_of, settings, reviews, loaded)
        rng = np.random.default_rng(1)
        for row in rows:
            if row["vector"] is not None:
                row["vector"] = rng.normal(0, 1, 2)
        return rows

    run_dir = root / "experiments" / "run"
    cb.generate_shards(
        plan,
        pairs,
        run_dir=run_dir,
        shards=[0],
        backend=None,
        loaded=None,
        settings=None,
        rows_fn=rows_fn,
        run_meta={"model_name": "m", "model_revision": "a" * 40, "dtype": "bfloat16"},
    )
    review_dir = root / "reviews" / "run-ai-v1"
    cb.write_queue(plan, run_dir, review_dir)
    queue = cb.read_queue(review_dir)
    labels = ["FAITHFUL"] * per_class + ["HALLUCINATED"] * per_class
    rows = []
    for position, (index, item) in enumerate(sorted(queue.items())):
        label = labels[position] if position < len(labels) else "UNVALIDATED"
        rows.append(
            [
                index,
                label,
                item["knowledge"].split(" says ")[0],
                "The passage decides this row.",
                [],
            ]
        )
    (review_dir / "decisions-000.json").write_text(json.dumps(rows), encoding="utf-8")
    _probe(tmp_path / "probe")
    return root, plan, run_dir, review_dir


def _meta(review_dir: Path, **values) -> None:
    payload = {
        "reviewer": "tester",
        "reviewer_type": "AI",
        "human_reviewed": False,
        "policy_version": cb.POLICY_VERSION,
    }
    payload.update(values)
    (review_dir / "review-meta.json").write_text(json.dumps(payload), encoding="utf-8")


def test_gate_runs_at_the_fixed_threshold_and_leaves_the_probe_alone(tmp_path) -> None:
    root, plan, run_dir, review_dir = _full_batch(tmp_path, per_class=25)
    _meta(review_dir)
    probe_dir = tmp_path / "probe"
    before = (probe_dir / "metadata.json").read_bytes(), (probe_dir / "probe.joblib").read_bytes()
    report = cb.evaluate_batch(
        plan=plan, run_dir=run_dir, review_dir=review_dir, probe_dir=probe_dir, artifacts=root
    )
    assert (probe_dir / "metadata.json").read_bytes() == before[0]
    assert (probe_dir / "probe.joblib").read_bytes() == before[1]
    assert report["threshold"] == 0.65 and report["fixed_gate_threshold"] == 0.65
    assert report["pair_probe_unchanged"] is True and report["tuned_on_this_batch"] is False
    assert report["reviewer_type"] == "AI" and report["human_reviewed"] is False
    assert report["eligible_n"] >= 40
    assert report["status"] in {"accepted", "rejected"}
    assert report["diagnostic_ece"] is not None
    assert len(report["reliability_bins"]) == 10
    assert report["shards_analyzed"] == [0]
    assert "probe score" in report["review_queue_withheld"]


def test_gate_stays_rejected_when_a_class_is_too_small(tmp_path) -> None:
    root, plan, run_dir, review_dir = _full_batch(tmp_path, per_class=12)
    _meta(review_dir)
    report = cb.evaluate_batch(
        plan=plan,
        run_dir=run_dir,
        review_dir=review_dir,
        probe_dir=tmp_path / "probe",
        artifacts=root,
    )
    assert report["status"] == "rejected"
    assert "below 20" in str(report["reason"])
    assert report["diagnostic_ece_is_gate_eligible"] is False


def test_gate_needs_the_reviewer_recorded_and_a_clean_batch(tmp_path) -> None:
    root, plan, run_dir, review_dir = _full_batch(tmp_path, per_class=25)
    args = {
        "plan": plan,
        "run_dir": run_dir,
        "review_dir": review_dir,
        "probe_dir": tmp_path / "probe",
        "artifacts": root,
    }
    with pytest.raises(ValueError, match="reviewer"):
        cb.evaluate_batch(**args)
    _meta(review_dir)
    used = plan["selected_question_ids"][0]
    _write(root / "experiments" / "later" / "predictions.json", {"rows": [{"question_id": used}]})
    with pytest.raises(ValueError, match="recorded probe or run file"):
        cb.evaluate_batch(**args)


def test_gate_refuses_a_run_from_a_different_plan(tmp_path) -> None:
    root, plan, run_dir, review_dir = _full_batch(tmp_path, per_class=25)
    _meta(review_dir)
    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["plan_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="different plan"):
        cb.evaluate_batch(
            plan=plan,
            run_dir=run_dir,
            review_dir=review_dir,
            probe_dir=tmp_path / "probe",
            artifacts=root,
        )


def test_gate_has_no_threshold_or_tuning_parameter() -> None:
    import inspect

    names = set(inspect.signature(cb.evaluate_batch).parameters)
    assert names == {"plan", "run_dir", "review_dir", "probe_dir", "artifacts"}


def test_shard_argument_parsing() -> None:
    assert cb._parse_shards(None, 3) == [0, 1, 2]
    assert cb._parse_shards("0-1,3", 5) == [0, 1, 3]
    with pytest.raises(ValueError):
        cb._parse_shards("9", 3)
