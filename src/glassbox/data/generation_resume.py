"""Resume generation of the missing v2 answers.

Cached generation-001 rows are not regenerated and that directory is not
rewritten. Each accepted answer is appended before the next question is
asked. An empty answer, a revision mismatch, or a missing hidden state is
a failure record, not a labeled example. This module does not train a probe
and does not choose a threshold.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from glassbox.data.generation_expand import HISTORICAL_GENERATION_RUN
from glassbox.probes.generation_eval import PROTECTED_PROBE_NAMES

ANSWERS_NAME = "answers.jsonl"
FAILURES_NAME = "failures.jsonl"
MANIFEST_NAME = "manifest.json"
ALLOWED_FILES = frozenset(
    {
        ANSWERS_NAME,
        FAILURES_NAME,
        MANIFEST_NAME,
        "predictions.json",
        "vectors",
        "integrity.json",
        "review_queue.json",
    }
)


def refuse_run_id(run_id: str) -> None:
    """Refuse the historical generation run and the live pair-probe names."""
    if run_id == HISTORICAL_GENERATION_RUN or run_id.startswith("final-hidden-state"):
        raise ValueError(f"Refusing protected run id {run_id}")
    if run_id in PROTECTED_PROBE_NAMES:
        raise ValueError(f"Refusing protected run id {run_id}")


def pending_question_ids(
    missing_ids: Sequence[str],
    *,
    cached_ids: set[str],
    completed_ids: set[str],
) -> list[str]:
    """Return missing plan ids that are neither cached nor already saved."""
    pending = []
    for question_id in missing_ids:
        if question_id in cached_ids or question_id in completed_ids:
            continue
        pending.append(question_id)
    return pending


def rejection_reason(
    *,
    answer: str,
    model_revision: str,
    expected_revision: str,
    generated_tokens: int,
    feature: np.ndarray | None,
) -> str | None:
    """Return why this generation cannot be stored, or None when it can."""
    if not str(answer).strip():
        return "empty answer"
    if model_revision != expected_revision:
        return f"revision {model_revision} does not match {expected_revision}"
    if generated_tokens < 1 or feature is None:
        return "hidden state or generated piece missing"
    vector = np.asarray(feature)
    if vector.ndim != 1 or vector.size < 1 or not np.isfinite(vector).all():
        return "hidden state is not a finite vector"
    return None


def load_completed_ids(path: Path) -> set[str]:
    """Read question ids already appended.

    A crash can leave one incomplete object on the last line. That line is
    ignored. A broken line anywhere earlier is an error.
    """
    if not path.is_file():
        return set()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    found: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == len(lines):
                continue
            raise ValueError(f"{path} line {line_number} is not JSON") from exc
        question_id = payload.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{path} line {line_number} has no question_id")
        if question_id in found:
            raise ValueError(f"{path} repeats question_id {question_id}")
        found.add(question_id)
    return found


def append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    """Append one object and flush it. Existing lines are left in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(payload), sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def assert_resume_directory(directory: Path) -> None:
    """Allow a resume directory that contains only this command's files."""
    if not directory.exists():
        return
    extra = sorted(path.name for path in directory.iterdir() if path.name not in ALLOWED_FILES)
    if extra:
        raise FileExistsError(
            f"{directory} contains {extra[0]}, which this resume command does not own"
        )


def manifest_matches(path: Path, expected: Mapping[str, object]) -> None:
    """Refuse to resume when the saved plan identity differs."""
    if not path.is_file():
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    for key in ("dataset_sha256", "seed", "max_questions", "model_revision"):
        if saved.get(key) != expected.get(key):
            raise ValueError(
                f"Resume manifest {key} is {saved.get(key)!r}, expected {expected.get(key)!r}"
            )


def generate_missing_command(args: object) -> int:
    """Generate plan questions that generation-001 does not already contain."""
    import sys

    from glassbox.config import repo_root
    from glassbox.data.evidence import label_halueval_generation
    from glassbox.data.generation_expand import (
        cached_answers_for_split,
        load_generation_rows,
        plan_expansion,
    )
    from glassbox.data.generation_v2 import _assignment_from_plan
    from glassbox.data.halueval import (
        PINNED_SHA256,
        prepare_pairs,
        read_pinned_snapshot,
        snapshot_sha256,
    )
    from glassbox.demo_config import load_demo_config
    from glassbox.models.backend import GenerationSettings, HuggingFaceBackend
    from glassbox.probes.generation_eval import halueval_snapshot_path
    from glassbox.probes.training import REGISTERED_POSITION
    from glassbox.reproducibility import require_commit_hash

    try:
        if not args.offline:
            raise ValueError("generate-v2-answers requires --offline and does not download weights")
        run_id = str(args.run_id)
        refuse_run_id(run_id)
        artifacts = Path(args.artifacts_root) if args.artifacts_root else repo_root() / "artifacts"
        directory = (artifacts / "experiments" / run_id).resolve()
        root = artifacts.resolve()
        if not directory.is_relative_to(root):
            raise ValueError(f"Refusing to write outside {root}")
        historical = (root / "experiments" / HISTORICAL_GENERATION_RUN).resolve()
        if directory == historical:
            raise ValueError("Refusing to write into the historical generation-001 directory")
        assert_resume_directory(directory)
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = Path(args.snapshot) if args.snapshot else halueval_snapshot_path()
        if not snapshot.is_file():
            raise FileNotFoundError(f"HaluEval snapshot is not at {snapshot}")
        digest = snapshot_sha256(snapshot)
        if digest != PINNED_SHA256:
            raise ValueError(
                f"HaluEval snapshot hash {digest} does not match pinned {PINNED_SHA256}"
            )
        cached_path = (
            Path(args.cached)
            if args.cached
            else root / "experiments" / HISTORICAL_GENERATION_RUN / "predictions.json"
        )
        if not cached_path.is_file():
            raise FileNotFoundError(f"Cached generation-001 predictions are not at {cached_path}")
        cached_rows = load_generation_rows(cached_path)
        prepared = prepare_pairs(read_pinned_snapshot(snapshot), seed=int(args.seed))
        plan = plan_expansion(
            prepared.pairs.values(),
            cached_rows,
            seed=int(args.seed),
            max_questions=int(args.max_questions),
        )
        selected = _assignment_from_plan(plan)
        _reused, missing = cached_answers_for_split(selected, cached_rows)
        config = load_demo_config(args.config, offline=True, model_name=args.model)
        expected_revision = config.spec.revision
        require_commit_hash(expected_revision, field="model.revision")
        identity = {
            "dataset_sha256": digest,
            "seed": int(args.seed),
            "max_questions": int(args.max_questions),
            "model_revision": expected_revision,
            "historical_run": HISTORICAL_GENERATION_RUN,
            "rewrites_historical_run": False,
        }
        manifest_path = directory / MANIFEST_NAME
        manifest_matches(manifest_path, identity)
        if not manifest_path.is_file():
            manifest_path.write_text(
                json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        answers_path = directory / ANSWERS_NAME
        failures_path = directory / FAILURES_NAME
        completed = load_completed_ids(answers_path)
        failed_ids = load_completed_ids(failures_path)
        cached_ids = {str(row["question_id"]) for row in cached_rows}
        if completed & cached_ids:
            raise ValueError("A saved answer repeats a cached generation-001 question")
        pending = pending_question_ids(
            missing,
            cached_ids=cached_ids,
            completed_ids=completed | failed_ids,
        )
        limit = int(args.limit) if args.limit else len(pending)
        pending = pending[:limit]
        if not pending:
            print(
                f"No pending questions. Saved answers: {len(completed)}. "
                f"Recorded failures: {len(failed_ids)}."
            )
            return 0
        print(
            f"Pending {len(pending)} of {len(missing)} missing questions. "
            f"Cached generation-001 ids: {len(cached_ids)}. "
            f"Already saved: {len(completed)}. Recorded failures: {len(failed_ids)}.",
            flush=True,
        )
        settings = GenerationSettings(
            max_new_tokens=config.generation.max_new_tokens,
            temperature=0.0,
            top_k=config.generation.top_k,
            layer_index=config.generation.layer_index,
            position=REGISTERED_POSITION,
            capture_all_layers=False,
        )
        backend = HuggingFaceBackend(config.spec)
        print(f"Loading {config.spec.name} once for the missing answers...", flush=True)
        loaded = backend.ensure_loaded()
        if loaded.resolved_revision != expected_revision:
            raise RuntimeError(
                f"Loaded revision {loaded.resolved_revision} does not match {expected_revision}"
            )
        if any(parameter.requires_grad for parameter in loaded.model.parameters()):
            raise RuntimeError("Qwen parameters were not frozen")
        split_of = {
            question_id: split
            for split in ("train", "validation", "test")
            for question_id in selected.ids(split)
        }
        vector_dir = directory / "vectors"
        vector_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        failed = 0
        for index, question_id in enumerate(pending, start=1):
            if "/" in question_id or question_id.startswith("."):
                raise ValueError(f"Refusing question id {question_id!r}")
            pair = prepared.pairs[question_id]
            result = backend.generate(pair.question, settings)
            feature = (
                None if result.feature is None else np.asarray(result.feature, dtype=np.float64)
            )
            reason = rejection_reason(
                answer=result.answer,
                model_revision=str(result.model_revision),
                expected_revision=expected_revision,
                generated_tokens=int(result.generated_tokens),
                feature=feature,
            )
            if reason is not None:
                append_jsonl(
                    failures_path,
                    {
                        "question_id": question_id,
                        "reason": reason,
                        "answer": result.answer,
                        "model_revision": result.model_revision,
                        "split": split_of[question_id],
                    },
                )
                failed += 1
                print(f"Failed {index}/{len(pending)} {question_id}: {reason}", flush=True)
                continue
            decision = label_halueval_generation(
                question=pair.question,
                knowledge=pair.knowledge,
                right_answer=pair.right_answer,
                hallucinated_answer=pair.hallucinated_answer,
                answer=result.answer,
            )
            vector_path = vector_dir / f"{question_id}.npy"
            if vector_path.exists():
                # The JSONL row is the accepted answer. A vector without that
                # row was left by an interrupted write and is not an answer.
                vector_path.unlink()
            np.save(vector_path, feature)
            append_jsonl(
                answers_path,
                {
                    "question_id": question_id,
                    "split": split_of[question_id],
                    "question": pair.question,
                    "answer": result.answer,
                    "knowledge": pair.knowledge,
                    "right_answer": pair.right_answer,
                    "hallucinated_answer": pair.hallucinated_answer,
                    "label": int(decision.label.value),
                    "label_name": decision.label.name,
                    "model_name": result.model_name,
                    "model_revision": result.model_revision,
                    "dtype": result.dtype,
                    "position": result.position,
                    "feature_layer": result.feature_layer,
                    "feature_dim": int(feature.shape[0]),
                    "generated_piece_count": int(result.generated_tokens),
                    "prompt_piece_count": int(result.prompt_tokens),
                    "source": "generated_v2",
                    "label_origin": "halueval_generation_evidence_v1",
                    "human_confirmed": False,
                    "vector_file": str(vector_path.name),
                    "calibration_validated": False,
                    "seed": int(args.seed),
                    "temperature": 0.0,
                    "answer_piece_limit": int(config.generation.max_new_tokens),
                    "prompt_mode": result.prompt_mode,
                },
            )
            saved += 1
            if index % 10 == 0 or index == len(pending):
                print(
                    f"Generated {index}/{len(pending)} {question_id} {decision.label.name}",
                    flush=True,
                )
        print(
            f"Saved {saved} new answers, {failed} failures, "
            f"{len(completed) + saved} total in {answers_path}"
        )
        return 0
    except (OSError, ValueError, RuntimeError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"generate-v2-answers stopped: {exc}", file=sys.stderr)
        return 1
