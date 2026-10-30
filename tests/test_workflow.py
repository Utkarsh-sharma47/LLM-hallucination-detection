"""Interactive questions, narrow reports, and session export. No model download."""

import json

from glassbox.cli import main, render_command
from glassbox.config import repo_root
from glassbox.models.backend import GeneratedToken, GenerationResult, GenerationSettings
from glassbox.models.distributions import TokenAlternative
from glassbox.present import render_turn_text
from glassbox.probes.logistic import PROBE_NOT_TRAINED
from glassbox.session import DemoSession
from glassbox.verification import ReferenceResult, load_curated_facts


def _generation(
    question: str,
    token_text: str = "Paris",
    alternative: str = "Lyon",
) -> GenerationResult:
    token = GeneratedToken(
        token_id=3,
        text=token_text,
        probability=0.7,
        top_alternatives=(
            TokenAlternative(token_id=3, text=token_text, probability=0.7),
            TokenAlternative(token_id=4, text=alternative, probability=0.2),
        ),
        is_argmax=True,
        normalized_entropy=0.2,
        margin=0.5,
    )
    return GenerationResult(
        question=question,
        prompt_text=f"Question: {question}\nAnswer:",
        prompt_mode="chat_template",
        answer="A generated answer.",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        model_revision="cached",
        device="cpu",
        dtype="float32",
        latency_seconds=0.4,
        prompt_tokens=8,
        generated_tokens=1,
        tokens=(token,),
        hidden_states_available=True,
        logits_available=True,
        feature=None,
        feature_layer=24,
        hidden_state_count=25,
        position="last_prompt_token",
        layer_vectors={},
        load_count=1,
    )


class _Backend:
    def __init__(self) -> None:
        self.load_count = 0
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.device = "cpu"
        self.questions: list[str] = []

    def ensure_loaded(self):
        self.load_count += 1
        return self

    def generate(self, question, settings, on_token=None):
        del settings, on_token
        self.questions.append(question)
        return _generation(question)


def _install_backend(monkeypatch) -> _Backend:
    backend = _Backend()
    monkeypatch.setattr("glassbox.cli.HuggingFaceBackend", lambda _spec: backend)
    monkeypatch.setattr(
        "glassbox.cli.make_console",
        lambda: __import__("rich.console", fromlist=["Console"]).Console(
            force_terminal=False, highlight=False, width=80
        ),
    )
    return backend


def test_default_mode_accepts_an_arbitrary_question_and_keeps_the_model(monkeypatch, capsys):
    backend = _install_backend(monkeypatch)
    answers = iter(["Who wrote The Odyssey?", "n", "What is 2 plus 2?", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert main([]) == 0
    output = capsys.readouterr().out
    assert backend.questions == ["Who wrote The Odyssey?", "What is 2 plus 2?"]
    assert backend.load_count == 1
    assert "UNVALIDATED" in output or "NO REFERENCE" in output
    assert "PROBE NOT TRAINED" in output or "NOT AVAILABLE" in output
    assert "What is the capital of Australia?" not in backend.questions


def test_menu_legend_matches_the_command_handlers(monkeypatch, capsys):
    _install_backend(monkeypatch)
    answers = iter(["Who wrote The Odyssey?", "t", "p", "v", "d", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert main([]) == 0
    output = capsys.readouterr().out
    for line in (
        "n = ask another question",
        "r = show full report",
        "t = token probability analysis",
        "p = probe analysis",
        "v = reference verification",
        "d = diagnostics",
        "e = export report",
        "q = quit",
    ):
        assert line in output


def test_once_answers_one_question_without_a_menu(monkeypatch, capsys):
    backend = _install_backend(monkeypatch)

    def forbidden(_prompt=""):
        raise AssertionError("interactive prompt")

    monkeypatch.setattr("builtins.input", forbidden)
    assert main(["--once", "--question", "Who wrote The Odyssey?"]) == 0
    output = capsys.readouterr().out
    assert backend.questions == ["Who wrote The Odyssey?"]
    assert "UNVALIDATED" in output or "NO REFERENCE" in output
    assert "PROBE NOT TRAINED" in output


def test_narrow_report_truncates_long_alternatives_and_stays_within_width():
    turn_generation = _generation("A free question?", token_text="  \n", alternative="B" * 180)
    from glassbox.reporting import TurnRecord

    turn = TurnRecord(
        generation=turn_generation,
        probe_status=PROBE_NOT_TRAINED,
        experimental_score=None,
        score_note="No trained probe is loaded.",
        factual_risk_score=None,
        reference=ReferenceResult(
            configured=False,
            status="not_configured",
            detail="NO REFERENCE AVAILABLE. Outside the curated list.",
        ),
    )
    text = render_turn_text(turn, width=48)
    assert "B" * 80 not in text
    assert "·" in text
    assert "⏎" in text
    assert "NO REFERENCE AVAILABLE" in text
    assert "RISK NOT ASSESSED" in text
    for line in text.splitlines():
        assert len(line) <= 48


def test_session_export_writes_json_and_markdown(tmp_path):
    facts = load_curated_facts(repo_root() / "configs" / "curated_facts.yaml")
    session = DemoSession(backend=_Backend(), settings=GenerationSettings(), facts=facts)
    session.ask("Who wrote The Odyssey?")
    json_path = session.export(tmp_path)
    markdown_path = json_path.with_suffix(".md")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    report = markdown_path.read_text(encoding="utf-8")
    assert payload["turns"][0]["question"] == "Who wrote The Odyssey?"
    assert payload["turns"][0]["answer"] == "A generated answer."
    assert payload["turns"][0]["output_steps"][0]["probability"] == 0.7
    table = (json_path.parent / "distributions-topk.csv").read_text(encoding="utf-8")
    assert "piece_id" in table
    assert "0.7" in table
    assert "NO REFERENCE AVAILABLE" in report
    assert "PROBE NOT TRAINED" in report
    assert "Who wrote The Odyssey?" in report
    message = render_command(session, "e", root=tmp_path)
    assert message is not None
    assert "JSON report:" in message
    assert "Markdown report:" in message
    assert "less" in message
