"""Modes that must not load weights or invent a probe."""

from glassbox.cli import main
from glassbox.demo_config import load_demo_config
from glassbox.present import EXTERNAL_PROBE_NOTE


def test_samples_do_not_construct_a_backend(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("model backend was constructed")

    monkeypatch.setattr("glassbox.cli.HuggingFaceBackend", forbidden)
    assert main(["--samples"]) == 0
    output = capsys.readouterr().out
    assert "What is the capital of Australia?" in output
    assert "Why is Sydney the capital of Australia?" in output


def test_analyze_text_does_not_probe_hidden_states(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("model backend was constructed")

    monkeypatch.setattr("glassbox.cli.HuggingFaceBackend", forbidden)
    code = main(
        [
            "--analyze-text",
            "--question",
            "What is the capital of Australia?",
            "--answer",
            "Canberra is the capital of Australia.",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "CORRECT" in output
    assert "Hidden-state probing cannot be performed on this text alone." in output
    assert EXTERNAL_PROBE_NOTE.split(".")[0] in output
    assert "PROBE NOT TRAINED" in output


def test_misleading_sample_can_fail_the_reference_check(monkeypatch, capsys):
    monkeypatch.setattr("glassbox.cli.HuggingFaceBackend", lambda *_a, **_k: None)
    code = main(
        [
            "--analyze-text",
            "--question",
            "Why is Sydney the capital of Australia?",
            "--answer",
            "Sydney is the capital of Australia.",
        ]
    )
    assert code == 0
    assert "FALSE CLAIM ASSERTED" in capsys.readouterr().out


def test_offline_config_refuses_a_download():
    config = load_demo_config(offline=True, model_name="glassbox/not-a-real-model")
    assert config.spec.local_files_only is True
    assert config.spec.allow_download is False
