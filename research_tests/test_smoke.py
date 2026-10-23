"""Smoke-test coverage."""

from __future__ import annotations

from glassbox.smoke import SMOKE_OK, main, run_smoke_checks


def test_smoke_checks_pass_offline() -> None:
    config = run_smoke_checks()
    assert config.phase == "scaffold"
    assert config.model.allow_download is False
    assert config.data.allow_download is False


def test_smoke_entry_point_prints_ok(capsys) -> None:
    assert main() == 0
    output = capsys.readouterr().out
    assert output.startswith(SMOKE_OK)
    assert "downloads=disabled" in output
    assert "Qwen/Qwen2.5-3B-Instruct" in output
