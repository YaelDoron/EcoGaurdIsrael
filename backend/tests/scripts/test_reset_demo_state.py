"""Tests for the scripts/reset_demo_state.py CLI entry point."""
from __future__ import annotations

from io import StringIO

import pytest

from scripts.reset_demo_state import main, parse_args, print_result
from src.simulation.demo_state_reset_service import DemoStateResetDisabledError, DemoStateResetResult


def test_parse_args_confirm_flag_defaults_false():
    args = parse_args([])
    assert args.confirm is False


def test_parse_args_confirm_flag_can_be_set():
    args = parse_args(["--confirm"])
    assert args.confirm is True


def test_main_refuses_without_confirm(monkeypatch, capsys):
    monkeypatch.setattr("scripts.reset_demo_state.initialize_database", lambda: None)

    exit_code = main([])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Refused" in captured.err
    assert "--confirm" in captured.err


def test_main_reports_database_configuration_error(monkeypatch, capsys):
    from src.database.connection import DatabaseConfigurationError

    def _raise():
        raise DatabaseConfigurationError("DATABASE_URL is not configured.")

    monkeypatch.setattr("scripts.reset_demo_state.initialize_database", _raise)

    exit_code = main(["--confirm"])

    assert exit_code == 2
    assert "Database configuration error" in capsys.readouterr().err


def test_main_refuses_when_service_reports_disabled(monkeypatch, capsys):
    monkeypatch.setattr("scripts.reset_demo_state.initialize_database", lambda: None)

    class _DisabledService:
        def reset_demo_state(self):
            raise DemoStateResetDisabledError("ENABLE_DEMO_DATA_RESET is not enabled.")

    monkeypatch.setattr("scripts.reset_demo_state.DemoStateResetService", lambda: _DisabledService())

    exit_code = main(["--confirm"])

    assert exit_code == 2
    assert "Refused" in capsys.readouterr().err


def test_main_exits_nonzero_when_reset_raises_unexpectedly(monkeypatch, capsys):
    monkeypatch.setattr("scripts.reset_demo_state.initialize_database", lambda: None)

    class _FailingService:
        def reset_demo_state(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("scripts.reset_demo_state.DemoStateResetService", lambda: _FailingService())

    exit_code = main(["--confirm"])

    assert exit_code == 1
    assert "Demo state reset failed" in capsys.readouterr().err


def test_main_succeeds_and_reports_counts(monkeypatch, capsys):
    monkeypatch.setattr("scripts.reset_demo_state.initialize_database", lambda: None)

    result = DemoStateResetResult(deleted_counts={"fire_events": 2, "satellite_hotspots": 6}, resources_restored=12)

    class _OkService:
        def reset_demo_state(self):
            return result

    monkeypatch.setattr("scripts.reset_demo_state.DemoStateResetService", lambda: _OkService())

    exit_code = main(["--confirm"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Road network cache: preserved" in output
    assert "fire_events: 2" in output
    assert "satellite_hotspots: 6" in output
    assert "Resources restored to AVAILABLE: 12" in output
    assert "Demo operational state is clean." in output
    assert "DATABASE_URL" not in output  # never prints credentials


def test_print_result_never_prints_credentials():
    output = StringIO()
    result = DemoStateResetResult(deleted_counts={"fire_events": 1}, resources_restored=5)

    print_result(result, output)

    text = output.getvalue()
    assert "postgres" not in text.lower()
    assert "@" not in text
