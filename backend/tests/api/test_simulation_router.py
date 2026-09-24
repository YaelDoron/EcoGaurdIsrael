"""API tests for the Simulation Control endpoints (Task A3, Part 30).

Uses FastAPI's dependency_overrides to replace SimulationRunManager with a
fake exposing only get_current_snapshot()/start_run() - no real
DemoSimulationRunner, ThreadPoolExecutor, or DB access happens in these
tests. ENABLE_SIMULATION_CONTROL_API is toggled via monkeypatch, replacing
the whole frozen `settings` singleton object (see
tests/services/simulation_control/test_simulation_run_manager.py for why a
direct `sys.modules` lookup - not `import src.config.settings as x` - is
required to reliably reach the real settings *submodule* object here:
`src/config/__init__.py` does `from src.config.settings import settings`,
which shadows the package's own `settings` attribute with the Settings
*instance*, and Python's `import a.b.c as x` resolves `x` via that same
attribute chain).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_simulation_run_manager
from src.services.simulation_control.simulation_run_manager import (
    SimulationAlreadyRunningError,
    SimulationCurrentEventSnapshot,
    SimulationPresetNotFoundError,
    SimulationResetDisabledError,
    SimulationRunError,
    SimulationRunState,
    _idle_snapshot,
)
from src.simulation.simulation_presets import SIMULATION_PRESETS

_settings_module = sys.modules["src.config.settings"]

PRESETS_ENDPOINT = "/api/v1/simulation/presets"
CURRENT_ENDPOINT = "/api/v1/simulation/runs/current"
START_ENDPOINT = "/api/v1/simulation/runs"

STARTED_AT = datetime(2026, 9, 19, 8, 0, 0, tzinfo=timezone.utc)


def enable_control(monkeypatch, *, enable_reset: bool = False) -> None:
    monkeypatch.setattr(
        _settings_module,
        "settings",
        SimpleNamespace(ENABLE_SIMULATION_CONTROL_API=True, ENABLE_DEMO_DATA_RESET=enable_reset),
    )


def disable_control(monkeypatch) -> None:
    monkeypatch.setattr(
        _settings_module,
        "settings",
        SimpleNamespace(ENABLE_SIMULATION_CONTROL_API=False, ENABLE_DEMO_DATA_RESET=False),
    )


class FakeSimulationRunManager:
    def __init__(self, snapshot=None, start_result=None, start_exception: Exception | None = None):
        self._snapshot = snapshot if snapshot is not None else _idle_snapshot()
        self._start_result = start_result
        self._start_exception = start_exception
        self.start_calls: list[dict] = []

    def get_current_snapshot(self):
        return self._snapshot

    def start_run(self, *, preset_id, seed, reset_demo_state):
        self.start_calls.append(
            {"preset_id": preset_id, "seed": seed, "reset_demo_state": reset_demo_state}
        )
        if self._start_exception is not None:
            raise self._start_exception
        return self._start_result


def client_for(manager, *, raise_server_exceptions: bool = True) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_simulation_run_manager] = lambda: manager
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def make_preparing_snapshot(**overrides) -> object:
    from dataclasses import replace

    base = replace(
        _idle_snapshot(),
        run_id="run-1",
        state=SimulationRunState.PREPARING,
        preset_id="operations_demo",
        seed=42,
        last_message="Preparing simulation run.",
    )
    return replace(base, **overrides) if overrides else base


def make_running_snapshot() -> object:
    from dataclasses import replace

    return replace(
        make_preparing_snapshot(),
        state=SimulationRunState.RUNNING,
        simulation_duration_seconds=120,
        events_total=18,
        events_completed=7,
        events_succeeded=7,
        events_failed=0,
        current_event=SimulationCurrentEventSnapshot(
            event_index=7,
            incident_id="incident-carmel-01",
            event_type="weather",
            timestamp_offset_sec=60,
        ),
        started_at=STARTED_AT,
        last_message="Started weather for incident-carmel-01.",
    )


def make_failed_snapshot() -> object:
    from dataclasses import replace

    return replace(
        make_preparing_snapshot(),
        state=SimulationRunState.FAILED,
        completed_at=STARTED_AT,
        last_message="Simulation run failed.",
        error=SimulationRunError(code="SIMULATION_RUN_FAILED", message="Simulation run failed."),
    )


# ---------------------------------------------------------------------------
# Feature flag gating
# ---------------------------------------------------------------------------


def test_presets_disabled_returns_403_envelope(monkeypatch):
    disable_control(monkeypatch)
    client = client_for(FakeSimulationRunManager())

    response = client.get(PRESETS_ENDPOINT)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "SIMULATION_CONTROL_DISABLED"


def test_current_disabled_returns_403_envelope(monkeypatch):
    disable_control(monkeypatch)
    client = client_for(FakeSimulationRunManager())

    response = client.get(CURRENT_ENDPOINT)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "SIMULATION_CONTROL_DISABLED"


def test_start_disabled_returns_403_and_never_calls_manager(monkeypatch):
    disable_control(monkeypatch)
    manager = FakeSimulationRunManager()
    client = client_for(manager)

    response = client.post(START_ENDPOINT, json={"preset": "operations_demo", "seed": 1})

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "SIMULATION_CONTROL_DISABLED"
    assert manager.start_calls == []


# ---------------------------------------------------------------------------
# GET /presets
# ---------------------------------------------------------------------------


def test_presets_enabled_lists_the_real_registry(monkeypatch):
    enable_control(monkeypatch)
    client = client_for(FakeSimulationRunManager())

    response = client.get(PRESETS_ENDPOINT)

    assert response.status_code == 200
    body = response.json()
    assert {preset["id"] for preset in body["presets"]} == {preset.id for preset in SIMULATION_PRESETS}
    for preset in body["presets"]:
        assert isinstance(preset["simulation_duration_seconds"], int)
        assert set(preset.keys()) == {"id", "display_name", "simulation_duration_seconds"}


# ---------------------------------------------------------------------------
# GET /runs/current
# ---------------------------------------------------------------------------


def test_current_idle_returns_200_not_404(monkeypatch):
    enable_control(monkeypatch)
    client = client_for(FakeSimulationRunManager(snapshot=_idle_snapshot()))

    response = client.get(CURRENT_ENDPOINT)

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "idle"
    assert body["run_id"] is None


def test_current_running_reflects_progress_and_current_event(monkeypatch):
    enable_control(monkeypatch)
    client = client_for(FakeSimulationRunManager(snapshot=make_running_snapshot()))

    body = client.get(CURRENT_ENDPOINT).json()

    assert body["state"] == "running"
    assert body["events_completed"] == 7
    assert body["events_total"] == 18
    assert body["current_event"] == {
        "event_index": 7,
        "incident_id": "incident-carmel-01",
        "event_type": "weather",
        "timestamp_offset_sec": 60,
    }


def test_current_failed_reflects_sanitized_error(monkeypatch):
    enable_control(monkeypatch)
    client = client_for(FakeSimulationRunManager(snapshot=make_failed_snapshot()))

    body = client.get(CURRENT_ENDPOINT).json()

    assert body["state"] == "failed"
    assert body["error"] == {"code": "SIMULATION_RUN_FAILED", "message": "Simulation run failed."}


def test_current_response_contains_only_defined_dto_fields(monkeypatch):
    enable_control(monkeypatch)
    client = client_for(FakeSimulationRunManager(snapshot=make_running_snapshot()))

    body = client.get(CURRENT_ENDPOINT).json()

    assert set(body.keys()) == {
        "run_id",
        "state",
        "preset_id",
        "seed",
        "mode",
        "simulation_duration_seconds",
        "events_total",
        "events_completed",
        "events_succeeded",
        "events_failed",
        "current_event",
        "started_at",
        "completed_at",
        "wall_clock_elapsed_seconds",
        "last_message",
        "error",
    }


# ---------------------------------------------------------------------------
# POST /runs - happy path
# ---------------------------------------------------------------------------


def test_start_happy_path_returns_202_with_preparing_snapshot(monkeypatch):
    enable_control(monkeypatch)
    manager = FakeSimulationRunManager(start_result=make_preparing_snapshot())
    client = client_for(manager)

    response = client.post(START_ENDPOINT, json={"preset": "operations_demo", "seed": 42})

    assert response.status_code == 202
    body = response.json()
    assert body["state"] == "preparing"
    assert body["run_id"] == "run-1"
    assert manager.start_calls == [{"preset_id": "operations_demo", "seed": 42, "reset_demo_state": True}]


def test_start_accepts_a_request_with_seed_omitted_entirely(monkeypatch):
    """seed is optional (Part K/S) - the router forwards whatever the
    request carries (including None) straight to the manager, which is the
    one place auto-generation actually happens."""
    enable_control(monkeypatch)
    manager = FakeSimulationRunManager(start_result=make_preparing_snapshot())
    client = client_for(manager)

    response = client.post(START_ENDPOINT, json={"preset": "operations_demo", "reset_demo_state": True})

    assert response.status_code == 202
    assert manager.start_calls == [{"preset_id": "operations_demo", "seed": None, "reset_demo_state": True}]


def test_start_still_uses_an_explicit_seed_exactly_as_given(monkeypatch):
    enable_control(monkeypatch)
    manager = FakeSimulationRunManager(start_result=make_preparing_snapshot())
    client = client_for(manager)

    client.post(START_ENDPOINT, json={"preset": "operations_demo", "seed": 42})

    assert manager.start_calls[0]["seed"] == 42


def test_start_defaults_reset_demo_state_to_true(monkeypatch):
    """Task 9A: the reset is mandatory, so the request field defaults to TRUE (it used to default to false)."""
    enable_control(monkeypatch)
    manager = FakeSimulationRunManager(start_result=make_preparing_snapshot())
    client = client_for(manager)

    client.post(START_ENDPOINT, json={"preset": "operations_demo", "seed": 42})

    assert manager.start_calls[0]["reset_demo_state"] is True


def test_start_refuses_an_explicit_opt_out_of_the_reset_with_422_and_never_reaches_the_manager(monkeypatch):
    enable_control(monkeypatch)
    manager = FakeSimulationRunManager(start_result=make_preparing_snapshot())
    client = client_for(manager)

    response = client.post(START_ENDPOINT, json={"preset": "operations_demo", "seed": 42, "reset_demo_state": False})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "SIMULATION_RESET_REQUIRED"
    assert manager.start_calls == []


def test_start_forwards_reset_demo_state_true(monkeypatch):
    enable_control(monkeypatch, enable_reset=True)
    manager = FakeSimulationRunManager(start_result=make_preparing_snapshot())
    client = client_for(manager)

    client.post(
        START_ENDPOINT, json={"preset": "operations_demo", "seed": 42, "reset_demo_state": True}
    )

    assert manager.start_calls[0]["reset_demo_state"] is True


# ---------------------------------------------------------------------------
# POST /runs - error mapping
# ---------------------------------------------------------------------------


def test_start_already_running_returns_409(monkeypatch):
    enable_control(monkeypatch)
    manager = FakeSimulationRunManager(
        start_exception=SimulationAlreadyRunningError("A simulation run is already running.")
    )
    client = client_for(manager)

    response = client.post(START_ENDPOINT, json={"preset": "operations_demo", "seed": 1})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "SIMULATION_ALREADY_RUNNING"


def test_start_unknown_preset_returns_404(monkeypatch):
    enable_control(monkeypatch)
    manager = FakeSimulationRunManager(start_exception=SimulationPresetNotFoundError("bogus"))
    client = client_for(manager)

    response = client.post(START_ENDPOINT, json={"preset": "bogus", "seed": 1})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SIMULATION_PRESET_NOT_FOUND"


def test_start_reset_disabled_returns_403(monkeypatch):
    enable_control(monkeypatch)
    manager = FakeSimulationRunManager(
        start_exception=SimulationResetDisabledError(
            "reset_demo_state was requested but ENABLE_DEMO_DATA_RESET is not enabled."
        )
    )
    client = client_for(manager)

    response = client.post(
        START_ENDPOINT, json={"preset": "operations_demo", "seed": 1, "reset_demo_state": True}
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "SIMULATION_RESET_DISABLED"


def test_start_unexpected_failure_returns_sanitized_500(monkeypatch):
    enable_control(monkeypatch)
    manager = FakeSimulationRunManager(
        start_exception=RuntimeError("DATABASE_URL=postgresql://secret leaked traceback")
    )
    client = client_for(manager, raise_server_exceptions=False)

    response = client.post(START_ENDPOINT, json={"preset": "operations_demo", "seed": 1})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "SIMULATION_START_FAILED"
    body_text = response.text
    for leaked in ("DATABASE_URL", "postgresql", "Traceback", "secret"):
        assert leaked not in body_text


def test_start_missing_required_fields_returns_422(monkeypatch):
    enable_control(monkeypatch)
    client = client_for(FakeSimulationRunManager())

    response = client.post(START_ENDPOINT, json={"seed": 1})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# OpenAPI registration
# ---------------------------------------------------------------------------


def test_all_three_endpoints_are_registered_in_openapi_schema():
    app = create_app()

    schema = app.openapi()

    assert PRESETS_ENDPOINT in schema["paths"]
    assert CURRENT_ENDPOINT in schema["paths"]
    assert START_ENDPOINT in schema["paths"]
    post_op = schema["paths"][START_ENDPOINT]["post"]
    assert "202" in post_op["responses"]
