"""Task A3, Part 31: one integration-style test wiring the real HTTP app,
the real `SimulationRunManager`, and a real background `ThreadPoolExecutor`
together end-to-end - proving POST returns immediately while the run
executes on another thread, and that GET /runs/current observes the
transition through to COMPLETED.

Uses a fake, near-instantaneous `DemoSimulationRunner` (never the real
operations_demo stack) so this test needs no Neon database and completes in
well under a second - only the polling loop below waits, and it is bounded
by a short timeout, not a fixed sleep. No `pytest.mark.integration` (see
pytest.ini: that marker specifically means "requires a live Neon/PostgreSQL
database", which this test does not).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace
import sys

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_simulation_run_manager
from src.services.simulation_control.simulation_run_manager import SimulationRunManager
from src.simulation.demo_simulation_runner import (
    DemoSimulationEventPhase,
    DemoSimulationEventProgress,
    DemoSimulationRunResult,
    DemoSimulationStatus,
)
from src.simulation.simulation_event import SimulationEventType
from src.simulation.simulation_scenario import build_operations_demo_scenario

_settings_module = sys.modules["src.config.settings"]

NOW = datetime(2026, 9, 19, 9, 0, 0, tzinfo=timezone.utc)


class InstantFakeRunner:
    """Emits one EVENT_STARTED/EVENT_COMPLETED pair then returns immediately -
    a stand-in for DemoSimulationRunner that performs no real analysis work,
    no DB access, and takes no real time."""

    def run(self, scenario, config, scenario_started_at=None, on_progress=None):
        if on_progress is not None:
            on_progress(
                DemoSimulationEventProgress(
                    phase=DemoSimulationEventPhase.EVENT_STARTED,
                    event_index=0,
                    events_total=1,
                    events_completed=0,
                    incident_id="incident-carmel-01",
                    event_type=SimulationEventType.WEATHER,
                    timestamp_offset_sec=0,
                    started_at=NOW,
                    now=NOW,
                )
            )
            on_progress(
                DemoSimulationEventProgress(
                    phase=DemoSimulationEventPhase.EVENT_COMPLETED,
                    event_index=0,
                    events_total=1,
                    events_completed=1,
                    incident_id="incident-carmel-01",
                    event_type=SimulationEventType.WEATHER,
                    timestamp_offset_sec=0,
                    started_at=NOW,
                    now=NOW,
                    success=True,
                )
            )
        return DemoSimulationRunResult(
            status=DemoSimulationStatus.COMPLETED,
            simulation_started_at=NOW,
            simulation_completed_at=NOW,
            simulation_duration_seconds=120,
            wall_clock_elapsed_seconds=0.01,
            events_total=1,
            events_executed=1,
            events_succeeded=1,
            events_failed=0,
            incident_ids=("incident-carmel-01",),
        )


def poll_until(client: TestClient, predicate, *, timeout: float = 5.0, interval: float = 0.01) -> dict:
    deadline = time.monotonic() + timeout
    body = None
    while time.monotonic() < deadline:
        body = client.get("/api/v1/simulation/runs/current").json()
        if predicate(body):
            return body
        time.sleep(interval)
    raise AssertionError(f"condition never became true; last body was {body}")


def test_start_run_executes_in_background_and_becomes_pollable_to_completion(monkeypatch):
    monkeypatch.setattr(
        _settings_module,
        "settings",
        SimpleNamespace(ENABLE_SIMULATION_CONTROL_API=True, ENABLE_DEMO_DATA_RESET=True),
    )

    class NoopReset:  # the mandatory reset (Task 9A) must not touch a real database in this test
        def reset_demo_state(self):
            return None

    manager = SimulationRunManager(
        runner_factory=InstantFakeRunner,
        reset_service_factory=NoopReset,
        executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-simulation-run"),
    )
    app = create_app()
    app.dependency_overrides[get_simulation_run_manager] = lambda: manager
    client = TestClient(app)

    start_response = client.post(
        "/api/v1/simulation/runs",
        json={"preset": "operations_demo", "seed": 7},
    )

    # POST returns immediately - it never blocks on the runner, so the
    # response body must already reflect a reserved run before the
    # background thread could possibly have finished it.
    assert start_response.status_code == 202
    started_body = start_response.json()
    assert started_body["run_id"] is not None
    assert started_body["state"] in ("preparing", "running", "completed")

    final_body = poll_until(client, lambda body: body["state"] == "completed")

    # simulation_duration_seconds is set from the real scenario built for
    # this seed (operations_demo now computes duration dynamically per
    # seed - see build_operations_demo_scenario) - never a fixed constant,
    # and never taken from the fake runner's own result.
    expected_duration_seconds = build_operations_demo_scenario(seed=7).duration_seconds

    assert final_body["run_id"] == started_body["run_id"]
    assert final_body["events_completed"] == 1
    assert final_body["events_succeeded"] == 1
    assert final_body["events_failed"] == 0
    assert final_body["simulation_duration_seconds"] == expected_duration_seconds
    assert final_body["wall_clock_elapsed_seconds"] is not None

    # A second GET after completion still returns the same finished run,
    # unchanged, rather than reverting to IDLE or 404ing.
    still_there = client.get("/api/v1/simulation/runs/current").json()
    assert still_there == final_body
