"""POST /api/v1/simulation/runs/current/stop and the presentation preset in /presets."""
from __future__ import annotations

from dataclasses import replace

from src.services.simulation_control.simulation_run_manager import SimulationNotRunningError, SimulationRunState
from tests.api.test_simulation_router import (
    FakeSimulationRunManager,
    client_for,
    disable_control,
    enable_control,
    make_running_snapshot,
)


class StoppableFakeManager(FakeSimulationRunManager):
    def __init__(self, snapshot, stop_exception: Exception | None = None):
        super().__init__(snapshot=snapshot)
        self.stop_calls = 0
        self._stop_exception = stop_exception

    def stop_run(self):
        self.stop_calls += 1
        if self._stop_exception is not None:
            raise self._stop_exception
        return replace(self._snapshot, last_message="Stopping after the current event.")


def test_stop_returns_202_with_the_stopping_snapshot(monkeypatch):
    enable_control(monkeypatch)
    manager = StoppableFakeManager(make_running_snapshot())

    response = client_for(manager).post("/api/v1/simulation/runs/current/stop")

    assert response.status_code == 202
    assert manager.stop_calls == 1
    assert response.json()["state"] == "running"
    assert response.json()["last_message"] == "Stopping after the current event."


def test_stop_without_an_active_run_returns_409(monkeypatch):
    enable_control(monkeypatch)
    manager = StoppableFakeManager(make_running_snapshot(), stop_exception=SimulationNotRunningError("No simulation run is active."))

    response = client_for(manager).post("/api/v1/simulation/runs/current/stop")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "SIMULATION_NOT_RUNNING"


def test_stop_is_refused_when_simulation_control_is_disabled(monkeypatch):
    disable_control(monkeypatch)
    manager = StoppableFakeManager(make_running_snapshot())

    response = client_for(manager).post("/api/v1/simulation/runs/current/stop")

    assert response.status_code == 403
    assert manager.stop_calls == 0


def test_a_stopped_run_is_reported_as_stopped(monkeypatch):
    enable_control(monkeypatch)
    stopped = replace(make_running_snapshot(), state=SimulationRunState.STOPPED, last_message="Simulation stopped by the operator.")

    response = client_for(FakeSimulationRunManager(snapshot=stopped)).get("/api/v1/simulation/runs/current")

    assert response.status_code == 200
    assert response.json()["state"] == "stopped"


def test_presets_list_the_presentation_preset_with_its_long_duration(monkeypatch):
    enable_control(monkeypatch)

    presets = client_for(FakeSimulationRunManager()).get("/api/v1/simulation/presets").json()["presets"]

    presentation = next(preset for preset in presets if preset["id"] == "presentation_demo")
    assert presentation["simulation_duration_seconds"] == 1800


def test_a_stopping_run_is_reported_as_stopping(monkeypatch):
    enable_control(monkeypatch)
    stopping = replace(make_running_snapshot(), state=SimulationRunState.STOPPING)

    response = client_for(FakeSimulationRunManager(snapshot=stopping)).get("/api/v1/simulation/runs/current")

    assert response.status_code == 200
    assert response.json()["state"] == "stopping"
