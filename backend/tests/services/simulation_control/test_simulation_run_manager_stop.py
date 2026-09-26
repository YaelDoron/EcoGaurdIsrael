"""SimulationRunManager: Stop Simulation and the presentation preset's pinned seed.

Fake runner/reset service only (see test_simulation_run_manager.py) - no
database, no real simulation stack.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.services.simulation_control.simulation_run_manager import (
    SimulationNotRunningError,
    SimulationRunManager,
    SimulationRunState,
)
from src.simulation.demo_simulation_runner import DemoSimulationEventPhase, DemoSimulationStatus
from src.simulation.presentation_scenario import PRESENTATION_DEFAULT_SEED
from tests.services.simulation_control.test_simulation_run_manager import (
    FakeResetService,
    FakeRunner,
    enable_demo_reset,
    make_manager,
    make_progress,
    make_result,
)


@pytest.fixture(autouse=True)
def _demo_reset_is_enabled(monkeypatch):
    enable_demo_reset(monkeypatch)


class StopAwareRunner:
    """Executes one event, parks until released, then honours config.should_stop like the real loop."""

    def __init__(self, events_total: int = 5) -> None:
        self.first_event_done = threading.Event()
        self.release = threading.Event()
        self.events_total = events_total
        self.config = None

    def run(self, scenario, config, scenario_started_at=None, on_progress=None):
        self.config = config
        executed = 0
        for index in range(self.events_total):
            if config.should_stop is not None and config.should_stop():
                break
            if on_progress is not None:
                on_progress(
                    make_progress(
                        phase=DemoSimulationEventPhase.EVENT_COMPLETED,
                        event_index=index,
                        events_total=self.events_total,
                        events_completed=index + 1,
                        success=True,
                    )
                )
            executed += 1
            if index == 0:
                self.first_event_done.set()
                self.release.wait(timeout=5)
        return make_result(DemoSimulationStatus.COMPLETED, events_total=self.events_total, succeeded=executed)


def _threaded_manager(runner, reset_service) -> tuple[SimulationRunManager, ThreadPoolExecutor]:
    executor = ThreadPoolExecutor(max_workers=1)
    manager = SimulationRunManager(
        runner_factory=lambda: runner,
        reset_service_factory=lambda: reset_service,
        executor=executor,
    )
    return manager, executor


def _wait_for_state(manager: SimulationRunManager, state: SimulationRunState) -> None:
    for _ in range(500):
        if manager.get_current_snapshot().state is state:
            return
        threading.Event().wait(0.01)
    raise AssertionError(f"run never reached {state}: {manager.get_current_snapshot()}")


def test_stop_prevents_further_events_and_ends_in_stopped_without_resetting():
    runner = StopAwareRunner(events_total=5)
    reset_service = FakeResetService()
    manager, executor = _threaded_manager(runner, reset_service)
    try:
        manager.start_run(preset_id="presentation_demo")
        assert runner.first_event_done.wait(timeout=5)

        stopping = manager.stop_run()
        # Acknowledged at once: RUNNING -> STOPPING while the current event finishes.
        assert stopping.state is SimulationRunState.STOPPING
        assert manager.get_current_snapshot().state is SimulationRunState.STOPPING
        assert stopping.last_message == "Stopping simulation - finishing the current event."
        runner.release.set()
        _wait_for_state(manager, SimulationRunState.STOPPED)
    finally:
        runner.release.set()
        executor.shutdown(wait=True)

    snapshot = manager.get_current_snapshot()
    assert snapshot.state is SimulationRunState.STOPPED
    assert snapshot.events_completed == 1  # the in-progress event finished; none of the other 4 ran
    assert snapshot.last_message == "Simulation stopped by the operator."
    # Stopping never resets/deletes anything: only the start's mandatory reset ran.
    assert reset_service.calls == 1


def test_a_new_run_after_stop_still_runs_the_mandatory_reset_and_is_not_pre_stopped():
    reset_service = FakeResetService()
    first = StopAwareRunner(events_total=3)
    manager, executor = _threaded_manager(first, reset_service)
    try:
        manager.start_run(preset_id="presentation_demo")
        assert first.first_event_done.wait(timeout=5)
        manager.stop_run()
        first.release.set()
        _wait_for_state(manager, SimulationRunState.STOPPED)

        second = StopAwareRunner(events_total=3)
        manager._runner_factory = lambda: second  # noqa: SLF001 - swap the fake for the second run
        second.release.set()
        manager.start_run(preset_id="presentation_demo")
        _wait_for_state(manager, SimulationRunState.COMPLETED)
    finally:
        first.release.set()
        executor.shutdown(wait=True)

    assert reset_service.calls == 2
    assert second.config.should_stop() is False  # a stale stop never leaks into the next run
    assert manager.get_current_snapshot().events_completed == 3


def test_stop_without_an_active_run_is_rejected():
    manager = make_manager()
    with pytest.raises(SimulationNotRunningError):
        manager.stop_run()

    manager.start_run(preset_id="operations_demo", seed=1)  # SyncExecutor: finishes immediately
    assert manager.get_current_snapshot().state is SimulationRunState.COMPLETED
    with pytest.raises(SimulationNotRunningError):
        manager.stop_run()


def test_runner_receives_a_stop_callback_that_reflects_the_request():
    runner = StopAwareRunner(events_total=2)
    manager, executor = _threaded_manager(runner, FakeResetService())
    try:
        manager.start_run(preset_id="presentation_demo")
        assert runner.first_event_done.wait(timeout=5)
        assert runner.config.should_stop() is False
        manager.stop_run()
        assert runner.config.should_stop() is True
    finally:
        runner.release.set()
        executor.shutdown(wait=True)


def test_presentation_preset_pins_its_approved_seed_when_the_request_omits_one():
    manager = make_manager(runner=FakeRunner(make_result(DemoSimulationStatus.COMPLETED)))

    snapshot = manager.start_run(preset_id="presentation_demo")

    assert snapshot.seed == PRESENTATION_DEFAULT_SEED


def test_an_explicit_seed_still_wins_for_the_presentation_preset():
    manager = make_manager()

    assert manager.start_run(preset_id="presentation_demo", seed=7).seed == 7


def test_operations_demo_keeps_backend_chosen_random_seeds():
    manager = make_manager()

    seeds = {manager.start_run(preset_id="operations_demo").seed for _ in range(3)}

    assert len(seeds) == 3


def test_stop_returns_immediately_without_waiting_for_the_worker():
    import time

    runner = StopAwareRunner(events_total=5)
    manager, executor = _threaded_manager(runner, FakeResetService())
    try:
        manager.start_run(preset_id="presentation_demo")
        assert runner.first_event_done.wait(timeout=5)  # the worker is now blocked inside an event

        started = time.perf_counter()
        snapshot = manager.stop_run()
        elapsed = time.perf_counter() - started

        assert elapsed < 0.2
        assert snapshot.state is SimulationRunState.STOPPING
    finally:
        runner.release.set()
        executor.shutdown(wait=True)
    assert manager.get_current_snapshot().state is SimulationRunState.STOPPED


def test_double_stop_while_stopping_is_a_safe_no_op():
    runner = StopAwareRunner(events_total=5)
    manager, executor = _threaded_manager(runner, FakeResetService())
    try:
        manager.start_run(preset_id="presentation_demo")
        assert runner.first_event_done.wait(timeout=5)
        first = manager.stop_run()
        second = manager.stop_run()
        assert first == second
        assert second.state is SimulationRunState.STOPPING
    finally:
        runner.release.set()
        executor.shutdown(wait=True)
    assert manager.get_current_snapshot().state is SimulationRunState.STOPPED
    with pytest.raises(SimulationNotRunningError):
        manager.stop_run()


def test_a_new_run_cannot_start_while_the_previous_one_is_stopping():
    from src.services.simulation_control.simulation_run_manager import SimulationAlreadyRunningError

    runner = StopAwareRunner(events_total=3)
    manager, executor = _threaded_manager(runner, FakeResetService())
    try:
        manager.start_run(preset_id="presentation_demo")
        assert runner.first_event_done.wait(timeout=5)
        manager.stop_run()
        with pytest.raises(SimulationAlreadyRunningError):
            manager.start_run(preset_id="presentation_demo")
    finally:
        runner.release.set()
        executor.shutdown(wait=True)


def test_stop_during_preparing_never_starts_any_event():
    class GatedReset:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.calls = 0

        def reset_demo_state(self):
            self.calls += 1
            self.entered.set()
            self.release.wait(timeout=5)

    reset = GatedReset()
    runner = StopAwareRunner(events_total=4)
    runner.release.set()
    manager, executor = _threaded_manager(runner, reset)
    try:
        manager.start_run(preset_id="presentation_demo")
        assert reset.entered.wait(timeout=5)
        assert manager.stop_run().state is SimulationRunState.STOPPING
        reset.release.set()
        _wait_for_state(manager, SimulationRunState.STOPPED)
    finally:
        reset.release.set()
        executor.shutdown(wait=True)
    assert manager.get_current_snapshot().events_completed == 0
    assert not runner.first_event_done.is_set()
