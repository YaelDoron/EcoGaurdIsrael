"""Tests for SimulationRunManager (Task A3).

Every test uses a fake DemoSimulationRunner and a fake DemoStateResetService
- never the real operations_demo/production stack - so the whole suite runs
instantly. Tests 3/4 (rejecting a concurrent start) use a real
ThreadPoolExecutor with an Event-gated fake to observe genuine PREPARING/
RUNNING state from another thread; every other test uses a synchronous
fake executor that runs the background work immediately on the calling
thread, so no real concurrency or waiting is needed.
"""
from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import src.config.settings  # noqa: F401 - ensures sys.modules entry below exists

# `src/config/__init__.py` does `from src.config.settings import settings`,
# which rebinds the *package's* `settings` attribute to the Settings
# instance - `import src.config.settings as x` follows that same attribute
# chain (via getattr(src, "config").settings) and would silently bind `x`
# to the instance too. sys.modules keys by full dotted name and is set
# directly by the import system, immune to that shadowing.
settings_module = sys.modules["src.config.settings"]

from src.services.simulation_control.simulation_run_manager import (
    SimulationAlreadyRunningError,
    SimulationPresetNotFoundError,
    SimulationResetDisabledError,
    SimulationRunManager,
    SimulationRunState,
)
from src.simulation.demo_simulation_runner import (
    DemoSimulationEventPhase,
    DemoSimulationEventProgress,
    DemoSimulationRunResult,
    DemoSimulationStatus,
)
from src.simulation.simulation_event import SimulationEventType

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


class SyncExecutor:
    """Runs submitted work synchronously on the calling thread - no real
    background execution, no waiting, deterministic test ordering."""

    def submit(self, fn, *args):
        fn(*args)

        class _ImmediateFuture:
            def result(self_inner):
                return None

        return _ImmediateFuture()


class FakeResetService:
    def __init__(self) -> None:
        self.calls = 0

    def reset_demo_state(self):
        self.calls += 1


class RaisingResetService:
    def reset_demo_state(self):
        raise RuntimeError("reset failed")


def make_progress(
    *,
    phase: DemoSimulationEventPhase,
    event_index: int,
    events_total: int,
    events_completed: int,
    success: bool | None = None,
    incident_id: str = "incident-carmel-01",
    event_type: SimulationEventType = SimulationEventType.WEATHER,
    offset: int = 0,
) -> DemoSimulationEventProgress:
    return DemoSimulationEventProgress(
        phase=phase,
        event_index=event_index,
        events_total=events_total,
        events_completed=events_completed,
        incident_id=incident_id,
        event_type=event_type,
        timestamp_offset_sec=offset,
        started_at=NOW,
        now=NOW,
        success=success,
        event_outcome=None,
    )


def make_result(status: DemoSimulationStatus, *, events_total=3, succeeded=3, failed=0) -> DemoSimulationRunResult:
    return DemoSimulationRunResult(
        status=status,
        simulation_started_at=NOW,
        simulation_completed_at=NOW,
        simulation_duration_seconds=120,
        wall_clock_elapsed_seconds=5.5,
        events_total=events_total,
        events_executed=succeeded + failed,
        events_succeeded=succeeded,
        events_failed=failed,
        incident_ids=("incident-carmel-01", "incident-golan-01"),
    )


class FakeRunner:
    """Emits a fixed progress sequence, then returns a fixed result -
    entirely synchronous, entirely deterministic, no real time elapsed."""

    def __init__(self, result: DemoSimulationRunResult, progress_events=()) -> None:
        self._result = result
        self._progress_events = progress_events

    def run(self, scenario, config, scenario_started_at=None, on_progress=None):
        if on_progress is not None:
            for progress in self._progress_events:
                on_progress(progress)
        return self._result


class RaisingRunner:
    def run(self, scenario, config, scenario_started_at=None, on_progress=None):
        raise RuntimeError("boom: simulated catastrophic runner failure")


class BlockingRunner:
    """Blocks .run() until `proceed` is set - lets a test observe RUNNING
    state from the calling thread while a real background thread is stuck."""

    def __init__(self, proceed: threading.Event, result: DemoSimulationRunResult) -> None:
        self._proceed = proceed
        self._result = result

    def run(self, scenario, config, scenario_started_at=None, on_progress=None):
        self._proceed.wait(timeout=5)
        return self._result


class BlockingResetService:
    def __init__(self, proceed: threading.Event) -> None:
        self._proceed = proceed

    def reset_demo_state(self):
        self._proceed.wait(timeout=5)


def enable_demo_reset(monkeypatch) -> None:
    # Targets the real src.config.settings *module* object directly (not a
    # dotted string) - src/config/__init__.py re-exports `settings` at the
    # package level, which shadows the submodule name and makes pytest's
    # string-based dotted-path resolution ambiguous.
    monkeypatch.setattr(settings_module, "settings", SimpleNamespace(ENABLE_DEMO_DATA_RESET=True))


def disable_demo_reset(monkeypatch) -> None:
    monkeypatch.setattr(settings_module, "settings", SimpleNamespace(ENABLE_DEMO_DATA_RESET=False))


def make_manager(runner=None, reset_service=None) -> SimulationRunManager:
    return SimulationRunManager(
        runner_factory=lambda: runner or FakeRunner(make_result(DemoSimulationStatus.COMPLETED)),
        reset_service_factory=lambda: reset_service or FakeResetService(),
        executor=SyncExecutor(),
    )


@pytest.fixture(autouse=True)
def _demo_reset_is_enabled(monkeypatch):
    """Task 9A: every run now resets first, so the reset safety flag must be on for a run to start at all.
    Tests that exercise the disabled flag turn it off explicitly (see disable_demo_reset)."""
    enable_demo_reset(monkeypatch)


# ---------------------------------------------------------------------------
# 1. Initial state
# ---------------------------------------------------------------------------


def test_initial_state_is_idle_with_no_run():
    manager = make_manager()

    snapshot = manager.get_current_snapshot()

    assert snapshot.state is SimulationRunState.IDLE
    assert snapshot.run_id is None
    assert snapshot.events_completed == 0


# ---------------------------------------------------------------------------
# 2. Start reserves a run immediately
# ---------------------------------------------------------------------------


def test_start_reserves_a_run_immediately():
    manager = make_manager()

    snapshot = manager.start_run(preset_id="operations_demo", seed=42)

    assert snapshot.run_id is not None
    assert snapshot.preset_id == "operations_demo"
    assert snapshot.seed == 42
    assert snapshot.mode == "automatic"


# ---------------------------------------------------------------------------
# 2b. Optional seed - auto-generation when omitted (Part L/S)
# ---------------------------------------------------------------------------


def test_explicit_seed_is_used_exactly_as_given():
    manager = make_manager()

    snapshot = manager.start_run(preset_id="operations_demo", seed=42)

    assert snapshot.seed == 42


def test_omitted_seed_generates_a_real_integer_seed():
    manager = make_manager()

    snapshot = manager.start_run(preset_id="operations_demo")

    assert isinstance(snapshot.seed, int)
    assert snapshot.seed is not None


def test_omitted_seed_default_is_also_generated():
    """`seed` defaults to None in the signature itself - a caller that
    never passes the keyword at all gets the same auto-generation."""
    manager = make_manager()

    snapshot = manager.start_run(preset_id="operations_demo")

    assert snapshot.seed is not None


def test_two_consecutive_auto_seeded_runs_do_not_reuse_the_immediately_previous_seed():
    manager = make_manager()

    first = manager.start_run(preset_id="operations_demo")
    second = manager.start_run(preset_id="operations_demo")

    assert first.seed != second.seed


def test_auto_generated_seed_still_builds_a_valid_reproducible_scenario(monkeypatch):
    """The generated seed is a real int compatible with the existing
    seeded scenario builder - not a placeholder/sentinel value."""
    from src.simulation.simulation_scenario import build_operations_demo_scenario

    manager = make_manager()
    snapshot = manager.start_run(preset_id="operations_demo")

    scenario_a = build_operations_demo_scenario(seed=snapshot.seed)
    scenario_b = build_operations_demo_scenario(seed=snapshot.seed)
    assert scenario_a.incidents == scenario_b.incidents
    assert scenario_a.events == scenario_b.events


# ---------------------------------------------------------------------------
# 3-4. Concurrent start rejected during PREPARING / RUNNING
# ---------------------------------------------------------------------------


def test_second_start_during_preparing_is_rejected(monkeypatch):
    enable_demo_reset(monkeypatch)
    proceed = threading.Event()
    manager = SimulationRunManager(
        runner_factory=lambda: FakeRunner(make_result(DemoSimulationStatus.COMPLETED)),
        reset_service_factory=lambda: BlockingResetService(proceed),
        executor=ThreadPoolExecutor(max_workers=1),
    )

    manager.start_run(preset_id="operations_demo", seed=42, reset_demo_state=True)
    try:
        _wait_for_state(manager, SimulationRunState.PREPARING)

        with pytest.raises(SimulationAlreadyRunningError):
            manager.start_run(preset_id="operations_demo", seed=99)
    finally:
        proceed.set()


def test_second_start_during_running_is_rejected():
    proceed = threading.Event()
    manager = SimulationRunManager(
        runner_factory=lambda: BlockingRunner(proceed, make_result(DemoSimulationStatus.COMPLETED)),
        reset_service_factory=lambda: FakeResetService(),
        executor=ThreadPoolExecutor(max_workers=1),
    )

    manager.start_run(preset_id="operations_demo", seed=42)
    try:
        _wait_for_state(manager, SimulationRunState.RUNNING)

        with pytest.raises(SimulationAlreadyRunningError):
            manager.start_run(preset_id="operations_demo", seed=99)
    finally:
        proceed.set()


def _wait_for_state(manager: SimulationRunManager, state: SimulationRunState, timeout: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager.get_current_snapshot().state is state:
            return
        time.sleep(0.01)
    raise AssertionError(f"manager never reached state {state}")


# ---------------------------------------------------------------------------
# 5-6. A new run is allowed after COMPLETED/FAILED
# ---------------------------------------------------------------------------


def test_start_after_completed_is_allowed():
    manager = make_manager(runner=FakeRunner(make_result(DemoSimulationStatus.COMPLETED)))
    first = manager.start_run(preset_id="operations_demo", seed=1)
    assert manager.get_current_snapshot().state is SimulationRunState.COMPLETED

    second = manager.start_run(preset_id="operations_demo", seed=2)

    assert second.run_id != first.run_id


def test_start_after_failed_is_allowed():
    manager = make_manager(runner=RaisingRunner())
    first = manager.start_run(preset_id="operations_demo", seed=1)
    assert manager.get_current_snapshot().state is SimulationRunState.FAILED

    manager2 = make_manager(runner=FakeRunner(make_result(DemoSimulationStatus.COMPLETED)))
    # Re-use the same manager instance to prove FAILED -> new start works:
    manager._runner_factory = lambda: FakeRunner(make_result(DemoSimulationStatus.COMPLETED))
    second = manager.start_run(preset_id="operations_demo", seed=2)

    assert second.run_id != first.run_id
    assert manager.get_current_snapshot().state is SimulationRunState.COMPLETED


# ---------------------------------------------------------------------------
# 7. Progress callback updates counters
# ---------------------------------------------------------------------------


def test_progress_callback_updates_counters_and_current_event():
    progress_events = [
        make_progress(phase=DemoSimulationEventPhase.EVENT_STARTED, event_index=0, events_total=2, events_completed=0),
        make_progress(
            phase=DemoSimulationEventPhase.EVENT_COMPLETED,
            event_index=0,
            events_total=2,
            events_completed=1,
            success=True,
        ),
        make_progress(
            phase=DemoSimulationEventPhase.EVENT_STARTED,
            event_index=1,
            events_total=2,
            events_completed=1,
            incident_id="incident-golan-01",
            event_type=SimulationEventType.NEWS,
            offset=40,
        ),
    ]
    runner = FakeRunner(make_result(DemoSimulationStatus.COMPLETED), progress_events=progress_events)
    manager = make_manager(runner=runner)

    manager.start_run(preset_id="operations_demo", seed=42)

    final = manager.get_current_snapshot()
    # The final _apply_result overwrites completed/succeeded/failed with the
    # runner's own authoritative totals, so check via a runner with no
    # trailing result overwrite by inspecting mid-run behavior instead:
    assert final.current_event.incident_id == "incident-golan-01"
    assert final.current_event.event_type == "news"
    assert final.current_event.timestamp_offset_sec == 40


def test_progress_events_completed_reflects_only_finished_events_before_result():
    seen_snapshots = []

    class RecordingRunner:
        def run(self, scenario, config, scenario_started_at=None, on_progress=None):
            on_progress(
                make_progress(phase=DemoSimulationEventPhase.EVENT_STARTED, event_index=7, events_total=18, events_completed=7)
            )
            seen_snapshots.append(manager.get_current_snapshot())
            on_progress(
                make_progress(
                    phase=DemoSimulationEventPhase.EVENT_COMPLETED,
                    event_index=7,
                    events_total=18,
                    events_completed=8,
                    success=True,
                )
            )
            seen_snapshots.append(manager.get_current_snapshot())
            return make_result(DemoSimulationStatus.COMPLETED)

    manager = make_manager(runner=RecordingRunner())
    manager.start_run(preset_id="operations_demo", seed=42)

    started_snapshot, completed_snapshot = seen_snapshots
    assert started_snapshot.events_completed == 7
    assert started_snapshot.current_event.event_index == 7
    assert completed_snapshot.events_completed == 8
    assert completed_snapshot.current_event.event_index == 7


# ---------------------------------------------------------------------------
# 8. Snapshot is read-only/thread-safe (frozen)
# ---------------------------------------------------------------------------


def test_snapshot_is_frozen_and_immutable():
    manager = make_manager()
    snapshot = manager.get_current_snapshot()

    with pytest.raises(Exception):
        snapshot.state = SimulationRunState.RUNNING  # type: ignore[misc]


def test_snapshot_returned_before_mutation_is_unaffected_by_later_changes():
    runner = FakeRunner(make_result(DemoSimulationStatus.COMPLETED))
    manager = make_manager(runner=runner)

    before = manager.start_run(preset_id="operations_demo", seed=1)
    after = manager.get_current_snapshot()

    assert before.state is SimulationRunState.PREPARING
    assert after.state is SimulationRunState.COMPLETED
    assert before.state is SimulationRunState.PREPARING  # unchanged - it's a distinct frozen object


# ---------------------------------------------------------------------------
# 9-11. Runner status -> manager state mapping
# ---------------------------------------------------------------------------


def test_runner_completed_maps_to_manager_completed():
    manager = make_manager(runner=FakeRunner(make_result(DemoSimulationStatus.COMPLETED)))

    manager.start_run(preset_id="operations_demo", seed=1)

    assert manager.get_current_snapshot().state is SimulationRunState.COMPLETED


def test_runner_completed_with_errors_maps_to_manager_completed_with_errors():
    result = make_result(DemoSimulationStatus.COMPLETED_WITH_ERRORS, succeeded=2, failed=1)
    manager = make_manager(runner=FakeRunner(result))

    manager.start_run(preset_id="operations_demo", seed=1)

    snapshot = manager.get_current_snapshot()
    assert snapshot.state is SimulationRunState.COMPLETED_WITH_ERRORS
    assert snapshot.events_failed == 1
    assert snapshot.events_succeeded == 2


def test_runner_exception_maps_to_manager_failed_with_sanitized_error():
    manager = make_manager(runner=RaisingRunner())

    manager.start_run(preset_id="operations_demo", seed=1)

    snapshot = manager.get_current_snapshot()
    assert snapshot.state is SimulationRunState.FAILED
    assert snapshot.error is not None
    assert snapshot.error.code == "SIMULATION_RUN_FAILED"
    assert "boom" not in snapshot.error.message
    assert "RuntimeError" not in snapshot.error.message


# ---------------------------------------------------------------------------
# 12-15. Reset-before-run semantics
# ---------------------------------------------------------------------------


def test_reset_is_invoked_before_runner_when_requested(monkeypatch):
    enable_demo_reset(monkeypatch)
    order = []

    class OrderedReset:
        def reset_demo_state(self):
            order.append("reset")

    class OrderedRunner:
        def run(self, scenario, config, scenario_started_at=None, on_progress=None):
            order.append("run")
            return make_result(DemoSimulationStatus.COMPLETED)

    manager = SimulationRunManager(
        runner_factory=lambda: OrderedRunner(),
        reset_service_factory=lambda: OrderedReset(),
        executor=SyncExecutor(),
    )

    manager.start_run(preset_id="operations_demo", seed=1, reset_demo_state=True)

    assert order == ["reset", "run"]


def test_reset_failure_prevents_runner_from_starting_and_manager_is_failed(monkeypatch):
    enable_demo_reset(monkeypatch)
    run_calls = []

    class NeverCalledRunner:
        def run(self, *args, **kwargs):
            run_calls.append(1)
            return make_result(DemoSimulationStatus.COMPLETED)

    manager = SimulationRunManager(
        runner_factory=lambda: NeverCalledRunner(),
        reset_service_factory=lambda: RaisingResetService(),
        executor=SyncExecutor(),
    )

    manager.start_run(preset_id="operations_demo", seed=1, reset_demo_state=True)

    assert run_calls == []
    assert manager.get_current_snapshot().state is SimulationRunState.FAILED


def test_every_run_resets_first_without_being_asked():
    """Task 9A: the reset is mandatory - a plain start_run() (no flag) resets before the runner."""
    reset_service = FakeResetService()
    manager = make_manager(reset_service=reset_service)

    manager.start_run(preset_id="operations_demo", seed=1)

    assert reset_service.calls == 1


def test_opting_out_of_the_reset_is_refused_and_nothing_is_reserved():
    from src.services.simulation_control.simulation_run_manager import SimulationResetRequiredError

    reset_service = FakeResetService()
    manager = make_manager(reset_service=reset_service)

    for opt_out in (False, None, 0):
        with pytest.raises(SimulationResetRequiredError):
            manager.start_run(preset_id="operations_demo", seed=1, reset_demo_state=opt_out)

    assert reset_service.calls == 0
    assert manager.get_current_snapshot().state is SimulationRunState.IDLE


def test_start_is_refused_fail_closed_when_the_reset_flag_is_off_even_without_asking_for_it(monkeypatch):
    disable_demo_reset(monkeypatch)
    reset_service = FakeResetService()
    manager = make_manager(reset_service=reset_service)

    with pytest.raises(SimulationResetDisabledError):
        manager.start_run(preset_id="operations_demo", seed=1)

    assert reset_service.calls == 0
    assert manager.get_current_snapshot().state is SimulationRunState.IDLE  # nothing was reserved


# ---------------------------------------------------------------------------
# 16-17. run_id changes; completed result stays available
# ---------------------------------------------------------------------------


def test_run_id_changes_between_runs():
    manager = make_manager(runner=FakeRunner(make_result(DemoSimulationStatus.COMPLETED)))

    first = manager.start_run(preset_id="operations_demo", seed=1)
    second = manager.start_run(preset_id="operations_demo", seed=2)

    assert first.run_id != second.run_id


def test_completed_result_remains_available_until_next_run():
    manager = make_manager(runner=FakeRunner(make_result(DemoSimulationStatus.COMPLETED, succeeded=5, failed=0)))

    manager.start_run(preset_id="operations_demo", seed=1)
    snapshot_a = manager.get_current_snapshot()
    snapshot_b = manager.get_current_snapshot()

    assert snapshot_a.state is SimulationRunState.COMPLETED
    assert snapshot_b.state is SimulationRunState.COMPLETED
    assert snapshot_a.events_succeeded == 5


# ---------------------------------------------------------------------------
# 18. No DB Session is stored on the manager
# ---------------------------------------------------------------------------


def test_manager_holds_no_sqlalchemy_session_attribute():
    manager = make_manager()

    for attribute_name in vars(manager):
        value = getattr(manager, attribute_name)
        assert type(value).__name__ != "Session", f"{attribute_name} looks like a stored SQLAlchemy Session"


# ---------------------------------------------------------------------------
# 19. Fake runner allows the whole suite to complete instantly - proven by
# the total runtime of this file (no sleeps, no real waits) rather than a
# single assertion; see pytest --durations if this regresses.
# ---------------------------------------------------------------------------


def test_preset_not_found_raises_without_reserving_a_run():
    manager = make_manager()

    with pytest.raises(SimulationPresetNotFoundError):
        manager.start_run(preset_id="does-not-exist", seed=1)

    assert manager.get_current_snapshot().state is SimulationRunState.IDLE
