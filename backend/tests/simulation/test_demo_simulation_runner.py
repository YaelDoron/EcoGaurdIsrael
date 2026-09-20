"""Tests for DemoSimulationRunner (Task A2).

All coordinators here are fakes - these tests never touch a real database
and never sleep for real, even for automatic mode or the full
operations_demo timeline (~120 simulated seconds).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from src.simulation import (
    SimulationEventExecutionResult,
    SimulationMode,
    build_active_fire_scenario,
    build_carmel_golan_active_fire_scenario,
    build_operations_demo_scenario,
)
from src.simulation.analysis import (
    SimulationFireDangerResult,
    SimulationFireDetectionResult,
    SimulationRefreshResult,
)
from src.simulation.demo_simulation_runner import (
    DemoSimulationEventPhase,
    DemoSimulationRunConfig,
    DemoSimulationRunner,
    DemoSimulationStatus,
)

STARTED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


class FakeExecutor:
    def __init__(self, success: bool = True) -> None:
        self.calls: list[tuple] = []
        self._success = success

    def execute(self, scenario, event, event_timestamp):
        self.calls.append((event, event_timestamp))
        return SimulationEventExecutionResult(
            event=event,
            success=self._success,
            generated_count=1,
            saved_count=1 if self._success else 0,
            duplicates_skipped=0,
            failed_count=0 if self._success else 1,
            error_message=None if self._success else "simulated failure",
        )


class NoOpFireDangerCoordinator:
    def handle_event(self, **kwargs):
        return SimulationFireDangerResult(triggered=False, assessment_result=None, reason="noop")


class NoOpFireDetectionCoordinator:
    def handle_event(self, **kwargs):
        return SimulationFireDetectionResult(triggered=False, detection_result=None, reason="noop")


class NoOpRefreshCoordinator:
    def handle_event(self, **kwargs):
        return SimulationRefreshResult(triggered=False, reason="noop")


class NoOpOperationalCoordinator:
    def scramble_resource_availability(self, latitude, longitude, availability_ratio=0.9):
        return []


class FakeAutomaticService:
    """Mirrors the CLI test suite's own duck-typed automatic-mode fake service."""

    def __init__(self, due_batches) -> None:
        self._due_batches = list(due_batches)
        self.started_with = None

    @property
    def is_finished(self):
        return not self._due_batches

    def start(self, scenario, mode):
        self.started_with = (scenario, mode)

    def get_due_events(self):
        if not self._due_batches:
            return []
        return self._due_batches.pop(0)


def make_runner(executor=None, service=None, clock_fn=None) -> DemoSimulationRunner:
    kwargs = dict(
        executor=executor or FakeExecutor(),
        fire_danger_coordinator=NoOpFireDangerCoordinator(),
        fire_detection_coordinator=NoOpFireDetectionCoordinator(),
        operational_coordinator=NoOpOperationalCoordinator(),
        simulation_refresh_coordinator=NoOpRefreshCoordinator(),
        service=service,
    )
    if clock_fn is not None:
        kwargs["clock_fn"] = clock_fn
    return DemoSimulationRunner(**kwargs)


# ---------------------------------------------------------------------------
# 1-2: ordering and callback-optional
# ---------------------------------------------------------------------------


def test_runner_executes_events_in_order():
    scenario = build_active_fire_scenario(seed=42)
    executor = FakeExecutor()
    runner = make_runner(executor=executor)

    result = runner.run(scenario, DemoSimulationRunConfig(mode=SimulationMode.MANUAL), scenario_started_at=STARTED_AT)

    assert [call[0] for call in executor.calls] == list(scenario.events)
    assert result.events_executed == len(scenario.events)


def test_runner_works_without_a_progress_callback():
    scenario = build_active_fire_scenario(seed=42)
    runner = make_runner()

    result = runner.run(scenario, DemoSimulationRunConfig(mode=SimulationMode.MANUAL), scenario_started_at=STARTED_AT)

    assert result.events_executed == len(scenario.events)


# ---------------------------------------------------------------------------
# 3-4: progress ordering and required fields
# ---------------------------------------------------------------------------


def test_progress_emitted_in_correct_order():
    scenario = build_active_fire_scenario(seed=42)
    runner = make_runner()
    progresses = []

    runner.run(
        scenario,
        DemoSimulationRunConfig(mode=SimulationMode.MANUAL),
        scenario_started_at=STARTED_AT,
        on_progress=progresses.append,
    )

    # Task A3: each event now emits exactly two ticks - EVENT_STARTED then
    # EVENT_COMPLETED - never one.
    assert len(progresses) == 2 * len(scenario.events)
    assert [progress.phase for progress in progresses] == [
        phase
        for _ in scenario.events
        for phase in (DemoSimulationEventPhase.EVENT_STARTED, DemoSimulationEventPhase.EVENT_COMPLETED)
    ]
    assert [progress.event_index for progress in progresses] == [
        index for index in range(len(scenario.events)) for _ in range(2)
    ]
    assert [progress.timestamp_offset_sec for progress in progresses] == [
        event.offset_seconds for event in scenario.events for _ in range(2)
    ]
    completed_only = [p for p in progresses if p.phase is DemoSimulationEventPhase.EVENT_COMPLETED]
    assert [progress.events_completed for progress in completed_only] == list(range(1, len(scenario.events) + 1))


def test_event_started_tick_precedes_event_completed_and_carries_no_outcome_yet():
    scenario = build_active_fire_scenario(seed=42)
    runner = make_runner()
    progresses = []

    runner.run(
        scenario,
        DemoSimulationRunConfig(mode=SimulationMode.MANUAL),
        scenario_started_at=STARTED_AT,
        on_progress=progresses.append,
    )

    started, completed = progresses[0], progresses[1]
    assert started.phase is DemoSimulationEventPhase.EVENT_STARTED
    assert started.success is None
    assert started.event_outcome is None
    assert started.events_completed == 0  # this event isn't counted as completed yet

    assert completed.phase is DemoSimulationEventPhase.EVENT_COMPLETED
    assert completed.success is True
    assert completed.event_outcome is not None
    assert completed.events_completed == 1


def test_progress_contains_required_fields():
    scenario = build_active_fire_scenario(seed=42)
    runner = make_runner()
    progresses = []

    runner.run(
        scenario,
        DemoSimulationRunConfig(mode=SimulationMode.MANUAL),
        scenario_started_at=STARTED_AT,
        on_progress=progresses.append,
    )

    completed_only = [p for p in progresses if p.phase is DemoSimulationEventPhase.EVENT_COMPLETED]
    first = completed_only[0]
    first_event = scenario.events[0]
    assert first.event_index == 0
    assert first.events_total == len(scenario.events)
    assert first.incident_id == first_event.incident_id
    assert first.event_type == first_event.event_type
    assert first.timestamp_offset_sec == first_event.offset_seconds
    assert first.success is True
    assert first.started_at == STARTED_AT
    assert isinstance(first.now, datetime)


# ---------------------------------------------------------------------------
# 5: final result counts
# ---------------------------------------------------------------------------


def test_final_result_counts_correct_for_all_success():
    scenario = build_active_fire_scenario(seed=42)
    runner = make_runner()

    result = runner.run(scenario, DemoSimulationRunConfig(mode=SimulationMode.MANUAL), scenario_started_at=STARTED_AT)

    assert result.events_total == len(scenario.events)
    assert result.events_executed == len(scenario.events)
    assert result.events_succeeded == len(scenario.events)
    assert result.events_failed == 0
    assert result.status is DemoSimulationStatus.COMPLETED
    assert result.incident_ids == tuple(incident.incident_id for incident in scenario.incidents)
    assert len(result.event_summaries) == len(scenario.events)


# ---------------------------------------------------------------------------
# 6-8: simulation duration vs. wall-clock duration are never conflated
# ---------------------------------------------------------------------------


def test_wall_clock_and_simulation_duration_are_measured_independently():
    scenario = build_active_fire_scenario(seed=42)
    clock_values = iter([1000.0, 1237.5])
    runner = make_runner(clock_fn=lambda: next(clock_values))

    result = runner.run(scenario, DemoSimulationRunConfig(mode=SimulationMode.MANUAL), scenario_started_at=STARTED_AT)

    assert result.simulation_duration_seconds == scenario.duration_seconds
    assert result.wall_clock_elapsed_seconds == 237.5
    assert result.simulation_duration_seconds != result.wall_clock_elapsed_seconds


# ---------------------------------------------------------------------------
# 9-10: automatic mode uses scenario-offset due events and an injected sleeper
# ---------------------------------------------------------------------------


def test_automatic_mode_uses_due_events_and_injected_sleeper_never_sleeps_for_real():
    scenario = build_active_fire_scenario(seed=42)
    first_event, second_event, third_event = scenario.events[:3]
    service = FakeAutomaticService(due_batches=[[first_event], [second_event, third_event]])
    executor = FakeExecutor()
    runner = make_runner(executor=executor, service=service)
    sleeps: list[float] = []
    config = DemoSimulationRunConfig(mode=SimulationMode.AUTOMATIC, sleep_fn=sleeps.append, poll_interval_seconds=0.25)

    started = time.monotonic()
    result = runner.run(scenario, config, scenario_started_at=STARTED_AT)
    real_elapsed = time.monotonic() - started

    assert service.started_with == (scenario, SimulationMode.AUTOMATIC)
    assert [call[0] for call in executor.calls] == [first_event, second_event, third_event]
    assert sleeps == [0.25]
    assert result.events_executed == 3
    assert real_elapsed < 1.0


# ---------------------------------------------------------------------------
# 11: manual mode never calls input() inside the runner
# ---------------------------------------------------------------------------


def test_manual_mode_runs_without_any_before_event_callback():
    """No before_event callback is supplied, so nothing resembling input()
    can be invoked - the runner itself has no input() call at all."""
    scenario = build_active_fire_scenario(seed=42)
    runner = make_runner()

    result = runner.run(scenario, DemoSimulationRunConfig(mode=SimulationMode.MANUAL), scenario_started_at=STARTED_AT)

    assert result.events_executed == len(scenario.events)


def test_manual_mode_before_event_receives_the_upcoming_event():
    scenario = build_active_fire_scenario(seed=42)
    runner = make_runner()
    seen_events = []

    config = DemoSimulationRunConfig(mode=SimulationMode.MANUAL, before_event=seen_events.append)
    runner.run(scenario, config, scenario_started_at=STARTED_AT)

    assert seen_events == list(scenario.events)


# ---------------------------------------------------------------------------
# 13: event-level failure preserves continue (not abort) semantics
# ---------------------------------------------------------------------------


def test_event_level_failure_continues_and_is_recorded():
    scenario = build_active_fire_scenario(seed=42)
    executor = FakeExecutor(success=False)
    runner = make_runner(executor=executor)

    result = runner.run(scenario, DemoSimulationRunConfig(mode=SimulationMode.MANUAL), scenario_started_at=STARTED_AT)

    assert result.events_executed == len(scenario.events)
    assert result.events_failed == len(scenario.events)
    assert result.events_succeeded == 0
    assert result.status is DemoSimulationStatus.COMPLETED_WITH_ERRORS
    assert all(not summary.success for summary in result.event_summaries)


# ---------------------------------------------------------------------------
# 14: runner-level catastrophic failure surfaces rather than pretending success
# ---------------------------------------------------------------------------


def test_unexpected_exception_during_execution_propagates():
    scenario = build_active_fire_scenario(seed=42)

    class BoomExecutor:
        def execute(self, scenario, event, event_timestamp):
            raise RuntimeError("boom")

    runner = make_runner(executor=BoomExecutor())

    with pytest.raises(RuntimeError, match="boom"):
        runner.run(scenario, DemoSimulationRunConfig(mode=SimulationMode.MANUAL), scenario_started_at=STARTED_AT)


# ---------------------------------------------------------------------------
# 15-16: existing presets (including operations_demo) run through the runner
# with fakes, without waiting real scenario-timeline time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build_scenario_fn",
    [build_active_fire_scenario, build_carmel_golan_active_fire_scenario, build_operations_demo_scenario],
)
def test_existing_presets_run_through_the_runner_without_real_delay(build_scenario_fn):
    scenario = build_scenario_fn(seed=42)
    executor = FakeExecutor()
    runner = make_runner(executor=executor)

    started = time.monotonic()
    result = runner.run(scenario, DemoSimulationRunConfig(mode=SimulationMode.MANUAL), scenario_started_at=STARTED_AT)
    real_elapsed = time.monotonic() - started

    assert result.events_total == len(scenario.events)
    assert result.events_executed == len(scenario.events)
    assert real_elapsed < 5.0, f"took {real_elapsed:.2f}s for a fake-only run of {len(scenario.events)} events"


# ---------------------------------------------------------------------------
# Part 14 prep: an optional should_stop predicate can end the loop early
# ---------------------------------------------------------------------------


def test_should_stop_predicate_ends_the_loop_early():
    scenario = build_active_fire_scenario(seed=42)
    executor = FakeExecutor()
    runner = make_runner(executor=executor)
    calls = {"count": 0}

    def should_stop() -> bool:
        calls["count"] += 1
        return calls["count"] > 2

    config = DemoSimulationRunConfig(mode=SimulationMode.MANUAL, should_stop=should_stop)
    result = runner.run(scenario, config, scenario_started_at=STARTED_AT)

    assert result.events_executed < len(scenario.events)
