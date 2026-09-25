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
    resolve_stale_simulation_events,
)

STARTED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _stub_stale_simulation_event_resolver(monkeypatch):
    """run() resolves stale simulation FireEvents through real repositories
    by default; keep these fake-based tests away from any real database.
    The stale-event tests below inject a SQLite-backed resolver explicitly
    (the module-level import above still refers to the real function)."""
    monkeypatch.setattr(
        "src.simulation.demo_simulation_runner.resolve_stale_simulation_events",
        lambda as_of: (),
    )


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


# ---------------------------------------------------------------------------
# Stale simulation-only FireEvent cleanup at run start
# ---------------------------------------------------------------------------

from datetime import timedelta  # noqa: E402
import itertools  # noqa: E402

from src.calculators.fire_detection.fire_detection_config import (  # noqa: E402
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.database.models.satellite_hotspot_db import SatelliteHotspotDB  # noqa: E402
from src.database.models.wildfire_report_db import WildfireReportDB  # noqa: E402
from src.models import FireEvent, FireEventStatus, FireEvidenceRef, FireEvidenceType  # noqa: E402
from src.repositories.fire_event_config import ACTIVE_EVENT_MATCH_WINDOW_HOURS  # noqa: E402
from src.repositories.fire_event_repository import FireEventRepository  # noqa: E402
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository  # noqa: E402
from src.services.fire_event_lifecycle.fire_event_lifecycle_service import FireEventLifecycleService  # noqa: E402
from src.simulation.generators.news_data_generator import SIMULATED_NEWS_SOURCE_FEED  # noqa: E402
from src.simulation.generators.satellite_data_generator import SIMULATED_SATELLITE  # noqa: E402

MATCH_WINDOW = timedelta(hours=ACTIVE_EVENT_MATCH_WINDOW_HOURS)
STALE_UPDATED_AT = STARTED_AT - MATCH_WINDOW - timedelta(minutes=1)
RECENT_UPDATED_AT = STARTED_AT - MATCH_WINDOW + timedelta(minutes=1)
REAL_SATELLITE = "N20"
REAL_NEWS_FEED = "Ynet"
_evidence_counter = itertools.count(1)


def _persist_event(
    session_factory,
    *,
    updated_at,
    satellites=(SIMULATED_SATELLITE,),
    news_feeds=(SIMULATED_NEWS_SOURCE_FEED,),
    status=FireEventStatus.CONFIRMED,
) -> int:
    refs = []
    with session_factory() as session:
        for satellite in satellites:
            hotspot = SatelliteHotspotDB(
                detection_key=f"test-hotspot-{next(_evidence_counter)}",
                latitude=32.731,
                longitude=35.046,
                detected_at=updated_at,
                satellite=satellite,
            )
            session.add(hotspot)
            session.flush()
            refs.append(FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot.id))
        for feed in news_feeds:
            report = WildfireReportDB(
                source_url=f"https://example.test/report/{next(_evidence_counter)}",
                source_feed=feed,
                title="Wildfire reported",
                fetched_at=updated_at,
            )
            session.add(report)
            session.flush()
            refs.append(FireEvidenceRef(FireEvidenceType.NEWS, report.id))
        session.commit()

    event = FireEvent(
        latitude=32.731,
        longitude=35.046,
        detected_at=updated_at - timedelta(minutes=10),
        updated_at=updated_at,
        status=status,
        detection_confidence=0.8,
        methodology=FIRE_DETECTION_METHODOLOGY_NAME,
        methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
        location_name="Carmel Demo Area",
    )
    return FireEventRepository(session_factory=session_factory).create_event(event, tuple(refs)).id


def _lifecycle_service(session_factory) -> FireEventLifecycleService:
    return FireEventLifecycleService(
        fire_event_repository=FireEventRepository(session_factory=session_factory),
        resource_commitment_repository=ResourceCommitmentRepository(session_factory),
        session_factory=session_factory,
    )


def _sqlite_resolver(session_factory):
    repository = FireEventRepository(session_factory=session_factory)
    lifecycle_service = _lifecycle_service(session_factory)
    return lambda as_of: resolve_stale_simulation_events(
        as_of, fire_event_repository=repository, lifecycle_service=lifecycle_service
    )


def _run_with_resolver(resolver, started_at=STARTED_AT):
    runner = DemoSimulationRunner(
        executor=FakeExecutor(),
        fire_danger_coordinator=NoOpFireDangerCoordinator(),
        fire_detection_coordinator=NoOpFireDetectionCoordinator(),
        operational_coordinator=NoOpOperationalCoordinator(),
        simulation_refresh_coordinator=NoOpRefreshCoordinator(),
        stale_simulation_event_resolver=resolver,
    )
    return runner.run(
        build_active_fire_scenario(seed=42),
        DemoSimulationRunConfig(mode=SimulationMode.MANUAL),
        scenario_started_at=started_at,
    )


def _event(session_factory, fire_event_id):
    return FireEventRepository(session_factory=session_factory).get_by_id(fire_event_id).event


def test_stale_simulation_only_event_is_resolved_at_run_start(sqlite_session_factory):
    stale_id = _persist_event(sqlite_session_factory, updated_at=STALE_UPDATED_AT)

    _run_with_resolver(_sqlite_resolver(sqlite_session_factory))

    event = _event(sqlite_session_factory, stale_id)
    assert event.status is FireEventStatus.RESOLVED
    assert event.updated_at == STARTED_AT


def test_recent_simulation_event_inside_match_window_stays_active_and_matchable(sqlite_session_factory):
    recent_id = _persist_event(sqlite_session_factory, updated_at=RECENT_UPDATED_AT)

    _run_with_resolver(_sqlite_resolver(sqlite_session_factory))

    assert _event(sqlite_session_factory, recent_id).status is FireEventStatus.CONFIRMED
    match = FireEventRepository(session_factory=sqlite_session_factory).find_matching_active_event(
        latitude=32.731, longitude=35.046, observed_at=STARTED_AT
    )
    assert match is not None and match.id == recent_id


def test_real_evidence_event_is_never_resolved(sqlite_session_factory):
    real_id = _persist_event(
        sqlite_session_factory, updated_at=STALE_UPDATED_AT, satellites=(REAL_SATELLITE,), news_feeds=(REAL_NEWS_FEED,)
    )

    _run_with_resolver(_sqlite_resolver(sqlite_session_factory))

    assert _event(sqlite_session_factory, real_id).status is FireEventStatus.CONFIRMED


@pytest.mark.parametrize(
    ("satellites", "news_feeds"),
    [
        ((SIMULATED_SATELLITE, REAL_SATELLITE), (SIMULATED_NEWS_SOURCE_FEED,)),
        ((SIMULATED_SATELLITE,), (SIMULATED_NEWS_SOURCE_FEED, REAL_NEWS_FEED)),
        ((SIMULATED_SATELLITE,), (REAL_NEWS_FEED,)),
        ((SIMULATED_SATELLITE, None), ()),
        ((SIMULATED_SATELLITE,), (None,)),
    ],
)
def test_mixed_or_unclassifiable_evidence_event_is_never_resolved(sqlite_session_factory, satellites, news_feeds):
    mixed_id = _persist_event(
        sqlite_session_factory, updated_at=STALE_UPDATED_AT, satellites=satellites, news_feeds=news_feeds
    )

    _run_with_resolver(_sqlite_resolver(sqlite_session_factory))

    assert _event(sqlite_session_factory, mixed_id).status is FireEventStatus.CONFIRMED


@pytest.mark.parametrize("transition", ["resolve_event", "dismiss_event"])
def test_already_inactive_simulation_event_is_untouched(sqlite_session_factory, transition):
    event_id = _persist_event(sqlite_session_factory, updated_at=STALE_UPDATED_AT - timedelta(hours=1))
    closed_at = STALE_UPDATED_AT
    getattr(_lifecycle_service(sqlite_session_factory), transition)(event_id, as_of=closed_at)
    before = _event(sqlite_session_factory, event_id)

    _run_with_resolver(_sqlite_resolver(sqlite_session_factory))

    after = _event(sqlite_session_factory, event_id)
    assert after.status is before.status
    assert after.updated_at == closed_at


def test_no_stale_simulation_events_is_a_no_op(sqlite_session_factory):
    assert _sqlite_resolver(sqlite_session_factory)(STARTED_AT) == ()


def test_repeated_runs_without_reset_do_not_accumulate_obsolete_simulation_events(sqlite_session_factory):
    resolver = _sqlite_resolver(sqlite_session_factory)
    # Leftover from an earlier run, plus one created by "run 1" itself.
    leftover_id = _persist_event(sqlite_session_factory, updated_at=STALE_UPDATED_AT)
    _run_with_resolver(resolver, started_at=STARTED_AT)
    run_one_event_id = _persist_event(sqlite_session_factory, updated_at=STARTED_AT + timedelta(minutes=5))

    # Run 2 inside the window: run 1's event stays active for reuse.
    _run_with_resolver(resolver, started_at=STARTED_AT + timedelta(hours=1))
    assert _event(sqlite_session_factory, run_one_event_id).status is FireEventStatus.CONFIRMED

    # Run 3 after the window: run 1's event is resolved too; the leftover
    # (already resolved by run 1) is not modified again.
    run_three_started_at = STARTED_AT + MATCH_WINDOW + timedelta(hours=1)
    _run_with_resolver(resolver, started_at=run_three_started_at)

    active_ids = FireEventRepository(session_factory=sqlite_session_factory).get_active_fire_event_ids()
    assert run_one_event_id not in active_ids and leftover_id not in active_ids
    assert _event(sqlite_session_factory, leftover_id).updated_at == STARTED_AT
    assert _event(sqlite_session_factory, run_one_event_id).updated_at == run_three_started_at


def test_cleanup_resolves_through_lifecycle_service_with_scenario_start_time():
    class FakeRepository:
        def __init__(self):
            self.queries = []

        def get_active_event_ids_with_only_marked_evidence_updated_before(self, **kwargs):
            self.queries.append(kwargs)
            return (7, 9)

    class RecordingLifecycleService:
        def __init__(self):
            self.calls = []

        def resolve_event(self, fire_event_id, *, as_of):
            self.calls.append((fire_event_id, as_of))

    repository = FakeRepository()
    lifecycle_service = RecordingLifecycleService()

    _run_with_resolver(
        lambda as_of: resolve_stale_simulation_events(
            as_of, fire_event_repository=repository, lifecycle_service=lifecycle_service
        )
    )

    assert repository.queries == [
        dict(
            updated_before=STARTED_AT - MATCH_WINDOW,
            satellite_name=SIMULATED_SATELLITE,
            news_source_feed=SIMULATED_NEWS_SOURCE_FEED,
        )
    ]
    assert lifecycle_service.calls == [(7, STARTED_AT), (9, STARTED_AT)]


def test_stale_event_cleanup_runs_before_the_scenario_starts():
    order: list = []

    class RecordingService(FakeAutomaticService):
        def start(self, scenario, mode):
            order.append("start")
            super().start(scenario, mode)

    runner = DemoSimulationRunner(
        executor=FakeExecutor(),
        fire_danger_coordinator=NoOpFireDangerCoordinator(),
        fire_detection_coordinator=NoOpFireDetectionCoordinator(),
        operational_coordinator=NoOpOperationalCoordinator(),
        simulation_refresh_coordinator=NoOpRefreshCoordinator(),
        service=RecordingService(due_batches=[]),
        stale_simulation_event_resolver=lambda as_of: order.append(("cleanup", as_of)) or (),
    )
    runner.run(
        build_active_fire_scenario(seed=42),
        DemoSimulationRunConfig(mode=SimulationMode.AUTOMATIC),
        scenario_started_at=STARTED_AT,
    )

    assert order[:2] == [("cleanup", STARTED_AT), "start"]


def test_stale_event_cleanup_failure_does_not_abort_the_run():
    def failing_resolver(as_of):
        raise RuntimeError("database unavailable")

    result = _run_with_resolver(failing_resolver)

    assert result.events_executed == len(build_active_fire_scenario(seed=42).events)
