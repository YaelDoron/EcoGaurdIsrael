"""Reusable, UI-independent execution of a demo SimulationScenario.

Task A2: extracts the orchestration that `scripts/run_demo_simulation.py`'s
`run_manual`/`run_automatic`/`execute_and_report_event` used to own directly
(build the analysis stack, execute events in order, invoke the existing
coordinators, aggregate results) into one authoritative, reusable runner.

DemoSimulationRunner:
- does NOT parse CLI args (argparse.Namespace never appears here),
- does NOT call input() (manual-mode stepping is an injected callback -
  only the CLI owns interactive input),
- does NOT know FastAPI/HTTP,
- does NOT build/derive Fire Danger, Fire Detection, Severity, Spread,
  Routing, or GA logic itself - it only calls the existing production
  coordinators/agents, exactly as the old CLI script did,
- does NOT reset demo state (see src/simulation/demo_state_reset_service.py -
  that remains a fully separate concern; the caller decides when to reset,
  before ever constructing/running a scenario).

Two clocks are deliberately kept distinct throughout this module (see
DemoSimulationRunResult): `scenario.duration_seconds` / `event.offset_seconds`
describe the SIMULATED timeline; `wall_clock_elapsed_seconds` describes how
long real execution actually took. For a scenario like operations_demo,
these can differ by more than 2x, because each event can trigger real
Severity/Spread/routing/global-replanning work against a real database.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import time
from typing import Callable

from src.agents.analysis import (
    FireDangerAssessmentAgent,
    FireDetectionAgent,
    FireSeverityAssessmentAgent,
    FireSpreadPredictionAgent,
    ResponseTargetGenerationAgent,
)
from src.calculators.fire_danger.ffwi_calculator import FFWICalculator
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_severity.fire_severity_calculator import FireSeverityCalculator
from src.calculators.fire_spread import FireSpreadCalculator
from src.calculators.response_target import ResponseTargetCalculator
from src.repositories.fire_danger_assessment_repository import FireDangerAssessmentRepository
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_spread_prediction_repository import FireSpreadPredictionRepository
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_danger import FireDangerInputService
from src.services.fire_detection import FireDetectionEvidenceService
from src.services.fire_severity import FireSeverityInputService
from src.services.fire_spread import FireSpreadInputService
from src.services.operational import OperationalContextService
from src.services.operational_planning_refresh.operational_planning_refresh_production_factory import (
    build_operational_planning_refresh_coordinator,
)
from src.services.response_target import ResponseTargetInputService
from src.simulation.analysis import (
    SimulationFireDangerCoordinator,
    SimulationFireDangerResult,
    SimulationFireDetectionCoordinator,
    SimulationFireDetectionResult,
    SimulationFireSeverityCoordinator,
    SimulationFireSeverityResult,
    SimulationFireSpreadCoordinator,
    SimulationFireSpreadResult,
    SimulationOperationalCoordinator,
    SimulationRefreshCoordinator,
    SimulationRefreshResult,
    SimulationResponseTargetCoordinator,
    SimulationResponseTargetResult,
)
from src.simulation.simulated_incident import SimulatedIncident
from src.simulation.simulation_event import SimulationEvent, SimulationEventType
from src.simulation.simulation_event_executor import (
    SimulationEventExecutionResult,
    SimulationEventExecutor,
    simulation_event_timestamp,
)
from src.simulation.simulation_scenario import SimulationScenario
from src.simulation.simulation_scenario_service import SimulationMode, SimulationScenarioService

DEFAULT_POLL_INTERVAL_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Default production stack factories (moved here from scripts/run_demo_simulation.py
# so both the CLI and a future API can build the same coordinators from one
# place - re-exported by the CLI module for backward compatibility).
# ---------------------------------------------------------------------------


def build_fire_danger_coordinator() -> SimulationFireDangerCoordinator:
    """Build the simulation fire-danger analysis stack using shared repositories."""
    weather_repository = WeatherRepository()
    input_service = FireDangerInputService(weather_repository=weather_repository)
    calculator = FFWICalculator()
    assessment_repository = FireDangerAssessmentRepository()
    agent = FireDangerAssessmentAgent(
        input_service=input_service,
        calculator=calculator,
        repository=assessment_repository,
    )
    return SimulationFireDangerCoordinator(assessment_agent=agent)


def build_fire_detection_coordinator() -> SimulationFireDetectionCoordinator:
    """Build the simulation fire-detection analysis stack using shared repositories."""
    satellite_repository = SatelliteHotspotRepository()
    news_repository = NewsRepository()
    evidence_service = FireDetectionEvidenceService(
        satellite_repository=satellite_repository,
        news_repository=news_repository,
    )
    calculator = FireDetectionCalculator()
    fire_event_repository = FireEventRepository()
    agent = FireDetectionAgent(
        evidence_service=evidence_service,
        calculator=calculator,
        fire_event_repository=fire_event_repository,
        satellite_repository=satellite_repository,
        news_repository=news_repository,
    )
    return SimulationFireDetectionCoordinator(detection_agent=agent)


def build_operational_coordinator() -> SimulationOperationalCoordinator:
    """Build the simulation operational-context stack using shared repositories."""
    fire_station_repository = FireStationRepository()
    firefighting_resource_repository = FirefightingResourceRepository()
    operational_context_service = OperationalContextService(
        fire_station_repository=fire_station_repository,
        firefighting_resource_repository=firefighting_resource_repository,
    )
    return SimulationOperationalCoordinator(
        operational_context_service=operational_context_service,
        firefighting_resource_repository=firefighting_resource_repository,
    )


def build_fire_severity_coordinator() -> SimulationFireSeverityCoordinator:
    """Build the simulation fire-severity analysis stack using shared repositories."""
    fire_event_repository = FireEventRepository()
    input_service = FireSeverityInputService(fire_event_repository=fire_event_repository)
    calculator = FireSeverityCalculator()
    assessment_repository = FireSeverityAssessmentRepository()
    agent = FireSeverityAssessmentAgent(
        input_service=input_service,
        calculator=calculator,
        repository=assessment_repository,
    )
    return SimulationFireSeverityCoordinator(
        severity_agent=agent,
        fire_event_repository=fire_event_repository,
    )


def build_fire_spread_coordinator() -> SimulationFireSpreadCoordinator:
    """Build the simulation fire-spread prediction stack using shared repositories.

    Reuses the exact same production FireSpreadInputService/FireSpreadCalculator/
    FireSpreadPredictionRepository as the live pipeline -- no separate
    simulation-only spread mathematics.
    """
    input_service = FireSpreadInputService()
    calculator = FireSpreadCalculator()
    prediction_repository = FireSpreadPredictionRepository()
    agent = FireSpreadPredictionAgent(
        input_service=input_service,
        calculator=calculator,
        repository=prediction_repository,
    )
    return SimulationFireSpreadCoordinator(spread_agent=agent)


def build_response_target_coordinator() -> SimulationResponseTargetCoordinator:
    """Build the simulation response-target stack using the production agent."""
    input_service = ResponseTargetInputService()
    calculator = ResponseTargetCalculator()
    repository = ResponseTargetRepository()
    agent = ResponseTargetGenerationAgent(
        input_service=input_service,
        calculator=calculator,
        repository=repository,
    )
    return SimulationResponseTargetCoordinator(generation_agent=agent)


def build_simulation_refresh_coordinator(
    operational_coordinator: SimulationOperationalCoordinator | None = None,
) -> SimulationRefreshCoordinator:
    """Build the central US 4.4 -> US 5.4 simulation refresh coordinator.

    Delegates the entire operational-refresh-then-planning-refresh sequence
    to the same production OperationalPlanningRefreshCoordinator normal
    runtime uses - never re-derives the US4.4 -> US5.4 sequence, and never
    constructs RoutePlanningAgent/ResponseOptimizationAgent/
    BaselineComparisonService itself.
    """
    operational_planning_refresh_coordinator = build_operational_planning_refresh_coordinator()

    return SimulationRefreshCoordinator(
        operational_planning_refresh=operational_planning_refresh_coordinator,
        fire_event_repository=FireEventRepository(),
        operational_coordinator=operational_coordinator or build_operational_coordinator(),
    )


# ---------------------------------------------------------------------------
# Structured, ORM-free models
# ---------------------------------------------------------------------------


class DemoSimulationEventPhase(Enum):
    """Which point in one event's lifecycle a DemoSimulationEventProgress tick describes.

    Task A3: the original A2 callback fired only after an event finished,
    so a consumer (e.g. a status-polling API) could never distinguish
    "about to process event 8" from "still processing event 7" - both looked
    identical between ticks. EVENT_STARTED closes that gap with a minimal,
    additive change: the callback contract now fires twice per event
    (EVENT_STARTED right before execute_simulation_event, EVENT_COMPLETED
    right after), never once. `event_outcome`/`success` are only meaningful
    on EVENT_COMPLETED - see DemoSimulationEventProgress.
    """

    EVENT_STARTED = "event_started"
    EVENT_COMPLETED = "event_completed"


class DemoSimulationStatus(Enum):
    """Overall outcome of one DemoSimulationRunner.run() call.

    Event-level failures do NOT change this to FAILED - that preserves the
    pre-existing CLI semantics where one failed event is recorded and the
    run continues (see test_manual_mode_aggregates_failures_and_continues).
    FAILED is reserved for a run that could not produce a normal completion
    result at all; a setup/construction failure instead raises (see the
    module docstring and Part 8 of the Task A2 audit) rather than being
    reported as this status.
    """

    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


@dataclass(frozen=True)
class SimulationEventOutcome:
    """Everything computed for one executed event - the reusable orchestration
    result. This is domain/dataclass data (the same result objects the
    coordinators already return), never a raw SQLAlchemy/ORM object.

    CLI (and, later, API) renderers consume this for detailed output; it is
    intentionally richer than DemoSimulationEventProgress's canonical summary
    fields, since terminal diagnostics (Part 12) need the full sub-results.
    """

    event: SimulationEvent
    incident: SimulatedIncident
    event_timestamp: datetime
    execution_result: SimulationEventExecutionResult
    fire_danger_result: SimulationFireDangerResult | None = None
    fire_detection_result: SimulationFireDetectionResult | None = None
    refresh_result: SimulationRefreshResult | None = None
    fire_severity_result: SimulationFireSeverityResult | None = None
    fire_spread_result: SimulationFireSpreadResult | None = None
    response_target_result: SimulationResponseTargetResult | None = None
    operational_context_scrambled: bool = False
    depleted_resource_count: int = 0
    affected_fire_event_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class DemoSimulationEventProgress:
    """One structured progress tick for one event.

    Fires exactly twice per event, in this order:

    1. `phase=EVENT_STARTED`, emitted immediately before the event is
       executed. `success` and `event_outcome` are `None` - the event
       hasn't run yet. `events_completed` still reflects only PRIOR events -
       this event is not yet one of them.
    2. `phase=EVENT_COMPLETED`, emitted immediately after the event
       finishes. `success` and `event_outcome` are populated.
       `events_completed` now includes this event.

    A consumer that only cares about finished events (e.g. the pre-A3 CLI
    renderer) filters on `phase is DemoSimulationEventPhase.EVENT_COMPLETED`
    and ignores EVENT_STARTED ticks.

    Contains only plain values and existing domain-level result objects
    (via `event_outcome`) - never raw ORM rows. `event_outcome` is the
    CLI-rendering detail; the remaining fields are the canonical summary a
    future API can safely serialize on their own.
    """

    phase: DemoSimulationEventPhase
    event_index: int
    events_total: int
    events_completed: int
    incident_id: str
    event_type: SimulationEventType
    timestamp_offset_sec: int
    started_at: datetime
    now: datetime
    success: bool | None = None
    event_outcome: SimulationEventOutcome | None = None
    message: str | None = None


@dataclass(frozen=True)
class DemoSimulationEventSummary:
    """Per-event execution summary retained in the final result.

    Preserves the existing generated/saved/duplicates/failed semantics from
    SimulationEventExecutionResult unchanged - see SimulationEventExecutor.
    """

    event_index: int
    incident_id: str
    event_type: SimulationEventType
    timestamp_offset_sec: int
    success: bool
    generated_count: int
    saved_count: int
    duplicates_skipped: int
    failed_count: int
    error_message: str | None


@dataclass(frozen=True)
class DemoSimulationRunResult:
    """Structured final result of one DemoSimulationRunner.run() call.

    Simulated-timeline duration and real wall-clock duration are kept as
    two clearly separate fields on purpose (Task A2, Part 5) - for
    operations_demo, simulation_duration_seconds is ~120 while
    wall_clock_elapsed_seconds can exceed 300+, because each event can
    trigger real Severity/Spread/routing/global-replanning work. Neither
    field is an estimate of the other.
    """

    status: DemoSimulationStatus
    simulation_started_at: datetime
    simulation_completed_at: datetime
    simulation_duration_seconds: int
    wall_clock_elapsed_seconds: float
    events_total: int
    events_executed: int
    events_succeeded: int
    events_failed: int
    incident_ids: tuple[str, ...]
    event_summaries: tuple[DemoSimulationEventSummary, ...] = ()


@dataclass(frozen=True)
class DemoSimulationRunConfig:
    """Execution configuration for one DemoSimulationRunner.run() call.

    Deliberately does not carry a seed: the seed belongs to scenario
    generation (see src.simulation.simulation_scenario builders), not to
    executing an already-built SimulationScenario - this runner accepts a
    fully constructed SimulationScenario, never a preset name/seed.
    """

    mode: SimulationMode
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    sleep_fn: Callable[[float], None] = time.sleep
    before_event: Callable[[SimulationEvent], None] | None = None
    should_stop: Callable[[], bool] | None = None


# ---------------------------------------------------------------------------
# Shared per-event orchestration (the single authoritative implementation)
# ---------------------------------------------------------------------------


def execute_simulation_event(
    *,
    scenario: SimulationScenario,
    event: SimulationEvent,
    scenario_started_at: datetime,
    executor: SimulationEventExecutor,
    fire_danger_coordinator: SimulationFireDangerCoordinator | None = None,
    fire_detection_coordinator: SimulationFireDetectionCoordinator | None = None,
    operational_coordinator: SimulationOperationalCoordinator | None = None,
    fire_severity_coordinator: SimulationFireSeverityCoordinator | None = None,
    simulation_refresh_coordinator: SimulationRefreshCoordinator | None = None,
    fire_spread_coordinator_factory: Callable[[], SimulationFireSpreadCoordinator] | None = None,
    response_target_coordinator_factory: Callable[[], SimulationResponseTargetCoordinator] | None = None,
) -> SimulationEventOutcome:
    """Execute one simulation event through the real production pipeline.

    This is the exact orchestration `scripts/run_demo_simulation.py`'s
    `execute_and_report_event` used to perform inline, minus printing: run
    the executor, then Fire Danger (weather events), then Fire Detection
    (satellite/news events, plus resource scrambling on a newly created
    FireEvent), then either the central SimulationRefreshCoordinator or the
    legacy Severity->Spread pair, then Response Targets for any affected
    FireEvent. No business algorithm lives here - only calls into the
    existing coordinators/agents.

    `fire_spread_coordinator_factory`/`response_target_coordinator_factory`
    are resolved lazily, only inside the legacy severity/spread branch or
    when a FireEvent was actually affected - mirroring the original code's
    lazy `get_fire_spread_coordinator()`/`get_response_target_coordinator()`
    singletons without this reusable function owning any singleton itself.
    """
    incident = scenario.get_incident(event.incident_id)
    event_timestamp = simulation_event_timestamp(scenario_started_at, event)
    affected_fire_event_ids: list[int] = []

    execution_result = executor.execute(scenario=scenario, event=event, event_timestamp=event_timestamp)

    fire_danger_result: SimulationFireDangerResult | None = None
    if fire_danger_coordinator is not None:
        fire_danger_result = fire_danger_coordinator.handle_event(
            scenario=scenario,
            event=event,
            execution_result=execution_result,
            event_timestamp=event_timestamp,
        )

    fire_detection_result: SimulationFireDetectionResult | None = None
    operational_context_scrambled = False
    depleted_resource_count = 0
    if fire_detection_coordinator is not None and event.event_type.name in {"SATELLITE", "NEWS"}:
        fire_detection_result = fire_detection_coordinator.handle_event(
            scenario=scenario,
            event=event,
            execution_result=execution_result,
            event_timestamp=event_timestamp,
        )
        if fire_detection_result.triggered:
            detection_result = fire_detection_result.detection_result
            if detection_result is not None and detection_result.success:
                affected_fire_event_ids.extend(detection_result.event_ids)
            if operational_coordinator is not None:
                should_scramble = (
                    detection_result is not None
                    and detection_result.success
                    and detection_result.events_created > 0
                )
                if should_scramble:
                    operational_context_scrambled = True
                    depleted_resource_count = _scramble_resources_for_new_fire_events(
                        incident, operational_coordinator
                    )

    detection_result_for_downstream = (
        fire_detection_result.detection_result
        if fire_detection_result is not None and fire_detection_result.triggered
        else None
    )

    refresh_result: SimulationRefreshResult | None = None
    fire_severity_result: SimulationFireSeverityResult | None = None
    fire_spread_result: SimulationFireSpreadResult | None = None
    if simulation_refresh_coordinator is not None:
        refresh_result = simulation_refresh_coordinator.handle_event(
            scenario=scenario,
            event=event,
            execution_result=execution_result,
            event_timestamp=event_timestamp,
            detection_result=detection_result_for_downstream,
        )
    elif fire_severity_coordinator is not None:
        fire_severity_result = fire_severity_coordinator.handle_event(
            scenario=scenario,
            event=event,
            execution_result=execution_result,
            event_timestamp=event_timestamp,
            detection_result=detection_result_for_downstream,
        )
        if fire_severity_result.triggered:
            affected_fire_event_ids.extend(
                stored.assessment.fire_event_id for stored in fire_severity_result.assessment_results
            )
            affected_fire_event_ids.extend(fire_severity_result.failed_fire_event_ids)
            spread_coordinator = (fire_spread_coordinator_factory or build_fire_spread_coordinator)()
            fire_spread_result = spread_coordinator.handle_severity_result(
                severity_result=fire_severity_result,
                event_timestamp=event_timestamp,
            )
            if fire_spread_result.triggered:
                affected_fire_event_ids.extend(
                    stored.prediction.fire_event_id for stored in fire_spread_result.prediction_results
                )
                affected_fire_event_ids.extend(fire_spread_result.failed_fire_event_ids)

    response_target_result: SimulationResponseTargetResult | None = None
    if affected_fire_event_ids:
        target_coordinator = (response_target_coordinator_factory or build_response_target_coordinator)()
        response_target_result = target_coordinator.generate_for_fire_events(
            fire_event_ids=affected_fire_event_ids,
            as_of=event_timestamp,
        )

    return SimulationEventOutcome(
        event=event,
        incident=incident,
        event_timestamp=event_timestamp,
        execution_result=execution_result,
        fire_danger_result=fire_danger_result,
        fire_detection_result=fire_detection_result,
        refresh_result=refresh_result,
        fire_severity_result=fire_severity_result,
        fire_spread_result=fire_spread_result,
        response_target_result=response_target_result,
        operational_context_scrambled=operational_context_scrambled,
        depleted_resource_count=depleted_resource_count,
        affected_fire_event_ids=tuple(affected_fire_event_ids),
    )


def _scramble_resources_for_new_fire_events(
    incident: SimulatedIncident,
    operational_coordinator: SimulationOperationalCoordinator,
) -> int:
    """Deplete resource availability near an incident that just became an active fire.

    Caller has already confirmed detection created a new FireEvent. Returns
    the number of resources actually changed (may legitimately be 0, e.g. no
    stations found nearby) - the caller still reports that the attempt was made.
    """
    depleted_resources = operational_coordinator.scramble_resource_availability(
        incident.location.latitude,
        incident.location.longitude,
    )
    return len(depleted_resources)


# ---------------------------------------------------------------------------
# The reusable runner
# ---------------------------------------------------------------------------


class DemoSimulationRunner:
    """Execute a fully-constructed SimulationScenario through the real
    production analysis/refresh stack, with structured progress and a
    structured final result.

    Owns no argparse, no FastAPI, no HTTP, no UI concept, and no hidden
    module-level singleton state - each instance builds and holds its own
    coordinators (or accepts injected ones), matching Part 15's requirement
    that the runner instance own its execution state. Does not hold a
    SQLAlchemy Session as a member; each coordinator/repository retains its
    own existing session-per-call ownership pattern.
    """

    def __init__(
        self,
        *,
        executor: SimulationEventExecutor | None = None,
        fire_danger_coordinator: SimulationFireDangerCoordinator | None = None,
        fire_detection_coordinator: SimulationFireDetectionCoordinator | None = None,
        operational_coordinator: SimulationOperationalCoordinator | None = None,
        fire_severity_coordinator: SimulationFireSeverityCoordinator | None = None,
        simulation_refresh_coordinator: SimulationRefreshCoordinator | None = None,
        fire_spread_coordinator: SimulationFireSpreadCoordinator | None = None,
        response_target_coordinator: SimulationResponseTargetCoordinator | None = None,
        fire_spread_coordinator_factory: Callable[[], SimulationFireSpreadCoordinator] | None = None,
        response_target_coordinator_factory: Callable[[], SimulationResponseTargetCoordinator] | None = None,
        service: SimulationScenarioService | None = None,
        clock_fn: Callable[[], float] = time.monotonic,
        wall_clock_now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._executor = executor or SimulationEventExecutor()
        self._fire_danger_coordinator = fire_danger_coordinator or build_fire_danger_coordinator()
        self._fire_detection_coordinator = fire_detection_coordinator or build_fire_detection_coordinator()
        self._operational_coordinator = operational_coordinator or build_operational_coordinator()

        # Preserves the pre-existing dual-path resolution exactly: when
        # neither is supplied, build the central refresh coordinator (the
        # production default); when a caller supplies fire_severity_coordinator
        # without simulation_refresh_coordinator, honor the legacy
        # Severity->Spread path instead of silently overriding it with a real
        # refresh coordinator (which would bypass an injected fake and reach
        # for a real database in tests that only intended to exercise Severity).
        if simulation_refresh_coordinator is None and fire_severity_coordinator is None:
            simulation_refresh_coordinator = build_simulation_refresh_coordinator(self._operational_coordinator)
        elif simulation_refresh_coordinator is None:
            fire_severity_coordinator = fire_severity_coordinator or build_fire_severity_coordinator()
        self._simulation_refresh_coordinator = simulation_refresh_coordinator
        self._fire_severity_coordinator = fire_severity_coordinator

        # Resolved lazily on first actual use, exactly like the pre-Task-A2
        # CLI's own get_fire_spread_coordinator()/get_response_target_coordinator():
        # most events never affect a FireEvent, so most runs never need
        # these at all. Building them eagerly here would needlessly touch
        # real repositories/DB wiring on every run() call, including tests
        # that only exercise Fire Danger/Fire Detection.
        #
        # A caller-supplied factory (e.g. the CLI's own lazily-cached
        # get_fire_spread_coordinator/get_response_target_coordinator
        # singletons) takes priority, so a caller that already owns a lazy
        # singleton - and monkeypatches it in tests - keeps full control;
        # otherwise this runner instance caches its own lazily-built
        # coordinator (instance-scoped state, never a module-level global).
        self._fire_spread_coordinator = fire_spread_coordinator
        self._response_target_coordinator = response_target_coordinator
        self._fire_spread_coordinator_factory = fire_spread_coordinator_factory
        self._response_target_coordinator_factory = response_target_coordinator_factory
        self._service = service
        self._clock_fn = clock_fn
        self._wall_clock_now_fn = wall_clock_now_fn

    def _get_fire_spread_coordinator(self) -> SimulationFireSpreadCoordinator:
        if self._fire_spread_coordinator_factory is not None:
            return self._fire_spread_coordinator_factory()
        if self._fire_spread_coordinator is None:
            self._fire_spread_coordinator = build_fire_spread_coordinator()
        return self._fire_spread_coordinator

    def _get_response_target_coordinator(self) -> SimulationResponseTargetCoordinator:
        if self._response_target_coordinator_factory is not None:
            return self._response_target_coordinator_factory()
        if self._response_target_coordinator is None:
            self._response_target_coordinator = build_response_target_coordinator()
        return self._response_target_coordinator

    def run(
        self,
        scenario: SimulationScenario,
        config: DemoSimulationRunConfig,
        scenario_started_at: datetime | None = None,
        on_progress: Callable[[DemoSimulationEventProgress], None] | None = None,
    ) -> DemoSimulationRunResult:
        """Execute every event in `scenario` and return a structured result.

        This is the single authoritative execution loop: both automatic and
        manual mode are handled here, branching only on `config.mode`. The
        CLI (and, later, an API) must not maintain a second copy of this
        loop - they only supply configuration/callbacks.
        """
        service = self._service or SimulationScenarioService()
        scenario_started_at = scenario_started_at or self._wall_clock_now_fn()
        wall_clock_start = self._clock_fn()

        service.start(scenario, mode=config.mode)

        event_summaries: list[DemoSimulationEventSummary] = []
        events_total = len(scenario.events)

        def handle_event(event: SimulationEvent, event_index: int) -> None:
            if on_progress is not None:
                on_progress(
                    DemoSimulationEventProgress(
                        phase=DemoSimulationEventPhase.EVENT_STARTED,
                        event_index=event_index,
                        events_total=events_total,
                        events_completed=len(event_summaries),
                        incident_id=event.incident_id,
                        event_type=event.event_type,
                        timestamp_offset_sec=event.offset_seconds,
                        started_at=scenario_started_at,
                        now=self._wall_clock_now_fn(),
                    )
                )

            outcome = execute_simulation_event(
                scenario=scenario,
                event=event,
                scenario_started_at=scenario_started_at,
                executor=self._executor,
                fire_danger_coordinator=self._fire_danger_coordinator,
                fire_detection_coordinator=self._fire_detection_coordinator,
                operational_coordinator=self._operational_coordinator,
                fire_severity_coordinator=self._fire_severity_coordinator,
                simulation_refresh_coordinator=self._simulation_refresh_coordinator,
                fire_spread_coordinator_factory=self._get_fire_spread_coordinator,
                response_target_coordinator_factory=self._get_response_target_coordinator,
            )
            result = outcome.execution_result
            event_summaries.append(
                DemoSimulationEventSummary(
                    event_index=event_index,
                    incident_id=event.incident_id,
                    event_type=event.event_type,
                    timestamp_offset_sec=event.offset_seconds,
                    success=result.success,
                    generated_count=result.generated_count,
                    saved_count=result.saved_count,
                    duplicates_skipped=result.duplicates_skipped,
                    failed_count=result.failed_count,
                    error_message=result.error_message,
                )
            )
            if on_progress is not None:
                on_progress(
                    DemoSimulationEventProgress(
                        phase=DemoSimulationEventPhase.EVENT_COMPLETED,
                        event_index=event_index,
                        events_total=events_total,
                        events_completed=len(event_summaries),
                        incident_id=event.incident_id,
                        event_type=event.event_type,
                        timestamp_offset_sec=event.offset_seconds,
                        success=result.success,
                        started_at=scenario_started_at,
                        now=self._wall_clock_now_fn(),
                        event_outcome=outcome,
                    )
                )

        event_index = 0
        if config.mode is SimulationMode.MANUAL:
            while not service.is_finished:
                if config.before_event is not None:
                    upcoming_event = service.current_scenario.events[service.current_event_index]
                    config.before_event(upcoming_event)
                if config.should_stop is not None and config.should_stop():
                    break
                event = service.advance()
                if event is None:
                    break
                handle_event(event, event_index)
                event_index += 1
        else:
            while not service.is_finished:
                if config.should_stop is not None and config.should_stop():
                    break
                due_events = service.get_due_events()
                for event in due_events:
                    handle_event(event, event_index)
                    event_index += 1
                if not service.is_finished:
                    config.sleep_fn(config.poll_interval_seconds)

        wall_clock_elapsed_seconds = self._clock_fn() - wall_clock_start
        simulation_completed_at = self._wall_clock_now_fn()
        events_succeeded = sum(1 for summary in event_summaries if summary.success)
        events_failed = len(event_summaries) - events_succeeded
        status = (
            DemoSimulationStatus.COMPLETED
            if events_failed == 0
            else DemoSimulationStatus.COMPLETED_WITH_ERRORS
        )

        return DemoSimulationRunResult(
            status=status,
            simulation_started_at=scenario_started_at,
            simulation_completed_at=simulation_completed_at,
            simulation_duration_seconds=scenario.duration_seconds,
            wall_clock_elapsed_seconds=wall_clock_elapsed_seconds,
            events_total=events_total,
            events_executed=len(event_summaries),
            events_succeeded=events_succeeded,
            events_failed=events_failed,
            incident_ids=tuple(incident.incident_id for incident in scenario.incidents),
            event_summaries=tuple(event_summaries),
        )
