"""Run an EcoGuard demo simulation from the command line.

Example:
    python -m scripts.run_demo_simulation --scenario active_fire --location carmel --mode manual --seed 42
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Callable, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import settings
from src.database.connection import DatabaseConfigurationError, init_db
from src.agents.analysis import (
    FireDangerAssessmentAgent,
    FireDetectionAgent,
    FireSeverityAssessmentAgent,
    ResponseTargetGenerationAgent,
)
from src.agents.analysis import FireSpreadPredictionAgent
from src.calculators.fire_danger.ffwi_calculator import FFWICalculator
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.response_target import ResponseTargetCalculator
from src.calculators.fire_severity.fire_severity_calculator import FireSeverityCalculator
from src.calculators.fire_spread import FireSpreadCalculator
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_danger_assessment_repository import FireDangerAssessmentRepository
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_spread_prediction_repository import FireSpreadPredictionRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_danger import FireDangerInputService
from src.services.fire_detection import FireDetectionEvidenceService
from src.services.operational import OperationalContextService
from src.services.operational_planning_refresh.operational_planning_refresh_production_factory import (
    build_operational_planning_refresh_coordinator,
)
from src.services.operational_refresh import OperationalRefreshResult
from src.services.response_planning.planning_refresh_result import PlanningRefreshResult
from src.services.response_planning.response_plan_details_service import ResponsePlanDetailsService
from src.services.response_target import ResponseTargetInputService
from src.services.fire_severity import FireSeverityInputService
from src.services.fire_spread import FireSpreadInputService
from src.simulation import (
    SIMULATION_LOCATIONS,
    ScenarioType,
    SimulatedIncident,
    SimulationEvent,
    SimulationEventExecutionResult,
    SimulationEventExecutor,
    SimulationMode,
    SimulationOperationalCoordinator,
    SimulationScenario,
    SimulationScenarioService,
    SimulationFireDangerCoordinator,
    SimulationFireDangerResult,
    SimulationFireDetectionCoordinator,
    SimulationFireDetectionResult,
    SimulationFireSeverityCoordinator,
    SimulationFireSeverityResult,
    SimulationResponseTargetCoordinator,
    SimulationResponseTargetResult,
    SimulationRefreshCoordinator,
    SimulationRefreshResult,
    build_carmel_golan_active_fire_scenario,
    build_active_fire_resource_refresh_scenario,
    build_scenario,
    get_simulation_location,
    simulation_event_timestamp,
)
from src.simulation import SimulationFireSpreadCoordinator, SimulationFireSpreadResult

DEFAULT_SEED = 42
DEFAULT_MODE = "manual"
DEFAULT_POLL_INTERVAL_SECONDS = 0.5
SUPPORTED_PRESETS = ("carmel_golan_active_fire", "active_fire_resource_refresh")


@dataclass
class RunSummary:
    events_executed: int = 0
    successful_events: int = 0
    failed_events: int = 0
    generated_records: int = 0
    saved_records: int = 0
    duplicates_skipped: int = 0
    failed_records: int = 0

    def add(self, result: SimulationEventExecutionResult) -> None:
        self.events_executed += 1
        if result.success:
            self.successful_events += 1
        else:
            self.failed_events += 1
        self.generated_records += result.generated_count
        self.saved_records += result.saved_count
        self.duplicates_skipped += result.duplicates_skipped
        self.failed_records += result.failed_count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EcoGuard SIMULATION MODE events and persist generated domain data."
    )
    parser.add_argument(
        "--scenario",
        choices=[scenario_type.value for scenario_type in ScenarioType],
        help="Single-incident scenario to run.",
    )
    parser.add_argument(
        "--location",
        choices=sorted(SIMULATION_LOCATIONS),
        help="Predefined simulation location for single-incident scenarios.",
    )
    parser.add_argument(
        "--preset",
        choices=SUPPORTED_PRESETS,
        help="Predefined multi-incident scenario preset.",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in SimulationMode],
        default=DEFAULT_MODE,
        help=f"Timeline execution mode. Default: {DEFAULT_MODE}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic simulation seed. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help=(
            "Automatic-mode polling interval in seconds. "
            f"Default: {DEFAULT_POLL_INTERVAL_SECONDS}."
        ),
    )
    args = parser.parse_args(argv)

    if args.poll_interval <= 0:
        parser.error("--poll-interval must be greater than 0.")

    if args.preset:
        if args.scenario or args.location:
            parser.error("--preset cannot be combined with --scenario or --location.")
    else:
        if not args.scenario:
            parser.error("--scenario is required unless --preset is supplied.")
        if not args.location:
            parser.error("--location is required unless --preset is supplied.")

    return args


def build_scenario_from_args(args: argparse.Namespace) -> SimulationScenario:
    if args.preset == "carmel_golan_active_fire":
        return build_carmel_golan_active_fire_scenario(seed=args.seed)
    if args.preset == "active_fire_resource_refresh":
        return build_active_fire_resource_refresh_scenario(seed=args.seed)
    if args.preset:
        raise ValueError(f"Unsupported preset: {args.preset!r}")

    scenario_type = ScenarioType(args.scenario)
    location = get_simulation_location(args.location)
    return build_scenario(scenario_type=scenario_type, location=location, seed=args.seed)


def initialize_database() -> None:
    if not settings.DATABASE_URL:
        raise DatabaseConfigurationError("DATABASE_URL is not configured; cannot run persistence demo.")
    init_db()


def run_manual(
    scenario: SimulationScenario,
    executor: SimulationEventExecutor | None = None,
    fire_danger_coordinator: SimulationFireDangerCoordinator | None = None,
    fire_detection_coordinator: SimulationFireDetectionCoordinator | None = None,
    operational_coordinator: SimulationOperationalCoordinator | None = None,
    fire_severity_coordinator: SimulationFireSeverityCoordinator | None = None,
    simulation_refresh_coordinator: SimulationRefreshCoordinator | None = None,
    service: SimulationScenarioService | None = None,
    scenario_started_at: datetime | None = None,
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> RunSummary:
    executor = executor or SimulationEventExecutor()
    fire_danger_coordinator = fire_danger_coordinator or build_fire_danger_coordinator()
    fire_detection_coordinator = fire_detection_coordinator or build_fire_detection_coordinator()
    operational_coordinator = operational_coordinator or build_operational_coordinator()
    if simulation_refresh_coordinator is None and fire_severity_coordinator is None:
        simulation_refresh_coordinator = build_simulation_refresh_coordinator(operational_coordinator)
    elif simulation_refresh_coordinator is None:
        fire_severity_coordinator = fire_severity_coordinator or build_fire_severity_coordinator()
    service = service or SimulationScenarioService()
    scenario_started_at = scenario_started_at or datetime.now(timezone.utc)
    summary = RunSummary()

    service.start(scenario, mode=SimulationMode.MANUAL)
    print_startup_summary(scenario, SimulationMode.MANUAL, scenario_started_at, output)

    while not service.is_finished:
        try:
            input_func("Press ENTER to execute next event...")
        except EOFError:
            print("No input available; continuing with next event.", file=output)

        event = service.advance()
        if event is None:
            break
        result = execute_and_report_event(
            scenario,
            event,
            scenario_started_at,
            executor,
            output,
            fire_danger_coordinator,
            fire_detection_coordinator,
            operational_coordinator,
            fire_severity_coordinator,
            simulation_refresh_coordinator,
        )
        summary.add(result)

    print_run_summary(summary, output)
    return summary


def run_automatic(
    scenario: SimulationScenario,
    executor: SimulationEventExecutor | None = None,
    fire_danger_coordinator: SimulationFireDangerCoordinator | None = None,
    fire_detection_coordinator: SimulationFireDetectionCoordinator | None = None,
    operational_coordinator: SimulationOperationalCoordinator | None = None,
    fire_severity_coordinator: SimulationFireSeverityCoordinator | None = None,
    simulation_refresh_coordinator: SimulationRefreshCoordinator | None = None,
    service: SimulationScenarioService | None = None,
    scenario_started_at: datetime | None = None,
    sleep_func: Callable[[float], None] = time.sleep,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    output: TextIO = sys.stdout,
) -> RunSummary:
    executor = executor or SimulationEventExecutor()
    fire_danger_coordinator = fire_danger_coordinator or build_fire_danger_coordinator()
    fire_detection_coordinator = fire_detection_coordinator or build_fire_detection_coordinator()
    operational_coordinator = operational_coordinator or build_operational_coordinator()
    if simulation_refresh_coordinator is None and fire_severity_coordinator is None:
        simulation_refresh_coordinator = build_simulation_refresh_coordinator(operational_coordinator)
    elif simulation_refresh_coordinator is None:
        fire_severity_coordinator = fire_severity_coordinator or build_fire_severity_coordinator()
    service = service or SimulationScenarioService()
    scenario_started_at = scenario_started_at or datetime.now(timezone.utc)
    summary = RunSummary()

    service.start(scenario, mode=SimulationMode.AUTOMATIC)
    print_startup_summary(scenario, SimulationMode.AUTOMATIC, scenario_started_at, output)

    while not service.is_finished:
        due_events = service.get_due_events()
        for event in due_events:
            result = execute_and_report_event(
                scenario,
                event,
                scenario_started_at,
                executor,
                output,
                fire_danger_coordinator,
                fire_detection_coordinator,
                operational_coordinator,
                fire_severity_coordinator,
                simulation_refresh_coordinator,
            )
            summary.add(result)

        if not service.is_finished:
            sleep_func(poll_interval_seconds)

    print_run_summary(summary, output)
    return summary


def execute_and_report_event(
    scenario: SimulationScenario,
    event: SimulationEvent,
    scenario_started_at: datetime,
    executor: SimulationEventExecutor,
    output: TextIO = sys.stdout,
    fire_danger_coordinator: SimulationFireDangerCoordinator | None = None,
    fire_detection_coordinator: SimulationFireDetectionCoordinator | None = None,
    operational_coordinator: SimulationOperationalCoordinator | None = None,
    fire_severity_coordinator: SimulationFireSeverityCoordinator | None = None,
    simulation_refresh_coordinator: SimulationRefreshCoordinator | None = None,
) -> SimulationEventExecutionResult:
    incident = scenario.get_incident(event.incident_id)
    event_timestamp = simulation_event_timestamp(scenario_started_at, event)
    affected_fire_event_ids: list[int] = []

    print(
        f"[T+{event.offset_seconds}s] {event.event_type.value.upper()} | "
        f"{event.incident_id} | {incident.location.name}",
        file=output,
    )
    result = executor.execute(scenario=scenario, event=event, event_timestamp=event_timestamp)
    print(format_execution_result(result), file=output)
    if result.error_message:
        print(f"Error: {result.error_message}", file=output)
    if fire_danger_coordinator is not None:
        fire_danger_result = fire_danger_coordinator.handle_event(
            scenario=scenario,
            event=event,
            execution_result=result,
            event_timestamp=event_timestamp,
        )
        if fire_danger_result.triggered:
            print_fire_danger_result(fire_danger_result, output)
    if fire_detection_coordinator is not None and event.event_type.name in {"SATELLITE", "NEWS"}:
        fire_detection_result = fire_detection_coordinator.handle_event(
            scenario=scenario,
            event=event,
            execution_result=result,
            event_timestamp=event_timestamp,
        )
        if fire_detection_result.triggered:
            print_fire_detection_result(fire_detection_result, output)
            detection_result = fire_detection_result.detection_result
            if detection_result is not None and detection_result.success:
                affected_fire_event_ids.extend(detection_result.event_ids)
            if operational_coordinator is not None:
                scramble_resources_for_new_fire_events(
                    fire_detection_result,
                    incident,
                    operational_coordinator,
                    output,
                )
    else:
        fire_detection_result = None

    if simulation_refresh_coordinator is not None:
        refresh_result = simulation_refresh_coordinator.handle_event(
            scenario=scenario,
            event=event,
            execution_result=result,
            event_timestamp=event_timestamp,
            detection_result=(
                fire_detection_result.detection_result
                if fire_detection_result is not None and fire_detection_result.triggered
                else None
            ),
        )
        if refresh_result.triggered:
            print_simulation_refresh_result(refresh_result, output)
    elif fire_severity_coordinator is not None:
        severity_result = fire_severity_coordinator.handle_event(
            scenario=scenario,
            event=event,
            execution_result=result,
            event_timestamp=event_timestamp,
            detection_result=(
                fire_detection_result.detection_result
                if fire_detection_result is not None and fire_detection_result.triggered
                else None
            ),
        )
        if severity_result.triggered:
            print_fire_severity_result(severity_result, output)
            affected_fire_event_ids.extend(
                stored.assessment.fire_event_id for stored in severity_result.assessment_results
            )
            affected_fire_event_ids.extend(severity_result.failed_fire_event_ids)
            spread_result = get_fire_spread_coordinator().handle_severity_result(
                severity_result=severity_result,
                event_timestamp=event_timestamp,
            )
            if spread_result.triggered:
                print_fire_spread_result(spread_result, output)
                affected_fire_event_ids.extend(
                    stored.prediction.fire_event_id for stored in spread_result.prediction_results
                )
                affected_fire_event_ids.extend(spread_result.failed_fire_event_ids)
    if affected_fire_event_ids:
        response_target_result = get_response_target_coordinator().generate_for_fire_events(
            fire_event_ids=affected_fire_event_ids,
            as_of=event_timestamp,
        )
        if response_target_result.triggered:
            print_response_target_result(response_target_result, output)
    print("", file=output)
    return result


def scramble_resources_for_new_fire_events(
    fire_detection_result: SimulationFireDetectionResult,
    incident: SimulatedIncident,
    operational_coordinator: SimulationOperationalCoordinator,
    output: TextIO = sys.stdout,
) -> None:
    """Deplete resource availability near an incident once it becomes an active fire.

    Only fires when detection actually created a new fire event (not merely
    updated an existing one or found no event), so the operational context
    is scrambled once, at the moment an incident location genuinely becomes
    an active wildfire.
    """
    detection_result = fire_detection_result.detection_result
    if detection_result is None or not detection_result.success or detection_result.events_created <= 0:
        return

    depleted_resources = operational_coordinator.scramble_resource_availability(
        incident.location.latitude,
        incident.location.longitude,
    )
    print("OPERATIONAL CONTEXT", file=output)
    print(f"depleted_resources={len(depleted_resources)}", file=output)


def format_execution_result(result: SimulationEventExecutionResult) -> str:
    return (
        f"generated={result.generated_count} "
        f"saved={result.saved_count} "
        f"duplicates={result.duplicates_skipped} "
        f"failed={result.failed_count} "
        f"success={result.success}"
    )


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
    runtime uses (build_operational_planning_refresh_coordinator) - this
    script does not construct RoutePlanningAgent, ResponseOptimizationAgent,
    or BaselineComparisonService itself, and does not re-derive the US4.4 ->
    US5.4 sequence.
    """
    operational_planning_refresh_coordinator = build_operational_planning_refresh_coordinator()

    return SimulationRefreshCoordinator(
        operational_planning_refresh=operational_planning_refresh_coordinator,
        fire_event_repository=FireEventRepository(),
        operational_coordinator=operational_coordinator or build_operational_coordinator(),
    )


_fire_spread_coordinator: SimulationFireSpreadCoordinator | None = None
_response_target_coordinator: SimulationResponseTargetCoordinator | None = None
_response_plan_details_service: ResponsePlanDetailsService | None = None


def get_fire_spread_coordinator() -> SimulationFireSpreadCoordinator:
    """Return the fire-spread simulation coordinator, built lazily on first use.

    Kept as a lazily-built module-level accessor (mirroring
    src/database/connection.py's get_engine()/get_session_factory() pattern)
    rather than a new parameter on run_manual/run_automatic/
    execute_and_report_event, so this teammate-owned script's existing
    function signatures never need to change for User Story 4.2. Tests that
    need a fake coordinator monkeypatch this function directly.
    """
    global _fire_spread_coordinator
    if _fire_spread_coordinator is None:
        _fire_spread_coordinator = build_fire_spread_coordinator()
    return _fire_spread_coordinator


def get_response_target_coordinator() -> SimulationResponseTargetCoordinator:
    """Return the response-target simulation coordinator, built lazily on first use."""
    global _response_target_coordinator
    if _response_target_coordinator is None:
        _response_target_coordinator = build_response_target_coordinator()
    return _response_target_coordinator


def get_response_plan_details_service() -> ResponsePlanDetailsService:
    """Return the US 5.5 read service used to enrich planning-refresh output, built lazily."""
    global _response_plan_details_service
    if _response_plan_details_service is None:
        _response_plan_details_service = ResponsePlanDetailsService()
    return _response_plan_details_service


def print_fire_danger_result(
    fire_danger_result: SimulationFireDangerResult,
    output: TextIO = sys.stdout,
) -> None:
    assessment_result = fire_danger_result.assessment_result
    print("FIRE DANGER ASSESSMENT", file=output)
    if assessment_result is None or not assessment_result.success:
        message = (
            assessment_result.error_message
            if assessment_result is not None and assessment_result.error_message
            else "Fire-danger assessment failed."
        )
        print("status=ERROR", file=output)
        print(f"message={message}", file=output)
        return

    assessment = assessment_result.assessment
    print(f"area={assessment.area_name}", file=output)
    print(f"status={assessment.status.name}", file=output)
    print(f"score={_format_optional_score(assessment.score)}", file=output)
    print(f"level={assessment.level.name if assessment.level is not None else '-'}", file=output)
    print(f"assessment_id={assessment_result.stored_assessment_id}", file=output)


def print_fire_detection_result(
    fire_detection_result: SimulationFireDetectionResult,
    output: TextIO = sys.stdout,
) -> None:
    detection_result = fire_detection_result.detection_result
    print("FIRE DETECTION", file=output)
    if detection_result is None or not detection_result.success:
        message = (
            detection_result.error_message
            if detection_result is not None and detection_result.error_message
            else "Fire detection failed."
        )
        print("status=ERROR", file=output)
        print(f"message={message}", file=output)
        return

    print(f"success={detection_result.success}", file=output)
    print(f"candidates_processed={detection_result.candidates_processed}", file=output)
    print(f"no_event_count={detection_result.no_event_count}", file=output)
    print(f"events_created={detection_result.events_created}", file=output)
    print(f"events_updated={detection_result.events_updated}", file=output)
    print(f"event_ids={_format_event_ids(detection_result.event_ids)}", file=output)


def print_fire_severity_result(
    fire_severity_result: SimulationFireSeverityResult,
    output: TextIO = sys.stdout,
) -> None:
    print("FIRE SEVERITY ASSESSMENT", file=output)
    for stored_assessment in fire_severity_result.assessment_results:
        assessment = stored_assessment.assessment
        print(f"event_id={assessment.fire_event_id}", file=output)
        print(f"status={assessment.status.name}", file=output)
        print(f"score={_format_optional_score(assessment.score)}", file=output)
        print(f"level={assessment.level.name if assessment.level is not None else '-'}", file=output)
        print(f"assessment_id={stored_assessment.assessment_id}", file=output)
    for fire_event_id, message in zip(
        fire_severity_result.failed_fire_event_ids,
        fire_severity_result.error_messages,
    ):
        print(f"event_id={fire_event_id}", file=output)
        print("status=ERROR", file=output)
        print(f"message={message}", file=output)


def print_fire_spread_result(
    fire_spread_result: SimulationFireSpreadResult,
    output: TextIO = sys.stdout,
) -> None:
    print("FIRE SPREAD PREDICTION", file=output)
    for stored_prediction in fire_spread_result.prediction_results:
        prediction = stored_prediction.prediction
        print(f"event_id={prediction.fire_event_id}", file=output)
        print(f"horizon_minutes={prediction.horizon_minutes}", file=output)
        print(f"status={prediction.status.name}", file=output)
        print(f"cells={len(prediction.cells)}", file=output)
        print(f"prediction_id={stored_prediction.id}", file=output)
    for fire_event_id, message in zip(
        fire_spread_result.failed_fire_event_ids,
        fire_spread_result.error_messages,
    ):
        print(f"event_id={fire_event_id}", file=output)
        print("status=ERROR", file=output)
        print(f"message={message}", file=output)


def print_response_target_result(
    response_target_result: SimulationResponseTargetResult,
    output: TextIO = sys.stdout,
) -> None:
    print("RESPONSE TARGETS", file=output)
    print(f"fire_events_requested={response_target_result.fire_events_requested}", file=output)
    print(f"target_sets_generated={response_target_result.target_sets_generated}", file=output)
    print(f"inactive_events={response_target_result.inactive_events}", file=output)
    print(f"failed_events={response_target_result.failed_events}", file=output)
    print(f"targets={response_target_result.target_count}", file=output)
    for generation_result in response_target_result.generation_results:
        print(f"event_id={generation_result.fire_event_id}", file=output)
        print(f"status={generation_result.status.name}", file=output)
        target_set_id = generation_result.target_set_id if generation_result.target_set_id is not None else "-"
        print(f"target_set_id={target_set_id}", file=output)
        print(f"target_count={generation_result.target_count}", file=output)
        if generation_result.error_message:
            print(f"message={generation_result.error_message}", file=output)
        for index, target in enumerate(generation_result.targets, start=1):
            print(f"target_{index}_type={target.target_type.name}", file=output)
            print(f"target_{index}_priority={target.priority_score:.2f}", file=output)
            print(f"target_{index}_lat={target.latitude:.6f}", file=output)
            print(f"target_{index}_lon={target.longitude:.6f}", file=output)
            if target.prediction_horizon_minutes is not None:
                print(f"target_{index}_horizon_minutes={target.prediction_horizon_minutes}", file=output)
            if target.spread_prediction_id is not None:
                print(f"target_{index}_spread_prediction_id={target.spread_prediction_id}", file=output)
            if target.spread_prediction_cell_id is not None:
                print(f"target_{index}_spread_prediction_cell_id={target.spread_prediction_cell_id}", file=output)


def print_simulation_refresh_result(
    simulation_refresh_result: SimulationRefreshResult,
    output: TextIO = sys.stdout,
) -> None:
    for refresh_result in simulation_refresh_result.refresh_results:
        if refresh_result.trigger_type.value == "resource_status_update":
            print_resource_refresh_result(refresh_result, output)
        else:
            print_operational_refresh_result(refresh_result, output)
    for planning_result in simulation_refresh_result.planning_results:
        print_planning_refresh_result(planning_result, output)


def print_planning_refresh_result(
    planning_result: PlanningRefreshResult,
    output: TextIO = sys.stdout,
) -> None:
    """Show whether US 5.4 planning refresh ran, and its outcome, for one FireEvent.

    Prints only fields already present on PlanningRefreshResult itself, plus
    (when a response_plan_id exists) a compact enrichment reusing the
    existing US 5.5 ResponsePlanDetailsService read - never recomputing
    routes, optimization, or scores here.
    """
    print("RESPONSE PLANNING REFRESH", file=output)
    print(f"fire_event_id={planning_result.fire_event_id}", file=output)
    print(f"status={planning_result.status.name}", file=output)
    if planning_result.route_planning_run_id is not None:
        print(f"route_planning_run_id={planning_result.route_planning_run_id}", file=output)
    if planning_result.response_plan_id is not None:
        print(f"response_plan_id={planning_result.response_plan_id}", file=output)
    if planning_result.comparison_id is not None:
        print(f"comparison_id={planning_result.comparison_id}", file=output)
    if planning_result.error:
        print(f"message={planning_result.error}", file=output)

    if planning_result.response_plan_id is None:
        return
    plan_details = get_response_plan_details_service().get_plan_details_by_id(planning_result.response_plan_id)
    if plan_details is None:
        return
    average_eta = (
        f"{plan_details.average_eta_seconds:.1f}" if plan_details.average_eta_seconds is not None else "-"
    )
    print(f"plan_score={plan_details.plan_score:.2f}", file=output)
    print(f"coverage_score={plan_details.coverage_score:.2f}", file=output)
    print(f"average_eta_seconds={average_eta}", file=output)
    print(f"response_actions={len(plan_details.actions)}", file=output)
    print(f"uncovered_targets={len(plan_details.uncovered_target_ids)}", file=output)


def print_operational_refresh_result(
    refresh_result: OperationalRefreshResult,
    output: TextIO = sys.stdout,
) -> None:
    print("OPERATIONAL REFRESH", file=output)
    print(f"trigger={refresh_result.trigger_type.name}", file=output)
    print(f"fire_event_id={refresh_result.fire_event_id}", file=output)
    print(f"status={refresh_result.status.name}", file=output)
    print(f"success={refresh_result.success}", file=output)
    if refresh_result.severity_result is not None:
        severity = refresh_result.severity_result.assessment
        print("Severity:", file=output)
        print(f"severity_status={severity.status.name}", file=output)
        print(f"severity_score={_format_optional_score(severity.score)}", file=output)
        print(f"severity_assessment_id={refresh_result.severity_result.assessment_id}", file=output)
    if refresh_result.spread_refresh_result is not None:
        print("Spread:", file=output)
        for horizon_result in refresh_result.spread_refresh_result.horizon_results:
            prediction_id = horizon_result.prediction_id if horizon_result.prediction_id is not None else "-"
            print(f"{horizon_result.horizon_minutes}m={horizon_result.status.name}", file=output)
            print(f"{horizon_result.horizon_minutes}m_prediction_id={prediction_id}", file=output)
    if refresh_result.response_target_result is not None:
        targets = refresh_result.response_target_result
        target_set_id = targets.target_set_id if targets.target_set_id is not None else "-"
        print("Response Targets:", file=output)
        print(f"target_status={targets.status.name}", file=output)
        print(f"target_set_id={target_set_id}", file=output)
        print(f"target_count={targets.target_count}", file=output)
    if refresh_result.error_message:
        print(f"message={refresh_result.error_message}", file=output)


def print_resource_refresh_result(
    refresh_result: OperationalRefreshResult,
    output: TextIO = sys.stdout,
) -> None:
    status_result = refresh_result.resource_status_result
    print("RESOURCE STATUS UPDATE", file=output)
    if status_result is None:
        print(f"status={refresh_result.status.name}", file=output)
        if refresh_result.error_message:
            print(f"message={refresh_result.error_message}", file=output)
        return
    previous_status = status_result.previous_status.name if status_result.previous_status is not None else "-"
    current_status = status_result.current_status.name if status_result.current_status is not None else "-"
    print(f"resource_id={status_result.resource_id}", file=output)
    print(f"{previous_status} -> {current_status}", file=output)
    print(f"status={refresh_result.status.name}", file=output)
    print(f"available_resources={len(refresh_result.available_resources)}", file=output)
    if refresh_result.error_message:
        print(f"message={refresh_result.error_message}", file=output)


def _format_optional_score(score: float | None) -> str:
    if score is None:
        return "-"
    return f"{score:.2f}"


def _format_event_ids(event_ids: tuple[int, ...]) -> str:
    if not event_ids:
        return "-"
    return ",".join(str(event_id) for event_id in event_ids)


def print_startup_summary(
    scenario: SimulationScenario,
    mode: SimulationMode,
    scenario_started_at: datetime,
    output: TextIO = sys.stdout,
) -> None:
    print("EcoGuard Demo Simulation", file=output)
    print("SIMULATION MODE", file=output)
    print(f"Mode: {mode.value.upper()}", file=output)
    print(f"Seed: {scenario.seed}", file=output)
    print(f"Started at: {scenario_started_at.isoformat()}", file=output)
    print(f"Duration: {scenario.duration_seconds}s", file=output)
    print("Incidents:", file=output)
    for incident in scenario.incidents:
        print(
            f"- {incident.incident_id} | {incident.scenario_type.value} | {incident.location.name}",
            file=output,
        )
    print("", file=output)


def print_run_summary(summary: RunSummary, output: TextIO = sys.stdout) -> None:
    print("Simulation completed", file=output)
    print(f"Events executed: {summary.events_executed}", file=output)
    print(f"Successful events: {summary.successful_events}", file=output)
    print(f"Failed events: {summary.failed_events}", file=output)
    print(f"Generated records: {summary.generated_records}", file=output)
    print(f"Saved records: {summary.saved_records}", file=output)
    print(f"Duplicates skipped: {summary.duplicates_skipped}", file=output)
    print(f"Failed records: {summary.failed_records}", file=output)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenario = build_scenario_from_args(args)

    try:
        initialize_database()
    except DatabaseConfigurationError as exc:
        print(f"Database configuration error: {exc}", file=sys.stderr)
        return 2

    mode = SimulationMode(args.mode)
    if mode is SimulationMode.MANUAL:
        run_manual(scenario)
    else:
        run_automatic(scenario, poll_interval_seconds=args.poll_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
