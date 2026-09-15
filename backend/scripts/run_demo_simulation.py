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
from src.agents.analysis import FireDangerAssessmentAgent, FireDetectionAgent, FireSeverityAssessmentAgent
from src.calculators.fire_danger.ffwi_calculator import FFWICalculator
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_severity.fire_severity_calculator import FireSeverityCalculator
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_danger_assessment_repository import FireDangerAssessmentRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_danger import FireDangerInputService
from src.services.fire_detection import FireDetectionEvidenceService
from src.services.fire_severity import FireSeverityInputService
from src.simulation import (
    SIMULATION_LOCATIONS,
    ScenarioType,
    SimulationEvent,
    SimulationEventExecutionResult,
    SimulationEventExecutor,
    SimulationMode,
    SimulationScenario,
    SimulationScenarioService,
    SimulationFireDangerCoordinator,
    SimulationFireDangerResult,
    SimulationFireDetectionCoordinator,
    SimulationFireDetectionResult,
    SimulationFireSeverityCoordinator,
    SimulationFireSeverityResult,
    build_carmel_golan_active_fire_scenario,
    build_scenario,
    get_simulation_location,
    simulation_event_timestamp,
)

DEFAULT_SEED = 42
DEFAULT_MODE = "manual"
DEFAULT_POLL_INTERVAL_SECONDS = 0.5
SUPPORTED_PRESETS = ("carmel_golan_active_fire",)


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
    fire_severity_coordinator: SimulationFireSeverityCoordinator | None = None,
    service: SimulationScenarioService | None = None,
    scenario_started_at: datetime | None = None,
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> RunSummary:
    executor = executor or SimulationEventExecutor()
    fire_danger_coordinator = fire_danger_coordinator or build_fire_danger_coordinator()
    fire_detection_coordinator = fire_detection_coordinator or build_fire_detection_coordinator()
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
            fire_severity_coordinator,
        )
        summary.add(result)

    print_run_summary(summary, output)
    return summary


def run_automatic(
    scenario: SimulationScenario,
    executor: SimulationEventExecutor | None = None,
    fire_danger_coordinator: SimulationFireDangerCoordinator | None = None,
    fire_detection_coordinator: SimulationFireDetectionCoordinator | None = None,
    fire_severity_coordinator: SimulationFireSeverityCoordinator | None = None,
    service: SimulationScenarioService | None = None,
    scenario_started_at: datetime | None = None,
    sleep_func: Callable[[float], None] = time.sleep,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    output: TextIO = sys.stdout,
) -> RunSummary:
    executor = executor or SimulationEventExecutor()
    fire_danger_coordinator = fire_danger_coordinator or build_fire_danger_coordinator()
    fire_detection_coordinator = fire_detection_coordinator or build_fire_detection_coordinator()
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
                fire_severity_coordinator,
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
    fire_severity_coordinator: SimulationFireSeverityCoordinator | None = None,
) -> SimulationEventExecutionResult:
    incident = scenario.get_incident(event.incident_id)
    event_timestamp = simulation_event_timestamp(scenario_started_at, event)

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
    if fire_detection_coordinator is not None:
        fire_detection_result = fire_detection_coordinator.handle_event(
            scenario=scenario,
            event=event,
            execution_result=result,
            event_timestamp=event_timestamp,
        )
        if fire_detection_result.triggered:
            print_fire_detection_result(fire_detection_result, output)
    else:
        fire_detection_result = None
    if fire_severity_coordinator is not None:
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
    print("", file=output)
    return result


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
