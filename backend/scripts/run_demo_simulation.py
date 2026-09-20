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
from src.calculators.global_response_optimization.global_assignment_change_calculator import AssignmentChangeType
from src.services.global_planning.global_planning_refresh_coordinator import (
    GlobalPlanningRefreshResult,
    GlobalPlanningRefreshStatus,
)
from src.services.operational_refresh import OperationalRefreshResult
from src.services.response_planning.response_plan_details_service import ResponsePlanDetailsService
from src.simulation import (
    SIMULATION_LOCATIONS,
    ScenarioType,
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
    build_operations_demo_scenario,
    build_scenario,
    get_simulation_location,
    simulation_event_timestamp,
)
from src.simulation import SimulationFireSpreadCoordinator, SimulationFireSpreadResult
from src.simulation.demo_state_reset_service import DemoStateResetDisabledError, DemoStateResetService
from src.simulation.demo_simulation_runner import (
    DemoSimulationEventPhase,
    DemoSimulationEventProgress,
    DemoSimulationRunConfig,
    DemoSimulationRunner,
    SimulationEventOutcome,
    build_fire_danger_coordinator,
    build_fire_detection_coordinator,
    build_fire_severity_coordinator,
    build_fire_spread_coordinator,
    build_operational_coordinator,
    build_response_target_coordinator,
    build_simulation_refresh_coordinator,
    execute_simulation_event,
)

DEFAULT_SEED = 42
DEFAULT_MODE = "manual"
DEFAULT_POLL_INTERVAL_SECONDS = 0.5
SUPPORTED_PRESETS = ("carmel_golan_active_fire", "active_fire_resource_refresh", "operations_demo")


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
    parser.add_argument(
        "--reset-demo-state",
        action="store_true",
        help=(
            "DESTRUCTIVE: reset runtime demo state via DemoStateResetService before "
            "running (see scripts/reset_demo_state.py). Requires ENABLE_DEMO_DATA_RESET=true; "
            "intended only for a dedicated demo database, never the shared team database."
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
    if args.preset == "operations_demo":
        return build_operations_demo_scenario(seed=args.seed)
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
    """Thin CLI adapter: configures DemoSimulationRunner for manual stepping.

    input() lives only here (inside the `before_event` closure) - the
    runner itself never calls input(). This is the CLI's only remaining
    event loop responsibility: translating runner progress into terminal
    output and driving manual acknowledgement.
    """
    scenario_started_at = scenario_started_at or datetime.now(timezone.utc)
    summary = RunSummary()

    def before_event(_event: SimulationEvent) -> None:
        try:
            input_func("Press ENTER to execute next event...")
        except EOFError:
            print("No input available; continuing with next event.", file=output)

    def on_progress(progress: DemoSimulationEventProgress) -> None:
        # The runner also emits an EVENT_STARTED tick before each event
        # (Task A3); the CLI's terminal rendering only needs the completed
        # outcome, so it ignores that first tick.
        if progress.phase is not DemoSimulationEventPhase.EVENT_COMPLETED:
            return
        render_event_outcome(progress.event_outcome, output)
        summary.add(progress.event_outcome.execution_result)

    runner = DemoSimulationRunner(
        executor=executor,
        fire_danger_coordinator=fire_danger_coordinator,
        fire_detection_coordinator=fire_detection_coordinator,
        operational_coordinator=operational_coordinator,
        fire_severity_coordinator=fire_severity_coordinator,
        simulation_refresh_coordinator=simulation_refresh_coordinator,
        fire_spread_coordinator_factory=get_fire_spread_coordinator,
        response_target_coordinator_factory=get_response_target_coordinator,
        service=service,
    )
    config = DemoSimulationRunConfig(mode=SimulationMode.MANUAL, before_event=before_event)

    print_startup_summary(scenario, SimulationMode.MANUAL, scenario_started_at, output)
    runner.run(scenario, config, scenario_started_at=scenario_started_at, on_progress=on_progress)

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
    """Thin CLI adapter: configures DemoSimulationRunner for automatic polling."""
    scenario_started_at = scenario_started_at or datetime.now(timezone.utc)
    summary = RunSummary()

    def on_progress(progress: DemoSimulationEventProgress) -> None:
        if progress.phase is not DemoSimulationEventPhase.EVENT_COMPLETED:
            return
        render_event_outcome(progress.event_outcome, output)
        summary.add(progress.event_outcome.execution_result)

    runner = DemoSimulationRunner(
        executor=executor,
        fire_danger_coordinator=fire_danger_coordinator,
        fire_detection_coordinator=fire_detection_coordinator,
        operational_coordinator=operational_coordinator,
        fire_severity_coordinator=fire_severity_coordinator,
        simulation_refresh_coordinator=simulation_refresh_coordinator,
        fire_spread_coordinator_factory=get_fire_spread_coordinator,
        response_target_coordinator_factory=get_response_target_coordinator,
        service=service,
    )
    config = DemoSimulationRunConfig(
        mode=SimulationMode.AUTOMATIC,
        sleep_fn=sleep_func,
        poll_interval_seconds=poll_interval_seconds,
    )

    print_startup_summary(scenario, SimulationMode.AUTOMATIC, scenario_started_at, output)
    runner.run(scenario, config, scenario_started_at=scenario_started_at, on_progress=on_progress)

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
    """Execute one event and print its outcome - a thin wrapper around the
    single authoritative orchestration in src.simulation.demo_simulation_runner.
    """
    outcome = execute_simulation_event(
        scenario=scenario,
        event=event,
        scenario_started_at=scenario_started_at,
        executor=executor,
        fire_danger_coordinator=fire_danger_coordinator,
        fire_detection_coordinator=fire_detection_coordinator,
        operational_coordinator=operational_coordinator,
        fire_severity_coordinator=fire_severity_coordinator,
        simulation_refresh_coordinator=simulation_refresh_coordinator,
        fire_spread_coordinator_factory=get_fire_spread_coordinator,
        response_target_coordinator_factory=get_response_target_coordinator,
    )
    render_event_outcome(outcome, output)
    return outcome.execution_result


def render_event_outcome(outcome: SimulationEventOutcome, output: TextIO = sys.stdout) -> None:
    """Print one event's outcome to the terminal - the CLI's only rendering
    responsibility. Reproduces the exact terminal blocks the pre-Task-A2 CLI
    printed, now driven by structured runner state instead of interleaving
    printing with orchestration."""
    incident = outcome.incident
    event = outcome.event
    result = outcome.execution_result

    print(
        f"[T+{event.offset_seconds}s] {event.event_type.value.upper()} | "
        f"{event.incident_id} | {incident.location.name}",
        file=output,
    )
    print(format_execution_result(result), file=output)
    if result.error_message:
        print(f"Error: {result.error_message}", file=output)

    if outcome.fire_danger_result is not None and outcome.fire_danger_result.triggered:
        print_fire_danger_result(outcome.fire_danger_result, output)

    if outcome.fire_detection_result is not None and outcome.fire_detection_result.triggered:
        print_fire_detection_result(outcome.fire_detection_result, output)
        if outcome.operational_context_scrambled:
            print("OPERATIONAL CONTEXT", file=output)
            print(f"depleted_resources={outcome.depleted_resource_count}", file=output)

    if outcome.refresh_result is not None and outcome.refresh_result.triggered:
        print_simulation_refresh_result(outcome.refresh_result, output)
    elif outcome.fire_severity_result is not None and outcome.fire_severity_result.triggered:
        print_fire_severity_result(outcome.fire_severity_result, output)
        if outcome.fire_spread_result is not None and outcome.fire_spread_result.triggered:
            print_fire_spread_result(outcome.fire_spread_result, output)

    if outcome.response_target_result is not None and outcome.response_target_result.triggered:
        print_response_target_result(outcome.response_target_result, output)

    print("", file=output)


def format_execution_result(result: SimulationEventExecutionResult) -> str:
    return (
        f"generated={result.generated_count} "
        f"saved={result.saved_count} "
        f"duplicates={result.duplicates_skipped} "
        f"failed={result.failed_count} "
        f"success={result.success}"
    )


# build_fire_danger_coordinator / build_fire_detection_coordinator /
# build_operational_coordinator / build_fire_severity_coordinator /
# build_fire_spread_coordinator / build_response_target_coordinator /
# build_simulation_refresh_coordinator now live in
# src.simulation.demo_simulation_runner (Task A2) and are imported above -
# this keeps exactly one place that builds the production analysis stack,
# reusable by this CLI and any future caller (e.g. a Task A3 API), rather
# than the CLI owning its own copy.

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
    if simulation_refresh_result.global_planning_result is not None:
        print_global_planning_refresh_result(simulation_refresh_result.global_planning_result, output)


def print_global_planning_refresh_result(
    result: GlobalPlanningRefreshResult,
    output: TextIO = sys.stdout,
) -> None:
    """Show the Stage 6 Global GA's authoritative planning-cycle outcome.

    Deliberately compact - GlobalPlanningRun id, active-event membership,
    planning status, per-event minimum/desired/assigned, global shortage,
    which resources are already dispatched vs newly assigned vs released,
    and NO_OP - never the internal GA search-space/population details
    (Task 12: keep the demo understandable).
    """
    print("GLOBAL PLANNING REFRESH", file=output)
    print(f"global_planning_run_id={result.global_planning_run_id}", file=output)
    print(f"trigger={result.trigger}", file=output)
    print(f"status={result.status.name}", file=output)
    if result.retry_count:
        print(f"retry_count={result.retry_count}", file=output)

    if result.status is GlobalPlanningRefreshStatus.NO_OP:
        print("no_op=true (semantic input and policy unchanged - nothing rewritten)", file=output)
        return
    if result.status is not GlobalPlanningRefreshStatus.ACTIVATED:
        return

    print(f"active_events={sorted(result.response_plan_ids_by_event)}", file=output)

    if result.shortage is not None:
        shortage = result.shortage
        print(
            "shortage="
            f"required({shortage.total_required}/{shortage.unmet_required} unmet) "
            f"desired({shortage.total_desired}/{shortage.unmet_desired} unmet) "
            f"assigned={shortage.total_assigned} "
            f"candidate_supply={shortage.candidate_assignable_resource_count}",
            file=output,
        )

    for event_result in result.event_results:
        demand = event_result.demand_result
        print(
            f"event_id={event_result.fire_event_id} "
            f"minimum={demand.minimum_resources} desired={demand.desired_resources} "
            f"assigned={demand.suppression_resources_assigned} "
            f"unmet_required={demand.unmet_minimum} unmet_desired={demand.unmet_desired}",
            file=output,
        )

    changes_by_type: dict[AssignmentChangeType, list[str]] = {}
    for change in result.assignment_changes:
        changes_by_type.setdefault(change.change_type, []).append(change.resource_id)
    for change_type in (
        AssignmentChangeType.NEW_ASSIGNMENT,
        AssignmentChangeType.REASSIGNED,
        AssignmentChangeType.RELEASED,
        AssignmentChangeType.UNCHANGED,
    ):
        resource_ids = changes_by_type.get(change_type)
        if resource_ids:
            print(f"{change_type.value}={sorted(resource_ids)}", file=output)

    for fire_event_id, response_plan_id in sorted(result.response_plan_ids_by_event.items()):
        plan_details = get_response_plan_details_service().get_plan_details_by_id(response_plan_id)
        if plan_details is None:
            continue
        average_eta = (
            f"{plan_details.average_eta_seconds:.1f}" if plan_details.average_eta_seconds is not None else "-"
        )
        print(
            f"event_id={fire_event_id} response_plan_id={response_plan_id} "
            f"coverage_score={plan_details.coverage_score:.2f} average_eta_seconds={average_eta} "
            f"response_actions={len(plan_details.actions)}",
            file=output,
        )


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

    if args.reset_demo_state:
        try:
            reset_result = DemoStateResetService().reset_demo_state()
        except DemoStateResetDisabledError as exc:
            print(f"Refused: {exc}", file=sys.stderr)
            return 2
        print("Demo state reset before run:")
        for table_name, count in reset_result.deleted_counts.items():
            print(f"- {table_name}: {count}")
        print(f"Resources restored to AVAILABLE: {reset_result.resources_restored}")
        print("")

    mode = SimulationMode(args.mode)
    if mode is SimulationMode.MANUAL:
        run_manual(scenario)
    else:
        run_automatic(scenario, poll_interval_seconds=args.poll_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
