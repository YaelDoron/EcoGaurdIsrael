"""Tests for the demo simulation runner."""
from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
from io import StringIO

import pytest

from scripts.run_demo_simulation import (
    DEFAULT_MODE,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_SEED,
    build_scenario_from_args,
    execute_and_report_event,
    format_execution_result,
    parse_args,
    run_automatic,
    run_manual,
)
from src.agents.analysis.fire_danger_assessment_result import FireDangerAssessmentResult
from src.calculators.fire_danger.ffwi_config import FFWI_METHODOLOGY_NAME, FFWI_METHODOLOGY_VERSION
from src.models import FireDangerAssessment, FireDangerAssessmentStatus, FireDangerLevel
from src.simulation.analysis.simulation_fire_danger_result import SimulationFireDangerResult
from src.simulation import (
    CARMEL_LOCATION,
    GOLAN_LOCATION,
    ScenarioType,
    SimulationEventExecutionResult,
    SimulationEventType,
    SimulationMode,
    build_active_fire_scenario,
    build_carmel_golan_active_fire_scenario,
    simulation_event_timestamp,
)

STARTED_AT = datetime(2026, 9, 12, 14, 0, 0, tzinfo=timezone.utc)


class FakeExecutor:
    def __init__(self, success=True) -> None:
        self.calls = []
        self._success = success

    def execute(self, scenario, event, event_timestamp):
        self.calls.append((scenario, event, event_timestamp))
        return SimulationEventExecutionResult(
            event=event,
            success=self._success,
            generated_count=1,
            saved_count=1 if self._success else 0,
            duplicates_skipped=0,
            failed_count=0 if self._success else 1,
            error_message=None if self._success else "simulated failure",
        )


class FakeFireDangerCoordinator:
    def __init__(self, mode="valid") -> None:
        self.mode = mode
        self.calls = []

    def handle_event(self, scenario, event, execution_result, event_timestamp):
        self.calls.append((scenario, event, execution_result, event_timestamp))
        if event.event_type is not SimulationEventType.WEATHER:
            return SimulationFireDangerResult(
                triggered=False,
                assessment_result=None,
                reason="non_weather_event",
            )
        if self.mode == "insufficient":
            assessment = make_assessment(status=FireDangerAssessmentStatus.INSUFFICIENT_DATA)
            assessment_result = FireDangerAssessmentResult(
                assessment=assessment,
                stored_assessment_id=124,
                success=True,
            )
        elif self.mode == "failure":
            assessment_result = FireDangerAssessmentResult(
                assessment=None,
                stored_assessment_id=None,
                success=False,
                error_message="Fire-danger assessment persistence failed.",
            )
        else:
            assessment = make_assessment(status=FireDangerAssessmentStatus.VALID)
            assessment_result = FireDangerAssessmentResult(
                assessment=assessment,
                stored_assessment_id=123,
                success=True,
            )
        return SimulationFireDangerResult(
            triggered=True,
            assessment_result=assessment_result,
            reason=None,
        )


class FakeAutomaticService:
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


def make_assessment(status=FireDangerAssessmentStatus.VALID, area_name="Carmel Demo Area"):
    return FireDangerAssessment(
        area_id="simulation-carmel",
        area_name=area_name,
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        assessed_at=STARTED_AT,
        status=status,
        score=47.83 if status is FireDangerAssessmentStatus.VALID else None,
        level=FireDangerLevel.VERY_HIGH if status is FireDangerAssessmentStatus.VALID else None,
        methodology=FFWI_METHODOLOGY_NAME,
        methodology_version=FFWI_METHODOLOGY_VERSION,
    )


def test_parse_valid_single_incident_args():
    args = parse_args(
        [
            "--scenario",
            "active_fire",
            "--location",
            "carmel",
            "--mode",
            "manual",
            "--seed",
            "99",
        ]
    )

    assert args.scenario == "active_fire"
    assert args.location == "carmel"
    assert args.mode == "manual"
    assert args.seed == 99
    assert args.preset is None


def test_parse_valid_preset_args_uses_defaults():
    args = parse_args(["--preset", "carmel_golan_active_fire"])

    assert args.preset == "carmel_golan_active_fire"
    assert args.scenario is None
    assert args.location is None
    assert args.mode == DEFAULT_MODE
    assert args.seed == DEFAULT_SEED
    assert args.poll_interval == DEFAULT_POLL_INTERVAL_SECONDS


@pytest.mark.parametrize(
    "argv",
    [
        ["--scenario", "invalid", "--location", "carmel"],
        ["--scenario", "active_fire", "--location", "invalid"],
        ["--scenario", "active_fire", "--location", "carmel", "--mode", "invalid"],
        ["--scenario", "active_fire", "--location", "carmel", "--seed", "not-int"],
        ["--preset", "unknown"],
        ["--preset", "carmel_golan_active_fire", "--scenario", "active_fire"],
        ["--preset", "carmel_golan_active_fire", "--location", "carmel"],
        ["--scenario", "active_fire"],
        ["--location", "carmel"],
    ],
)
def test_parse_invalid_args_fail_clearly(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_build_active_fire_carmel_single_incident_scenario():
    args = Namespace(scenario="active_fire", location="carmel", preset=None, seed=42)

    scenario = build_scenario_from_args(args)

    assert len(scenario.incidents) == 1
    assert scenario.incidents[0].scenario_type is ScenarioType.ACTIVE_FIRE
    assert scenario.incidents[0].location == CARMEL_LOCATION


def test_build_high_risk_golan_single_incident_scenario():
    args = Namespace(scenario="high_risk_no_fire", location="golan", preset=None, seed=99)

    scenario = build_scenario_from_args(args)

    assert scenario.seed == 99
    assert len(scenario.incidents) == 1
    assert scenario.incidents[0].scenario_type is ScenarioType.HIGH_RISK_NO_FIRE
    assert scenario.incidents[0].location == GOLAN_LOCATION


def test_build_preset_creates_carmel_golan_scenario():
    args = Namespace(scenario=None, location=None, preset="carmel_golan_active_fire", seed=42)

    scenario = build_scenario_from_args(args)

    assert [incident.incident_id for incident in scenario.incidents] == [
        "incident-carmel-01",
        "incident-golan-01",
    ]
    assert [incident.location for incident in scenario.incidents] == [CARMEL_LOCATION, GOLAN_LOCATION]


def test_manual_mode_executes_events_in_order_with_expected_timestamps():
    scenario = build_active_fire_scenario(seed=42)
    executor = FakeExecutor()
    output = StringIO()
    prompts = []

    summary = run_manual(
        scenario=scenario,
        executor=executor,
        fire_danger_coordinator=FakeFireDangerCoordinator(),
        scenario_started_at=STARTED_AT,
        input_func=lambda prompt: prompts.append(prompt) or "",
        output=output,
    )

    assert [call[1] for call in executor.calls] == list(scenario.events)
    assert [call[2] for call in executor.calls] == [
        simulation_event_timestamp(STARTED_AT, event) for event in scenario.events
    ]
    assert len(prompts) == len(scenario.events)
    assert summary.events_executed == len(scenario.events)
    assert summary.successful_events == len(scenario.events)
    assert "Simulation completed" in output.getvalue()


def test_manual_mode_aggregates_failures_and_continues():
    scenario = build_active_fire_scenario(seed=42)
    executor = FakeExecutor(success=False)
    output = StringIO()

    summary = run_manual(
        scenario=scenario,
        executor=executor,
        fire_danger_coordinator=FakeFireDangerCoordinator(),
        scenario_started_at=STARTED_AT,
        input_func=lambda prompt: "",
        output=output,
    )

    assert len(executor.calls) == len(scenario.events)
    assert summary.successful_events == 0
    assert summary.failed_events == len(scenario.events)
    assert "simulated failure" in output.getvalue()


def test_automatic_mode_executes_due_batches_and_sleeps_between_polls():
    scenario = build_active_fire_scenario(seed=42)
    first_event, second_event, third_event = scenario.events[:3]
    service = FakeAutomaticService(due_batches=[[first_event], [second_event, third_event]])
    executor = FakeExecutor()
    sleeps = []
    output = StringIO()

    summary = run_automatic(
        scenario=scenario,
        executor=executor,
        fire_danger_coordinator=FakeFireDangerCoordinator(),
        service=service,
        scenario_started_at=STARTED_AT,
        sleep_func=sleeps.append,
        poll_interval_seconds=0.25,
        output=output,
    )

    assert service.started_with == (scenario, SimulationMode.AUTOMATIC)
    assert [call[1] for call in executor.calls] == [first_event, second_event, third_event]
    assert sleeps == [0.25]
    assert summary.events_executed == 3


def test_multi_incident_preset_manual_order_makes_incidents_visible():
    scenario = build_carmel_golan_active_fire_scenario(seed=42)
    executor = FakeExecutor()
    output = StringIO()

    summary = run_manual(
        scenario=scenario,
        executor=executor,
        fire_danger_coordinator=FakeFireDangerCoordinator(),
        scenario_started_at=STARTED_AT,
        input_func=lambda prompt: "",
        output=output,
    )

    assert [call[1] for call in executor.calls] == list(scenario.events)
    assert executor.calls[0][1].incident_id == "incident-carmel-01"
    assert executor.calls[1][1].incident_id == "incident-golan-01"
    assert "incident-carmel-01" in output.getvalue()
    assert "incident-golan-01" in output.getvalue()
    assert "Carmel Demo Area" in output.getvalue()
    assert "Golan Heights Demo Area" in output.getvalue()
    assert summary.events_executed == len(scenario.events)


def test_format_execution_result_is_human_readable():
    scenario = build_active_fire_scenario(seed=42)
    event = scenario.events[0]
    result = SimulationEventExecutionResult(
        event=event,
        success=True,
        generated_count=3,
        saved_count=2,
        duplicates_skipped=1,
        failed_count=0,
    )

    assert format_execution_result(result) == "generated=3 saved=2 duplicates=1 failed=0 success=True"


def test_weather_event_output_includes_fire_danger_assessment_block_when_triggered():
    scenario = build_active_fire_scenario(seed=42)
    event = scenario.events[0]
    output = StringIO()

    execute_and_report_event(
        scenario=scenario,
        event=event,
        scenario_started_at=STARTED_AT,
        executor=FakeExecutor(),
        output=output,
        fire_danger_coordinator=FakeFireDangerCoordinator(),
    )

    text = output.getvalue()
    assert "generated=1 saved=1 duplicates=0 failed=0 success=True" in text
    assert "FIRE DANGER ASSESSMENT" in text
    assert "area=Carmel Demo Area" in text
    assert "status=VALID" in text
    assert "score=47.83" in text
    assert "level=VERY_HIGH" in text
    assert "assessment_id=123" in text


@pytest.mark.parametrize("event_type", [SimulationEventType.SATELLITE, SimulationEventType.NEWS])
def test_non_weather_event_output_does_not_include_fire_danger_assessment(event_type):
    scenario = build_active_fire_scenario(seed=42)
    event = next(event for event in scenario.events if event.event_type is event_type)
    output = StringIO()

    execute_and_report_event(
        scenario=scenario,
        event=event,
        scenario_started_at=STARTED_AT,
        executor=FakeExecutor(),
        output=output,
        fire_danger_coordinator=FakeFireDangerCoordinator(),
    )

    assert "FIRE DANGER ASSESSMENT" not in output.getvalue()


def test_insufficient_data_fire_danger_output_is_printed_clearly():
    scenario = build_active_fire_scenario(seed=42)
    output = StringIO()

    execute_and_report_event(
        scenario=scenario,
        event=scenario.events[0],
        scenario_started_at=STARTED_AT,
        executor=FakeExecutor(),
        output=output,
        fire_danger_coordinator=FakeFireDangerCoordinator(mode="insufficient"),
    )

    text = output.getvalue()
    assert "status=INSUFFICIENT_DATA" in text
    assert "score=-" in text
    assert "level=-" in text
    assert "LOW" not in text


def test_operational_assessment_failure_is_printed_separately_from_simulation_failure():
    scenario = build_active_fire_scenario(seed=42)
    output = StringIO()

    execute_and_report_event(
        scenario=scenario,
        event=scenario.events[0],
        scenario_started_at=STARTED_AT,
        executor=FakeExecutor(success=True),
        output=output,
        fire_danger_coordinator=FakeFireDangerCoordinator(mode="failure"),
    )

    text = output.getvalue()
    assert "generated=1 saved=1 duplicates=0 failed=0 success=True" in text
    assert "FIRE DANGER ASSESSMENT" in text
    assert "status=ERROR" in text
    assert "message=Fire-danger assessment persistence failed." in text
    assert "Error: simulated failure" not in text


def test_manual_mode_invokes_fire_danger_after_each_event_without_changing_prompt_behavior():
    scenario = build_active_fire_scenario(seed=42)
    coordinator = FakeFireDangerCoordinator()
    prompts = []

    run_manual(
        scenario=scenario,
        executor=FakeExecutor(),
        fire_danger_coordinator=coordinator,
        scenario_started_at=STARTED_AT,
        input_func=lambda prompt: prompts.append(prompt) or "",
        output=StringIO(),
    )

    assert len(prompts) == len(scenario.events)
    assert [call[1] for call in coordinator.calls] == list(scenario.events)


def test_automatic_mode_invokes_fire_danger_after_due_events_without_changing_sleep_behavior():
    scenario = build_active_fire_scenario(seed=42)
    first_event, second_event = scenario.events[:2]
    service = FakeAutomaticService(due_batches=[[first_event], [second_event]])
    coordinator = FakeFireDangerCoordinator()
    sleeps = []

    run_automatic(
        scenario=scenario,
        executor=FakeExecutor(),
        fire_danger_coordinator=coordinator,
        service=service,
        scenario_started_at=STARTED_AT,
        sleep_func=sleeps.append,
        poll_interval_seconds=0.25,
        output=StringIO(),
    )

    assert [call[1] for call in coordinator.calls] == [first_event, second_event]
    assert sleeps == [0.25]


def test_multi_incident_output_associates_assessments_with_correct_event_areas():
    scenario = build_carmel_golan_active_fire_scenario(seed=42)
    executor = FakeExecutor()
    coordinator = FakeFireDangerCoordinator()
    output = StringIO()

    run_manual(
        scenario=scenario,
        executor=executor,
        fire_danger_coordinator=coordinator,
        scenario_started_at=STARTED_AT,
        input_func=lambda prompt: "",
        output=output,
    )

    text = output.getvalue()
    assert "incident-carmel-01 | Carmel Demo Area" in text
    assert "incident-golan-01 | Golan Heights Demo Area" in text
    assert [call[1] for call in coordinator.calls] == list(scenario.events)
