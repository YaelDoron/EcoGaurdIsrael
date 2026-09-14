"""Tests for SimulationFireDetectionCoordinator trigger policy."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from src.agents.analysis.fire_detection_result import FireDetectionResult
from src.simulation import (
    CARMEL_LOCATION,
    ScenarioType,
    SimulatedIncident,
    SimulationEvent,
    SimulationEventExecutionResult,
    SimulationEventType,
    SimulationScenario,
)
from src.simulation.analysis import SimulationFireDetectionCoordinator
from src.simulation.analysis.simulation_fire_detection_coordinator import (
    NO_SOURCE_DATA_AVAILABLE_REASON,
    NON_DETECTION_EVENT_REASON,
    SOURCE_EVENT_FAILED_REASON,
)
from src.simulation.analysis.simulation_fire_detection_result import SimulationFireDetectionResult

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


class FakeDetectionAgent:
    def __init__(self, result: FireDetectionResult) -> None:
        self.result = result
        self.calls = []

    def detect(self, as_of):
        self.calls.append({"as_of": as_of})
        return self.result


def make_agent_result(success=True) -> FireDetectionResult:
    if not success:
        return FireDetectionResult(
            success=False,
            candidates_processed=1,
            no_event_count=0,
            events_created=0,
            events_updated=0,
            event_ids=(),
            error_message="Fire detection orchestration failed.",
        )
    return FireDetectionResult(
        success=True,
        candidates_processed=1,
        no_event_count=0,
        events_created=1,
        events_updated=0,
        event_ids=(55,),
    )


def make_scenario() -> SimulationScenario:
    incident = SimulatedIncident(
        incident_id="incident-carmel-01",
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=CARMEL_LOCATION,
    )
    return SimulationScenario(
        duration_seconds=120,
        seed=42,
        incidents=(incident,),
        events=(
            SimulationEvent(0, SimulationEventType.WEATHER, incident.incident_id, 0),
            SimulationEvent(20, SimulationEventType.SATELLITE, incident.incident_id, 0),
            SimulationEvent(40, SimulationEventType.NEWS, incident.incident_id, 0),
        ),
    )


def make_result(event, success=True, saved_count=1, duplicates_skipped=0, failed_count=0):
    return SimulationEventExecutionResult(
        event=event,
        success=success,
        generated_count=1,
        saved_count=saved_count,
        duplicates_skipped=duplicates_skipped,
        failed_count=failed_count,
        error_message=None if success else "simulated failure",
    )


def handle(coordinator, scenario, event, result=None):
    return coordinator.handle_event(
        scenario=scenario,
        event=event,
        execution_result=result or make_result(event),
        event_timestamp=AS_OF,
    )


@pytest.mark.parametrize("event_type", [SimulationEventType.SATELLITE, SimulationEventType.NEWS])
def test_direct_evidence_successful_persistence_triggers_detection(event_type):
    scenario = make_scenario()
    event = next(event for event in scenario.events if event.event_type is event_type)
    agent = FakeDetectionAgent(make_agent_result())

    result = handle(SimulationFireDetectionCoordinator(agent), scenario, event)

    assert result.triggered is True
    assert result.detection_result.success is True
    assert len(agent.calls) == 1


def test_weather_does_not_trigger_detection_or_call_agent():
    scenario = make_scenario()
    event = scenario.events[0]
    agent = FakeDetectionAgent(make_agent_result())

    result = handle(SimulationFireDetectionCoordinator(agent), scenario, event)

    assert result.triggered is False
    assert result.reason == NON_DETECTION_EVENT_REASON
    assert agent.calls == []


@pytest.mark.parametrize("event_type", [SimulationEventType.SATELLITE, SimulationEventType.NEWS])
def test_direct_evidence_with_no_saved_or_duplicate_rows_does_not_trigger(event_type):
    scenario = make_scenario()
    event = next(event for event in scenario.events if event.event_type is event_type)
    agent = FakeDetectionAgent(make_agent_result())

    result = handle(
        SimulationFireDetectionCoordinator(agent),
        scenario,
        event,
        result=make_result(event, saved_count=0, duplicates_skipped=0),
    )

    assert result.triggered is False
    assert result.reason == NO_SOURCE_DATA_AVAILABLE_REASON
    assert agent.calls == []


@pytest.mark.parametrize("event_type", [SimulationEventType.SATELLITE, SimulationEventType.NEWS])
def test_duplicate_direct_evidence_rows_trigger_detection(event_type):
    scenario = make_scenario()
    event = next(event for event in scenario.events if event.event_type is event_type)
    agent = FakeDetectionAgent(make_agent_result())

    result = handle(
        SimulationFireDetectionCoordinator(agent),
        scenario,
        event,
        result=make_result(event, saved_count=0, duplicates_skipped=1),
    )

    assert result.triggered is True
    assert len(agent.calls) == 1


def test_failed_source_event_without_usable_data_does_not_trigger():
    scenario = make_scenario()
    event = scenario.events[1]
    agent = FakeDetectionAgent(make_agent_result())

    result = handle(
        SimulationFireDetectionCoordinator(agent),
        scenario,
        event,
        result=make_result(event, success=False, saved_count=0, failed_count=1),
    )

    assert result.triggered is False
    assert result.reason == SOURCE_EVENT_FAILED_REASON
    assert agent.calls == []


def test_triggered_event_calls_agent_once_with_exact_timestamp():
    scenario = make_scenario()
    event = scenario.events[1]
    agent = FakeDetectionAgent(make_agent_result())

    handle(SimulationFireDetectionCoordinator(agent), scenario, event)

    assert agent.calls == [{"as_of": AS_OF}]


def test_agent_success_returned_through_result():
    scenario = make_scenario()
    event = scenario.events[1]
    agent_result = make_agent_result()

    result = handle(SimulationFireDetectionCoordinator(FakeDetectionAgent(agent_result)), scenario, event)

    assert result.detection_result == agent_result


def test_agent_operational_failure_is_triggered_failure():
    scenario = make_scenario()
    event = scenario.events[1]
    agent_result = make_agent_result(success=False)

    result = handle(SimulationFireDetectionCoordinator(FakeDetectionAgent(agent_result)), scenario, event)

    assert result.triggered is True
    assert result.detection_result.success is False
    assert result.reason is None


def test_coordinator_does_not_pass_incident_id_to_agent():
    scenario = make_scenario()
    event = scenario.events[1]
    agent = FakeDetectionAgent(make_agent_result())

    handle(SimulationFireDetectionCoordinator(agent), scenario, event)

    assert "incident_id" not in agent.calls[0]
    assert agent.calls[0] == {"as_of": AS_OF}


def test_multi_incident_metadata_does_not_alter_agent_api():
    scenario = make_scenario()
    event = SimulationEvent(20, SimulationEventType.SATELLITE, "incident-carmel-01", 99)
    agent = FakeDetectionAgent(make_agent_result())

    handle(SimulationFireDetectionCoordinator(agent), scenario, event)

    assert agent.calls == [{"as_of": AS_OF}]


def test_manual_and_automatic_contexts_use_same_handle_event_contract():
    scenario = make_scenario()
    event = scenario.events[1]
    manual_agent = FakeDetectionAgent(make_agent_result())
    automatic_agent = FakeDetectionAgent(make_agent_result())

    manual_result = handle(SimulationFireDetectionCoordinator(manual_agent), scenario, event)
    automatic_result = handle(SimulationFireDetectionCoordinator(automatic_agent), scenario, event)

    assert manual_result == automatic_result
    assert manual_agent.calls == automatic_agent.calls


def test_simulation_fire_detection_result_is_immutable():
    result = SimulationFireDetectionResult(triggered=True, detection_result=make_agent_result())

    with pytest.raises(FrozenInstanceError):
        result.triggered = False


def test_non_triggered_result_requires_reason():
    with pytest.raises(ValueError):
        SimulationFireDetectionResult(triggered=False, detection_result=None)
