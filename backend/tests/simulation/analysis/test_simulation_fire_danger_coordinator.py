"""Tests for SimulationFireDangerCoordinator trigger policy."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from src.agents.analysis.fire_danger_assessment_result import FireDangerAssessmentResult
from src.calculators.fire_danger.ffwi_config import FFWI_METHODOLOGY_NAME, FFWI_METHODOLOGY_VERSION
from src.models import FireDangerAssessment, FireDangerAssessmentStatus, FireDangerLevel
from src.services.fire_danger.fire_danger_input_service import haversine_distance_km
from src.simulation import (
    CARMEL_LOCATION,
    GOLAN_LOCATION,
    ScenarioType,
    SimulatedIncident,
    SimulationEvent,
    SimulationEventExecutionResult,
    SimulationEventType,
    SimulationScenario,
    build_carmel_golan_active_fire_scenario,
)
from src.simulation.analysis import ASSESSMENT_RADIUS_KM, SimulationFireDangerCoordinator
from src.simulation.analysis.simulation_fire_danger_coordinator import (
    NON_WEATHER_EVENT_REASON,
    NO_WEATHER_SAVED_REASON,
    WEATHER_EVENT_FAILED_REASON,
)
from src.simulation.analysis.simulation_fire_danger_result import SimulationFireDangerResult
from src.simulation.generators.weather_data_generator import SIMULATED_WEATHER_STATION_OFFSETS

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


class FakeAssessmentAgent:
    def __init__(self, result: FireDangerAssessmentResult) -> None:
        self.result = result
        self.calls = []

    def assess(self, area, as_of):
        self.calls.append({"area": area, "as_of": as_of})
        return self.result


def make_assessment(area_name=CARMEL_LOCATION.name, status=FireDangerAssessmentStatus.VALID):
    return FireDangerAssessment(
        area_id="simulation-carmel",
        area_name=area_name,
        area_latitude=CARMEL_LOCATION.latitude,
        area_longitude=CARMEL_LOCATION.longitude,
        area_radius_km=ASSESSMENT_RADIUS_KM,
        assessed_at=AS_OF,
        status=status,
        score=42.5 if status is FireDangerAssessmentStatus.VALID else None,
        level=FireDangerLevel.VERY_HIGH if status is FireDangerAssessmentStatus.VALID else None,
        methodology=FFWI_METHODOLOGY_NAME,
        methodology_version=FFWI_METHODOLOGY_VERSION,
    )


def make_agent_result(status=FireDangerAssessmentStatus.VALID, success=True):
    if not success:
        return FireDangerAssessmentResult(
            assessment=None,
            stored_assessment_id=None,
            success=False,
            error_message="Fire-danger assessment persistence failed.",
        )
    return FireDangerAssessmentResult(
        assessment=make_assessment(status=status),
        stored_assessment_id=55,
        success=True,
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


def test_weather_successful_persistence_triggers_assessment():
    scenario = make_scenario()
    event = scenario.events[0]
    agent = FakeAssessmentAgent(make_agent_result())
    coordinator = SimulationFireDangerCoordinator(agent)

    result = handle(coordinator, scenario, event)

    assert result.triggered is True
    assert result.assessment_result.success is True
    assert len(agent.calls) == 1


@pytest.mark.parametrize("event_type", [SimulationEventType.SATELLITE, SimulationEventType.NEWS])
def test_non_weather_events_never_trigger_assessment(event_type):
    scenario = make_scenario()
    event = next(event for event in scenario.events if event.event_type is event_type)
    agent = FakeAssessmentAgent(make_agent_result())
    coordinator = SimulationFireDangerCoordinator(agent)

    result = handle(coordinator, scenario, event)

    assert result.triggered is False
    assert result.assessment_result is None
    assert result.reason == NON_WEATHER_EVENT_REASON
    assert agent.calls == []


def test_failed_weather_event_with_no_usable_persistence_does_not_trigger():
    scenario = make_scenario()
    event = scenario.events[0]
    agent = FakeAssessmentAgent(make_agent_result())
    coordinator = SimulationFireDangerCoordinator(agent)

    result = handle(
        coordinator,
        scenario,
        event,
        result=make_result(event, success=False, saved_count=0, failed_count=1),
    )

    assert result.triggered is False
    assert result.reason == WEATHER_EVENT_FAILED_REASON
    assert agent.calls == []


def test_weather_with_no_saved_or_duplicate_rows_does_not_trigger():
    scenario = make_scenario()
    event = scenario.events[0]
    agent = FakeAssessmentAgent(make_agent_result())
    coordinator = SimulationFireDangerCoordinator(agent)

    result = handle(coordinator, scenario, event, result=make_result(event, saved_count=0))

    assert result.triggered is False
    assert result.reason == NO_WEATHER_SAVED_REASON
    assert agent.calls == []


def test_weather_duplicate_rows_trigger_assessment_because_rows_are_persisted_inputs():
    scenario = make_scenario()
    event = scenario.events[0]
    agent = FakeAssessmentAgent(make_agent_result())
    coordinator = SimulationFireDangerCoordinator(agent)

    result = handle(
        coordinator,
        scenario,
        event,
        result=make_result(event, saved_count=0, duplicates_skipped=3),
    )

    assert result.triggered is True
    assert len(agent.calls) == 1


def test_triggered_weather_calls_agent_exactly_once_and_passes_timestamp():
    scenario = make_scenario()
    event = scenario.events[0]
    agent = FakeAssessmentAgent(make_agent_result())
    coordinator = SimulationFireDangerCoordinator(agent)

    handle(coordinator, scenario, event)

    assert len(agent.calls) == 1
    assert agent.calls[0]["as_of"] == AS_OF


def test_correct_assessment_area_is_constructed_from_simulation_location():
    scenario = make_scenario()
    event = scenario.events[0]
    agent = FakeAssessmentAgent(make_agent_result())
    coordinator = SimulationFireDangerCoordinator(agent)

    handle(coordinator, scenario, event)

    area = agent.calls[0]["area"]
    assert area.id == "simulation-carmel"
    assert area.name == CARMEL_LOCATION.name
    assert area.latitude == CARMEL_LOCATION.latitude
    assert area.longitude == CARMEL_LOCATION.longitude
    assert area.radius_km == ASSESSMENT_RADIUS_KM


def test_incident_id_is_not_present_in_assessment_area():
    scenario = make_scenario()
    event = scenario.events[0]
    agent = FakeAssessmentAgent(make_agent_result())
    coordinator = SimulationFireDangerCoordinator(agent)

    handle(coordinator, scenario, event)

    area = agent.calls[0]["area"]
    assert not hasattr(area, "incident_id")
    assert event.incident_id not in (area.id, area.name)


def test_carmel_weather_assesses_carmel_only():
    scenario = build_carmel_golan_active_fire_scenario(seed=42)
    carmel_event = scenario.events[0]
    agent = FakeAssessmentAgent(make_agent_result())
    coordinator = SimulationFireDangerCoordinator(agent)

    handle(coordinator, scenario, carmel_event)

    assert agent.calls[0]["area"].name == CARMEL_LOCATION.name


def test_golan_weather_assesses_golan_only():
    scenario = build_carmel_golan_active_fire_scenario(seed=42)
    golan_event = scenario.events[1]
    agent = FakeAssessmentAgent(make_agent_result())
    coordinator = SimulationFireDangerCoordinator(agent)

    handle(coordinator, scenario, golan_event)

    assert agent.calls[0]["area"].name == GOLAN_LOCATION.name


def test_multi_incident_carmel_event_does_not_assess_golan():
    scenario = build_carmel_golan_active_fire_scenario(seed=42)
    carmel_event = scenario.events[0]
    agent = FakeAssessmentAgent(make_agent_result())
    coordinator = SimulationFireDangerCoordinator(agent)

    handle(coordinator, scenario, carmel_event)

    assert agent.calls[0]["area"].name != GOLAN_LOCATION.name


def test_valid_agent_result_is_returned_through_simulation_result():
    scenario = make_scenario()
    event = scenario.events[0]
    agent_result = make_agent_result()
    coordinator = SimulationFireDangerCoordinator(FakeAssessmentAgent(agent_result))

    result = handle(coordinator, scenario, event)

    assert result.assessment_result == agent_result


def test_insufficient_data_agent_result_is_still_triggered():
    scenario = make_scenario()
    event = scenario.events[0]
    agent_result = make_agent_result(status=FireDangerAssessmentStatus.INSUFFICIENT_DATA)
    coordinator = SimulationFireDangerCoordinator(FakeAssessmentAgent(agent_result))

    result = handle(coordinator, scenario, event)

    assert result.triggered is True
    assert result.assessment_result.success is True
    assert result.assessment_result.assessment.status is FireDangerAssessmentStatus.INSUFFICIENT_DATA


def test_operational_agent_failure_is_triggered_failure_not_non_triggered():
    scenario = make_scenario()
    event = scenario.events[0]
    agent_result = make_agent_result(success=False)
    coordinator = SimulationFireDangerCoordinator(FakeAssessmentAgent(agent_result))

    result = handle(coordinator, scenario, event)

    assert result.triggered is True
    assert result.assessment_result.success is False
    assert result.reason is None


def test_generated_assessment_area_uses_centralized_radius_constant():
    scenario = make_scenario()
    event = scenario.events[0]
    agent = FakeAssessmentAgent(make_agent_result())

    handle(SimulationFireDangerCoordinator(agent), scenario, event)

    assert agent.calls[0]["area"].radius_km == ASSESSMENT_RADIUS_KM


def test_simulated_local_weather_station_offsets_fit_within_assessment_radius():
    for latitude_offset, longitude_offset in SIMULATED_WEATHER_STATION_OFFSETS:
        distance_km = haversine_distance_km(
            CARMEL_LOCATION.latitude,
            CARMEL_LOCATION.longitude,
            CARMEL_LOCATION.latitude + latitude_offset,
            CARMEL_LOCATION.longitude + longitude_offset,
        )
        assert distance_km < ASSESSMENT_RADIUS_KM


def test_simulation_fire_danger_result_is_immutable():
    result = SimulationFireDangerResult(
        triggered=True,
        assessment_result=make_agent_result(),
    )

    with pytest.raises(FrozenInstanceError):
        result.triggered = False


def test_non_triggered_result_requires_reason():
    with pytest.raises(ValueError):
        SimulationFireDangerResult(triggered=False, assessment_result=None)
