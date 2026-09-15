"""Tests for simulation scenario definitions."""

import pytest

from src.models.resource_status import ResourceStatus
from src.simulation import (
    CARMEL_LOCATION,
    GOLAN_LOCATION,
    MAX_SCENARIO_DURATION_SECONDS,
    ScenarioType,
    SimulatedIncident,
    SimulationEvent,
    SimulationEventType,
    SimulationScenario,
    build_active_fire_scenario,
    build_active_fire_resource_refresh_scenario,
    build_carmel_golan_active_fire_scenario,
    build_high_risk_no_fire_scenario,
    build_low_risk_no_fire_scenario,
    build_scenario,
)

INCIDENT_ID = "incident-carmel-01"
INCIDENT = SimulatedIncident(
    incident_id=INCIDENT_ID,
    scenario_type=ScenarioType.ACTIVE_FIRE,
    location=CARMEL_LOCATION,
)


def make_event(
    offset_seconds: int,
    event_type: SimulationEventType,
    incident_id: str = INCIDENT_ID,
    source_event_index: int = 0,
) -> SimulationEvent:
    return SimulationEvent(
        offset_seconds=offset_seconds,
        event_type=event_type,
        incident_id=incident_id,
        source_event_index=source_event_index,
    )


def summarize_events(scenario: SimulationScenario) -> list[tuple[int, SimulationEventType, str, int]]:
    return [
        (event.offset_seconds, event.event_type, event.incident_id, event.source_event_index)
        for event in scenario.events
    ]


def test_valid_120_second_scenario():
    scenario = SimulationScenario(
        duration_seconds=120,
        seed=123,
        incidents=(INCIDENT,),
        events=(make_event(0, SimulationEventType.WEATHER),),
    )

    assert scenario.duration_seconds == 120
    assert scenario.seed == 123
    assert scenario.incidents == (INCIDENT,)


@pytest.mark.parametrize("duration_seconds", [0, -1, 121, None, "120", True])
def test_invalid_duration_is_rejected(duration_seconds):
    with pytest.raises(ValueError):
        SimulationScenario(
            duration_seconds=duration_seconds,
            seed=42,
            incidents=(INCIDENT,),
            events=(),
        )


def test_invalid_seed_is_rejected():
    with pytest.raises(ValueError):
        SimulationScenario(
            duration_seconds=120,
            seed=True,
            incidents=(INCIDENT,),
            events=(),
        )


def test_invalid_incident_is_rejected():
    with pytest.raises(ValueError):
        SimulationScenario(
            duration_seconds=120,
            seed=42,
            incidents=("incident-carmel-01",),
            events=(),
        )


def test_duplicate_incident_ids_are_rejected():
    duplicate = SimulatedIncident(
        incident_id=INCIDENT_ID,
        scenario_type=ScenarioType.HIGH_RISK_NO_FIRE,
        location=GOLAN_LOCATION,
    )

    with pytest.raises(ValueError):
        SimulationScenario(
            duration_seconds=120,
            seed=42,
            incidents=(INCIDENT, duplicate),
            events=(),
        )


def test_event_beyond_duration_is_rejected():
    with pytest.raises(ValueError):
        SimulationScenario(
            duration_seconds=20,
            seed=42,
            incidents=(INCIDENT,),
            events=(make_event(21, SimulationEventType.NEWS),),
        )


def test_event_referencing_unknown_incident_is_rejected():
    with pytest.raises(ValueError):
        SimulationScenario(
            duration_seconds=120,
            seed=42,
            incidents=(INCIDENT,),
            events=(make_event(20, SimulationEventType.SATELLITE, incident_id="incident-golan-01"),),
        )


def test_scenario_with_no_events_is_allowed():
    scenario = SimulationScenario(
        duration_seconds=30,
        seed=42,
        incidents=(INCIDENT,),
        events=(),
    )

    assert scenario.events == ()
    assert scenario.incidents == (INCIDENT,)


def test_events_are_sorted_chronologically():
    weather = make_event(65, SimulationEventType.WEATHER, source_event_index=1)
    satellite = make_event(20, SimulationEventType.SATELLITE)
    news = make_event(40, SimulationEventType.NEWS)

    scenario = SimulationScenario(
        duration_seconds=120,
        seed=42,
        incidents=(INCIDENT,),
        events=(weather, satellite, news),
    )

    assert scenario.events == (satellite, news, weather)


def test_events_with_same_offset_keep_stable_order():
    first = make_event(20, SimulationEventType.WEATHER)
    second = make_event(20, SimulationEventType.SATELLITE)
    third = make_event(20, SimulationEventType.NEWS)

    scenario = SimulationScenario(
        duration_seconds=120,
        seed=42,
        incidents=(INCIDENT,),
        events=(first, second, third),
    )

    assert scenario.events == (first, second, third)


def test_get_incident_returns_matching_incident():
    scenario = SimulationScenario(
        duration_seconds=120,
        seed=42,
        incidents=(INCIDENT,),
        events=(),
    )

    assert scenario.get_incident(INCIDENT_ID) == INCIDENT


def test_get_incident_rejects_unknown_id():
    scenario = SimulationScenario(
        duration_seconds=120,
        seed=42,
        incidents=(INCIDENT,),
        events=(),
    )

    with pytest.raises(KeyError):
        scenario.get_incident("incident-golan-01")


def test_seed_is_preserved():
    scenario = build_active_fire_scenario(seed=999)

    assert scenario.seed == 999


def test_active_fire_default_timeline():
    scenario = build_active_fire_scenario(
        location=CARMEL_LOCATION,
        incident_id=INCIDENT_ID,
    )

    assert scenario.incidents == (INCIDENT,)
    assert scenario.duration_seconds == MAX_SCENARIO_DURATION_SECONDS
    assert summarize_events(scenario) == [
        (0, SimulationEventType.WEATHER, INCIDENT_ID, 0),
        (20, SimulationEventType.SATELLITE, INCIDENT_ID, 0),
        (40, SimulationEventType.NEWS, INCIDENT_ID, 0),
        (65, SimulationEventType.WEATHER, INCIDENT_ID, 1),
        (80, SimulationEventType.SATELLITE, INCIDENT_ID, 1),
        (100, SimulationEventType.NEWS, INCIDENT_ID, 1),
    ]


def test_low_risk_no_fire_default_timeline_is_weather_only():
    scenario = build_low_risk_no_fire_scenario(
        location=GOLAN_LOCATION,
        seed=99,
        incident_id="incident-golan-low",
    )

    assert scenario.incidents[0].scenario_type is ScenarioType.LOW_RISK_NO_FIRE
    assert scenario.incidents[0].location == GOLAN_LOCATION
    assert scenario.duration_seconds == MAX_SCENARIO_DURATION_SECONDS
    assert summarize_events(scenario) == [
        (0, SimulationEventType.WEATHER, "incident-golan-low", 0),
        (60, SimulationEventType.WEATHER, "incident-golan-low", 1),
    ]


def test_high_risk_no_fire_default_timeline_is_weather_only():
    scenario = build_high_risk_no_fire_scenario(
        location=GOLAN_LOCATION,
        seed=99,
        incident_id="incident-golan-high",
    )

    assert scenario.incidents[0].scenario_type is ScenarioType.HIGH_RISK_NO_FIRE
    assert scenario.incidents[0].location == GOLAN_LOCATION
    assert scenario.duration_seconds == MAX_SCENARIO_DURATION_SECONDS
    assert summarize_events(scenario) == [
        (0, SimulationEventType.WEATHER, "incident-golan-high", 0),
        (60, SimulationEventType.WEATHER, "incident-golan-high", 1),
    ]


@pytest.mark.parametrize(
    ("scenario_type", "builder"),
    [
        (ScenarioType.LOW_RISK_NO_FIRE, build_low_risk_no_fire_scenario),
        (ScenarioType.HIGH_RISK_NO_FIRE, build_high_risk_no_fire_scenario),
        (ScenarioType.ACTIVE_FIRE, build_active_fire_scenario),
    ],
)
def test_build_scenario_dispatches_to_matching_builder(scenario_type, builder):
    assert build_scenario(
        scenario_type=scenario_type,
        location=GOLAN_LOCATION,
        seed=99,
        incident_id="incident-golan-01",
    ) == builder(location=GOLAN_LOCATION, seed=99, incident_id="incident-golan-01")


def test_build_scenario_rejects_invalid_scenario_type():
    with pytest.raises(ValueError):
        build_scenario("active_fire")


def test_carmel_golan_active_fire_scenario_builds_interleaved_incidents():
    scenario = build_carmel_golan_active_fire_scenario(seed=123)

    assert scenario.seed == 123
    assert [incident.incident_id for incident in scenario.incidents] == [
        "incident-carmel-01",
        "incident-golan-01",
    ]
    assert [incident.location for incident in scenario.incidents] == [
        CARMEL_LOCATION,
        GOLAN_LOCATION,
    ]
    assert summarize_events(scenario) == [
        (0, SimulationEventType.WEATHER, "incident-carmel-01", 0),
        (10, SimulationEventType.WEATHER, "incident-golan-01", 0),
        (20, SimulationEventType.SATELLITE, "incident-carmel-01", 0),
        (35, SimulationEventType.SATELLITE, "incident-golan-01", 0),
        (40, SimulationEventType.NEWS, "incident-carmel-01", 0),
        (55, SimulationEventType.NEWS, "incident-golan-01", 0),
        (65, SimulationEventType.WEATHER, "incident-carmel-01", 1),
        (75, SimulationEventType.WEATHER, "incident-golan-01", 1),
        (80, SimulationEventType.SATELLITE, "incident-carmel-01", 1),
        (95, SimulationEventType.SATELLITE, "incident-golan-01", 1),
        (100, SimulationEventType.NEWS, "incident-carmel-01", 1),
        (115, SimulationEventType.NEWS, "incident-golan-01", 1),
    ]


def test_active_fire_resource_refresh_scenario_adds_resource_status_cycle():
    scenario = build_active_fire_resource_refresh_scenario(seed=42)

    resource_events = [event for event in scenario.events if event.event_type is SimulationEventType.RESOURCE_STATUS]

    assert [event.offset_seconds for event in resource_events] == [60, 90]
    assert resource_events[0].resource_status_change.new_status is ResourceStatus.UNAVAILABLE
    assert resource_events[1].resource_status_change.new_status is ResourceStatus.AVAILABLE
    assert resource_events[0].resource_status_change.selection_key == resource_events[1].resource_status_change.selection_key
