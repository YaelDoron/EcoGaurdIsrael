"""Tests for SimulationFireSeverityCoordinator trigger policy."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from src.agents.analysis.fire_detection_result import FireDetectionResult
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.models import FireEvent, FireEventStatus, FireSeverityAssessment, FireSeverityAssessmentStatus, FireSeverityLevel
from src.repositories.fire_event_repository import StoredFireEvent
from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment
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
from src.simulation.analysis import SimulationFireSeverityCoordinator
from src.simulation.analysis.simulation_fire_severity_coordinator import (
    DETECTION_FAILED_REASON,
    NO_ACTIVE_FIRE_EVENTS_NEAR_WEATHER_REASON,
    NO_AFFECTED_FIRE_EVENTS_REASON,
    NO_DETECTION_RESULT_REASON,
    NO_SOURCE_DATA_AVAILABLE_REASON,
    NON_SEVERITY_EVENT_REASON,
    SOURCE_EVENT_FAILED_REASON,
)
from src.simulation.analysis.simulation_fire_severity_result import SimulationFireSeverityResult

STARTED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
T0 = STARTED_AT
T20 = STARTED_AT + timedelta(seconds=20)
T65 = STARTED_AT + timedelta(seconds=65)
T80 = STARTED_AT + timedelta(seconds=80)


class FakeSeverityAgent:
    def __init__(self, fail_ids=()) -> None:
        self.fail_ids = set(fail_ids)
        self.calls = []

    def assess(self, fire_event_id, assessed_at):
        self.calls.append({"fire_event_id": fire_event_id, "assessed_at": assessed_at})
        if fire_event_id in self.fail_ids:
            raise RuntimeError(f"severity failed for {fire_event_id}")
        return StoredFireSeverityAssessment(
            assessment_id=fire_event_id + 1000,
            assessment=FireSeverityAssessment(
                fire_event_id=fire_event_id,
                assessed_at=assessed_at,
                status=FireSeverityAssessmentStatus.VALID,
                score=72.5,
                level=FireSeverityLevel.HIGH,
                methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
                methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
            ),
            weather_observation_ids=(1,),
            satellite_hotspot_ids=(2,),
            selected_frp_hotspot_id=2,
        )


class FakeFireEventRepository:
    def __init__(self, events=()) -> None:
        self.events = tuple(events)
        self.calls = []

    def get_response_eligible_events_near(self, latitude, longitude, radius_km, as_of):  # Task 9A: severity is response work
        self.calls.append(
            {
                "latitude": latitude,
                "longitude": longitude,
                "radius_km": radius_km,
                "as_of": as_of,
            }
        )
        return self.events

    def get_by_id(self, fire_event_id):
        """A registered event, else a CONFIRMED one (these tests are about severity of response-eligible fires)."""
        self.by_id_calls = getattr(self, "by_id_calls", []) + [fire_event_id]
        return next((e for e in self.events if e.id == fire_event_id), None) or stored_event(fire_event_id)


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
            SimulationEvent(65, SimulationEventType.WEATHER, incident.incident_id, 1),
            SimulationEvent(80, SimulationEventType.SATELLITE, incident.incident_id, 1),
        ),
    )


def make_execution_result(event, success=True, saved_count=1, duplicates_skipped=0, failed_count=0):
    return SimulationEventExecutionResult(
        event=event,
        success=success,
        generated_count=1,
        saved_count=saved_count,
        duplicates_skipped=duplicates_skipped,
        failed_count=failed_count,
        error_message=None if success else "simulated failure",
    )


def make_detection_result(event_ids=(55,), success=True) -> FireDetectionResult:
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
        event_ids=event_ids,
    )


def stored_event(event_id, latitude=CARMEL_LOCATION.latitude, longitude=CARMEL_LOCATION.longitude):
    return StoredFireEvent(
        id=event_id,
        event=FireEvent(
            latitude=latitude,
            longitude=longitude,
            detected_at=T20,
            updated_at=T20,
            status=FireEventStatus.CONFIRMED,  # only response-eligible events are assessed (Task 9A)
            detection_confidence=0.6,
            methodology="ECOGUARD_ACTIVE_FIRE_DETECTION",
            methodology_version="1.0",
        ),
    )


def handle(coordinator, scenario, event, timestamp, execution_result=None, detection_result=None):
    return coordinator.handle_event(
        scenario=scenario,
        event=event,
        execution_result=execution_result or make_execution_result(event),
        event_timestamp=timestamp,
        detection_result=detection_result,
    )


def event_for(
    scenario: SimulationScenario,
    incident_id: str,
    event_type: SimulationEventType,
    offset_seconds: int,
) -> SimulationEvent:
    return next(
        event
        for event in scenario.events
        if event.incident_id == incident_id
        and event.event_type is event_type
        and event.offset_seconds == offset_seconds
    )


def test_weather_before_fire_invokes_no_severity_when_no_active_fire_event_exists():
    scenario = make_scenario()
    event = scenario.events[0]
    severity_agent = FakeSeverityAgent()
    repository = FakeFireEventRepository(events=())

    result = handle(SimulationFireSeverityCoordinator(severity_agent, repository), scenario, event, T0)

    assert result.triggered is False
    assert result.reason == NO_ACTIVE_FIRE_EVENTS_NEAR_WEATHER_REASON
    assert severity_agent.calls == []
    assert repository.calls[0]["as_of"] == T0


def test_satellite_created_event_triggers_severity_after_detection_result():
    scenario = make_scenario()
    event = scenario.events[1]
    severity_agent = FakeSeverityAgent()

    result = handle(
        SimulationFireSeverityCoordinator(severity_agent, FakeFireEventRepository()),
        scenario,
        event,
        T20,
        detection_result=make_detection_result(event_ids=(55,)),
    )

    assert result.triggered is True
    assert severity_agent.calls == [{"fire_event_id": 55, "assessed_at": T20}]
    assert tuple(stored.assessment.fire_event_id for stored in result.assessment_results) == (55,)


def test_news_event_runs_no_severity_even_with_detection_event_ids():
    scenario = make_scenario()
    event = scenario.events[2]
    severity_agent = FakeSeverityAgent()

    result = handle(
        SimulationFireSeverityCoordinator(severity_agent, FakeFireEventRepository()),
        scenario,
        event,
        STARTED_AT + timedelta(seconds=40),
        detection_result=make_detection_result(event_ids=(55,)),
    )

    assert result.triggered is False
    assert result.reason == NON_SEVERITY_EVENT_REASON
    assert severity_agent.calls == []


def test_weather_reassessment_uses_nearby_active_events_and_exact_timestamp():
    scenario = make_scenario()
    event = scenario.events[3]
    severity_agent = FakeSeverityAgent()
    repository = FakeFireEventRepository(events=(stored_event(55),))

    result = handle(SimulationFireSeverityCoordinator(severity_agent, repository), scenario, event, T65)

    assert result.triggered is True
    assert severity_agent.calls == [{"fire_event_id": 55, "assessed_at": T65}]
    assert repository.calls[0]["latitude"] == CARMEL_LOCATION.latitude
    assert repository.calls[0]["longitude"] == CARMEL_LOCATION.longitude
    assert repository.calls[0]["as_of"] == T65


def test_weather_ignores_distant_events_by_using_repository_relevance_result():
    scenario = make_scenario()
    event = scenario.events[3]
    severity_agent = FakeSeverityAgent()
    repository = FakeFireEventRepository(events=())

    result = handle(SimulationFireSeverityCoordinator(severity_agent, repository), scenario, event, T65)

    assert result.triggered is False
    assert result.reason == NO_ACTIVE_FIRE_EVENTS_NEAR_WEATHER_REASON
    assert severity_agent.calls == []


def test_inactive_events_are_not_assessed_when_repository_excludes_them():
    scenario = make_scenario()
    event = scenario.events[3]
    severity_agent = FakeSeverityAgent()
    repository = FakeFireEventRepository(events=())

    result = handle(SimulationFireSeverityCoordinator(severity_agent, repository), scenario, event, T65)

    assert result.triggered is False
    assert severity_agent.calls == []


def test_second_satellite_update_assesses_same_event_once_at_new_timestamp():
    scenario = make_scenario()
    event = scenario.events[4]
    severity_agent = FakeSeverityAgent()

    result = handle(
        SimulationFireSeverityCoordinator(severity_agent, FakeFireEventRepository()),
        scenario,
        event,
        T80,
        detection_result=make_detection_result(event_ids=(55,)),
    )

    assert result.triggered is True
    assert severity_agent.calls == [{"fire_event_id": 55, "assessed_at": T80}]


def test_multiple_affected_events_are_deduplicated_and_sorted():
    scenario = make_scenario()
    event = scenario.events[1]
    severity_agent = FakeSeverityAgent()

    handle(
        SimulationFireSeverityCoordinator(severity_agent, FakeFireEventRepository()),
        scenario,
        event,
        T20,
        detection_result=make_detection_result(event_ids=(9, 5, 9)),
    )

    assert severity_agent.calls == [
        {"fire_event_id": 5, "assessed_at": T20},
        {"fire_event_id": 9, "assessed_at": T20},
    ]


def test_multi_incident_weather_uses_current_incident_location_without_passing_incident_id_to_agent():
    scenario = build_carmel_golan_active_fire_scenario(seed=42)
    carmel_weather = event_for(scenario, "incident-carmel-01", SimulationEventType.WEATHER, 65)
    golan_weather = event_for(scenario, "incident-golan-01", SimulationEventType.WEATHER, 75)
    carmel_agent = FakeSeverityAgent()
    golan_agent = FakeSeverityAgent()

    handle(
        SimulationFireSeverityCoordinator(carmel_agent, FakeFireEventRepository(events=(stored_event(10),))),
        scenario,
        carmel_weather,
        T65,
    )
    handle(
        SimulationFireSeverityCoordinator(golan_agent, FakeFireEventRepository(events=(stored_event(20),))),
        scenario,
        golan_weather,
        T65,
    )

    assert carmel_agent.calls == [{"fire_event_id": 10, "assessed_at": T65}]
    assert golan_agent.calls == [{"fire_event_id": 20, "assessed_at": T65}]


def test_multi_incident_weather_query_coordinates_are_area_specific():
    scenario = build_carmel_golan_active_fire_scenario(seed=42)
    carmel_weather = event_for(scenario, "incident-carmel-01", SimulationEventType.WEATHER, 65)
    golan_weather = event_for(scenario, "incident-golan-01", SimulationEventType.WEATHER, 75)
    carmel_repository = FakeFireEventRepository(events=(stored_event(10),))
    golan_repository = FakeFireEventRepository(events=(stored_event(20),))

    handle(SimulationFireSeverityCoordinator(FakeSeverityAgent(), carmel_repository), scenario, carmel_weather, T65)
    handle(SimulationFireSeverityCoordinator(FakeSeverityAgent(), golan_repository), scenario, golan_weather, T65)

    assert carmel_repository.calls[0]["latitude"] == CARMEL_LOCATION.latitude
    assert golan_repository.calls[0]["latitude"] == GOLAN_LOCATION.latitude


def test_failed_or_empty_source_persistence_does_not_trigger_severity():
    scenario = make_scenario()
    event = scenario.events[1]
    severity_agent = FakeSeverityAgent()

    failed = handle(
        SimulationFireSeverityCoordinator(severity_agent, FakeFireEventRepository()),
        scenario,
        event,
        T20,
        execution_result=make_execution_result(event, success=False, saved_count=0, failed_count=1),
        detection_result=make_detection_result(),
    )
    empty = handle(
        SimulationFireSeverityCoordinator(severity_agent, FakeFireEventRepository()),
        scenario,
        event,
        T20,
        execution_result=make_execution_result(event, saved_count=0, duplicates_skipped=0),
        detection_result=make_detection_result(),
    )

    assert failed.reason == SOURCE_EVENT_FAILED_REASON
    assert empty.reason == NO_SOURCE_DATA_AVAILABLE_REASON
    assert severity_agent.calls == []


def test_duplicate_source_rows_remain_usable_for_severity_trigger():
    scenario = make_scenario()
    event = scenario.events[1]
    severity_agent = FakeSeverityAgent()

    result = handle(
        SimulationFireSeverityCoordinator(severity_agent, FakeFireEventRepository()),
        scenario,
        event,
        T20,
        execution_result=make_execution_result(event, saved_count=0, duplicates_skipped=1),
        detection_result=make_detection_result(event_ids=(55,)),
    )

    assert result.triggered is True
    assert severity_agent.calls == [{"fire_event_id": 55, "assessed_at": T20}]


@pytest.mark.parametrize(
    ("detection_result", "expected_reason"),
    [
        (None, NO_DETECTION_RESULT_REASON),
        (make_detection_result(success=False), DETECTION_FAILED_REASON),
        (make_detection_result(event_ids=()), NO_AFFECTED_FIRE_EVENTS_REASON),
    ],
)
def test_satellite_requires_successful_detection_with_affected_events(detection_result, expected_reason):
    scenario = make_scenario()
    event = scenario.events[1]
    severity_agent = FakeSeverityAgent()

    result = handle(
        SimulationFireSeverityCoordinator(severity_agent, FakeFireEventRepository()),
        scenario,
        event,
        T20,
        detection_result=detection_result,
    )

    assert result.triggered is False
    assert result.reason == expected_reason
    assert severity_agent.calls == []


def test_severity_agent_failure_is_recorded_without_retrying():
    scenario = make_scenario()
    event = scenario.events[1]
    severity_agent = FakeSeverityAgent(fail_ids=(55,))

    result = handle(
        SimulationFireSeverityCoordinator(severity_agent, FakeFireEventRepository()),
        scenario,
        event,
        T20,
        detection_result=make_detection_result(event_ids=(55,)),
    )

    assert result.triggered is True
    assert result.assessment_results == ()
    assert result.failed_fire_event_ids == (55,)
    assert result.error_messages == ("severity failed for 55",)
    assert severity_agent.calls == [{"fire_event_id": 55, "assessed_at": T20}]


def test_partial_failure_records_successes_and_failures():
    scenario = make_scenario()
    event = scenario.events[1]
    severity_agent = FakeSeverityAgent(fail_ids=(9,))

    result = handle(
        SimulationFireSeverityCoordinator(severity_agent, FakeFireEventRepository()),
        scenario,
        event,
        T20,
        detection_result=make_detection_result(event_ids=(5, 9)),
    )

    assert tuple(item.assessment.fire_event_id for item in result.assessment_results) == (5,)
    assert result.failed_fire_event_ids == (9,)
    assert result.error_messages == ("severity failed for 9",)


def test_simulation_fire_severity_result_is_immutable():
    result = SimulationFireSeverityResult(
        triggered=True,
        assessment_results=(FakeSeverityAgent().assess(55, T20),),
    )

    with pytest.raises(FrozenInstanceError):
        result.triggered = False


def test_non_triggered_result_requires_reason():
    with pytest.raises(ValueError):
        SimulationFireSeverityResult(triggered=False)


# --- Task 9A: SUSPECTED events are active for monitoring but not assessed ---


def test_a_suspected_event_from_detection_is_not_assessed_but_a_confirmed_one_is():
    from dataclasses import replace

    suspected = replace(stored_event(7), event=replace(stored_event(7).event, status=FireEventStatus.SUSPECTED))
    repository = FakeFireEventRepository(events=(suspected, stored_event(8)))
    agent = FakeSeverityAgent()
    coordinator = SimulationFireSeverityCoordinator(severity_agent=agent, fire_event_repository=repository)
    scenario = make_scenario()
    satellite = next(e for e in scenario.events if e.event_type is SimulationEventType.SATELLITE)

    result = handle(coordinator, scenario, satellite, T20, detection_result=make_detection_result((7, 8)))

    assert [call["fire_event_id"] for call in agent.calls] == [8]  # only the CONFIRMED event was assessed
    assert result.triggered


def test_only_suspected_events_trigger_no_severity_work_at_all():
    from dataclasses import replace

    suspected = replace(stored_event(7), event=replace(stored_event(7).event, status=FireEventStatus.SUSPECTED))
    agent = FakeSeverityAgent()
    coordinator = SimulationFireSeverityCoordinator(severity_agent=agent, fire_event_repository=FakeFireEventRepository(events=(suspected,)))
    scenario = make_scenario()
    satellite = next(e for e in scenario.events if e.event_type is SimulationEventType.SATELLITE)

    result = handle(coordinator, scenario, satellite, T20, detection_result=make_detection_result((7,)))

    assert agent.calls == [] and not result.triggered and result.reason == "no_response_eligible_fire_events"
