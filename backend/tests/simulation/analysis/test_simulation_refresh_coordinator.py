"""Tests for simulation integration with the production US4.4 -> US5.4 bridge."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from src.agents.analysis.fire_detection_result import FireDetectionResult
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.models.resource_status import ResourceStatus
from src.repositories.fire_event_repository import StoredFireEvent
from src.services.operational_planning_refresh.operational_planning_refresh_result import (
    OperationalPlanningRefreshResult,
)
from src.services.operational_refresh import OperationalRefreshResult, OperationalRefreshStatus
from src.services.response_planning import PlanningRefreshResult, PlanningRefreshStatus
from src.simulation import (
    CARMEL_LOCATION,
    ScenarioType,
    SimulatedIncident,
    SimulationEvent,
    SimulationEventExecutionResult,
    SimulationEventType,
    SimulationRefreshCoordinator,
    SimulationResourceStatusChange,
    SimulationScenario,
)

AS_OF = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


class FakeOperationalPlanningRefreshCoordinator:
    """Fake OperationalPlanningRefreshCoordinator: records calls, returns/raises as configured.

    Returns a real PlanningRefreshResult for successful fire-event refreshes
    (SimulationRefreshResult validates planning_results against that exact
    class), standing in for what ResponsePlanningRefreshOrchestrator would
    produce in production.
    """

    def __init__(self, fail_ids=()) -> None:
        self.fire_calls = []
        self.resource_calls = []
        self.fail_ids = set(fail_ids)

    def refresh_fire_event(self, *, fire_event_id, trigger_type, as_of):
        self.fire_calls.append({"fire_event_id": fire_event_id, "trigger_type": trigger_type, "as_of": as_of})
        if fire_event_id in self.fail_ids:
            raise RuntimeError(f"refresh failed for {fire_event_id}")
        operational_result = OperationalRefreshResult(
            trigger_type=trigger_type,
            status=OperationalRefreshStatus.REFRESHED,
            success=True,
            fire_event_id=fire_event_id,
            as_of=as_of,
        )
        planning_result = PlanningRefreshResult(
            status=PlanningRefreshStatus.REFRESHED,
            fire_event_id=fire_event_id,
            route_planning_run_id=100 + fire_event_id,
            response_plan_id=200 + fire_event_id,
            comparison_id=300 + fire_event_id,
        )
        return OperationalPlanningRefreshResult(
            operational_result=operational_result,
            planning_results=(planning_result,),
        )

    def refresh_resource(self, *, resource_id, new_status, as_of):
        self.resource_calls.append({"resource_id": resource_id, "new_status": new_status, "as_of": as_of})
        operational_result = OperationalRefreshResult(
            trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE,
            status=OperationalRefreshStatus.RESOURCE_UPDATED,
            success=True,
        )
        return OperationalPlanningRefreshResult(operational_result=operational_result, planning_results=())


class FakeFireEventRepository:
    def __init__(self, event_ids=(3, 9, 4)) -> None:
        self.event_ids = tuple(event_ids)
        self.calls = []

    def get_active_events_near(self, *, latitude, longitude, radius_km, as_of):
        self.calls.append({"latitude": latitude, "longitude": longitude, "radius_km": radius_km, "as_of": as_of})
        return tuple(
            StoredFireEvent(
                id=event_id,
                event=FireEvent(
                    latitude=latitude,
                    longitude=longitude,
                    detected_at=as_of,
                    updated_at=as_of,
                    status=FireEventStatus.CONFIRMED,
                    detection_confidence=0.8,
                    methodology="test",
                    methodology_version="1.0",
                ),
            )
            for event_id in self.event_ids
        )


@dataclass
class ResourceRecord:
    id: str
    status: ResourceStatus


class FakeOperationalCoordinator:
    def __init__(self, resources) -> None:
        self.resources = list(resources)
        self.calls = []

    def select_available_resource(self, latitude, longitude):
        self.calls.append({"latitude": latitude, "longitude": longitude})
        available = sorted(
            (resource for resource in self.resources if resource.status is ResourceStatus.AVAILABLE),
            key=lambda resource: resource.id,
        )
        return available[0] if available else None


def make_scenario() -> SimulationScenario:
    return SimulationScenario(
        duration_seconds=120,
        seed=42,
        incidents=(
            SimulatedIncident(
                incident_id="incident-carmel-01",
                scenario_type=ScenarioType.ACTIVE_FIRE,
                location=CARMEL_LOCATION,
            ),
        ),
        events=(
            SimulationEvent(0, SimulationEventType.WEATHER, "incident-carmel-01", 0),
        ),
    )


def execution_result(event: SimulationEvent, *, success=True) -> SimulationEventExecutionResult:
    return SimulationEventExecutionResult(
        event=event,
        success=success,
        generated_count=1,
        saved_count=1 if success else 0,
        duplicates_skipped=0,
        failed_count=0 if success else 1,
        error_message=None if success else "source failed",
    )


def detection_result(event_ids=(9, 3, 9, 4), *, success=True) -> FireDetectionResult:
    return FireDetectionResult(
        success=success,
        candidates_processed=1,
        no_event_count=0,
        events_created=1 if event_ids else 0,
        events_updated=0,
        event_ids=tuple(event_ids),
        error_message=None if success else "detection failed",
    )


def make_coordinator(operational_planning_refresh=None, event_repository=None, operational_coordinator=None):
    return SimulationRefreshCoordinator(
        operational_planning_refresh=operational_planning_refresh or FakeOperationalPlanningRefreshCoordinator(),
        fire_event_repository=event_repository or FakeFireEventRepository(),
        operational_coordinator=operational_coordinator,
    )


def test_weather_source_event_maps_to_weather_update_with_exact_timestamp():
    scenario = make_scenario()
    event = scenario.events[0]
    bridge = FakeOperationalPlanningRefreshCoordinator()
    coordinator = make_coordinator(operational_planning_refresh=bridge, event_repository=FakeFireEventRepository((5,)))

    result = coordinator.handle_event(
        scenario=scenario,
        event=event,
        execution_result=execution_result(event),
        event_timestamp=AS_OF,
    )

    assert result.triggered is True
    assert result.fire_event_ids == (5,)
    assert bridge.fire_calls == [
        {"fire_event_id": 5, "trigger_type": OperationalRefreshTriggerType.WEATHER_UPDATE, "as_of": AS_OF}
    ]


@pytest.mark.parametrize("event_type", [SimulationEventType.SATELLITE, SimulationEventType.NEWS])
def test_detection_events_map_to_fire_event_update(event_type):
    scenario = make_scenario()
    event = SimulationEvent(20, event_type, "incident-carmel-01", 0)
    bridge = FakeOperationalPlanningRefreshCoordinator()
    coordinator = make_coordinator(operational_planning_refresh=bridge)

    result = coordinator.handle_event(
        scenario=scenario,
        event=event,
        execution_result=execution_result(event),
        event_timestamp=AS_OF,
        detection_result=detection_result((12,)),
    )

    assert result.fire_event_ids == (12,)
    assert bridge.fire_calls == [
        {"fire_event_id": 12, "trigger_type": OperationalRefreshTriggerType.FIRE_EVENT_UPDATE, "as_of": AS_OF}
    ]


def test_fire_event_ids_are_deduped_and_processed_in_ascending_order():
    bridge = FakeOperationalPlanningRefreshCoordinator()
    coordinator = make_coordinator(operational_planning_refresh=bridge)

    result = coordinator.refresh_fire_events(
        fire_event_ids=(9, 3, 9, 4),
        trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        as_of=AS_OF,
    )

    assert result.fire_event_ids == (3, 4, 9)
    assert [call["fire_event_id"] for call in bridge.fire_calls] == [3, 4, 9]


def test_one_fire_event_failure_does_not_prevent_other_refreshes():
    bridge = FakeOperationalPlanningRefreshCoordinator(fail_ids={4})
    coordinator = make_coordinator(operational_planning_refresh=bridge)

    result = coordinator.refresh_fire_events(
        fire_event_ids=(3, 4, 9),
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )

    assert [call["fire_event_id"] for call in bridge.fire_calls] == [3, 4, 9]
    assert result.failures == 1
    assert [item.status for item in result.refresh_results] == [
        OperationalRefreshStatus.REFRESHED,
        OperationalRefreshStatus.FAILED,
        OperationalRefreshStatus.REFRESHED,
    ]
    # The failed FireEvent produced no planning result; the other two did.
    assert [item.fire_event_id for item in result.planning_results] == [3, 9]


def test_successful_fire_event_refresh_surfaces_its_planning_result():
    bridge = FakeOperationalPlanningRefreshCoordinator()
    coordinator = make_coordinator(operational_planning_refresh=bridge)

    result = coordinator.refresh_fire_events(
        fire_event_ids=(3,),
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )

    assert len(result.planning_results) == 1
    assert result.planning_results[0].status is PlanningRefreshStatus.REFRESHED
    assert result.planning_results[0].fire_event_id == 3


def test_resource_event_maps_to_resource_refresh_not_fire_refresh():
    scenario = make_scenario()
    event = SimulationEvent(
        60,
        SimulationEventType.RESOURCE_STATUS,
        "incident-carmel-01",
        0,
        SimulationResourceStatusChange(new_status=ResourceStatus.UNAVAILABLE, resource_id="TRUCK-A"),
    )
    bridge = FakeOperationalPlanningRefreshCoordinator()
    coordinator = make_coordinator(operational_planning_refresh=bridge)

    result = coordinator.handle_event(
        scenario=scenario,
        event=event,
        execution_result=execution_result(event),
        event_timestamp=AS_OF,
    )

    assert result.triggered is True
    assert bridge.resource_calls == [{"resource_id": "TRUCK-A", "new_status": ResourceStatus.UNAVAILABLE, "as_of": AS_OF}]
    assert bridge.fire_calls == []


def test_resource_event_passes_the_event_timestamp_as_of_to_the_bridge():
    scenario = make_scenario()
    event = SimulationEvent(
        60,
        SimulationEventType.RESOURCE_STATUS,
        "incident-carmel-01",
        0,
        SimulationResourceStatusChange(new_status=ResourceStatus.UNAVAILABLE, resource_id="TRUCK-A"),
    )
    bridge = FakeOperationalPlanningRefreshCoordinator()
    coordinator = make_coordinator(operational_planning_refresh=bridge)
    later = AS_OF.replace(minute=1)

    coordinator.handle_event(scenario=scenario, event=event, execution_result=execution_result(event), event_timestamp=later)

    assert bridge.resource_calls[0]["as_of"] == later


def test_resource_selection_is_deterministic_and_reused_for_return_transition():
    scenario = make_scenario()
    operational = FakeOperationalCoordinator(
        [
            ResourceRecord("TRUCK-B", ResourceStatus.AVAILABLE),
            ResourceRecord("TRUCK-A", ResourceStatus.AVAILABLE),
        ]
    )
    bridge = FakeOperationalPlanningRefreshCoordinator()
    coordinator = make_coordinator(operational_planning_refresh=bridge, operational_coordinator=operational)
    first = SimulationEvent(
        60,
        SimulationEventType.RESOURCE_STATUS,
        "incident-carmel-01",
        0,
        SimulationResourceStatusChange(new_status=ResourceStatus.UNAVAILABLE, selection_key="primary"),
    )
    second = SimulationEvent(
        90,
        SimulationEventType.RESOURCE_STATUS,
        "incident-carmel-01",
        1,
        SimulationResourceStatusChange(new_status=ResourceStatus.AVAILABLE, selection_key="primary"),
    )

    coordinator.handle_event(scenario=scenario, event=first, execution_result=execution_result(first), event_timestamp=AS_OF)
    operational.resources[0].status = ResourceStatus.UNAVAILABLE
    coordinator.handle_event(scenario=scenario, event=second, execution_result=execution_result(second), event_timestamp=AS_OF)

    assert [
        {"resource_id": call["resource_id"], "new_status": call["new_status"]} for call in bridge.resource_calls
    ] == [
        {"resource_id": "TRUCK-A", "new_status": ResourceStatus.UNAVAILABLE},
        {"resource_id": "TRUCK-A", "new_status": ResourceStatus.AVAILABLE},
    ]
    assert len(operational.calls) == 1


@pytest.mark.parametrize(
    "new_status",
    [ResourceStatus.UNAVAILABLE, ResourceStatus.AVAILABLE, ResourceStatus.ASSIGNED, ResourceStatus.AVAILABLE],
)
def test_resource_transitions_are_forwarded_to_the_bridge(new_status):
    scenario = make_scenario()
    event = SimulationEvent(
        60,
        SimulationEventType.RESOURCE_STATUS,
        "incident-carmel-01",
        0,
        SimulationResourceStatusChange(new_status=new_status, resource_id="TRUCK-A"),
    )
    bridge = FakeOperationalPlanningRefreshCoordinator()
    coordinator = make_coordinator(operational_planning_refresh=bridge)

    coordinator.handle_event(scenario=scenario, event=event, execution_result=execution_result(event), event_timestamp=AS_OF)

    assert bridge.resource_calls == [{"resource_id": "TRUCK-A", "new_status": new_status, "as_of": AS_OF}]


def test_resource_event_does_not_use_detection_result_or_fire_refresh():
    scenario = make_scenario()
    event = SimulationEvent(
        60,
        SimulationEventType.RESOURCE_STATUS,
        "incident-carmel-01",
        0,
        SimulationResourceStatusChange(new_status=ResourceStatus.ASSIGNED, resource_id="TRUCK-A"),
    )
    bridge = FakeOperationalPlanningRefreshCoordinator()
    coordinator = make_coordinator(operational_planning_refresh=bridge)

    coordinator.handle_event(
        scenario=scenario,
        event=event,
        execution_result=execution_result(event),
        event_timestamp=AS_OF,
        detection_result=detection_result((100,)),
    )

    assert bridge.resource_calls == [{"resource_id": "TRUCK-A", "new_status": ResourceStatus.ASSIGNED, "as_of": AS_OF}]
    assert bridge.fire_calls == []
