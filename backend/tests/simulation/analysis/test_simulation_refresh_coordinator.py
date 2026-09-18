"""Tests for simulation integration with the production US4.4 -> Stage 6
global planning bridge.

Stage 6 Task 48 (simulation cutover): SimulationRefreshCoordinator now
triggers exactly ONE global planning refresh per logical batch (never one
per FireEvent), via OperationalPlanningRefreshCoordinator.refresh_fire_events_batch/
refresh_resource - the exact same production path every other trigger uses.
"""
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
    OperationalPlanningRefreshBatchResult,
    OperationalPlanningRefreshResult,
)
from src.services.operational_refresh import OperationalRefreshResult, OperationalRefreshStatus
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


@dataclass(frozen=True)
class FakeGlobalPlanningRefreshResult:
    """Stand-in for GlobalPlanningRefreshResult."""

    status: str
    trigger: str


class FakeOperationalPlanningRefreshCoordinator:
    """Fake OperationalPlanningRefreshCoordinator: records calls, returns/raises as configured.

    Implements refresh_fire_events_batch (one call per logical batch) and
    refresh_resource, matching the real coordinator's Stage 6 interface -
    never a per-FireEvent refresh_fire_event fan-out.
    """

    def __init__(self, fail_ids=()) -> None:
        self.batch_calls = []
        self.resource_calls = []
        self.fail_ids = set(fail_ids)

    def refresh_fire_events_batch(self, *, fire_event_ids, trigger_type, as_of):
        self.batch_calls.append({"fire_event_ids": fire_event_ids, "trigger_type": trigger_type, "as_of": as_of})
        operational_results = []
        for fire_event_id in fire_event_ids:
            if fire_event_id in self.fail_ids:
                operational_results.append(
                    OperationalRefreshResult(
                        trigger_type=trigger_type,
                        status=OperationalRefreshStatus.FAILED,
                        success=False,
                        fire_event_id=fire_event_id,
                        as_of=as_of,
                        error_message=f"refresh failed for {fire_event_id}",
                    )
                )
            else:
                operational_results.append(
                    OperationalRefreshResult(
                        trigger_type=trigger_type,
                        status=OperationalRefreshStatus.REFRESHED,
                        success=True,
                        fire_event_id=fire_event_id,
                        as_of=as_of,
                    )
                )
        any_success = any(result.success for result in operational_results)
        global_planning_result = (
            FakeGlobalPlanningRefreshResult(status="activated", trigger=trigger_type.value) if any_success else None
        )
        return OperationalPlanningRefreshBatchResult(
            operational_results=tuple(operational_results), global_planning_result=global_planning_result
        )

    def refresh_resource(self, *, resource_id, new_status, as_of):
        self.resource_calls.append({"resource_id": resource_id, "new_status": new_status, "as_of": as_of})
        operational_result = OperationalRefreshResult(
            trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE,
            status=OperationalRefreshStatus.RESOURCE_UPDATED,
            success=True,
        )
        global_planning_result = FakeGlobalPlanningRefreshResult(
            status="activated", trigger=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE.value
        )
        return OperationalPlanningRefreshResult(
            operational_result=operational_result, global_planning_result=global_planning_result
        )


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
    assert bridge.batch_calls == [
        {"fire_event_ids": (5,), "trigger_type": OperationalRefreshTriggerType.WEATHER_UPDATE, "as_of": AS_OF}
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
    assert bridge.batch_calls == [
        {"fire_event_ids": (12,), "trigger_type": OperationalRefreshTriggerType.FIRE_EVENT_UPDATE, "as_of": AS_OF}
    ]


def test_fire_event_ids_are_deduped_and_processed_in_ascending_order_within_one_batch():
    bridge = FakeOperationalPlanningRefreshCoordinator()
    coordinator = make_coordinator(operational_planning_refresh=bridge)

    result = coordinator.refresh_fire_events(
        fire_event_ids=(9, 3, 9, 4),
        trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        as_of=AS_OF,
    )

    assert result.fire_event_ids == (3, 4, 9)
    # Exactly ONE batch call, never a per-FireEvent fan-out.
    assert len(bridge.batch_calls) == 1
    assert bridge.batch_calls[0]["fire_event_ids"] == (3, 4, 9)


def test_several_fire_events_sharing_one_update_trigger_exactly_one_global_refresh():
    """Task 34: several FireEvents sharing one logical upstream update must
    never each trigger their own separate global replan."""
    bridge = FakeOperationalPlanningRefreshCoordinator()
    coordinator = make_coordinator(operational_planning_refresh=bridge)

    result = coordinator.refresh_fire_events(
        fire_event_ids=(3, 4, 9),
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )

    assert len(bridge.batch_calls) == 1
    assert result.global_planning_result == FakeGlobalPlanningRefreshResult(
        status="activated", trigger=OperationalRefreshTriggerType.WEATHER_UPDATE.value
    )


def test_one_fire_event_failure_does_not_prevent_the_batch_global_refresh():
    bridge = FakeOperationalPlanningRefreshCoordinator(fail_ids={4})
    coordinator = make_coordinator(operational_planning_refresh=bridge)

    result = coordinator.refresh_fire_events(
        fire_event_ids=(3, 4, 9),
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )

    assert len(bridge.batch_calls) == 1
    assert result.failures == 1
    assert [item.status for item in result.refresh_results] == [
        OperationalRefreshStatus.REFRESHED,
        OperationalRefreshStatus.FAILED,
        OperationalRefreshStatus.REFRESHED,
    ]
    # At least one success in the batch -> the single global refresh still ran.
    assert result.global_planning_result is not None


def test_all_fire_events_failing_produces_no_global_planning_result():
    bridge = FakeOperationalPlanningRefreshCoordinator(fail_ids={3, 4, 9})
    coordinator = make_coordinator(operational_planning_refresh=bridge)

    result = coordinator.refresh_fire_events(
        fire_event_ids=(3, 4, 9),
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )

    assert result.failures == 3
    assert result.global_planning_result is None


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
    assert bridge.batch_calls == []


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
    assert bridge.batch_calls == []
