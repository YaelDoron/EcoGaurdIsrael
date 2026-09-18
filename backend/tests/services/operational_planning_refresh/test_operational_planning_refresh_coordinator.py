"""Tests for OperationalPlanningRefreshCoordinator (the US4.4 -> Stage 6
global-planning bridge).

Uses fakes for both the operational orchestrator and the global planning
collaborator: this suite proves the coordinator's own composition logic
(when it calls the global refresh, with what arguments, how it aggregates
results) without re-deriving or re-testing either US4.4's or
GlobalPlanningRefreshCoordinator's own already-covered internal behavior.

Stage 6 Task 47 (production cutover): this coordinator now calls a
GlobalPlanningRefreshPort exactly ONCE per operational change - never a
per-FireEvent fan-out. The fake below never imports the real
GlobalPlanningRefreshResult/GlobalPlanningRefreshStatus classes: the
coordinator itself never inspects their shape beyond passing them through,
so a lightweight local stand-in is sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.models.resource_status import ResourceStatus
from src.services.operational_planning_refresh.operational_planning_refresh_coordinator import (
    OperationalPlanningRefreshCoordinator,
)
from src.services.operational_refresh.operational_refresh_result import (
    OperationalRefreshResult,
    OperationalRefreshStatus,
)

AS_OF = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
FIRE_EVENT_ID = 42


@dataclass(frozen=True)
class FakeGlobalPlanningRefreshResult:
    """Stand-in for GlobalPlanningRefreshResult - see module docstring."""

    status: str
    trigger: str
    global_planning_run_id: int | None = None


class FakeGlobalPlanningRefresh:
    """Fake GlobalPlanningRefreshPort: records calls, returns a canned/mapped result."""

    def __init__(self, *, result: FakeGlobalPlanningRefreshResult | None = None) -> None:
        self.calls: list[dict] = []
        self._result = result

    def refresh(self, *, trigger: str, as_of: datetime) -> FakeGlobalPlanningRefreshResult:
        self.calls.append({"trigger": trigger, "as_of": as_of})
        if self._result is not None:
            return self._result
        return FakeGlobalPlanningRefreshResult(status="no_op", trigger=trigger)


class FakeOperationalRefreshOrchestrator:
    """Fake OperationalRefreshOrchestrator: records calls, returns/raises as configured."""

    def __init__(
        self,
        *,
        fire_event_result: OperationalRefreshResult | None = None,
        resource_result: OperationalRefreshResult | None = None,
        raise_on_fire_event: Exception | None = None,
        raise_on_resource: Exception | None = None,
        call_log: list[str] | None = None,
    ) -> None:
        self.fire_event_calls: list[dict] = []
        self.resource_calls: list[dict] = []
        self._fire_event_result = fire_event_result
        self._resource_result = resource_result
        self._raise_on_fire_event = raise_on_fire_event
        self._raise_on_resource = raise_on_resource
        self._call_log = call_log

    def refresh_fire_event(self, *, fire_event_id: int, trigger_type: OperationalRefreshTriggerType, as_of: datetime):
        self.fire_event_calls.append({"fire_event_id": fire_event_id, "trigger_type": trigger_type, "as_of": as_of})
        if self._call_log is not None:
            self._call_log.append("operational")
        if self._raise_on_fire_event is not None:
            raise self._raise_on_fire_event
        return self._fire_event_result

    def refresh_resource(self, *, resource_id, new_status: ResourceStatus):
        self.resource_calls.append({"resource_id": resource_id, "new_status": new_status})
        if self._call_log is not None:
            self._call_log.append("operational")
        if self._raise_on_resource is not None:
            raise self._raise_on_resource
        return self._resource_result


class LoggingGlobalPlanningRefresh(FakeGlobalPlanningRefresh):
    """FakeGlobalPlanningRefresh that also appends to a shared call-order log."""

    def __init__(self, *, call_log: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._call_log = call_log

    def refresh(self, *, trigger: str, as_of: datetime) -> FakeGlobalPlanningRefreshResult:
        self._call_log.append("planning")
        return super().refresh(trigger=trigger, as_of=as_of)


def operational_success(
    *,
    trigger_type: OperationalRefreshTriggerType,
    fire_event_id: int | None = None,
    as_of: datetime | None = None,
    status: OperationalRefreshStatus = OperationalRefreshStatus.REFRESHED,
) -> OperationalRefreshResult:
    return OperationalRefreshResult(
        trigger_type=trigger_type,
        status=status,
        success=True,
        fire_event_id=fire_event_id,
        as_of=as_of,
    )


def operational_failure(
    *,
    trigger_type: OperationalRefreshTriggerType,
    fire_event_id: int | None = None,
    as_of: datetime | None = None,
    status: OperationalRefreshStatus = OperationalRefreshStatus.FAILED,
    error_message: str = "operational refresh failed",
) -> OperationalRefreshResult:
    return OperationalRefreshResult(
        trigger_type=trigger_type,
        status=status,
        success=False,
        fire_event_id=fire_event_id,
        as_of=as_of,
        error_message=error_message,
    )


def make_coordinator(
    *,
    operational: FakeOperationalRefreshOrchestrator,
    planning: FakeGlobalPlanningRefresh,
) -> OperationalPlanningRefreshCoordinator:
    return OperationalPlanningRefreshCoordinator(
        operational_refresh_orchestrator=operational,
        global_planning_refresh=planning,
    )


# ---------------------------------------------------------------------------
# refresh_fire_event: environmental / FireEvent trigger
# ---------------------------------------------------------------------------


def test_fire_event_refresh_triggers_exactly_one_global_refresh():
    operational = FakeOperationalRefreshOrchestrator(
        fire_event_result=operational_success(
            trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE, fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
        )
    )
    planning = FakeGlobalPlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE, as_of=AS_OF
    )

    assert len(planning.calls) == 1
    assert planning.calls[0] == {"trigger": OperationalRefreshTriggerType.WEATHER_UPDATE.value, "as_of": AS_OF}
    assert result.global_planning_result == FakeGlobalPlanningRefreshResult(
        status="no_op", trigger=OperationalRefreshTriggerType.WEATHER_UPDATE.value
    )


def test_fire_event_refresh_scenario_a_changed_targets_surfaces_the_activated_result_untouched():
    """The global planning collaborator reports an ACTIVATED cycle; the
    coordinator must surface that exact result, not reinterpret or duplicate it."""
    activated = FakeGlobalPlanningRefreshResult(
        status="activated", trigger=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE.value, global_planning_run_id=901
    )
    operational = FakeOperationalRefreshOrchestrator(
        fire_event_result=operational_success(trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE)
    )
    planning = FakeGlobalPlanningRefresh(result=activated)
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE, as_of=AS_OF
    )

    assert result.global_planning_result == activated
    assert len(planning.calls) == 1


def test_fire_event_refresh_scenario_b_unchanged_state_surfaces_the_no_op_result_untouched():
    """The global planning collaborator reports NO_OP (fingerprint
    unchanged); the coordinator still calls it exactly once and surfaces the
    NO_OP result as-is - it never decides NO_OP itself."""
    no_op = FakeGlobalPlanningRefreshResult(
        status="no_op", trigger=OperationalRefreshTriggerType.SEVERITY_UPDATE.value, global_planning_run_id=5
    )
    operational = FakeOperationalRefreshOrchestrator(
        fire_event_result=operational_success(trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE)
    )
    planning = FakeGlobalPlanningRefresh(result=no_op)
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE, as_of=AS_OF
    )

    assert result.global_planning_result == no_op
    assert len(planning.calls) == 1


def test_fire_event_refresh_skips_global_planning_when_operational_refresh_fails():
    operational = FakeOperationalRefreshOrchestrator(
        fire_event_result=operational_failure(trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE)
    )
    planning = FakeGlobalPlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE, as_of=AS_OF
    )

    assert result.operational_result.success is False
    assert result.global_planning_result is None
    assert planning.calls == []


def test_fire_event_refresh_wraps_unexpected_operational_exception_into_failed_result_and_skips_planning():
    operational = FakeOperationalRefreshOrchestrator(raise_on_fire_event=RuntimeError("boom"))
    planning = FakeGlobalPlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE, as_of=AS_OF
    )

    assert result.operational_result.success is False
    assert result.operational_result.status is OperationalRefreshStatus.FAILED
    assert result.operational_result.error_message == "boom"
    assert result.global_planning_result is None
    assert planning.calls == []


def test_fire_event_refresh_calls_operational_before_planning():
    call_log: list[str] = []
    operational = FakeOperationalRefreshOrchestrator(
        fire_event_result=operational_success(trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE),
        call_log=call_log,
    )
    planning = LoggingGlobalPlanningRefresh(call_log=call_log)
    coordinator = make_coordinator(operational=operational, planning=planning)

    coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE, as_of=AS_OF
    )

    assert call_log == ["operational", "planning"]


def test_fire_event_refresh_passes_through_the_same_as_of_used_for_the_operational_refresh():
    operational = FakeOperationalRefreshOrchestrator(
        fire_event_result=operational_success(trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE)
    )
    planning = FakeGlobalPlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE, as_of=AS_OF
    )

    assert operational.fire_event_calls[0]["as_of"] == AS_OF
    assert planning.calls[0]["as_of"] == AS_OF


def test_fire_event_refresh_passes_the_trigger_type_value_as_the_global_refresh_trigger():
    operational = FakeOperationalRefreshOrchestrator(
        fire_event_result=operational_success(trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE)
    )
    planning = FakeGlobalPlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE, as_of=AS_OF
    )

    assert planning.calls[0]["trigger"] == "severity_update"


# ---------------------------------------------------------------------------
# refresh_resource: resource-status trigger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "new_status",
    [ResourceStatus.UNAVAILABLE, ResourceStatus.ASSIGNED, ResourceStatus.AVAILABLE],
)
def test_resource_refresh_triggers_exactly_one_global_refresh_regardless_of_active_event_count(new_status):
    """AVAILABLE->UNAVAILABLE, AVAILABLE->ASSIGNED, and back to AVAILABLE
    must all trigger exactly ONE global planning refresh - never a fan-out
    loop, and the bridge does not special-case which transition occurred or
    how many FireEvents are active (that is GlobalPlanningRefreshCoordinator's
    own concern)."""
    operational = FakeOperationalRefreshOrchestrator(
        resource_result=operational_success(trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE)
    )
    planning = FakeGlobalPlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_resource(resource_id="truck-1", new_status=new_status, as_of=AS_OF)

    assert operational.resource_calls == [{"resource_id": "truck-1", "new_status": new_status}]
    assert len(planning.calls) == 1
    assert planning.calls[0]["trigger"] == "resource_status_update"
    assert result.global_planning_result is not None


def test_resource_refresh_skips_global_planning_when_operational_refresh_fails():
    operational = FakeOperationalRefreshOrchestrator(
        resource_result=operational_failure(trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE)
    )
    planning = FakeGlobalPlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_resource(resource_id="truck-1", new_status=ResourceStatus.UNAVAILABLE, as_of=AS_OF)

    assert planning.calls == []
    assert result.global_planning_result is None
    assert result.operational_result.success is False


def test_resource_refresh_still_triggers_global_planning_on_resource_no_op():
    """ResourceStatusUpdateService's existing NO_OP outcome still succeeds,
    so the bridge still asks the global planner to check - but since
    nothing relevant changed, the fake (standing in for the Global GA's own
    fingerprint check) reports NO_OP, proving no unnecessary new plan
    results from a same-status update."""
    operational = FakeOperationalRefreshOrchestrator(
        resource_result=OperationalRefreshResult(
            trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE,
            status=OperationalRefreshStatus.RESOURCE_NO_OP,
            success=True,
        )
    )
    planning = FakeGlobalPlanningRefresh()  # defaults to a "no_op" result
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_resource(resource_id="truck-1", new_status=ResourceStatus.AVAILABLE, as_of=AS_OF)

    assert len(planning.calls) == 1
    assert result.global_planning_result.status == "no_op"


def test_resource_refresh_skips_planning_when_resource_not_found():
    operational = FakeOperationalRefreshOrchestrator(
        resource_result=OperationalRefreshResult(
            trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE,
            status=OperationalRefreshStatus.RESOURCE_NOT_FOUND,
            success=False,
            error_message="Firefighting resource 'truck-1' was not found.",
        )
    )
    planning = FakeGlobalPlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_resource(resource_id="truck-1", new_status=ResourceStatus.UNAVAILABLE, as_of=AS_OF)

    assert result.operational_result.success is False
    assert result.global_planning_result is None
    assert planning.calls == []


def test_resource_refresh_wraps_unexpected_operational_exception_into_failed_result_and_skips_planning():
    operational = FakeOperationalRefreshOrchestrator(raise_on_resource=RuntimeError("db unavailable"))
    planning = FakeGlobalPlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_resource(resource_id="truck-1", new_status=ResourceStatus.UNAVAILABLE, as_of=AS_OF)

    assert result.operational_result.success is False
    assert result.operational_result.status is OperationalRefreshStatus.FAILED
    assert result.operational_result.trigger_type is OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE
    assert result.operational_result.error_message == "db unavailable"
    assert result.global_planning_result is None
    assert planning.calls == []


def test_resource_refresh_never_triggers_the_environmental_fire_event_refresh_path():
    """A resource-only update must not rerun Fire Detection/Severity/
    Spread/Response Targets. Proven here by making refresh_fire_event raise
    if the coordinator ever calls it from refresh_resource - it must not."""

    class ExplodingOnFireEventRefresh(FakeOperationalRefreshOrchestrator):
        def refresh_fire_event(self, **kwargs):
            raise AssertionError("refresh_resource must never call refresh_fire_event")

    operational = ExplodingOnFireEventRefresh(
        resource_result=operational_success(trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE)
    )
    planning = FakeGlobalPlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    coordinator.refresh_resource(resource_id="truck-1", new_status=ResourceStatus.UNAVAILABLE, as_of=AS_OF)


def test_resource_refresh_calls_operational_before_planning():
    call_log: list[str] = []
    operational = FakeOperationalRefreshOrchestrator(
        resource_result=operational_success(trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE),
        call_log=call_log,
    )
    planning = LoggingGlobalPlanningRefresh(call_log=call_log)
    coordinator = make_coordinator(operational=operational, planning=planning)

    coordinator.refresh_resource(resource_id="truck-1", new_status=ResourceStatus.UNAVAILABLE, as_of=AS_OF)

    assert call_log == ["operational", "planning"]
