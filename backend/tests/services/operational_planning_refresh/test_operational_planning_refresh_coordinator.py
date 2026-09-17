"""Tests for OperationalPlanningRefreshCoordinator (the US4.4 -> US5.4 bridge).

Uses fakes for both the operational orchestrator and the planning
collaborator, matching how ResponsePlanningRefreshOrchestrator's own tests
use fake routing/optimization/baseline collaborators: this suite proves the
coordinator's own composition logic (when it calls planning, with what
arguments, how it aggregates results) without re-deriving or re-testing
either US4.4's or US5.4's already-covered internal behavior.

The planning-side fake never imports the real PlanningRefreshResult /
PlanningRefreshStatus classes (see operational_planning_refresh_ports.py's
docstring): those live in a package with an optional osmnx dependency, and
the coordinator itself never inspects their shape beyond passing them
through, so a lightweight local stand-in is sufficient and keeps this
suite runnable without osmnx installed.
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
class FakePlanningRefreshResult:
    """Stand-in for PlanningRefreshResult - see module docstring."""

    status: str
    fire_event_id: int
    route_planning_run_id: int | None = None
    response_plan_id: int | None = None
    comparison_id: int | None = None


class FakePlanningRefresh:
    """Fake PlanningRefreshPort: records calls, returns a canned/mapped result."""

    def __init__(self, *, results_by_fire_event_id: dict[int, FakePlanningRefreshResult] | None = None) -> None:
        self.calls: list[dict] = []
        self._results_by_fire_event_id = results_by_fire_event_id or {}

    def refresh(self, *, fire_event_id: int, as_of: datetime) -> FakePlanningRefreshResult:
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        if fire_event_id in self._results_by_fire_event_id:
            return self._results_by_fire_event_id[fire_event_id]
        return FakePlanningRefreshResult(status="no_op", fire_event_id=fire_event_id)


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


class LoggingPlanningRefresh(FakePlanningRefresh):
    """FakePlanningRefresh that also appends to a shared call-order log."""

    def __init__(self, *, call_log: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._call_log = call_log

    def refresh(self, *, fire_event_id: int, as_of: datetime) -> FakePlanningRefreshResult:
        self._call_log.append("planning")
        return super().refresh(fire_event_id=fire_event_id, as_of=as_of)


class FakeFireEventRepository:
    def __init__(self, active_ids: tuple[int, ...] = ()) -> None:
        self.active_ids = active_ids
        self.calls = 0

    def get_active_fire_event_ids(self) -> tuple[int, ...]:
        self.calls += 1
        return self.active_ids


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
    planning: FakePlanningRefresh,
    fire_event_repository: FakeFireEventRepository | None = None,
) -> OperationalPlanningRefreshCoordinator:
    return OperationalPlanningRefreshCoordinator(
        operational_refresh_orchestrator=operational,
        planning_refresh=planning,
        fire_event_repository=fire_event_repository or FakeFireEventRepository(),
    )


# ---------------------------------------------------------------------------
# refresh_fire_event: environmental / FireEvent trigger
# ---------------------------------------------------------------------------


def test_fire_event_refresh_triggers_planning_for_the_same_fire_event():
    operational = FakeOperationalRefreshOrchestrator(
        fire_event_result=operational_success(
            trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE, fire_event_id=FIRE_EVENT_ID, as_of=AS_OF
        )
    )
    planning = FakePlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE, as_of=AS_OF
    )

    assert len(planning.calls) == 1
    assert planning.calls[0] == {"fire_event_id": FIRE_EVENT_ID, "as_of": AS_OF}
    assert result.planning_results == (FakePlanningRefreshResult(status="no_op", fire_event_id=FIRE_EVENT_ID),)


def test_fire_event_refresh_scenario_a_changed_targets_surfaces_the_refreshed_plan_untouched():
    """Task 6 Scenario A: the planning collaborator reports a full replan (new
    RoutePlanningRun/ResponsePlan/comparison ids); the coordinator must
    surface that exact result, not reinterpret or duplicate it."""
    refreshed = FakePlanningRefreshResult(
        status="refreshed",
        fire_event_id=FIRE_EVENT_ID,
        route_planning_run_id=901,
        response_plan_id=902,
        comparison_id=903,
    )
    operational = FakeOperationalRefreshOrchestrator(
        fire_event_result=operational_success(trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE)
    )
    planning = FakePlanningRefresh(results_by_fire_event_id={FIRE_EVENT_ID: refreshed})
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE, as_of=AS_OF
    )

    assert result.planning_results == (refreshed,)
    assert len(planning.calls) == 1


def test_fire_event_refresh_scenario_b_unchanged_state_surfaces_the_no_op_result_untouched():
    """Task 6 Scenario B: the planning collaborator reports NO_OP (fingerprint
    unchanged); the coordinator still calls it exactly once and surfaces the
    NO_OP result as-is - it never decides NO_OP itself."""
    no_op = FakePlanningRefreshResult(
        status="no_op", fire_event_id=FIRE_EVENT_ID, route_planning_run_id=1, response_plan_id=2, comparison_id=3
    )
    operational = FakeOperationalRefreshOrchestrator(
        fire_event_result=operational_success(trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE)
    )
    planning = FakePlanningRefresh(results_by_fire_event_id={FIRE_EVENT_ID: no_op})
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE, as_of=AS_OF
    )

    assert result.planning_results == (no_op,)
    assert len(planning.calls) == 1


def test_fire_event_refresh_skips_planning_when_operational_refresh_fails():
    operational = FakeOperationalRefreshOrchestrator(
        fire_event_result=operational_failure(trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE)
    )
    planning = FakePlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE, as_of=AS_OF
    )

    assert result.operational_result.success is False
    assert result.planning_results == ()
    assert planning.calls == []


def test_fire_event_refresh_wraps_unexpected_operational_exception_into_failed_result_and_skips_planning():
    operational = FakeOperationalRefreshOrchestrator(raise_on_fire_event=RuntimeError("boom"))
    planning = FakePlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    result = coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE, as_of=AS_OF
    )

    assert result.operational_result.success is False
    assert result.operational_result.status is OperationalRefreshStatus.FAILED
    assert result.operational_result.error_message == "boom"
    assert result.planning_results == ()
    assert planning.calls == []


def test_fire_event_refresh_calls_operational_before_planning():
    call_log: list[str] = []
    operational = FakeOperationalRefreshOrchestrator(
        fire_event_result=operational_success(trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE),
        call_log=call_log,
    )
    planning = LoggingPlanningRefresh(call_log=call_log)
    coordinator = make_coordinator(operational=operational, planning=planning)

    coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE, as_of=AS_OF
    )

    assert call_log == ["operational", "planning"]


def test_fire_event_refresh_passes_through_the_same_as_of_used_for_the_operational_refresh():
    operational = FakeOperationalRefreshOrchestrator(
        fire_event_result=operational_success(trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE)
    )
    planning = FakePlanningRefresh()
    coordinator = make_coordinator(operational=operational, planning=planning)

    coordinator.refresh_fire_event(
        fire_event_id=FIRE_EVENT_ID, trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE, as_of=AS_OF
    )

    assert operational.fire_event_calls[0]["as_of"] == AS_OF
    assert planning.calls[0]["as_of"] == AS_OF


# ---------------------------------------------------------------------------
# refresh_resource: resource-status trigger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "new_status",
    [ResourceStatus.UNAVAILABLE, ResourceStatus.ASSIGNED, ResourceStatus.AVAILABLE],
)
def test_resource_refresh_triggers_planning_for_every_active_fire_event(new_status):
    """Task 7: AVAILABLE->UNAVAILABLE, AVAILABLE->ASSIGNED, and back to
    AVAILABLE must all trigger a planning check for every active FireEvent -
    the bridge does not special-case which transition occurred."""
    operational = FakeOperationalRefreshOrchestrator(
        resource_result=operational_success(trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE)
    )
    planning = FakePlanningRefresh()
    fire_event_repository = FakeFireEventRepository(active_ids=(1, 5, 9))
    coordinator = make_coordinator(operational=operational, planning=planning, fire_event_repository=fire_event_repository)

    result = coordinator.refresh_resource(resource_id="truck-1", new_status=new_status, as_of=AS_OF)

    assert operational.resource_calls == [{"resource_id": "truck-1", "new_status": new_status}]
    assert [call["fire_event_id"] for call in planning.calls] == [1, 5, 9]
    assert len(result.planning_results) == 3


def test_resource_refresh_checks_active_fire_events_in_deterministic_ascending_order():
    operational = FakeOperationalRefreshOrchestrator(
        resource_result=operational_success(trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE)
    )
    planning = FakePlanningRefresh()
    fire_event_repository = FakeFireEventRepository(active_ids=(2, 3, 7))
    coordinator = make_coordinator(operational=operational, planning=planning, fire_event_repository=fire_event_repository)

    coordinator.refresh_resource(resource_id="truck-1", new_status=ResourceStatus.UNAVAILABLE, as_of=AS_OF)

    assert [call["fire_event_id"] for call in planning.calls] == [2, 3, 7]


def test_resource_refresh_with_no_active_fire_events_calls_no_planning():
    operational = FakeOperationalRefreshOrchestrator(
        resource_result=operational_success(trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE)
    )
    planning = FakePlanningRefresh()
    coordinator = make_coordinator(
        operational=operational, planning=planning, fire_event_repository=FakeFireEventRepository(active_ids=())
    )

    result = coordinator.refresh_resource(resource_id="truck-1", new_status=ResourceStatus.UNAVAILABLE, as_of=AS_OF)

    assert planning.calls == []
    assert result.planning_results == ()
    assert result.operational_result.success is True


def test_resource_refresh_still_checks_planning_on_resource_no_op_but_causes_no_unnecessary_new_plan():
    """Task 7 'same status repeated': ResourceStatusUpdateService's existing
    NO_OP outcome still succeeds, so the bridge still asks US5.4 to check -
    but since nothing relevant changed, the fake (standing in for US5.4's own
    fingerprint check) reports NO_OP for every FireEvent, proving no
    unnecessary new plan results from a same-status update."""
    operational = FakeOperationalRefreshOrchestrator(
        resource_result=OperationalRefreshResult(
            trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE,
            status=OperationalRefreshStatus.RESOURCE_NO_OP,
            success=True,
        )
    )
    planning = FakePlanningRefresh()  # defaults every fire_event_id to a "no_op" result
    fire_event_repository = FakeFireEventRepository(active_ids=(1,))
    coordinator = make_coordinator(operational=operational, planning=planning, fire_event_repository=fire_event_repository)

    result = coordinator.refresh_resource(resource_id="truck-1", new_status=ResourceStatus.AVAILABLE, as_of=AS_OF)

    assert len(planning.calls) == 1
    assert result.planning_results[0].status == "no_op"


def test_resource_refresh_skips_planning_when_resource_not_found():
    operational = FakeOperationalRefreshOrchestrator(
        resource_result=OperationalRefreshResult(
            trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE,
            status=OperationalRefreshStatus.RESOURCE_NOT_FOUND,
            success=False,
            error_message="Firefighting resource 'truck-1' was not found.",
        )
    )
    planning = FakePlanningRefresh()
    fire_event_repository = FakeFireEventRepository(active_ids=(1, 2))
    coordinator = make_coordinator(operational=operational, planning=planning, fire_event_repository=fire_event_repository)

    result = coordinator.refresh_resource(resource_id="truck-1", new_status=ResourceStatus.UNAVAILABLE, as_of=AS_OF)

    assert result.operational_result.success is False
    assert result.planning_results == ()
    assert planning.calls == []
    assert fire_event_repository.calls == 0


def test_resource_refresh_wraps_unexpected_operational_exception_into_failed_result_and_skips_planning():
    operational = FakeOperationalRefreshOrchestrator(raise_on_resource=RuntimeError("db unavailable"))
    planning = FakePlanningRefresh()
    fire_event_repository = FakeFireEventRepository(active_ids=(1,))
    coordinator = make_coordinator(operational=operational, planning=planning, fire_event_repository=fire_event_repository)

    result = coordinator.refresh_resource(resource_id="truck-1", new_status=ResourceStatus.UNAVAILABLE, as_of=AS_OF)

    assert result.operational_result.success is False
    assert result.operational_result.status is OperationalRefreshStatus.FAILED
    assert result.operational_result.trigger_type is OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE
    assert result.operational_result.error_message == "db unavailable"
    assert result.planning_results == ()
    assert planning.calls == []


def test_resource_refresh_never_triggers_the_environmental_fire_event_refresh_path():
    """Task 5: a resource-only update must not rerun Fire Detection/Severity/
    Spread/Response Targets. Proven here by making refresh_fire_event raise
    if the coordinator ever calls it from refresh_resource - it must not."""

    class ExplodingOnFireEventRefresh(FakeOperationalRefreshOrchestrator):
        def refresh_fire_event(self, **kwargs):
            raise AssertionError("refresh_resource must never call refresh_fire_event")

    operational = ExplodingOnFireEventRefresh(
        resource_result=operational_success(trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE)
    )
    planning = FakePlanningRefresh()
    fire_event_repository = FakeFireEventRepository(active_ids=(1, 2))
    coordinator = make_coordinator(operational=operational, planning=planning, fire_event_repository=fire_event_repository)

    coordinator.refresh_resource(resource_id="truck-1", new_status=ResourceStatus.UNAVAILABLE, as_of=AS_OF)
