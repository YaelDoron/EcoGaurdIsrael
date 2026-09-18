"""Tests for GlobalPlanningInput (Stage 3 of the Global Multi-Incident Optimizer refactor)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models.current_global_assignment import CurrentGlobalAssignment
from src.models.demand_source import DemandSource
from src.models.dispatch_state import DispatchState
from src.models.global_incident_demand import GlobalIncidentDemand
from src.models.global_planning_input import GlobalPlanningInput
from src.models.global_planning_resource import GlobalPlanningResource
from src.models.global_planning_target import GlobalPlanningTarget
from src.models.global_route_matrix import GlobalRouteMatrix
from src.models.global_route_option import GlobalRouteOption
from src.models.resource_status import ResourceStatus
from src.models.response_target_type import ResponseTargetType

AS_OF = datetime(2026, 9, 21, 8, 0, tzinfo=timezone.utc)


def make_assignment(**overrides) -> CurrentGlobalAssignment:
    defaults = dict(
        resource_id="R1",
        fire_event_id=1,
        response_target_id=10,
        response_plan_id=1,
        dispatch_state=DispatchState.DISPATCHED,
    )
    defaults.update(overrides)
    return CurrentGlobalAssignment(**defaults)


def make_demand(fire_event_id: int, **overrides) -> GlobalIncidentDemand:
    defaults = dict(
        fire_event_id=fire_event_id,
        severity_assessment_id=None,
        severity_level=None,
        severity_score=None,
        minimum_resources=0,
        desired_resources=1,
        demand_source=DemandSource.INSUFFICIENT_SEVERITY,
        policy_methodology="ecoguard_demo_severity_demand_policy",
        policy_version="1.0",
    )
    defaults.update(overrides)
    return GlobalIncidentDemand(**defaults)


def make_target(**overrides) -> GlobalPlanningTarget:
    defaults = dict(
        fire_event_id=1,
        response_target_id=10,
        target_order=0,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=32.7,
        longitude=35.0,
        priority_score=100.0,
    )
    defaults.update(overrides)
    return GlobalPlanningTarget(**defaults)


def make_resource(**overrides) -> GlobalPlanningResource:
    defaults = dict(
        resource_id="R1",
        station_id="S1",
        station_name="Station 1",
        station_latitude=32.7,
        station_longitude=35.0,
        operational_status=ResourceStatus.AVAILABLE,
    )
    defaults.update(overrides)
    return GlobalPlanningResource(**defaults)


def make_option(**overrides) -> GlobalRouteOption:
    defaults = dict(
        resource_id="R1", fire_event_id=1, response_target_id=10, eta_seconds=60.0,
        route_distance_meters=500.0, node_path=(1, 2),
    )
    defaults.update(overrides)
    return GlobalRouteOption(**defaults)


def make_input(**overrides) -> GlobalPlanningInput:
    active_fire_event_ids = overrides.get("active_fire_event_ids", (1,))
    defaults = dict(
        global_planning_run_id=1,
        as_of=AS_OF,
        active_fire_event_ids=active_fire_event_ids,
        targets=(make_target(),),
        resources=(make_resource(),),
        route_matrix=GlobalRouteMatrix(options=(make_option(),)),
        event_target_set_ids={1: 100},
        incident_demands=tuple(make_demand(fire_event_id) for fire_event_id in active_fire_event_ids),
        input_fingerprint="f" * 64,
    )
    defaults.update(overrides)
    return GlobalPlanningInput(**defaults)


def test_valid_input_round_trips():
    result = make_input()
    assert result.global_planning_run_id == 1
    assert len(result.targets) == 1
    assert len(result.resources) == 1
    assert len(result.route_matrix) == 1


def test_rejects_target_belonging_to_inactive_event():
    with pytest.raises(ValueError):
        make_input(active_fire_event_ids=(2,), event_target_set_ids={2: 100})


def test_rejects_route_referencing_unknown_resource():
    with pytest.raises(ValueError):
        make_input(route_matrix=GlobalRouteMatrix(options=(make_option(resource_id="GHOST"),)))


def test_rejects_route_referencing_unknown_target():
    with pytest.raises(ValueError):
        make_input(route_matrix=GlobalRouteMatrix(options=(make_option(response_target_id=999),)))


def test_rejects_route_with_mismatched_fire_event_id():
    target = make_target(fire_event_id=1, response_target_id=10)
    other_target = make_target(fire_event_id=2, response_target_id=20)
    with pytest.raises(ValueError):
        make_input(
            active_fire_event_ids=(1, 2),
            targets=(target, other_target),
            event_target_set_ids={1: 100, 2: 200},
            route_matrix=GlobalRouteMatrix(
                options=(make_option(resource_id="R1", fire_event_id=1, response_target_id=20),)
            ),
        )


def test_rejects_duplicate_resources():
    with pytest.raises(ValueError):
        make_input(resources=(make_resource(), make_resource()))


def test_rejects_duplicate_targets():
    with pytest.raises(ValueError):
        make_input(targets=(make_target(), make_target()))


def test_rejects_missing_target_set_id_for_event_with_targets():
    with pytest.raises(ValueError):
        make_input(event_target_set_ids={})


def test_allows_active_event_with_no_targets_and_no_target_set_entry():
    """Task 5: an active event with no usable ResponseTargetSet is
    represented honestly - zero targets, no event_target_set_ids entry -
    not an error."""
    result = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(fire_event_id=1),),
        event_target_set_ids={1: 100},
        route_matrix=GlobalRouteMatrix(options=(make_option(fire_event_id=1, response_target_id=10),)),
    )
    assert 2 in result.active_fire_event_ids
    assert 2 not in result.event_target_set_ids


def test_rejects_naive_as_of():
    with pytest.raises(ValueError):
        make_input(as_of=datetime(2026, 1, 1))


def test_rejects_empty_fingerprint():
    with pytest.raises(ValueError):
        make_input(input_fingerprint="")


def test_rejects_duplicate_active_fire_event_ids():
    with pytest.raises(ValueError):
        make_input(active_fire_event_ids=(1, 1))


# ---------------------------------------------------------------------------
# Stage 6 - current_assignments
# ---------------------------------------------------------------------------


def test_current_assignments_defaults_to_empty():
    result = make_input()
    assert result.current_assignments == ()


def test_accepts_valid_current_assignment():
    result = make_input(current_assignments=(make_assignment(),))
    assert result.current_assignments[0].dispatch_state is DispatchState.DISPATCHED


def test_rejects_current_assignment_referencing_unknown_resource():
    with pytest.raises(ValueError):
        make_input(current_assignments=(make_assignment(resource_id="GHOST"),))


def test_rejects_current_assignment_referencing_unknown_target():
    with pytest.raises(ValueError):
        make_input(current_assignments=(make_assignment(response_target_id=999),))


def test_rejects_current_assignment_for_inactive_event():
    with pytest.raises(ValueError):
        make_input(
            active_fire_event_ids=(1, 2),
            targets=(make_target(fire_event_id=1), make_target(fire_event_id=2, response_target_id=20)),
            event_target_set_ids={1: 100, 2: 200},
            route_matrix=GlobalRouteMatrix(
                options=(make_option(fire_event_id=1, response_target_id=10), make_option(fire_event_id=2, response_target_id=20))
            ),
            current_assignments=(make_assignment(fire_event_id=3, response_target_id=10),),
        )


def test_rejects_current_assignment_with_mismatched_fire_event_id():
    target_a = make_target(fire_event_id=1, response_target_id=10)
    target_b = make_target(fire_event_id=2, response_target_id=20)
    with pytest.raises(ValueError):
        make_input(
            active_fire_event_ids=(1, 2),
            targets=(target_a, target_b),
            event_target_set_ids={1: 100, 2: 200},
            route_matrix=GlobalRouteMatrix(
                options=(make_option(fire_event_id=1, response_target_id=10), make_option(fire_event_id=2, response_target_id=20))
            ),
            # response_target_id 20 actually belongs to fire_event_id 2, not 1.
            current_assignments=(make_assignment(fire_event_id=1, response_target_id=20, resource_id="R1"),),
        )


def test_rejects_duplicate_current_assignment_resource_ids():
    with pytest.raises(ValueError):
        make_input(current_assignments=(make_assignment(), make_assignment()))
