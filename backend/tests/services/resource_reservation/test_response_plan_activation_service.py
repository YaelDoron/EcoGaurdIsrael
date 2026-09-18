"""Tests for ResponsePlanActivationService (Global Multi-Incident Optimizer, Stage 1),
using real repositories against SQLite (atomicity/locking needs a real
transactional engine - fakes cannot exercise SELECT ... FOR UPDATE or a
DB-level unique-constraint rollback honestly).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.database.models.route_result_db import RouteResultDB
from src.models import GraphNode, ResponseAction, ResponsePlan, ResponsePlanStatus
from src.models.resource_status import ResourceStatus
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.services.resource_reservation import ResourceCommitmentConflict, ResponsePlanActivationService

AS_OF = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
STATION_ID = "STATION-1"
SOURCE_NODE_ID = 8001
TARGET_NODE_ID = 8002
FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64
FINGERPRINT_C = "c" * 64


def _persist_road_network(session) -> None:
    RoadNetworkRepository().save_network(
        session,
        nodes=[
            GraphNode(id=SOURCE_NODE_ID, latitude=32.7, longitude=35.0),
            GraphNode(id=TARGET_NODE_ID, latitude=32.7, longitude=35.01),
        ],
        edges=[],
    )


def _persist_fire_event(session, **overrides) -> int:
    defaults = dict(
        latitude=32.7,
        longitude=35.0,
        detected_at=AS_OF,
        updated_at=AS_OF,
        status="confirmed",
        detection_confidence=0.9,
        methodology="m",
        methodology_version="1.0",
    )
    defaults.update(overrides)
    event = FireEventDB(**defaults)
    session.add(event)
    session.flush()
    return event.id


def _persist_resources(session, resource_ids, station_id=STATION_ID, status=ResourceStatus.AVAILABLE) -> None:
    if session.get(FireStationDB, station_id) is None:
        session.add(FireStationDB(id=station_id, name="Station", latitude=32.7, longitude=35.0))
        session.flush()
    for resource_id in resource_ids:
        session.add(FirefightingResourceDB(id=resource_id, station_id=station_id, status=status))


def _persist_planning_prerequisites(session, fire_event_id: int) -> tuple[int, int]:
    target_set = ResponseTargetSetDB(
        fire_event_id=fire_event_id, generated_at=AS_OF, methodology="m", methodology_version="1.0"
    )
    session.add(target_set)
    session.flush()
    route_run = RoutePlanningRunDB(
        fire_event_id=fire_event_id,
        response_target_set_id=target_set.id,
        planned_at=AS_OF,
        methodology="m",
        methodology_version="1.0",
        resource_ids=[],
    )
    session.add(route_run)
    session.flush()
    return target_set.id, route_run.id


def _persist_target_and_route_result(
    session, target_set_id: int, fire_event_id: int, target_order: int, resource_id: str, route_run_id: int
) -> tuple[int, int]:
    target = ResponseTargetDB(
        response_target_set_id=target_set_id,
        fire_event_id=fire_event_id,
        target_order=target_order,
        target_type="active_fire",
        latitude=32.7,
        longitude=35.0,
        priority_score=100.0,
    )
    session.add(target)
    session.flush()
    route_result = RouteResultDB(
        route_planning_run_id=route_run_id,
        resource_id=resource_id,
        response_target_id=target.id,
        status="reachable",
        source_node_id=SOURCE_NODE_ID,
        target_node_id=TARGET_NODE_ID,
        node_path=[SOURCE_NODE_ID, TARGET_NODE_ID],
        distance_meters=500.0,
        travel_time_seconds=60.0,
    )
    session.add(route_result)
    session.flush()
    return target.id, route_result.id


@pytest.fixture
def wiring(sqlite_session_factory):
    response_plan_repository = ResponsePlanRepository(sqlite_session_factory)
    resource_commitment_repository = ResourceCommitmentRepository(sqlite_session_factory)
    activation_service = ResponsePlanActivationService(
        response_plan_repository=response_plan_repository,
        resource_commitment_repository=resource_commitment_repository,
        session_factory=sqlite_session_factory,
    )
    return response_plan_repository, resource_commitment_repository, activation_service


@pytest.fixture
def scenario(sqlite_session_factory):
    """Two FireEvents (A, B) sharing one station with resources R1, R2, R3 AVAILABLE."""
    session = sqlite_session_factory()
    _persist_road_network(session)
    event_a = _persist_fire_event(session, latitude=32.7, longitude=35.0)
    event_b = _persist_fire_event(session, latitude=32.71, longitude=35.01)
    _persist_resources(session, ("R1", "R2", "R3"))
    target_set_a_id, run_a_id = _persist_planning_prerequisites(session, event_a)
    target_set_b_id, run_b_id = _persist_planning_prerequisites(session, event_b)
    session.commit()
    session.close()
    return {
        "event_a": event_a,
        "event_b": event_b,
        "target_set_a_id": target_set_a_id,
        "target_set_b_id": target_set_b_id,
        "run_a_id": run_a_id,
        "run_b_id": run_b_id,
    }


def _make_plan(
    response_plan_repository,
    sqlite_session_factory,
    *,
    fire_event_id,
    target_set_id,
    run_id,
    resource_ids,
    generated_at=AS_OF,
) -> int:
    session = sqlite_session_factory()
    existing_max_order = session.query(ResponseTargetDB.target_order).filter(
        ResponseTargetDB.response_target_set_id == target_set_id
    ).order_by(ResponseTargetDB.target_order.desc()).first()
    order_start = (existing_max_order[0] + 1) if existing_max_order else 1
    actions = []
    for offset, resource_id in enumerate(resource_ids):
        target_id, route_result_id = _persist_target_and_route_result(
            session, target_set_id, fire_event_id, order_start + offset, resource_id, run_id
        )
        actions.append(ResponseAction(resource_id, target_id, route_result_id))
    session.commit()
    session.close()

    stored = response_plan_repository.save(
        ResponsePlan(
            fire_event_id=fire_event_id,
            response_target_set_id=target_set_id,
            route_planning_run_id=run_id,
            generated_at=generated_at,
            status=ResponsePlanStatus.COMPLETE,
            methodology="m",
            methodology_version="1.0",
            random_seed=1,
            actions=tuple(actions),
            uncovered_target_ids=(),
            plan_score=1.0,
            coverage_score=1.0,
            average_eta_seconds=1.0,
        )
    )
    return stored.id


# ---------------------------------------------------------------------------
# First activation
# ---------------------------------------------------------------------------


def test_first_activation_creates_commitments_and_returns_sidecar(wiring, scenario, sqlite_session_factory):
    response_plan_repository, resource_commitment_repository, activation_service = wiring
    plan_id = _make_plan(
        response_plan_repository,
        sqlite_session_factory,
        fire_event_id=scenario["event_a"],
        target_set_id=scenario["target_set_a_id"],
        run_id=scenario["run_a_id"],
        resource_ids=("R1", "R2"),
    )

    result = activation_service.activate(
        response_plan_id=plan_id, planning_effective_state_fingerprint=FINGERPRINT_A, as_of=AS_OF
    )

    assert result.response_plan_id == plan_id
    assert result.planning_effective_state_fingerprint == FINGERPRINT_A
    committed = {c.resource_id for c in resource_commitment_repository.get_for_fire_event(scenario["event_a"])}
    assert committed == {"R1", "R2"}


def test_historical_candidate_plan_owns_no_resources_before_activation(wiring, scenario, sqlite_session_factory):
    """A saved-but-not-yet-activated ResponsePlan (no planning-state sidecar)
    must not own any commitments merely by having been persisted."""
    response_plan_repository, resource_commitment_repository, _ = wiring
    _make_plan(
        response_plan_repository,
        sqlite_session_factory,
        fire_event_id=scenario["event_a"],
        target_set_id=scenario["target_set_a_id"],
        run_id=scenario["run_a_id"],
        resource_ids=("R1",),
    )

    assert resource_commitment_repository.get_for_fire_event(scenario["event_a"]) == ()


# ---------------------------------------------------------------------------
# Same-event replacement
# ---------------------------------------------------------------------------


def test_same_event_replacement_releases_dropped_resource_and_keeps_carried_over_one(
    wiring, scenario, sqlite_session_factory
):
    response_plan_repository, resource_commitment_repository, activation_service = wiring
    old_plan_id = _make_plan(
        response_plan_repository,
        sqlite_session_factory,
        fire_event_id=scenario["event_a"],
        target_set_id=scenario["target_set_a_id"],
        run_id=scenario["run_a_id"],
        resource_ids=("R1", "R2"),
    )
    activation_service.activate(
        response_plan_id=old_plan_id, planning_effective_state_fingerprint=FINGERPRINT_A, as_of=AS_OF
    )

    new_plan_id = _make_plan(
        response_plan_repository,
        sqlite_session_factory,
        fire_event_id=scenario["event_a"],
        target_set_id=scenario["target_set_a_id"],
        run_id=scenario["run_a_id"],
        resource_ids=("R1", "R3"),
        generated_at=AS_OF + timedelta(minutes=1),
    )
    activation_service.activate(
        response_plan_id=new_plan_id,
        planning_effective_state_fingerprint=FINGERPRINT_B,
        as_of=AS_OF + timedelta(minutes=1),
    )

    committed = {c.resource_id: c for c in resource_commitment_repository.get_for_fire_event(scenario["event_a"])}
    assert set(committed) == {"R1", "R3"}
    assert all(c.response_plan_id == new_plan_id for c in committed.values())


# ---------------------------------------------------------------------------
# Failed replacement rollback
# ---------------------------------------------------------------------------


def test_failed_replacement_leaves_old_plan_and_commitments_intact(wiring, scenario, sqlite_session_factory):
    response_plan_repository, resource_commitment_repository, activation_service = wiring
    old_plan_id = _make_plan(
        response_plan_repository,
        sqlite_session_factory,
        fire_event_id=scenario["event_a"],
        target_set_id=scenario["target_set_a_id"],
        run_id=scenario["run_a_id"],
        resource_ids=("R1", "R2"),
    )
    activation_service.activate(
        response_plan_id=old_plan_id, planning_effective_state_fingerprint=FINGERPRINT_A, as_of=AS_OF
    )

    # B claims R3 first.
    plan_b_id = _make_plan(
        response_plan_repository,
        sqlite_session_factory,
        fire_event_id=scenario["event_b"],
        target_set_id=scenario["target_set_b_id"],
        run_id=scenario["run_b_id"],
        resource_ids=("R3",),
    )
    activation_service.activate(
        response_plan_id=plan_b_id, planning_effective_state_fingerprint=FINGERPRINT_C, as_of=AS_OF
    )

    # A's new candidate plan wants R1 (already A's own) and R3 (now B's) - must fail atomically.
    new_plan_id = _make_plan(
        response_plan_repository,
        sqlite_session_factory,
        fire_event_id=scenario["event_a"],
        target_set_id=scenario["target_set_a_id"],
        run_id=scenario["run_a_id"],
        resource_ids=("R1", "R3"),
        generated_at=AS_OF + timedelta(minutes=1),
    )
    with pytest.raises(ResourceCommitmentConflict) as excinfo:
        activation_service.activate(
            response_plan_id=new_plan_id,
            planning_effective_state_fingerprint=FINGERPRINT_B,
            as_of=AS_OF + timedelta(minutes=1),
        )
    assert "R3" in excinfo.value.conflicts

    committed = {c.resource_id: c for c in resource_commitment_repository.get_for_fire_event(scenario["event_a"])}
    assert set(committed) == {"R1", "R2"}
    assert all(c.response_plan_id == old_plan_id for c in committed.values())


# ---------------------------------------------------------------------------
# Cross-event conflict
# ---------------------------------------------------------------------------


def test_cross_event_conflict_does_not_partially_commit_the_requested_set(
    wiring, scenario, sqlite_session_factory
):
    response_plan_repository, resource_commitment_repository, activation_service = wiring
    plan_a_id = _make_plan(
        response_plan_repository,
        sqlite_session_factory,
        fire_event_id=scenario["event_a"],
        target_set_id=scenario["target_set_a_id"],
        run_id=scenario["run_a_id"],
        resource_ids=("R1",),
    )
    activation_service.activate(
        response_plan_id=plan_a_id, planning_effective_state_fingerprint=FINGERPRINT_A, as_of=AS_OF
    )

    plan_b_id = _make_plan(
        response_plan_repository,
        sqlite_session_factory,
        fire_event_id=scenario["event_b"],
        target_set_id=scenario["target_set_b_id"],
        run_id=scenario["run_b_id"],
        resource_ids=("R1", "R2"),
    )
    with pytest.raises(ResourceCommitmentConflict):
        activation_service.activate(
            response_plan_id=plan_b_id, planning_effective_state_fingerprint=FINGERPRINT_B, as_of=AS_OF
        )

    # All-or-nothing: R2 must NOT have been committed to B just because R1 failed.
    assert resource_commitment_repository.get_for_fire_event(scenario["event_b"]) == ()
    assert resource_commitment_repository.get_by_resource_id("R2") is None


# ---------------------------------------------------------------------------
# Resource operational status vs commitment (Task 12)
# ---------------------------------------------------------------------------


def test_unavailable_resource_with_no_commitment_is_rejected(wiring, scenario, sqlite_session_factory):
    response_plan_repository, resource_commitment_repository, activation_service = wiring
    session = sqlite_session_factory()
    resource = session.get(FirefightingResourceDB, "R1")
    resource.status = ResourceStatus.UNAVAILABLE
    session.commit()
    session.close()

    plan_id = _make_plan(
        response_plan_repository,
        sqlite_session_factory,
        fire_event_id=scenario["event_a"],
        target_set_id=scenario["target_set_a_id"],
        run_id=scenario["run_a_id"],
        resource_ids=("R1",),
    )

    with pytest.raises(ResourceCommitmentConflict) as excinfo:
        activation_service.activate(
            response_plan_id=plan_id, planning_effective_state_fingerprint=FINGERPRINT_A, as_of=AS_OF
        )
    assert excinfo.value.conflicts["R1"] == "not_available"


def test_resource_already_owned_by_same_event_remains_eligible_even_if_unavailable(
    wiring, scenario, sqlite_session_factory
):
    """Task 12: operational status and commitment ownership are separate -
    a resource already committed to THIS event stays eligible for this
    event's own replan even if its raw status is no longer AVAILABLE."""
    response_plan_repository, resource_commitment_repository, activation_service = wiring
    plan_id = _make_plan(
        response_plan_repository,
        sqlite_session_factory,
        fire_event_id=scenario["event_a"],
        target_set_id=scenario["target_set_a_id"],
        run_id=scenario["run_a_id"],
        resource_ids=("R1",),
    )
    activation_service.activate(
        response_plan_id=plan_id, planning_effective_state_fingerprint=FINGERPRINT_A, as_of=AS_OF
    )

    session = sqlite_session_factory()
    resource = session.get(FirefightingResourceDB, "R1")
    resource.status = ResourceStatus.ASSIGNED
    session.commit()
    session.close()

    new_plan_id = _make_plan(
        response_plan_repository,
        sqlite_session_factory,
        fire_event_id=scenario["event_a"],
        target_set_id=scenario["target_set_a_id"],
        run_id=scenario["run_a_id"],
        resource_ids=("R1",),
        generated_at=AS_OF + timedelta(minutes=1),
    )

    result = activation_service.activate(
        response_plan_id=new_plan_id,
        planning_effective_state_fingerprint=FINGERPRINT_B,
        as_of=AS_OF + timedelta(minutes=1),
    )

    assert result.response_plan_id == new_plan_id
    committed = resource_commitment_repository.get_by_resource_id("R1")
    assert committed.response_plan_id == new_plan_id


def test_missing_resource_is_rejected(sqlite_session_factory):
    """A resource referenced by a saved plan's actions can vanish from
    firefighting_resources between plan generation and activation (a fleet
    record deleted out from under a stale candidate plan). firefighting_
    resources.id is a real FK target for response_actions/route_results, so
    that exact end-to-end scenario cannot be constructed without violating
    referential integrity elsewhere - this exercises the same conflict-
    detection code (_find_conflicts) the full activate() call delegates to,
    directly, the way ResponsePlanActivationService.activate() itself calls
    it (locked FireEvent row, requested ids, and the locked resource rows
    actually found in the DB)."""
    from src.services.resource_reservation.response_plan_activation_service import ResponsePlanActivationService

    session = sqlite_session_factory()
    conflicts = ResponsePlanActivationService._find_conflicts(
        session, fire_event_id=1, requested_resource_ids=("GHOST-RESOURCE",), db_resources=()
    )
    session.close()

    assert conflicts == {"GHOST-RESOURCE": "resource_not_found"}


def test_zero_action_plan_activates_and_releases_all_prior_commitments(wiring, scenario, sqlite_session_factory):
    response_plan_repository, resource_commitment_repository, activation_service = wiring
    plan_id = _make_plan(
        response_plan_repository,
        sqlite_session_factory,
        fire_event_id=scenario["event_a"],
        target_set_id=scenario["target_set_a_id"],
        run_id=scenario["run_a_id"],
        resource_ids=("R1",),
    )
    activation_service.activate(
        response_plan_id=plan_id, planning_effective_state_fingerprint=FINGERPRINT_A, as_of=AS_OF
    )

    empty_plan = response_plan_repository.save(
        ResponsePlan(
            fire_event_id=scenario["event_a"],
            response_target_set_id=scenario["target_set_a_id"],
            route_planning_run_id=scenario["run_a_id"],
            generated_at=AS_OF + timedelta(minutes=1),
            status=ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS,
            methodology="m",
            methodology_version="1.0",
            random_seed=1,
            actions=(),
            uncovered_target_ids=(),
            plan_score=0.0,
            coverage_score=0.0,
            average_eta_seconds=None,
        )
    )

    activation_service.activate(
        response_plan_id=empty_plan.id,
        planning_effective_state_fingerprint=FINGERPRINT_B,
        as_of=AS_OF + timedelta(minutes=1),
    )

    assert resource_commitment_repository.get_for_fire_event(scenario["event_a"]) == ()


def test_activate_rejects_unknown_response_plan_id(wiring):
    _, _, activation_service = wiring
    with pytest.raises(ValueError):
        activation_service.activate(
            response_plan_id=999999, planning_effective_state_fingerprint=FINGERPRINT_A, as_of=AS_OF
        )
