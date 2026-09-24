"""Tests for FireEventLifecycleService (Stage 1.1 of the Global
Multi-Incident Optimizer refactor): atomic inactive-transition + immediate
resource-commitment release.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.database.models.route_result_db import RouteResultDB
from src.models import GraphNode, ResponseAction, ResponsePlan, ResponsePlanStatus
from sqlalchemy import text
from src.models.fire_event_status import FireEventStatus
from src.models.resource_status import ResourceStatus
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.services.fire_event_lifecycle import FireEventLifecycleService
from src.services.resource_reservation import ResponsePlanActivationService

AS_OF = datetime(2026, 9, 19, 8, 0, tzinfo=timezone.utc)
LATER = AS_OF + timedelta(hours=1)
STATION_ID = "STATION-LC-1"
SOURCE_NODE_ID = 7001
TARGET_NODE_ID = 7002


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


def _persist_resources(session, resource_ids, station_id=STATION_ID) -> None:
    if session.get(FireStationDB, station_id) is None:
        session.add(FireStationDB(id=station_id, name="Station", latitude=32.7, longitude=35.0))
        session.flush()
    for resource_id in resource_ids:
        session.add(FirefightingResourceDB(id=resource_id, station_id=station_id, status=ResourceStatus.AVAILABLE))


@pytest.fixture
def wiring(sqlite_session_factory):
    fire_event_repository = FireEventRepository(sqlite_session_factory)
    resource_commitment_repository = ResourceCommitmentRepository(sqlite_session_factory)
    response_plan_repository = ResponsePlanRepository(sqlite_session_factory)
    lifecycle_service = FireEventLifecycleService(
        fire_event_repository=fire_event_repository,
        resource_commitment_repository=resource_commitment_repository,
        session_factory=sqlite_session_factory,
    )
    return {
        "fire_event_repository": fire_event_repository,
        "resource_commitment_repository": resource_commitment_repository,
        "response_plan_repository": response_plan_repository,
        "lifecycle_service": lifecycle_service,
    }


def _activate_plan_with_resources(sqlite_session_factory, response_plan_repository, fire_event_id, resource_ids):
    """Build+persist+activate a real ResponsePlan owning `resource_ids` for fire_event_id."""
    session = sqlite_session_factory()
    _persist_resources(session, resource_ids)
    RoadNetworkRepository().save_network(
        session,
        nodes=[
            GraphNode(id=SOURCE_NODE_ID, latitude=32.7, longitude=35.0),
            GraphNode(id=TARGET_NODE_ID, latitude=32.7, longitude=35.01),
        ],
        edges=[],
    )
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

    actions = []
    for order, resource_id in enumerate(resource_ids, start=1):
        target = ResponseTargetDB(
            response_target_set_id=target_set.id,
            fire_event_id=fire_event_id,
            target_order=order,
            target_type="active_fire",
            latitude=32.7,
            longitude=35.0,
            priority_score=100.0,
        )
        session.add(target)
        session.flush()
        route_result = RouteResultDB(
            route_planning_run_id=route_run.id,
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
        actions.append(ResponseAction(resource_id, target.id, route_result.id))
    session.commit()
    target_set_id, run_id = target_set.id, route_run.id
    session.close()

    stored_plan = response_plan_repository.save(
        ResponsePlan(
            fire_event_id=fire_event_id,
            response_target_set_id=target_set_id,
            route_planning_run_id=run_id,
            generated_at=AS_OF,
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
    activation_service = ResponsePlanActivationService(
        response_plan_repository=response_plan_repository,
        resource_commitment_repository=ResourceCommitmentRepository(sqlite_session_factory),
        session_factory=sqlite_session_factory,
    )
    activation_service.activate(
        response_plan_id=stored_plan.id, planning_effective_state_fingerprint="f" * 64, as_of=AS_OF
    )
    return stored_plan.id


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------


def test_resolve_releases_commitments_and_sets_status(wiring, sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session, status="confirmed")
    session.commit()
    session.close()
    _activate_plan_with_resources(
        sqlite_session_factory, wiring["response_plan_repository"], fire_event_id, ("R1", "R2")
    )
    assert len(wiring["resource_commitment_repository"].get_for_fire_event(fire_event_id)) == 2

    result = wiring["lifecycle_service"].resolve_event(fire_event_id, as_of=LATER)

    assert result.event.status is FireEventStatus.RESOLVED
    assert wiring["resource_commitment_repository"].get_for_fire_event(fire_event_id) == ()


def test_resolve_preserves_historical_response_plan_and_actions(wiring, sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session, status="confirmed")
    session.commit()
    session.close()
    plan_id = _activate_plan_with_resources(
        sqlite_session_factory, wiring["response_plan_repository"], fire_event_id, ("R1",)
    )

    wiring["lifecycle_service"].resolve_event(fire_event_id, as_of=LATER)

    stored_plan = wiring["response_plan_repository"].get_by_id(plan_id)
    assert stored_plan is not None
    assert len(stored_plan.plan.actions) == 1
    assert stored_plan.plan.actions[0].resource_id == "R1"


# ---------------------------------------------------------------------------
# Dismiss
# ---------------------------------------------------------------------------


def test_dismiss_releases_commitments_and_sets_status(wiring, sqlite_session_factory):
    # Task 9A: commitments can only be created for a CONFIRMED event. A SUSPECTED event that still owns commitments can
    # only be legacy data (committed before 9A) - reproduce that by demoting the status directly, then dismiss it.
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session, status="confirmed")
    session.commit()
    session.close()
    _activate_plan_with_resources(
        sqlite_session_factory, wiring["response_plan_repository"], fire_event_id, ("R1",)
    )
    with sqlite_session_factory() as session:
        session.execute(text("UPDATE fire_events SET status = 'suspected' WHERE id = :id"), {"id": fire_event_id})
        session.commit()

    result = wiring["lifecycle_service"].dismiss_event(fire_event_id, as_of=LATER)

    assert result.event.status is FireEventStatus.DISMISSED
    assert wiring["resource_commitment_repository"].get_for_fire_event(fire_event_id) == ()


# ---------------------------------------------------------------------------
# Active transition must NOT release (Task 6) - exercised via the EXISTING
# active-only path (FireEventRepository.update_event), since
# FireEventLifecycleService deliberately implements only the two inactive
# transitions (Task 2/3: no new lifecycle rules, no generic state machine).
# ---------------------------------------------------------------------------


def test_suspected_to_confirmed_transition_creates_no_commitments_and_only_then_allows_them(wiring, sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session, status="suspected")
    session.commit()
    session.close()
    commitments = wiring["resource_commitment_repository"]
    assert commitments.get_for_fire_event(fire_event_id) == ()  # Task 9A: a SUSPECTED event owns no commitments

    stored = wiring["fire_event_repository"].get_by_id(fire_event_id)
    from dataclasses import replace

    confirmed_event = replace(stored.event, status=FireEventStatus.CONFIRMED, updated_at=LATER)
    wiring["fire_event_repository"].update_event(fire_event_id, confirmed_event)

    assert commitments.get_for_fire_event(fire_event_id) == ()  # promotion itself commits nothing
    _activate_plan_with_resources(
        sqlite_session_factory, wiring["response_plan_repository"], fire_event_id, ("R1", "R2")
    )  # ...but now the response-eligible event may commit resources
    assert {c.resource_id for c in commitments.get_for_fire_event(fire_event_id)} == {"R1", "R2"}


# ---------------------------------------------------------------------------
# Atomic rollback
# ---------------------------------------------------------------------------


class _FailingReleaseResourceCommitmentRepository(ResourceCommitmentRepository):
    """Real repository whose in-session release is forced to fail, to prove
    the status write it's paired with rolls back too (Task 4/11)."""

    def release_for_fire_event_in_session(self, session, fire_event_id):
        raise RuntimeError("release failed")


def test_release_failure_rolls_back_the_status_transition_too(wiring, sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session, status="confirmed")
    session.commit()
    session.close()
    _activate_plan_with_resources(
        sqlite_session_factory, wiring["response_plan_repository"], fire_event_id, ("R1",)
    )

    failing_service = FireEventLifecycleService(
        fire_event_repository=wiring["fire_event_repository"],
        resource_commitment_repository=_FailingReleaseResourceCommitmentRepository(sqlite_session_factory),
        session_factory=sqlite_session_factory,
    )

    with pytest.raises(RuntimeError, match="release failed"):
        failing_service.resolve_event(fire_event_id, as_of=LATER)

    stored = wiring["fire_event_repository"].get_by_id(fire_event_id)
    assert stored.event.status is FireEventStatus.CONFIRMED  # unchanged - rolled back
    committed = {c.resource_id for c in wiring["resource_commitment_repository"].get_for_fire_event(fire_event_id)}
    assert committed == {"R1"}  # unchanged - rolled back


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_resolving_an_already_resolved_event_twice_is_safe(wiring, sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session, status="confirmed")
    session.commit()
    session.close()
    _activate_plan_with_resources(
        sqlite_session_factory, wiring["response_plan_repository"], fire_event_id, ("R1",)
    )

    first = wiring["lifecycle_service"].resolve_event(fire_event_id, as_of=LATER)
    second = wiring["lifecycle_service"].resolve_event(fire_event_id, as_of=LATER + timedelta(minutes=1))

    assert first.event.status is FireEventStatus.RESOLVED
    assert second.event.status is FireEventStatus.RESOLVED
    assert wiring["resource_commitment_repository"].get_for_fire_event(fire_event_id) == ()


def test_resolve_event_not_found_raises(wiring):
    with pytest.raises(ValueError):
        wiring["lifecycle_service"].resolve_event(999999, as_of=LATER)


def test_dismiss_event_not_found_raises(wiring):
    with pytest.raises(ValueError):
        wiring["lifecycle_service"].dismiss_event(999999, as_of=LATER)


def test_invalid_arguments_rejected(wiring):
    with pytest.raises(ValueError):
        wiring["lifecycle_service"].resolve_event(0, as_of=LATER)
    with pytest.raises(ValueError):
        wiring["lifecycle_service"].resolve_event(1, as_of=datetime(2026, 1, 1))  # naive
    with pytest.raises(ValueError):
        wiring["lifecycle_service"].dismiss_event(-1, as_of=LATER)
