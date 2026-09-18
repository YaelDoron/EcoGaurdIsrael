"""Tests for GlobalCandidateCollector (Stage 3 of the Global Multi-Incident
Optimizer refactor, Tasks 4/7/8/9/26)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.models.resource_status import ResourceStatus
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.services.global_planning.global_candidate_collector import GlobalCandidateCollector
from src.services.global_planning.global_candidate_resource_config import GlobalCandidateResourceConfig
from src.services.operational.operational_context_service import OperationalContextService

COMMITTED_AT = datetime(2026, 9, 21, 8, 0, tzinfo=timezone.utc)


def _persist_station(session, station_id, latitude, longitude, name=None) -> None:
    session.add(FireStationDB(id=station_id, name=name or station_id, latitude=latitude, longitude=longitude))


def _persist_resource(session, resource_id, station_id, status=ResourceStatus.AVAILABLE) -> None:
    session.add(FirefightingResourceDB(id=resource_id, station_id=station_id, status=status))


@pytest.fixture
def collector(sqlite_session_factory) -> GlobalCandidateCollector:
    return GlobalCandidateCollector(
        fire_station_repository=FireStationRepository(sqlite_session_factory),
        firefighting_resource_repository=FirefightingResourceRepository(sqlite_session_factory),
        resource_commitment_repository=ResourceCommitmentRepository(sqlite_session_factory),
        operational_context_service=OperationalContextService(
            fire_station_repository=FireStationRepository(sqlite_session_factory),
            firefighting_resource_repository=FirefightingResourceRepository(sqlite_session_factory),
        ),
    )


def test_one_resource_near_two_fires_appears_exactly_once(collector, sqlite_session_factory):
    session = sqlite_session_factory()
    _persist_station(session, "S1", 32.70, 35.00)
    _persist_resource(session, "R1", "S1")
    session.commit()
    session.close()

    # Two anchors close enough together that S1 is within range of both.
    resources = collector.collect([(1, 32.701, 35.001), (2, 32.702, 35.002)])

    assert [r.resource_id for r in resources] == ["R1"]


def test_resources_from_multiple_stations_are_both_included(collector, sqlite_session_factory):
    session = sqlite_session_factory()
    _persist_station(session, "S1", 32.70, 35.00)
    _persist_station(session, "S2", 32.90, 35.20)
    _persist_resource(session, "R1", "S1")
    _persist_resource(session, "R2", "S2")
    session.commit()
    session.close()

    resources = collector.collect([(1, 32.70, 35.00), (2, 32.90, 35.20)])

    assert {r.resource_id for r in resources} == {"R1", "R2"}


def test_farther_station_discoverable_via_max_fallback_stations(sqlite_session_factory):
    session = sqlite_session_factory()
    _persist_station(session, "S1", 32.70, 35.00)
    _persist_station(session, "S_FAR", 40.00, 45.00)  # far outside any progressive radius
    _persist_resource(session, "R1", "S1")
    _persist_resource(session, "R_FAR", "S_FAR")
    session.commit()
    session.close()

    narrow_collector = GlobalCandidateCollector(
        fire_station_repository=FireStationRepository(sqlite_session_factory),
        firefighting_resource_repository=FirefightingResourceRepository(sqlite_session_factory),
        resource_commitment_repository=ResourceCommitmentRepository(sqlite_session_factory),
        operational_context_service=OperationalContextService(
            fire_station_repository=FireStationRepository(sqlite_session_factory),
            firefighting_resource_repository=FirefightingResourceRepository(sqlite_session_factory),
        ),
        config=GlobalCandidateResourceConfig(max_fallback_stations=2),
    )

    resources = narrow_collector.collect([(1, 32.70, 35.00)])

    assert {r.resource_id for r in resources} == {"R1", "R_FAR"}


def test_unavailable_resource_is_collected_but_not_assignable(collector, sqlite_session_factory):
    session = sqlite_session_factory()
    _persist_station(session, "S1", 32.70, 35.00)
    _persist_resource(session, "R1", "S1", status=ResourceStatus.UNAVAILABLE)
    session.commit()
    session.close()

    (resource,) = collector.collect([(1, 32.70, 35.00)])

    assert resource.operational_status is ResourceStatus.UNAVAILABLE
    assert resource.is_assignable is False


def test_committed_resource_appears_with_ownership_metadata(collector, sqlite_session_factory):
    from src.database.models.fire_event_db import FireEventDB
    from src.database.models.response_plan_db import ResponsePlanDB
    from src.database.models.response_target_set_db import ResponseTargetSetDB
    from src.database.models.route_planning_run_db import RoutePlanningRunDB
    from src.repositories.resource_commitment_repository import ResourceCommitmentRepository as RCR

    session = sqlite_session_factory()
    _persist_station(session, "S1", 32.70, 35.00)
    _persist_resource(session, "R1", "S1", status=ResourceStatus.ASSIGNED)
    event = FireEventDB(
        latitude=32.7, longitude=35.0, detected_at=COMMITTED_AT, updated_at=COMMITTED_AT,
        status="confirmed", detection_confidence=0.9, methodology="m", methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    target_set = ResponseTargetSetDB(
        fire_event_id=event.id, generated_at=COMMITTED_AT, methodology="m", methodology_version="1.0"
    )
    session.add(target_set)
    session.flush()
    route_run = RoutePlanningRunDB(
        fire_event_id=event.id, response_target_set_id=target_set.id, planned_at=COMMITTED_AT,
        methodology="m", methodology_version="1.0", resource_ids=[],
    )
    session.add(route_run)
    session.flush()
    plan = ResponsePlanDB(
        fire_event_id=event.id, response_target_set_id=target_set.id, route_planning_run_id=route_run.id,
        generated_at=COMMITTED_AT, status="complete", methodology="m", methodology_version="1.0", random_seed=1,
    )
    session.add(plan)
    session.flush()
    event_id, plan_id = event.id, plan.id
    session.commit()
    session.close()

    # replace_commitments_for_plan uses a caller-owned session - commit it ourselves.
    session2 = sqlite_session_factory()
    RCR(sqlite_session_factory).replace_commitments_for_plan(
        session2, fire_event_id=event_id, response_plan_id=plan_id, resource_ids=("R1",), committed_at=COMMITTED_AT,
    )
    session2.commit()
    session2.close()

    (resource,) = collector.collect([(event_id, 32.70, 35.00)])

    assert resource.current_commitment_fire_event_id == event_id
    assert resource.current_commitment_response_plan_id == plan_id
    assert resource.is_committed is True
    assert resource.is_assignable is True


# ---------------------------------------------------------------------------
# Stage 5 demand-aware expansion / farther-station reinforcement (Tasks 11-12)
# ---------------------------------------------------------------------------


def test_desired_assignable_supply_pulls_in_a_farther_station(collector, sqlite_session_factory):
    """Task 31 evidence: local supply alone (2) is short of desired demand
    (4) - a genuinely farther station is discovered and its resources make
    up the shortfall, without fabricating any resource."""
    session = sqlite_session_factory()
    _persist_station(session, "S1", 32.70, 35.00)
    _persist_station(session, "S_FAR", 40.00, 45.00)
    _persist_resource(session, "R1", "S1")
    _persist_resource(session, "R2", "S1")
    _persist_resource(session, "R_FAR1", "S_FAR")
    _persist_resource(session, "R_FAR2", "S_FAR")
    _persist_resource(session, "R_FAR3", "S_FAR")
    session.commit()
    session.close()

    resources = collector.collect([(1, 32.70, 35.00)], desired_assignable_supply=4)

    resource_ids = {r.resource_id for r in resources}
    assert {"R1", "R2"}.issubset(resource_ids)
    assert resource_ids & {"R_FAR1", "R_FAR2", "R_FAR3"}
    assert len(resources) >= 4


def test_no_expansion_when_local_supply_already_meets_demand(collector, sqlite_session_factory):
    session = sqlite_session_factory()
    _persist_station(session, "S1", 32.70, 35.00)
    _persist_station(session, "S_FAR", 40.00, 45.00)
    _persist_resource(session, "R1", "S1")
    _persist_resource(session, "R2", "S1")
    _persist_resource(session, "R_FAR", "S_FAR")
    session.commit()
    session.close()

    resources = collector.collect([(1, 32.70, 35.00)], desired_assignable_supply=2)

    assert {r.resource_id for r in resources} == {"R1", "R2"}


def test_unavailable_resources_do_not_satisfy_demand_and_trigger_expansion(collector, sqlite_session_factory):
    """UNAVAILABLE resources are collected but never count toward
    assignable supply (Task 34's principle applied to demand-driven
    expansion): a station with only UNAVAILABLE resources must not stop
    the collector from reaching out to a farther, genuinely assignable
    station."""
    session = sqlite_session_factory()
    _persist_station(session, "S1", 32.70, 35.00)
    _persist_station(session, "S_FAR", 40.00, 45.00)
    _persist_resource(session, "R1", "S1", status=ResourceStatus.UNAVAILABLE)
    _persist_resource(session, "R_FAR", "S_FAR", status=ResourceStatus.AVAILABLE)
    session.commit()
    session.close()

    resources = collector.collect([(1, 32.70, 35.00)], desired_assignable_supply=1)

    assert {r.resource_id for r in resources} == {"R1", "R_FAR"}


def test_max_demand_driven_stations_caps_reinforcement(sqlite_session_factory):
    """Task 12's independent cap: even if demand is still unmet, the
    collector stops adding farther stations once the cap is reached rather
    than reaching out indefinitely."""
    session = sqlite_session_factory()
    _persist_station(session, "S1", 32.70, 35.00)
    _persist_station(session, "S_FAR1", 40.00, 45.00)
    _persist_station(session, "S_FAR2", 41.00, 46.00)
    _persist_resource(session, "R1", "S1")
    _persist_resource(session, "R_FAR1", "S_FAR1")
    _persist_resource(session, "R_FAR2", "S_FAR2")
    session.commit()
    session.close()

    capped_collector = GlobalCandidateCollector(
        fire_station_repository=FireStationRepository(sqlite_session_factory),
        firefighting_resource_repository=FirefightingResourceRepository(sqlite_session_factory),
        resource_commitment_repository=ResourceCommitmentRepository(sqlite_session_factory),
        operational_context_service=OperationalContextService(
            fire_station_repository=FireStationRepository(sqlite_session_factory),
            firefighting_resource_repository=FirefightingResourceRepository(sqlite_session_factory),
        ),
        config=GlobalCandidateResourceConfig(max_demand_driven_stations=1),
    )

    resources = capped_collector.collect([(1, 32.70, 35.00)], desired_assignable_supply=3)

    # Only one additional (nearest) farther station may be pulled in, so
    # supply stays short of the requested 3 even though a second farther
    # station genuinely exists.
    assert len(resources) == 2
    assert {r.resource_id for r in resources} == {"R1", "R_FAR1"}


# ---------------------------------------------------------------------------
# Stage 6 - required_resource_ids pulls in a hard-dispatched resource's
# station regardless of geography (Task 14)
# ---------------------------------------------------------------------------


def test_required_resource_ids_pulls_in_a_far_dispatched_resources_station(collector, sqlite_session_factory):
    session = sqlite_session_factory()
    _persist_station(session, "S1", 32.70, 35.00)
    _persist_station(session, "S_FAR", 40.00, 45.00)
    _persist_resource(session, "R1", "S1")
    _persist_resource(session, "R_FAR", "S_FAR")
    session.commit()
    session.close()

    resources = collector.collect([(1, 32.70, 35.00)], required_resource_ids=("R_FAR",))

    assert {r.resource_id for r in resources} == {"R1", "R_FAR"}


def test_required_resource_ids_is_a_no_op_when_already_within_normal_reach(collector, sqlite_session_factory):
    session = sqlite_session_factory()
    _persist_station(session, "S1", 32.70, 35.00)
    _persist_resource(session, "R1", "S1")
    session.commit()
    session.close()

    resources = collector.collect([(1, 32.70, 35.00)], required_resource_ids=("R1",))

    assert {r.resource_id for r in resources} == {"R1"}


def test_required_resource_ids_silently_ignores_a_resource_that_no_longer_exists(collector, sqlite_session_factory):
    session = sqlite_session_factory()
    _persist_station(session, "S1", 32.70, 35.00)
    _persist_resource(session, "R1", "S1")
    session.commit()
    session.close()

    resources = collector.collect([(1, 32.70, 35.00)], required_resource_ids=("GHOST",))

    assert {r.resource_id for r in resources} == {"R1"}
