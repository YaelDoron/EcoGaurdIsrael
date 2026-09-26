"""Tests for GlobalPlanningInputBuilder's Stage 3.1 fail-safe behavior:
GlobalPlanningInputUnstable instead of returning a known-stale input.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.models.global_planning_resource import GlobalPlanningResource
from src.models.resource_status import ResourceStatus
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.services.global_planning.current_global_assignment_loader import CurrentGlobalAssignmentLoader
from src.services.global_planning.global_candidate_collector import GlobalCandidateCollector
from src.services.global_planning.global_incident_demand_builder import GlobalIncidentDemandBuilder
from src.services.global_planning.global_planning_input_builder import GlobalPlanningInputBuilder
from src.services.global_planning.global_planning_input_unstable import GlobalPlanningInputUnstable
from src.services.global_planning.global_route_matrix_builder import GlobalRouteMatrixBuilder
from src.services.operational.operational_context_service import OperationalContextService
from src.services.response_planning.current_response_plan_resolver import CurrentResponsePlanResolver

AS_OF = datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Test doubles for precise, deterministic control-flow tests
# ---------------------------------------------------------------------------


def _resource(status=ResourceStatus.AVAILABLE, commitment_fire_event_id=None, commitment_plan_id=None):
    return GlobalPlanningResource(
        resource_id="R1",
        station_id="S1",
        station_name="S1",
        station_latitude=32.7,
        station_longitude=35.0,
        operational_status=status,
        current_commitment_fire_event_id=commitment_fire_event_id,
        current_commitment_response_plan_id=commitment_plan_id,
    )


class _FakeMember:
    def __init__(self, fire_event_id: int) -> None:
        self.member = type("M", (), {"fire_event_id": fire_event_id})()


class FakeGlobalPlanningRunRepository:
    def __init__(self, fire_event_ids):
        self._members = tuple(_FakeMember(fid) for fid in fire_event_ids)

    def get_members(self, global_planning_run_id):
        return self._members


class FakeResponseTargetRepository:
    """No FireEvent has a usable current target set - keeps these tests
    focused purely on resource-state stability, not routing/target
    content, which are already covered elsewhere in Stage 3's suite."""

    def get_latest_for_event_as_of(self, fire_event_id, as_of):
        return None


class FakeRoadNetworkRepository:
    def get_network_in_bbox(self, session, *args, **kwargs):
        return [], []

    def save_network(self, session, nodes, edges):
        raise AssertionError("GlobalPlanningInputBuilder must never write to the road network.")


class FakeRoadNetworkFetcher:
    def fetch_network_in_bbox(self, *args, **kwargs):
        return [], []

    def fetch_network_in_bbox_tiled(self, *args, **kwargs):
        return [], []


class SequencedCandidateCollector:
    """Returns a pre-programmed resource tuple on each successive .collect()
    call (clamped to the last entry once exhausted), so a test can dictate
    the exact before/after material-state sequence the builder observes."""

    def __init__(self, sequence):
        self._sequence = list(sequence)
        self.calls = 0

    def collect(self, anchors, desired_assignable_supply=None, required_resource_ids=None):
        index = min(self.calls, len(self._sequence) - 1)
        self.calls += 1
        return self._sequence[index]


def _make_builder(*, fire_event_ids, candidate_collector, session_factory) -> GlobalPlanningInputBuilder:
    return GlobalPlanningInputBuilder(
        global_planning_run_repository=FakeGlobalPlanningRunRepository(fire_event_ids),
        response_target_repository=FakeResponseTargetRepository(),
        candidate_collector=candidate_collector,
        incident_demand_builder=GlobalIncidentDemandBuilder(
            fire_severity_assessment_repository=FireSeverityAssessmentRepository(session_factory)
        ),
        current_global_assignment_loader=CurrentGlobalAssignmentLoader(
            current_response_plan_resolver=CurrentResponsePlanResolver(
                ResponsePlanRepository(session_factory), ResponsePlanPlanningStateRepository(session_factory)
            ),
            resource_commitment_repository=ResourceCommitmentRepository(session_factory),
        ),
        route_matrix_builder=GlobalRouteMatrixBuilder(),
        road_network_repository=FakeRoadNetworkRepository(),
        road_network_fetcher=FakeRoadNetworkFetcher(),
        session_factory=session_factory,
    )


# ---------------------------------------------------------------------------
# Stable first attempt
# ---------------------------------------------------------------------------


def test_stable_first_attempt_returns_input_normally(sqlite_session_factory):
    resources = (_resource(ResourceStatus.AVAILABLE),)
    collector = SequencedCandidateCollector([resources, resources])
    builder = _make_builder(fire_event_ids=(1,), candidate_collector=collector, session_factory=sqlite_session_factory)

    result = builder.build(global_planning_run_id=1, as_of=AS_OF)

    assert result.resources == resources
    assert collector.calls == 2  # 1 initial collect + 1 revalidation, no rebuild needed


# ---------------------------------------------------------------------------
# One change, then stable
# ---------------------------------------------------------------------------


def test_one_change_then_stable_returns_the_second_attempts_input(sqlite_session_factory):
    resources_v1 = (_resource(ResourceStatus.AVAILABLE),)
    resources_v2 = (_resource(ResourceStatus.ASSIGNED, commitment_fire_event_id=1, commitment_plan_id=1),)
    collector = SequencedCandidateCollector([resources_v1, resources_v2, resources_v2])
    builder = _make_builder(fire_event_ids=(1,), candidate_collector=collector, session_factory=sqlite_session_factory)

    result = builder.build(global_planning_run_id=1, as_of=AS_OF)

    assert result.resources == resources_v2
    assert collector.calls == 3  # initial(v1) + revalidate(v2, changed) + revalidate(v2, stable)


# ---------------------------------------------------------------------------
# Repeated change -> GlobalPlanningInputUnstable
# ---------------------------------------------------------------------------


def test_repeated_change_raises_unstable_and_returns_no_input(sqlite_session_factory):
    resources_v1 = (_resource(ResourceStatus.AVAILABLE),)
    resources_v2 = (_resource(ResourceStatus.ASSIGNED, commitment_fire_event_id=1, commitment_plan_id=1),)
    resources_v3 = (_resource(ResourceStatus.UNAVAILABLE),)
    collector = SequencedCandidateCollector([resources_v1, resources_v2, resources_v3])
    builder = _make_builder(fire_event_ids=(1,), candidate_collector=collector, session_factory=sqlite_session_factory)

    with pytest.raises(GlobalPlanningInputUnstable) as excinfo:
        builder.build(global_planning_run_id=1, as_of=AS_OF)

    assert excinfo.value.global_planning_run_id == 1
    assert excinfo.value.attempts == 2
    assert collector.calls == 3  # initial(v1) + revalidate(v2, changed) + revalidate(v3, changed again)


def test_unstable_exception_does_not_leak_repository_or_sql_details(sqlite_session_factory):
    resources_v1 = (_resource(ResourceStatus.AVAILABLE),)
    resources_v2 = (_resource(ResourceStatus.UNAVAILABLE),)
    resources_v3 = (_resource(ResourceStatus.AVAILABLE, commitment_fire_event_id=1, commitment_plan_id=1),)
    collector = SequencedCandidateCollector([resources_v1, resources_v2, resources_v3])
    builder = _make_builder(fire_event_ids=(1,), candidate_collector=collector, session_factory=sqlite_session_factory)

    with pytest.raises(GlobalPlanningInputUnstable) as excinfo:
        builder.build(global_planning_run_id=1, as_of=AS_OF)

    message = str(excinfo.value)
    for forbidden in ("SELECT", "sqlalchemy", "IntegrityError", "session"):
        assert forbidden.lower() not in message.lower()


# ---------------------------------------------------------------------------
# No DB writes on failure (real DB-backed scenario, always-changing state)
# ---------------------------------------------------------------------------


class AlwaysChangingCollector:
    """Wraps a REAL GlobalCandidateCollector but flips the first resource's
    operational_status on every call, forcing genuine, deterministic
    repeated instability against a real SQLite-backed candidate pool."""

    def __init__(self, real_collector: GlobalCandidateCollector) -> None:
        self._real_collector = real_collector
        self._toggle = False

    def collect(self, anchors, desired_assignable_supply=None, required_resource_ids=None):
        resources = self._real_collector.collect(anchors, desired_assignable_supply, required_resource_ids)
        self._toggle = not self._toggle
        status = ResourceStatus.UNAVAILABLE if self._toggle else ResourceStatus.AVAILABLE
        return tuple(replace(resource, operational_status=status) for resource in resources)


def test_failure_causes_no_db_mutation(sqlite_session_factory):
    session = sqlite_session_factory()
    event = FireEventDB(
        latitude=32.7, longitude=35.0, detected_at=AS_OF, updated_at=AS_OF, status="confirmed",
        detection_confidence=0.9, methodology="m", methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    station = FireStationDB(id="S1", name="S1", latitude=32.7, longitude=35.0)
    session.add(station)
    session.flush()
    session.add(FirefightingResourceDB(id="R1", station_id="S1", status=ResourceStatus.AVAILABLE))
    target_set = ResponseTargetSetDB(
        fire_event_id=event.id, generated_at=AS_OF, methodology="m", methodology_version="1.0"
    )
    session.add(target_set)
    session.flush()
    session.add(
        ResponseTargetDB(
            response_target_set_id=target_set.id, fire_event_id=event.id, target_order=1,
            target_type="active_fire", latitude=32.7, longitude=35.0, priority_score=100.0,
        )
    )
    session.commit()
    event_id = event.id
    session.close()

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    stored_run = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="legacy_per_event_orchestration",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=(event_id,),
    )

    fire_station_repository = FireStationRepository(sqlite_session_factory)
    firefighting_resource_repository = FirefightingResourceRepository(sqlite_session_factory)
    real_collector = GlobalCandidateCollector(
        fire_station_repository=fire_station_repository,
        firefighting_resource_repository=firefighting_resource_repository,
        resource_commitment_repository=ResourceCommitmentRepository(sqlite_session_factory),
        operational_context_service=OperationalContextService(
            fire_station_repository=fire_station_repository,
            firefighting_resource_repository=firefighting_resource_repository,
        ),
    )
    builder = GlobalPlanningInputBuilder(
        global_planning_run_repository=run_repository,
        response_target_repository=ResponseTargetRepository(sqlite_session_factory),
        candidate_collector=AlwaysChangingCollector(real_collector),
        incident_demand_builder=GlobalIncidentDemandBuilder(
            fire_severity_assessment_repository=FireSeverityAssessmentRepository(sqlite_session_factory)
        ),
        current_global_assignment_loader=CurrentGlobalAssignmentLoader(
            current_response_plan_resolver=CurrentResponsePlanResolver(
                ResponsePlanRepository(sqlite_session_factory),
                ResponsePlanPlanningStateRepository(sqlite_session_factory),
            ),
            resource_commitment_repository=ResourceCommitmentRepository(sqlite_session_factory),
        ),
        route_matrix_builder=GlobalRouteMatrixBuilder(),
        # In-memory test: the road network is irrelevant to "failure causes no DB
        # mutation", so it uses the module's fakes - never the real Neon road cache
        # or a live OpenStreetMap/Overpass fetch (which previously hung this test).
        road_network_repository=FakeRoadNetworkRepository(),
        road_network_fetcher=FakeRoadNetworkFetcher(),
        session_factory=sqlite_session_factory,
    )

    def _row_counts():
        session = sqlite_session_factory()
        try:
            plans = len(session.execute(select(ResponsePlanDB)).scalars().all())
            target_sets = len(session.execute(select(ResponseTargetSetDB)).scalars().all())
            targets = len(session.execute(select(ResponseTargetDB)).scalars().all())
            resources = len(session.execute(select(FirefightingResourceDB)).scalars().all())
            return plans, target_sets, targets, resources
        finally:
            session.close()

    before = _row_counts()

    with pytest.raises(GlobalPlanningInputUnstable):
        builder.build(global_planning_run_id=stored_run.id, as_of=AS_OF)

    after = _row_counts()
    assert before == after
    # And the one resource's real, persisted status is untouched by the fake toggling.
    assert firefighting_resource_repository.get_by_id("R1").status is ResourceStatus.AVAILABLE
