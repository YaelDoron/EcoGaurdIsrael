"""Unit tests for RoutePlanningRepository using SQLite in-memory."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.calculators.response_target.response_target_config import (
    RESPONSE_TARGET_METHODOLOGY_NAME,
    RESPONSE_TARGET_METHODOLOGY_VERSION,
)
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.database.models.route_result_db import RouteResultDB
from src.models import (
    FireEvent,
    FireEventStatus,
    FireEvidenceRef,
    FireEvidenceType,
    GraphNode,
    ResponseTarget,
    ResponseTargetSet,
    ResponseTargetType,
    RoutePlanningRun,
    RouteResult,
    RouteStatus,
    SatelliteHotspot,
)
from src.models.resource_status import ResourceStatus
from src.repositories.exceptions import RoutePlanningRepositoryError
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.repositories.route_planning_repository import RoutePlanningRepository, StoredRoutePlanningRun
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository

ROUTING_METHODOLOGY_NAME = "ECOGUARD_ROUTING_DIJKSTRA"
ROUTING_METHODOLOGY_VERSION = "1.0"

PLANNED_AT = datetime(2026, 9, 16, 12, 30, tzinfo=timezone.utc)
GENERATED_AT = PLANNED_AT - timedelta(minutes=5)
EVENT_TIME = GENERATED_AT - timedelta(minutes=40)

SOURCE_NODE_ID = 1001
TARGET_NODE_ID = 1002


@pytest.fixture
def repository(sqlite_session_factory) -> RoutePlanningRepository:
    return RoutePlanningRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def fire_event_repository(sqlite_session_factory) -> FireEventRepository:
    return FireEventRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def response_target_repository(sqlite_session_factory) -> ResponseTargetRepository:
    return ResponseTargetRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def satellite_repository(sqlite_session_factory) -> SatelliteHotspotRepository:
    return SatelliteHotspotRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def event_id(fire_event_repository, satellite_repository) -> int:
    return persist_event(fire_event_repository, satellite_repository)


@pytest.fixture
def response_target_set_id(event_id, response_target_repository) -> int:
    stored = response_target_repository.save_target_set(make_target_set(event_id, (make_active(event_id),)))
    return stored.id


@pytest.fixture
def response_target_id(event_id, response_target_set_id, sqlite_session_factory) -> int:
    session = sqlite_session_factory()
    row = session.execute(
        select(ResponseTargetDB).where(ResponseTargetDB.response_target_set_id == response_target_set_id)
    ).scalars().one()
    target_id = row.id
    session.close()
    return target_id


@pytest.fixture(autouse=True)
def graph_nodes(sqlite_session_factory) -> None:
    session = sqlite_session_factory()
    RoadNetworkRepository().save_network(
        session,
        nodes=[
            GraphNode(id=SOURCE_NODE_ID, latitude=32.700, longitude=35.000),
            GraphNode(id=TARGET_NODE_ID, latitude=32.700, longitude=35.010),
        ],
        edges=[],
    )
    session.close()


@pytest.fixture(autouse=True)
def firefighting_resources(sqlite_session_factory) -> None:
    """route_results.resource_id is now a real FK (FND-05); every resource_id
    literal ("truck-1", "truck-2") used across this file's routes needs a
    matching FirefightingResourceDB row."""
    session = sqlite_session_factory()
    session.add(FireStationDB(id="FIXTURE-STATION", name="Fixture Station", latitude=32.7, longitude=35.0))
    session.flush()
    session.add_all(
        [
            FirefightingResourceDB(id="truck-1", station_id="FIXTURE-STATION", status=ResourceStatus.AVAILABLE),
            FirefightingResourceDB(id="truck-2", station_id="FIXTURE-STATION", status=ResourceStatus.AVAILABLE),
        ]
    )
    session.commit()
    session.close()


def make_event(**overrides) -> FireEvent:
    defaults = dict(
        latitude=32.731,
        longitude=35.046,
        detected_at=EVENT_TIME,
        updated_at=GENERATED_AT - timedelta(minutes=5),
        status=FireEventStatus.CONFIRMED,
        detection_confidence=0.85,
        methodology=FIRE_DETECTION_METHODOLOGY_NAME,
        methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
    )
    defaults.update(overrides)
    return FireEvent(**defaults)


def persist_event(fire_event_repository, satellite_repository, **overrides) -> int:
    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=32.731,
            longitude=35.046,
            detected_at=EVENT_TIME,
            confidence="h",
            frp=72.0,
            satellite="N20",
        )
    )
    hotspot_id = satellite_repository.get_recent_hotspots(as_of=GENERATED_AT, lookback_minutes=360)[0].id
    stored = fire_event_repository.create_event(
        make_event(**overrides),
        supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),),
    )
    return stored.id


def make_active(fire_event_id: int, **overrides) -> ResponseTarget:
    defaults = dict(
        fire_event_id=fire_event_id,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=32.731,
        longitude=35.046,
        priority_score=150.0,
    )
    defaults.update(overrides)
    return ResponseTarget(**defaults)


def make_target_set(fire_event_id: int, targets: tuple[ResponseTarget, ...], **overrides) -> ResponseTargetSet:
    defaults = dict(
        fire_event_id=fire_event_id,
        generated_at=GENERATED_AT,
        methodology=RESPONSE_TARGET_METHODOLOGY_NAME,
        methodology_version=RESPONSE_TARGET_METHODOLOGY_VERSION,
        targets=targets,
    )
    defaults.update(overrides)
    return ResponseTargetSet(**defaults)


def make_reachable_route(resource_id: str, response_target_id: int, **overrides) -> RouteResult:
    defaults = dict(
        resource_id=resource_id,
        response_target_id=response_target_id,
        status=RouteStatus.REACHABLE,
        source_node_id=SOURCE_NODE_ID,
        target_node_id=TARGET_NODE_ID,
        node_path=(SOURCE_NODE_ID, TARGET_NODE_ID),
        distance_meters=1200.0,
        travel_time_seconds=90.0,
    )
    defaults.update(overrides)
    return RouteResult(**defaults)


def make_run(fire_event_id: int, response_target_set_id: int, routes: tuple[RouteResult, ...], **overrides) -> RoutePlanningRun:
    resource_ids = overrides.pop("resource_ids", tuple(sorted({route.resource_id for route in routes})))
    defaults = dict(
        fire_event_id=fire_event_id,
        response_target_set_id=response_target_set_id,
        planned_at=PLANNED_AT,
        methodology=ROUTING_METHODOLOGY_NAME,
        methodology_version=ROUTING_METHODOLOGY_VERSION,
        resource_ids=resource_ids,
        routes=routes,
    )
    defaults.update(overrides)
    return RoutePlanningRun(**defaults)


def get_run_rows(sqlite_session_factory):
    session = sqlite_session_factory()
    rows = session.execute(select(RoutePlanningRunDB)).scalars().all()
    session.close()
    return rows


def get_route_rows(sqlite_session_factory):
    session = sqlite_session_factory()
    rows = session.execute(select(RouteResultDB).order_by(RouteResultDB.id.asc())).scalars().all()
    session.close()
    return rows


def test_save_run_creates_header_and_route_row(repository, event_id, response_target_set_id, response_target_id, sqlite_session_factory):
    run = make_run(event_id, response_target_set_id, (make_reachable_route("truck-1", response_target_id),))

    stored = repository.save_run(run)

    assert isinstance(stored, StoredRoutePlanningRun)
    assert stored.id > 0
    assert len(get_run_rows(sqlite_session_factory)) == 1
    rows = get_route_rows(sqlite_session_factory)
    assert len(rows) == 1
    assert rows[0].resource_id == "truck-1"
    assert rows[0].status == "reachable"


def test_save_run_with_multiple_resources_routing_to_same_target(
    repository, event_id, response_target_set_id, response_target_id
):
    routes = (
        make_reachable_route("truck-1", response_target_id),
        make_reachable_route("truck-2", response_target_id, travel_time_seconds=150.0),
    )

    stored = repository.save_run(make_run(event_id, response_target_set_id, routes))

    assert len(stored.routes) == 2
    assert {route.route_result.resource_id for route in stored.routes} == {"truck-1", "truck-2"}


def test_round_trip_preserves_reachable_route_fields(repository, event_id, response_target_set_id, response_target_id):
    run = make_run(event_id, response_target_set_id, (make_reachable_route("truck-1", response_target_id),))

    saved = repository.save_run(run)
    found = repository.get_by_id(saved.id)

    assert found.run == run
    assert found.run.methodology == ROUTING_METHODOLOGY_NAME
    assert found.run.methodology_version == ROUTING_METHODOLOGY_VERSION
    assert found.run.planned_at == PLANNED_AT
    assert found.routes[0].route_result.node_path == (SOURCE_NODE_ID, TARGET_NODE_ID)
    assert found.routes[0].route_result.distance_meters == 1200.0
    assert found.routes[0].route_result.travel_time_seconds == 90.0


def test_round_trip_preserves_unreachable_route(repository, event_id, response_target_set_id, response_target_id):
    unreachable = make_reachable_route(
        "truck-1",
        response_target_id,
        status=RouteStatus.UNREACHABLE,
        node_path=(),
        distance_meters=None,
        travel_time_seconds=None,
    )
    run = make_run(event_id, response_target_set_id, (unreachable,))

    saved = repository.save_run(run)
    found = repository.get_by_id(saved.id)

    result = found.routes[0].route_result
    assert result.status is RouteStatus.UNREACHABLE
    assert result.node_path == ()
    assert result.distance_meters is None
    assert result.travel_time_seconds is None
    assert result.source_node_id == SOURCE_NODE_ID
    assert result.target_node_id == TARGET_NODE_ID


def test_round_trip_preserves_unmappable_route(repository, event_id, response_target_set_id, response_target_id):
    unmappable = make_reachable_route(
        "truck-1",
        response_target_id,
        status=RouteStatus.UNMAPPABLE,
        source_node_id=None,
        target_node_id=None,
        node_path=(),
        distance_meters=None,
        travel_time_seconds=None,
    )
    run = make_run(event_id, response_target_set_id, (unmappable,))

    saved = repository.save_run(run)
    found = repository.get_by_id(saved.id)

    result = found.routes[0].route_result
    assert result.status is RouteStatus.UNMAPPABLE
    assert result.source_node_id is None
    assert result.target_node_id is None


def test_run_with_no_routes_persists_resource_ids_only(repository, event_id, response_target_set_id):
    run = make_run(event_id, response_target_set_id, (), resource_ids=("truck-1", "truck-2"))

    stored = repository.save_run(run)
    found = repository.get_by_id(stored.id)

    assert found.run.resource_ids == ("truck-1", "truck-2")
    assert found.routes == ()


def test_get_by_id_returns_none_for_unknown_id(repository):
    assert repository.get_by_id(999999) is None


def test_missing_response_target_set_reference_fails(repository, event_id):
    run = make_run(event_id, 999999, ())

    with pytest.raises(RoutePlanningRepositoryError):
        repository.save_run(run)


def test_response_target_set_fire_event_mismatch_fails(
    repository, event_id, fire_event_repository, satellite_repository, response_target_set_id
):
    other_event_id = persist_event(fire_event_repository, satellite_repository)
    run = make_run(other_event_id, response_target_set_id, ())

    with pytest.raises(RoutePlanningRepositoryError):
        repository.save_run(run)


def test_missing_response_target_reference_fails(repository, event_id, response_target_set_id):
    run = make_run(event_id, response_target_set_id, (make_reachable_route("truck-1", 999999),))

    with pytest.raises(RoutePlanningRepositoryError):
        repository.save_run(run)


def test_response_target_from_wrong_set_fails(
    repository,
    event_id,
    fire_event_repository,
    satellite_repository,
    response_target_repository,
    response_target_set_id,
):
    other_event_id = persist_event(fire_event_repository, satellite_repository)
    other_set = response_target_repository.save_target_set(make_target_set(other_event_id, (make_active(other_event_id),)))
    other_target_id = other_set.targets[0].id

    run = make_run(event_id, response_target_set_id, (make_reachable_route("truck-1", other_target_id),))

    with pytest.raises(RoutePlanningRepositoryError):
        repository.save_run(run)


def test_unknown_source_node_id_fails(repository, event_id, response_target_set_id, response_target_id):
    route = make_reachable_route("truck-1", response_target_id, source_node_id=999999, node_path=(999999, TARGET_NODE_ID))
    run = make_run(event_id, response_target_set_id, (route,))

    with pytest.raises(RoutePlanningRepositoryError):
        repository.save_run(run)


def test_failure_rolls_back_entire_run(repository, event_id, response_target_set_id, sqlite_session_factory):
    run = make_run(event_id, response_target_set_id, (make_reachable_route("truck-1", 999999),))

    with pytest.raises(RoutePlanningRepositoryError):
        repository.save_run(run)

    assert get_run_rows(sqlite_session_factory) == []
    assert get_route_rows(sqlite_session_factory) == []


def test_append_only_saves_do_not_collapse(repository, event_id, response_target_set_id, response_target_id):
    first = repository.save_run(make_run(event_id, response_target_set_id, (make_reachable_route("truck-1", response_target_id),)))
    second = repository.save_run(
        make_run(
            event_id,
            response_target_set_id,
            (make_reachable_route("truck-1", response_target_id, travel_time_seconds=200.0),),
            planned_at=PLANNED_AT + timedelta(minutes=5),
        )
    )

    assert first.id != second.id
    assert repository.get_by_id(first.id) is not None
    assert repository.get_by_id(second.id) is not None


def test_latest_for_event_as_of_returns_latest_before_boundary(repository, event_id, response_target_set_id, response_target_id):
    older = repository.save_run(
        make_run(
            event_id,
            response_target_set_id,
            (make_reachable_route("truck-1", response_target_id),),
            planned_at=PLANNED_AT - timedelta(minutes=10),
        )
    )
    newer = repository.save_run(
        make_run(
            event_id,
            response_target_set_id,
            (make_reachable_route("truck-1", response_target_id),),
            planned_at=PLANNED_AT,
        )
    )
    repository.save_run(
        make_run(
            event_id,
            response_target_set_id,
            (make_reachable_route("truck-1", response_target_id),),
            planned_at=PLANNED_AT + timedelta(minutes=10),
        )
    )

    latest = repository.get_latest_for_event_as_of(event_id, PLANNED_AT)

    assert latest.id == newer.id
    assert latest.id != older.id


def test_latest_for_event_as_of_exact_boundary_included(repository, event_id, response_target_set_id, response_target_id):
    saved = repository.save_run(
        make_run(event_id, response_target_set_id, (make_reachable_route("truck-1", response_target_id),))
    )

    latest = repository.get_latest_for_event_as_of(event_id, PLANNED_AT)

    assert latest.id == saved.id


def test_latest_for_event_as_of_excludes_other_fire_events(
    repository, event_id, fire_event_repository, satellite_repository, response_target_repository, response_target_set_id, response_target_id
):
    other_event_id = persist_event(fire_event_repository, satellite_repository)
    other_set = response_target_repository.save_target_set(make_target_set(other_event_id, (make_active(other_event_id),)))

    expected = repository.save_run(
        make_run(event_id, response_target_set_id, (make_reachable_route("truck-1", response_target_id),))
    )
    repository.save_run(
        make_run(
            other_event_id,
            other_set.id,
            (),
            resource_ids=(),
            planned_at=PLANNED_AT + timedelta(minutes=1),
        )
    )

    latest = repository.get_latest_for_event_as_of(event_id, PLANNED_AT + timedelta(minutes=10))

    assert latest.id == expected.id


def test_latest_for_event_as_of_tie_break_uses_newest_id(repository, event_id, response_target_set_id, response_target_id):
    first = repository.save_run(
        make_run(event_id, response_target_set_id, (make_reachable_route("truck-1", response_target_id),))
    )
    second = repository.save_run(
        make_run(event_id, response_target_set_id, (make_reachable_route("truck-1", response_target_id),))
    )

    latest = repository.get_latest_for_event_as_of(event_id, PLANNED_AT)

    assert latest.id == second.id
    assert second.id > first.id


def test_latest_for_event_as_of_returns_none_when_no_earlier_run(repository, event_id, response_target_set_id):
    repository.save_run(
        make_run(event_id, response_target_set_id, (), planned_at=PLANNED_AT + timedelta(minutes=1))
    )

    assert repository.get_latest_for_event_as_of(event_id, PLANNED_AT) is None


def test_history_order_is_deterministic(repository, event_id, response_target_set_id):
    first = repository.save_run(make_run(event_id, response_target_set_id, (), planned_at=PLANNED_AT - timedelta(minutes=10)))
    second = repository.save_run(make_run(event_id, response_target_set_id, (), planned_at=PLANNED_AT))
    third = repository.save_run(make_run(event_id, response_target_set_id, (), planned_at=PLANNED_AT))

    history = repository.get_history_for_event(event_id)

    assert [item.id for item in history] == [third.id, second.id, first.id]


def test_invalid_repository_arguments_rejected(repository):
    with pytest.raises(RoutePlanningRepositoryError):
        repository.save_run("not-a-run")
    with pytest.raises(RoutePlanningRepositoryError):
        repository.get_by_id(0)
    with pytest.raises(RoutePlanningRepositoryError):
        repository.get_latest_for_event_as_of(0, PLANNED_AT)
    with pytest.raises(RoutePlanningRepositoryError):
        repository.get_latest_for_event_as_of(1, datetime(2026, 9, 16, 12, 0))
    with pytest.raises(RoutePlanningRepositoryError):
        repository.get_history_for_event(True)
