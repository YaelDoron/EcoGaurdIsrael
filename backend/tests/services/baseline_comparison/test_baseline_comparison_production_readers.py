"""Tests for the production reader adapters, against a real SQLite-backed repository stack."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.graph_node_db import GraphNodeDB
from src.database.models.response_target_db import ResponseTargetDB
from src.models import (
    ResponseAction,
    ResponsePlan,
    ResponsePlanStatus,
    ResponseTarget,
    ResponseTargetSet,
    ResponseTargetType,
)
from src.models.resource_status import ResourceStatus
from src.models.routing import RoutePlanningRun, RouteResult, RouteStatus
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.baseline_comparison.baseline_comparison_production_readers import (
    ResponsePlanOptimizedPlanReaderAdapter,
    RoutePlanningRunReaderAdapter,
)

GENERATED_AT = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def persisted_context(sqlite_session_factory):
    session = sqlite_session_factory()
    event = FireEventDB(
        latitude=32.731,
        longitude=35.046,
        detected_at=GENERATED_AT - timedelta(minutes=30),
        updated_at=GENERATED_AT - timedelta(minutes=5),
        status="confirmed",
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    fire_event_id = event.id
    # route_results.resource_id / response_actions.resource_id are now real
    # FKs (FND-05): every resource_id literal used below ("truck-1", "truck-2")
    # needs a matching FirefightingResourceDB row.
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

    stored_target_set = ResponseTargetRepository(sqlite_session_factory).save_target_set(
        ResponseTargetSet(
            fire_event_id=fire_event_id,
            generated_at=GENERATED_AT - timedelta(minutes=1),
            methodology="TEST_TARGETS",
            methodology_version="1.0",
            targets=(
                ResponseTarget(
                    fire_event_id=fire_event_id,
                    target_type=ResponseTargetType.ACTIVE_FIRE,
                    latitude=32.731,
                    longitude=35.046,
                    priority_score=150.0,
                ),
            ),
        )
    )

    # response_plan_uncovered_targets.response_target_id is now a real FK too
    # (FND-05), so "uncovered" tests need real target ids, not literal (20,
    # 30). ResponseTargetSet only allows exactly one ACTIVE_FIRE target, so
    # these extra rows are inserted directly rather than through another
    # domain-validated ResponseTargetSet.
    session = sqlite_session_factory()
    other_targets = [
        ResponseTargetDB(
            response_target_set_id=stored_target_set.id,
            fire_event_id=fire_event_id,
            target_order=order,
            target_type="active_fire",
            latitude=32.731,
            longitude=35.046,
            priority_score=100.0,
        )
        for order in (1, 2)
    ]
    session.add_all(other_targets)
    session.commit()
    other_target_ids = tuple(target.id for target in other_targets)
    session.close()

    return {
        "fire_event_id": fire_event_id,
        "response_target_set_id": stored_target_set.id,
        "response_target_id": stored_target_set.targets[0].id,
        "other_target_ids": other_target_ids,
    }


def make_route_planning_run(context, resource_ids=("truck-1",)) -> RoutePlanningRun:
    return RoutePlanningRun(
        fire_event_id=context["fire_event_id"],
        response_target_set_id=context["response_target_set_id"],
        planned_at=GENERATED_AT,
        methodology="ECOGUARD_ROUTING_DIJKSTRA",
        methodology_version="1.0",
        resource_ids=resource_ids,
        routes=(
            RouteResult(
                resource_id="truck-1",
                response_target_id=context["response_target_id"],
                status=RouteStatus.UNMAPPABLE,
                source_node_id=None,
                target_node_id=None,
                node_path=(),
                distance_meters=None,
                travel_time_seconds=None,
            ),
        ),
    )


def make_response_plan(context, route_planning_run_id, route_result_id, **overrides) -> ResponsePlan:
    defaults = dict(
        fire_event_id=context["fire_event_id"],
        response_target_set_id=context["response_target_set_id"],
        route_planning_run_id=route_planning_run_id,
        generated_at=GENERATED_AT,
        status=ResponsePlanStatus.COMPLETE,
        methodology="GENETIC_RESOURCE_ALLOCATION",
        methodology_version="1.0",
        random_seed=42,
        actions=(ResponseAction("truck-1", context["response_target_id"], route_result_id),),
        uncovered_target_ids=(),
        plan_score=90.0,
        coverage_score=100.0,
        average_eta_seconds=200.0,
    )
    defaults.update(overrides)
    return ResponsePlan(**defaults)


# ---------------------------------------------------------------------------
# RoutePlanningRunReaderAdapter
# ---------------------------------------------------------------------------


def test_route_planning_run_reader_returns_correct_shape(persisted_context, sqlite_session_factory):
    stored_run = RoutePlanningRepository(sqlite_session_factory).save_run(
        make_route_planning_run(persisted_context, resource_ids=("truck-1", "truck-2"))
    )
    reader = RoutePlanningRunReaderAdapter(RoutePlanningRepository(sqlite_session_factory))

    view = reader.get_by_id(stored_run.id)

    assert view.id == stored_run.id
    assert view.fire_event_id == persisted_context["fire_event_id"]
    assert view.response_target_set_id == persisted_context["response_target_set_id"]
    assert set(view.resource_ids) == {"truck-1", "truck-2"}
    assert len(view.route_results) == 1
    route_view = view.route_results[0]
    assert route_view.resource_id == "truck-1"
    assert route_view.response_target_id == persisted_context["response_target_id"]
    assert route_view.status == "unmappable"
    assert route_view.travel_time_seconds is None


def test_route_planning_run_reader_returns_none_for_missing_run(sqlite_session_factory):
    reader = RoutePlanningRunReaderAdapter(RoutePlanningRepository(sqlite_session_factory))

    assert reader.get_by_id(999999) is None


def test_route_planning_run_reader_carries_distance_meters_for_reachable_routes(
    persisted_context, sqlite_session_factory
):
    session = sqlite_session_factory()
    session.add_all([GraphNodeDB(id=1, latitude=32.7, longitude=35.0), GraphNodeDB(id=2, latitude=32.71, longitude=35.01)])
    session.commit()
    session.close()

    run = RoutePlanningRun(
        fire_event_id=persisted_context["fire_event_id"],
        response_target_set_id=persisted_context["response_target_set_id"],
        planned_at=GENERATED_AT,
        methodology="ECOGUARD_ROUTING_DIJKSTRA",
        methodology_version="1.0",
        resource_ids=("truck-1",),
        routes=(
            RouteResult(
                resource_id="truck-1",
                response_target_id=persisted_context["response_target_id"],
                status=RouteStatus.REACHABLE,
                source_node_id=1,
                target_node_id=2,
                node_path=(1, 2),
                distance_meters=1234.5,
                travel_time_seconds=88.0,
            ),
        ),
    )
    stored_run = RoutePlanningRepository(sqlite_session_factory).save_run(run)
    reader = RoutePlanningRunReaderAdapter(RoutePlanningRepository(sqlite_session_factory))

    view = reader.get_by_id(stored_run.id)

    route_view = view.route_results[0]
    assert route_view.status == "reachable"
    assert route_view.distance_meters == 1234.5
    assert route_view.travel_time_seconds == 88.0


# ---------------------------------------------------------------------------
# ResponsePlanOptimizedPlanReaderAdapter
# ---------------------------------------------------------------------------


def test_optimized_plan_reader_returns_correct_shape(persisted_context, sqlite_session_factory):
    stored_run = RoutePlanningRepository(sqlite_session_factory).save_run(
        make_route_planning_run(persisted_context)
    )
    stored_plan = ResponsePlanRepository(sqlite_session_factory).save(
        make_response_plan(
            persisted_context, stored_run.id, stored_run.routes[0].id, plan_score=91.5, coverage_score=100.0
        )
    )
    reader = ResponsePlanOptimizedPlanReaderAdapter(ResponsePlanRepository(sqlite_session_factory))

    view = reader.get_by_id(stored_plan.id)

    assert view.id == stored_plan.id
    assert view.fire_event_id == persisted_context["fire_event_id"]
    assert view.route_planning_run_id == stored_run.id
    assert view.response_target_set_id == persisted_context["response_target_set_id"]
    assert view.score.total_score == 91.5
    assert view.score.coverage_score == 100.0
    assert view.score.covered_target_count == 1
    assert view.score.total_target_count == 1
    assert view.score.uncovered_target_ids == ()


def test_optimized_plan_reader_computes_target_counts_from_actions_and_uncovered(
    persisted_context, sqlite_session_factory
):
    stored_run = RoutePlanningRepository(sqlite_session_factory).save_run(
        make_route_planning_run(persisted_context)
    )
    stored_plan = ResponsePlanRepository(sqlite_session_factory).save(
        make_response_plan(
            persisted_context,
            stored_run.id,
            stored_run.routes[0].id,
            status=ResponsePlanStatus.PARTIAL,
            uncovered_target_ids=persisted_context["other_target_ids"],
        )
    )
    reader = ResponsePlanOptimizedPlanReaderAdapter(ResponsePlanRepository(sqlite_session_factory))

    view = reader.get_by_id(stored_plan.id)

    assert view.score.covered_target_count == 1
    assert view.score.total_target_count == 3
    assert view.score.uncovered_target_ids == tuple(sorted(persisted_context["other_target_ids"]))


def test_optimized_plan_reader_returns_none_for_missing_plan(sqlite_session_factory):
    reader = ResponsePlanOptimizedPlanReaderAdapter(ResponsePlanRepository(sqlite_session_factory))

    assert reader.get_by_id(999999) is None
