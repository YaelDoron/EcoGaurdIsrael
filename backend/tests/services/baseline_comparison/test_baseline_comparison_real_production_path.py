"""Production-style test: real BaselineComparisonService + real Task 6/6.1 adapters.

No fake baseline scorer: the real ResponsePlanScorer computes the baseline
score, reached through the real ResponsePlanBaselineScorerAdapter, the real
RoutePlanningRunReaderAdapter, and the real ResponsePlanOptimizedPlanReaderAdapter -
everything BaselineComparisonService needs in production.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.calculators.response_optimization.response_plan_scorer import ResponsePlanScorer, eta_factor
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.graph_node_db import GraphNodeDB
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
from src.repositories.plan_comparison_repository import PlanComparisonRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.baseline_comparison.baseline_comparison_production_readers import (
    ResponsePlanOptimizedPlanReaderAdapter,
    RoutePlanningRunReaderAdapter,
)
from src.services.baseline_comparison.baseline_comparison_service import BaselineComparisonService
from src.services.baseline_comparison.response_plan_baseline_scorer_adapter import ResponsePlanBaselineScorerAdapter

GENERATED_AT = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def test_real_production_baseline_comparison_uses_shared_scorer(sqlite_session_factory):
    session = sqlite_session_factory()
    event = FireEventDB(
        latitude=32.731,
        longitude=35.046,
        detected_at=GENERATED_AT - timedelta(hours=1),
        updated_at=GENERATED_AT - timedelta(minutes=5),
        status="confirmed",
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    session.add(event)
    session.add_all(
        [GraphNodeDB(id=1, latitude=32.7, longitude=35.0), GraphNodeDB(id=2, latitude=32.71, longitude=35.01)]
    )
    # route_results.resource_id / response_actions.resource_id are now real
    # FKs (FND-05): "truck-1" needs a matching FirefightingResourceDB row.
    session.add(FireStationDB(id="FIXTURE-STATION", name="Fixture Station", latitude=32.7, longitude=35.0))
    session.flush()
    session.add(FirefightingResourceDB(id="truck-1", station_id="FIXTURE-STATION", status=ResourceStatus.AVAILABLE))
    session.commit()
    fire_event_id = event.id
    session.close()

    response_target_repository = ResponseTargetRepository(sqlite_session_factory)
    stored_target_set = response_target_repository.save_target_set(
        ResponseTargetSet(
            fire_event_id=fire_event_id,
            generated_at=GENERATED_AT,
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
    target_id = stored_target_set.targets[0].id

    route_planning_repository = RoutePlanningRepository(sqlite_session_factory)
    stored_run = route_planning_repository.save_run(
        RoutePlanningRun(
            fire_event_id=fire_event_id,
            response_target_set_id=stored_target_set.id,
            planned_at=GENERATED_AT,
            methodology="ECOGUARD_ROUTING_DIJKSTRA",
            methodology_version="1.0",
            resource_ids=("truck-1",),
            routes=(
                RouteResult(
                    resource_id="truck-1",
                    response_target_id=target_id,
                    status=RouteStatus.REACHABLE,
                    source_node_id=1,
                    target_node_id=2,
                    node_path=(1, 2),
                    distance_meters=800.0,
                    travel_time_seconds=60.0,
                ),
            ),
        )
    )
    route_result_id = stored_run.routes[0].id

    response_plan_repository = ResponsePlanRepository(sqlite_session_factory)
    stored_plan = response_plan_repository.save(
        ResponsePlan(
            fire_event_id=fire_event_id,
            response_target_set_id=stored_target_set.id,
            route_planning_run_id=stored_run.id,
            generated_at=GENERATED_AT,
            status=ResponsePlanStatus.COMPLETE,
            methodology="GENETIC_RESOURCE_ALLOCATION",
            methodology_version="1.0",
            random_seed=42,
            actions=(ResponseAction("truck-1", target_id, route_result_id),),
            uncovered_target_ids=(),
            plan_score=95.0,
            coverage_score=100.0,
            average_eta_seconds=60.0,
        )
    )

    plan_comparison_repository = PlanComparisonRepository(sqlite_session_factory)
    service = BaselineComparisonService(
        optimized_plan_reader=ResponsePlanOptimizedPlanReaderAdapter(response_plan_repository),
        route_planning_run_reader=RoutePlanningRunReaderAdapter(route_planning_repository),
        scorer=ResponsePlanBaselineScorerAdapter(ResponsePlanScorer()),
        response_target_set_reader=response_target_repository,
        plan_comparison_repository=plan_comparison_repository,
    )

    result = service.compare(response_plan_id=stored_plan.id)

    stored_rows = plan_comparison_repository.list_for_fire_event(fire_event_id)
    assert len(stored_rows) == 1
    assert stored_rows[0].comparison == result

    # Optimized side: existing persisted optimized score, unchanged.
    assert result.optimized_plan_id == stored_plan.id
    assert result.optimized_score == 95.0
    assert result.optimized_coverage_score == 100.0

    # Traceability preserved exactly.
    assert result.route_planning_run_id == stored_run.id
    assert result.response_target_set_id == stored_target_set.id

    # Baseline score comes from the REAL ResponsePlanScorer: one target
    # (priority 150), one reachable resource -> full coverage, score =
    # 100 * eta_factor(60s) since total_priority == covered_priority.
    expected_baseline_score = 100.0 * eta_factor(60.0)
    assert result.baseline_score == pytest.approx(expected_baseline_score)
    assert result.baseline_coverage_score == 100.0
    assert result.baseline_average_eta_seconds == 60.0
