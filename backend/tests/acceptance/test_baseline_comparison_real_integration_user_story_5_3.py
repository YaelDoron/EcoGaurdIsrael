"""Real US 5.1 -> US 5.2 -> US 5.3 end-to-end integration tests.

Unlike `test_baseline_comparison_user_story_5_3.py` (which proves Tasks 1-5's
own logic against fakes for the once-missing Company 1/2 boundaries), every
test here drives `BaselineComparisonService()` with its real, unmodified
default wiring: a real persisted `RoutePlanningRun`/`RouteResult` snapshot
(US 5.1), a real GA-optimized and persisted `ResponsePlan` (US 5.2, via
`ResponseOptimizationAgent.optimize_from_route_planning_run`), and the real
`ResponsePlanRepositoryOptimizedPlanReader` / `RoutePlanningRepositoryRunReader`
/ `ResponsePlanScorerBaselineAdapter` adapters -- no test doubles anywhere in
the production call path.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.calculators.baseline_plan.baseline_plan_calculator import BaselinePlanCalculator, RouteCandidate, TargetOrder
from src.calculators.response_optimization import ResponseOptimizationConfig, ResponsePlanScorer
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_spread_prediction_cell_db import FireSpreadPredictionCellDB
from src.database.models.fire_spread_prediction_db import FireSpreadPredictionDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.graph_node_db import GraphNodeDB
from src.models import ResponseTarget, ResponseTargetSet, ResponseTargetType
from src.models.resource_status import ResourceStatus
from src.models.response_action import ResponseAction
from src.models.routing import RoutePlanningRun, RouteResult, RouteStatus
from src.repositories.plan_comparison_repository import PlanComparisonRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.baseline_comparison import (
    BaselineComparisonService,
    ResponsePlanRepositoryOptimizedPlanReader,
    ResponsePlanScorerBaselineAdapter,
    RoutePlanningRepositoryRunReader,
)
from src.services.response_optimization import ResponseOptimizationInputService
from src.agents.analysis.response_optimization_agent import ResponseOptimizationAgent

AS_OF = datetime(2026, 9, 16, 17, 0, tzinfo=timezone.utc)
SOURCE_NODE_ID = 92001
TARGET_NODE_ID = 92002


@pytest.fixture
def stack(sqlite_session_factory):
    session = sqlite_session_factory()
    session.add_all(
        [
            GraphNodeDB(id=SOURCE_NODE_ID, latitude=32.7, longitude=35.0),
            GraphNodeDB(id=TARGET_NODE_ID, latitude=32.71, longitude=35.01),
        ]
    )
    session.add(FireStationDB(id="S-1", name="Station 1", latitude=32.7, longitude=35.0))
    session.flush()
    session.add_all(
        [
            FirefightingResourceDB(id="R1", station_id="S-1", status=ResourceStatus.AVAILABLE),
            FirefightingResourceDB(id="R2", station_id="S-1", status=ResourceStatus.AVAILABLE),
        ]
    )
    session.commit()
    session.close()

    targets = ResponseTargetRepository(sqlite_session_factory)
    routes = RoutePlanningRepository(sqlite_session_factory)
    plans = ResponsePlanRepository(sqlite_session_factory)
    comparisons = PlanComparisonRepository(sqlite_session_factory)
    input_service = ResponseOptimizationInputService(route_planning_repository=routes, response_target_repository=targets)
    agent = ResponseOptimizationAgent(plans, input_service=input_service)
    comparison_service = BaselineComparisonService(
        optimized_plan_reader=ResponsePlanRepositoryOptimizedPlanReader(plans),
        route_planning_run_reader=RoutePlanningRepositoryRunReader(routes),
        scorer=ResponsePlanScorerBaselineAdapter(input_service=input_service, scorer=ResponsePlanScorer()),
        response_target_set_reader=targets,
        plan_comparison_repository=comparisons,
    )
    return {
        "targets": targets,
        "routes": routes,
        "plans": plans,
        "comparisons": comparisons,
        "agent": agent,
        "comparison_service": comparison_service,
        "input_service": input_service,
        "session_factory": sqlite_session_factory,
    }


def persist_fire_event(sqlite_session_factory, *, offset: float = 0.0) -> dict:
    session = sqlite_session_factory()
    event = FireEventDB(
        latitude=32.7 + offset,
        longitude=35.0,
        detected_at=AS_OF - timedelta(minutes=30),
        updated_at=AS_OF - timedelta(minutes=5),
        status="confirmed",
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    spread_prediction = FireSpreadPredictionDB(
        fire_event_id=event.id,
        severity_assessment_id=None,
        predicted_at=AS_OF - timedelta(minutes=10),
        horizon_minutes=30,
        status="insufficient_data",
        methodology="TEST_SPREAD",
        methodology_version="1.0",
    )
    session.add(spread_prediction)
    session.flush()
    spread_cell = FireSpreadPredictionCellDB(
        prediction_id=spread_prediction.id,
        latitude=32.701 + offset,
        longitude=35.0,
        spread_probability=0.8,
        spread_risk_score=50.0,
        reached_step=1,
        reached_minutes=30,
    )
    session.add(spread_cell)
    session.commit()
    ids = {"event_id": event.id, "spread_prediction_id": spread_prediction.id, "spread_cell_id": spread_cell.id}
    session.close()
    return ids


def make_target_set(stack, fire_event, priorities=(100.0, 50.0)):
    """Build a real, persisted ResponseTargetSet: target 0 is the required
    ACTIVE_FIRE target, any further targets are PREDICTED_RISK."""
    return stack["targets"].save_target_set(
        ResponseTargetSet(
            fire_event_id=fire_event["event_id"],
            generated_at=AS_OF - timedelta(minutes=2),
            methodology="TARGETS",
            methodology_version="1.0",
            targets=tuple(
                ResponseTarget(
                    fire_event_id=fire_event["event_id"],
                    target_type=ResponseTargetType.ACTIVE_FIRE if index == 0 else ResponseTargetType.PREDICTED_RISK,
                    latitude=32.7 + index * 0.001,
                    longitude=35.0,
                    priority_score=priority,
                    prediction_horizon_minutes=30 if index != 0 else None,
                    spread_prediction_id=fire_event["spread_prediction_id"] if index != 0 else None,
                    spread_prediction_cell_id=fire_event["spread_cell_id"] if index != 0 else None,
                )
                for index, priority in enumerate(priorities)
            ),
        )
    )


def reachable(resource_id: str, target_id: int, eta: float) -> RouteResult:
    return RouteResult(
        resource_id=resource_id,
        response_target_id=target_id,
        status=RouteStatus.REACHABLE,
        source_node_id=SOURCE_NODE_ID,
        target_node_id=TARGET_NODE_ID,
        node_path=(SOURCE_NODE_ID, TARGET_NODE_ID),
        distance_meters=eta * 10.0,
        travel_time_seconds=eta,
    )


def unreachable(resource_id: str, target_id: int) -> RouteResult:
    return RouteResult(
        resource_id=resource_id,
        response_target_id=target_id,
        status=RouteStatus.UNREACHABLE,
        source_node_id=SOURCE_NODE_ID,
        target_node_id=TARGET_NODE_ID,
        node_path=(),
        distance_meters=None,
        travel_time_seconds=None,
    )


def unmappable(resource_id: str, target_id: int) -> RouteResult:
    return RouteResult(
        resource_id=resource_id,
        response_target_id=target_id,
        status=RouteStatus.UNMAPPABLE,
        source_node_id=None,
        target_node_id=None,
        node_path=(),
        distance_meters=None,
        travel_time_seconds=None,
    )


def save_run(stack, stored_target_set, routes, resource_ids=("R1", "R2"), fire_event_id=None):
    return stack["routes"].save_run(
        RoutePlanningRun(
            fire_event_id=fire_event_id if fire_event_id is not None else stored_target_set.target_set.fire_event_id,
            response_target_set_id=stored_target_set.id,
            planned_at=AS_OF - timedelta(minutes=1),
            methodology="ECOGUARD_ROUTING_DIJKSTRA",
            methodology_version="1.0",
            resource_ids=resource_ids,
            routes=tuple(routes),
        )
    )


def optimize(stack, stored_run, seed=5):
    result = stack["agent"].optimize_from_route_planning_run(
        stored_run.id,
        as_of=AS_OF,
        config=ResponseOptimizationConfig(population_size=6, generation_count=3, random_seed=seed),
    )
    return stack["plans"].get_by_id(result.response_plan_id)


def independent_real_baseline_score(stack, route_planning_run_id: int):
    """Recompute the baseline through the SAME real components, independently
    of `BaselineComparisonService`, to prove there is one shared methodology
    and one shared scorer -- not a duplicated formula."""
    optimization_input = stack["input_service"].build_from_route_planning_run(route_planning_run_id)
    candidates = tuple(
        RouteCandidate(
            route_result_id=option.route_result_id,
            resource_id=option.resource_id,
            response_target_id=option.response_target_id,
            status="reachable" if option.is_reachable else "unreachable",
            travel_time_seconds=option.travel_time_seconds,
            distance_meters=option.distance_meters,
        )
        for option in optimization_input.route_options
    )
    targets = tuple(
        TargetOrder(
            response_target_id=target.response_target_id,
            target_order=target.target_order,
            target_type=target.target_type,
            priority_score=target.priority_score,
        )
        for target in optimization_input.targets
    )
    allocation = BaselinePlanCalculator().allocate(targets=targets, route_candidates=candidates)
    actions = tuple(
        ResponseAction(
            resource_id=assignment.resource_id,
            response_target_id=assignment.response_target_id,
            route_result_id=assignment.route_result_id,
        )
        for assignment in allocation.assignments
    )
    return ResponsePlanScorer().evaluate(optimization_input, actions)


def test_real_chain_persists_plan_comparison_from_exact_persisted_snapshot(stack):
    fire_event = persist_fire_event(stack["session_factory"])
    stored_targets = make_target_set(stack, fire_event)
    t1, t2 = [stored_target.id for stored_target in stored_targets.targets]
    stored_run = save_run(stack, stored_targets, (reachable("R1", t1, 90.0), reachable("R2", t2, 150.0)))

    stored_plan = optimize(stack, stored_run)
    plan = stored_plan.plan

    comparison = stack["comparison_service"].compare(response_plan_id=stored_plan.id)

    assert comparison.fire_event_id == fire_event["event_id"]
    assert comparison.optimized_plan_id == stored_plan.id
    assert comparison.route_planning_run_id == stored_run.id
    assert comparison.response_target_set_id == stored_targets.id

    # Optimized metrics are copied verbatim from the persisted ResponsePlan,
    # never recomputed by US 5.3.
    assert comparison.optimized_score == plan.plan_score
    assert comparison.optimized_coverage_score == plan.coverage_score
    assert comparison.optimized_average_eta_seconds == plan.average_eta_seconds

    # Baseline metrics come from the real shared ResponsePlanScorer.
    expected = independent_real_baseline_score(stack, stored_run.id)
    assert comparison.baseline_score == expected.total_score
    assert comparison.baseline_coverage_score == expected.coverage_score
    assert comparison.baseline_average_eta_seconds == expected.average_eta_seconds

    assert comparison.score_difference == comparison.optimized_score - comparison.baseline_score

    stored_rows = stack["comparisons"].list_for_fire_event(fire_event["event_id"])
    assert len(stored_rows) == 1
    assert stored_rows[0].comparison == comparison

    real_route_result_ids = {stored_route.id for stored_route in stored_run.routes}
    assert real_route_result_ids  # sanity: real persisted route_result IDs exist and were used above


def test_route_status_filtering_only_reachable_routes_are_assignable(stack):
    fire_event = persist_fire_event(stack["session_factory"])
    stored_targets = make_target_set(stack, fire_event)
    t1, t2 = [stored_target.id for stored_target in stored_targets.targets]
    stored_run = save_run(
        stack,
        stored_targets,
        (
            reachable("R1", t1, 60.0),
            unreachable("R2", t1),
            unmappable("R2", t2),
        ),
    )
    stored_plan = optimize(stack, stored_run, seed=9)

    comparison = stack["comparison_service"].compare(response_plan_id=stored_plan.id)

    expected = independent_real_baseline_score(stack, stored_run.id)
    # Only target t1 can be covered (its sole reachable route); t2 has none.
    assert expected.covered_target_count == 1
    assert comparison.baseline_coverage_score == expected.coverage_score
    assert comparison.baseline_score == expected.total_score


def test_snapshot_isolation_compares_referenced_plan_not_a_newer_run(stack):
    fire_event_a = persist_fire_event(stack["session_factory"], offset=0.0)
    targets_a = make_target_set(stack, fire_event_a)
    ta1, ta2 = [t.id for t in targets_a.targets]
    run_a = save_run(stack, targets_a, (reachable("R1", ta1, 80.0), reachable("R2", ta2, 120.0)))
    plan_a = optimize(stack, run_a, seed=1)

    # Newer snapshot for a DIFFERENT fire event, created after plan A.
    fire_event_b = persist_fire_event(stack["session_factory"], offset=0.2)
    targets_b = make_target_set(stack, fire_event_b, priorities=(90.0,))
    tb1 = targets_b.targets[0].id
    run_b = save_run(stack, targets_b, (reachable("R1", tb1, 30.0),))
    plan_b = optimize(stack, run_b, seed=2)

    comparison = stack["comparison_service"].compare(response_plan_id=plan_a.id)

    assert comparison.fire_event_id == fire_event_a["event_id"]
    assert comparison.route_planning_run_id == run_a.id
    assert comparison.response_target_set_id == targets_a.id
    assert comparison.route_planning_run_id != run_b.id
    assert comparison.response_target_set_id != targets_b.id


def test_current_resource_availability_changes_do_not_affect_baseline(stack):
    fire_event = persist_fire_event(stack["session_factory"])
    stored_targets = make_target_set(stack, fire_event)
    t1, t2 = [t.id for t in stored_targets.targets]
    stored_run = save_run(stack, stored_targets, (reachable("R1", t1, 70.0), reachable("R2", t2, 110.0)))
    stored_plan = optimize(stack, stored_run, seed=3)

    before = stack["comparison_service"].compare(response_plan_id=stored_plan.id)

    # Flip current resource availability AFTER the historical snapshot was planned.
    session = stack["session_factory"]()
    session.query(FirefightingResourceDB).filter(FirefightingResourceDB.id == "R1").update(
        {"status": ResourceStatus.UNAVAILABLE}
    )
    session.query(FirefightingResourceDB).filter(FirefightingResourceDB.id == "R2").update(
        {"status": ResourceStatus.ASSIGNED}
    )
    session.commit()
    session.close()

    after = stack["comparison_service"].compare(response_plan_id=stored_plan.id)

    assert after.baseline_score == before.baseline_score
    assert after.baseline_coverage_score == before.baseline_coverage_score
    assert after.baseline_average_eta_seconds == before.baseline_average_eta_seconds

    statuses = {row.id: row.status for row in session_all_resources(stack)}
    assert statuses == {"R1": ResourceStatus.UNAVAILABLE, "R2": ResourceStatus.ASSIGNED}


def session_all_resources(stack):
    session = stack["session_factory"]()
    try:
        return session.query(FirefightingResourceDB).all()
    finally:
        session.close()


def test_append_only_history_two_real_comparisons_of_the_same_plan(stack):
    fire_event = persist_fire_event(stack["session_factory"])
    stored_targets = make_target_set(stack, fire_event, priorities=(100.0,))
    t1 = stored_targets.targets[0].id
    stored_run = save_run(stack, stored_targets, (reachable("R1", t1, 45.0),), resource_ids=("R1",))
    stored_plan = optimize(stack, stored_run, seed=4)

    first = stack["comparison_service"].compare(response_plan_id=stored_plan.id)
    second = stack["comparison_service"].compare(response_plan_id=stored_plan.id)

    rows = stack["comparisons"].list_for_fire_event(fire_event["event_id"])
    assert len(rows) == 2
    assert rows[0].id != rows[1].id
    assert rows[0].comparison == first
    assert rows[1].comparison == second
