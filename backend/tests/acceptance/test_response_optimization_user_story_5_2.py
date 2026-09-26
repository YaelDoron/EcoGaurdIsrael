"""Acceptance tests for US 5.2 response optimization over prepared input."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import ast
from pathlib import Path

import pytest

from src.agents.analysis.response_optimization_agent import ResponseOptimizationAgent
from src.agents.analysis.response_optimization_result import ResponseOptimizationStatus
from src.calculators.response_optimization import (
    METHODOLOGY,
    METHODOLOGY_VERSION,
    ResponseOptimizationConfig,
    ResponsePlanScorer,
)
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.database.models.route_result_db import RouteResultDB
from src.models import (
    GraphNode,
    OptimizationResource,
    OptimizationRouteOption,
    OptimizationTarget,
    ResponseOptimizationInput,
    ResponsePlanStatus,
    ResponseTargetType,
)
from src.models.resource_status import ResourceStatus
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.road_network_repository import RoadNetworkRepository

AS_OF = datetime(2026, 9, 16, 15, 0, tzinfo=timezone.utc)

# FND-05 gave response_plans.route_planning_run_id, response_actions.resource_id/
# response_target_id/route_result_id real FK constraints. These tests build
# ResponseOptimizationInput values with hand-picked ids for a pure calculator
# input, so `optimization_input()` also persists a minimal real row for every
# id it references (idempotently - the same id may recur across calls within
# one test) before ResponsePlanRepository ever needs to satisfy those FKs.
ROUTE_PLANNING_RUN_ID = 77
FIXTURE_STATION_ID = "FIXTURE-STATION"
FIXTURE_SOURCE_NODE_ID = 9001
FIXTURE_TARGET_NODE_ID = 9002


@pytest.fixture
def repository(sqlite_session_factory) -> ResponsePlanRepository:
    return ResponsePlanRepository(sqlite_session_factory)


@pytest.fixture
def persisted_context(sqlite_session_factory):
    session = sqlite_session_factory()
    event = FireEventDB(
        latitude=32.731,
        longitude=35.046,
        detected_at=AS_OF - timedelta(minutes=45),
        updated_at=AS_OF - timedelta(minutes=2),
        status="confirmed",
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    target_set = session.merge(
        __import__(
            "src.database.models.response_target_set_db",
            fromlist=["ResponseTargetSetDB"],
        ).ResponseTargetSetDB(
            fire_event_id=event.id,
            generated_at=AS_OF - timedelta(minutes=1),
            methodology="TEST_TARGETS",
            methodology_version="1.0",
        )
    )
    session.flush()
    RoadNetworkRepository().save_network(
        session,
        nodes=[
            GraphNode(id=FIXTURE_SOURCE_NODE_ID, latitude=32.700, longitude=35.000),
            GraphNode(id=FIXTURE_TARGET_NODE_ID, latitude=32.700, longitude=35.010),
        ],
        edges=[],
    )
    session.add(
        RoutePlanningRunDB(
            id=ROUTE_PLANNING_RUN_ID,
            fire_event_id=event.id,
            response_target_set_id=target_set.id,
            planned_at=AS_OF - timedelta(minutes=1),
            methodology="TEST_ROUTING",
            methodology_version="1.0",
            resource_ids=[],
        )
    )
    session.add(
        FireStationDB(id=FIXTURE_STATION_ID, name="Fixture Station", latitude=32.7, longitude=35.0)
    )
    session.commit()
    context = {
        "fire_event_id": event.id,
        "response_target_set_id": target_set.id,
        "route_planning_run_id": ROUTE_PLANNING_RUN_ID,
        "session_factory": sqlite_session_factory,
    }
    session.close()
    return context


def target(target_id: int, order: int, priority: float) -> OptimizationTarget:
    return OptimizationTarget(
        response_target_id=target_id,
        target_order=order,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        priority_score=priority,
    )


def resource(resource_id: str) -> OptimizationResource:
    return OptimizationResource(resource_id)


def route(
    route_id: int,
    resource_id: str,
    target_id: int,
    eta: float | None,
    *,
    reachable: bool = True,
) -> OptimizationRouteOption:
    return OptimizationRouteOption(
        route_result_id=route_id,
        resource_id=resource_id,
        response_target_id=target_id,
        is_reachable=reachable,
        travel_time_seconds=eta if reachable else None,
        distance_meters=1000.0 if reachable else None,
    )


def _persist_fk_prerequisites(
    context,
    targets: tuple[OptimizationTarget, ...],
    resources: tuple[OptimizationResource, ...],
    routes: tuple[OptimizationRouteOption, ...],
) -> None:
    """Persist a real row for every target/resource/route-result id these
    inputs reference, since ResponsePlanDB/ResponseActionDB's FKs (FND-05)
    now require the referenced rows to exist. Idempotent: an id may recur
    across multiple `optimization_input()` calls within one test."""
    session = context["session_factory"]()
    try:
        for target_input in targets:
            if session.get(ResponseTargetDB, target_input.response_target_id) is None:
                session.add(
                    ResponseTargetDB(
                        id=target_input.response_target_id,
                        response_target_set_id=context["response_target_set_id"],
                        fire_event_id=context["fire_event_id"],
                        target_order=target_input.target_order,
                        target_type=target_input.target_type.value,
                        latitude=32.731,
                        longitude=35.046,
                        priority_score=target_input.priority_score,
                    )
                )
        for resource_input in resources:
            if session.get(FirefightingResourceDB, resource_input.resource_id) is None:
                session.add(
                    FirefightingResourceDB(
                        id=resource_input.resource_id,
                        station_id=FIXTURE_STATION_ID,
                        status=ResourceStatus.AVAILABLE,
                    )
                )
        session.flush()
        for route_input in routes:
            if session.get(RouteResultDB, route_input.route_result_id) is None:
                session.add(
                    RouteResultDB(
                        id=route_input.route_result_id,
                        route_planning_run_id=context["route_planning_run_id"],
                        resource_id=route_input.resource_id,
                        response_target_id=route_input.response_target_id,
                        status="reachable" if route_input.is_reachable else "unreachable",
                        source_node_id=FIXTURE_SOURCE_NODE_ID,
                        target_node_id=FIXTURE_TARGET_NODE_ID,
                        node_path=[FIXTURE_SOURCE_NODE_ID, FIXTURE_TARGET_NODE_ID]
                        if route_input.is_reachable
                        else [],
                        distance_meters=route_input.distance_meters if route_input.is_reachable else None,
                        travel_time_seconds=route_input.travel_time_seconds if route_input.is_reachable else None,
                    )
                )
        session.commit()
    finally:
        session.close()


def optimization_input(
    context,
    *,
    targets: tuple[OptimizationTarget, ...],
    resources: tuple[OptimizationResource, ...],
    routes: tuple[OptimizationRouteOption, ...],
) -> ResponseOptimizationInput:
    _persist_fk_prerequisites(context, targets, resources, routes)
    return ResponseOptimizationInput(
        fire_event_id=context["fire_event_id"],
        response_target_set_id=context["response_target_set_id"],
        route_planning_run_id=context["route_planning_run_id"],
        targets=targets,
        resources=resources,
        route_options=routes,
    )


def run_plan(
    repository: ResponsePlanRepository,
    input_data: ResponseOptimizationInput,
    config: ResponseOptimizationConfig | None = None,
):
    result = ResponseOptimizationAgent(repository).optimize_from_input(
        input_data,
        as_of=AS_OF,
        config=config
        or ResponseOptimizationConfig(
            population_size=8,
            generation_count=4,
            random_seed=42,
            mutation_rate=0.0,
        ),
    )
    assert result.success is True
    assert result.status is ResponseOptimizationStatus.OPTIMIZED
    return repository.get_by_id(result.response_plan_id).plan


def logical_plan(plan):
    return (
        plan.status,
        plan.actions,
        plan.uncovered_target_ids,
        plan.plan_score,
        plan.coverage_score,
        plan.average_eta_seconds,
        plan.methodology,
        plan.methodology_version,
        plan.random_seed,
    )


def test_high_priority_target_is_preferred_with_limited_resource(repository, persisted_context):
    input_data = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0), target(20, 1, 20.0)),
        resources=(resource("TRUCK-A"),),
        routes=(route(100, "TRUCK-A", 10, 300.0), route(200, "TRUCK-A", 20, 300.0)),
    )

    plan = run_plan(repository, input_data)

    assert [(action.resource_id, action.response_target_id, action.route_result_id) for action in plan.actions] == [
        ("TRUCK-A", 10, 100)
    ]
    assert plan.uncovered_target_ids == (20,)


def test_eta_utility_changes_preferred_assignment(repository, persisted_context):
    input_data = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0), target(20, 1, 100.0)),
        resources=(resource("R1"), resource("R2")),
        routes=(
            route(100, "R1", 10, 100.0),
            route(101, "R2", 10, 1000.0),
            route(200, "R1", 20, 1000.0),
            route(201, "R2", 20, 100.0),
        ),
    )

    plan = run_plan(repository, input_data)
    score = ResponsePlanScorer().evaluate(input_data, plan.actions)

    assert {(action.resource_id, action.response_target_id) for action in plan.actions} == {("R1", 10), ("R2", 20)}
    assert plan.plan_score == pytest.approx(score.total_score)


def test_resource_and_target_uniqueness_invariants(repository, persisted_context):
    one_resource = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0), target(20, 1, 80.0), target(30, 2, 60.0)),
        resources=(resource("R1"),),
        routes=(route(100, "R1", 10, 100.0), route(200, "R1", 20, 100.0), route(300, "R1", 30, 100.0)),
    )
    one_target = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0),),
        resources=(resource("R1"), resource("R2"), resource("R3")),
        routes=(route(100, "R1", 10, 100.0), route(101, "R2", 10, 100.0), route(102, "R3", 10, 100.0)),
    )

    resource_limited = run_plan(repository, one_resource)
    target_limited = run_plan(repository, one_target)

    assert len({action.resource_id for action in resource_limited.actions}) == len(resource_limited.actions)
    assert len(resource_limited.actions) <= 1
    assert len({action.response_target_id for action in target_limited.actions}) == len(target_limited.actions)
    assert len(target_limited.actions) <= 1


def test_more_targets_than_resources_produces_partial_plan(repository, persisted_context):
    input_data = optimization_input(
        persisted_context,
        targets=tuple(target(10 + index * 10, index, 100.0 - index) for index in range(4)),
        resources=(resource("R1"), resource("R2")),
        routes=(
            route(100, "R1", 10, 100.0),
            route(200, "R2", 20, 100.0),
            route(300, "R1", 30, 100.0),
            route(400, "R2", 40, 100.0),
        ),
    )

    plan = run_plan(repository, input_data)

    assert plan.status is ResponsePlanStatus.PARTIAL
    assert len(plan.actions) <= 2
    assert len({action.resource_id for action in plan.actions}) == len(plan.actions)
    assert set(plan.uncovered_target_ids) == {target.response_target_id for target in input_data.targets} - {
        action.response_target_id for action in plan.actions
    }


def test_more_resources_than_targets_has_no_artificial_assignments_and_resource_status_unchanged(
    repository,
    persisted_context,
    sqlite_session_factory,
):
    session = sqlite_session_factory()
    session.add(FireStationDB(id="S-1", name="Station 1", latitude=32.7, longitude=35.0))
    session.flush()
    for resource_id in ("R1", "R2", "R3", "R4", "R5"):
        session.add(FirefightingResourceDB(id=resource_id, station_id="S-1", status=ResourceStatus.AVAILABLE))
    session.commit()
    session.close()

    input_data = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0), target(20, 1, 90.0)),
        resources=tuple(resource(f"R{index}") for index in range(1, 6)),
        routes=tuple(route(100 + index, f"R{index}", 10 if index % 2 else 20, 100.0) for index in range(1, 6)),
    )

    plan = run_plan(repository, input_data)

    assert len(plan.actions) <= 2
    assert len({action.response_target_id for action in plan.actions}) == len(plan.actions)
    session = sqlite_session_factory()
    statuses = {row.id: row.status for row in session.query(FirefightingResourceDB).all()}
    session.close()
    assert set(statuses.values()) == {ResourceStatus.AVAILABLE}


def test_unreachable_target_is_uncovered_but_reachable_target_persists(repository, persisted_context):
    input_data = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0), target(20, 1, 90.0)),
        resources=(resource("R1"), resource("R2")),
        routes=(route(100, "R1", 10, 100.0), route(200, "R2", 20, None, reachable=False)),
    )

    plan = run_plan(repository, input_data)

    assert plan.status is ResponsePlanStatus.PARTIAL
    assert [action.response_target_id for action in plan.actions] == [10]
    assert plan.uncovered_target_ids == (20,)


def test_no_feasible_assignments_is_successful_persisted_result(repository, persisted_context):
    input_data = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0), target(20, 1, 50.0)),
        resources=(resource("R1"), resource("R2")),
        routes=(route(100, "R1", 10, None, reachable=False), route(200, "R2", 20, None, reachable=False)),
    )

    plan = run_plan(repository, input_data)

    assert plan.status is ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS
    assert plan.actions == ()
    assert plan.uncovered_target_ids == (10, 20)
    assert plan.plan_score == pytest.approx(0.0)
    assert plan.coverage_score == pytest.approx(0.0)
    assert plan.average_eta_seconds is None


def test_complete_and_partial_plan_business_outcomes(repository, persisted_context):
    complete_input = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0), target(20, 1, 50.0)),
        resources=(resource("R1"), resource("R2")),
        routes=(route(100, "R1", 10, 100.0), route(200, "R2", 20, 200.0)),
    )
    partial_input = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0), target(20, 1, 50.0)),
        resources=(resource("R1"),),
        routes=(route(100, "R1", 10, 100.0),),
    )

    complete = run_plan(repository, complete_input)
    partial = run_plan(repository, partial_input)

    assert complete.status is ResponsePlanStatus.COMPLETE
    assert complete.uncovered_target_ids == ()
    assert {action.route_result_id for action in complete.actions} == {100, 200}
    assert complete.coverage_score == pytest.approx(100.0)
    assert partial.status is ResponsePlanStatus.PARTIAL
    assert partial.actions
    assert partial.uncovered_target_ids == (20,)
    assert partial.coverage_score < 100.0


def test_deterministic_logical_result_and_different_input_order(repository, persisted_context):
    canonical = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0), target(20, 1, 50.0)),
        resources=(resource("R1"), resource("R2")),
        routes=(route(100, "R1", 10, 100.0), route(200, "R2", 20, 200.0)),
    )
    shuffled = optimization_input(
        persisted_context,
        targets=(target(20, 1, 50.0), target(10, 0, 100.0)),
        resources=(resource("R2"), resource("R1")),
        routes=(route(200, "R2", 20, 200.0), route(100, "R1", 10, 100.0)),
    )
    config = ResponseOptimizationConfig(population_size=6, generation_count=2, random_seed=9)

    first = run_plan(repository, canonical, config)
    second = run_plan(repository, canonical, config)
    third = run_plan(repository, shuffled, config)

    assert first.generated_at == second.generated_at == third.generated_at == AS_OF
    assert logical_plan(first) == logical_plan(second) == logical_plan(third)


def test_append_only_history_and_exact_traceability(repository, persisted_context):
    input_data = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0),),
        resources=(resource("R1"),),
        routes=(route(555, "R1", 10, 100.0),),
    )

    first = run_plan(repository, input_data)
    second = run_plan(repository, input_data)
    history = repository.get_for_fire_event(persisted_context["fire_event_id"])

    assert len(history) == 2
    assert history[0].id != history[1].id
    assert repository.get_by_id(history[1].id).plan == first
    assert repository.get_by_id(history[0].id).plan == second
    assert first.fire_event_id == persisted_context["fire_event_id"]
    assert first.response_target_set_id == persisted_context["response_target_set_id"]
    assert first.route_planning_run_id == 77
    assert first.methodology == METHODOLOGY
    assert first.methodology_version == METHODOLOGY_VERSION
    assert first.random_seed == ResponseOptimizationConfig().random_seed
    assert first.actions[0].resource_id == "R1"
    assert first.actions[0].response_target_id == 10
    assert first.actions[0].route_result_id == 555


def test_explicit_time_and_naive_datetime_behavior(repository, persisted_context):
    input_data = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0),),
        resources=(resource("R1"),),
        routes=(route(100, "R1", 10, 100.0),),
    )

    plan = run_plan(repository, input_data)

    assert plan.generated_at == AS_OF
    with pytest.raises(ValueError):
        ResponseOptimizationAgent(repository).optimize_from_input(input_data, as_of=datetime(2026, 9, 16, 15, 0))


def test_score_traceability_uses_existing_scorer(repository, persisted_context):
    input_data = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0), target(20, 1, 50.0)),
        resources=(resource("R1"), resource("R2")),
        routes=(route(100, "R1", 10, 300.0), route(200, "R2", 20, 900.0)),
    )

    plan = run_plan(repository, input_data)
    score = ResponsePlanScorer().evaluate(input_data, plan.actions)

    assert plan.plan_score == pytest.approx(score.total_score)
    assert plan.coverage_score == pytest.approx(score.coverage_score)
    assert plan.average_eta_seconds == pytest.approx(score.average_eta_seconds)
    assert plan.status is score.status


def test_agent_technical_failure_is_not_no_feasible_assignments(persisted_context):
    class FailingRepository:
        def save(self, plan):  # noqa: ANN001
            raise RuntimeError("database down")

    input_data = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0),),
        resources=(resource("R1"),),
        routes=(route(100, "R1", 10, 100.0),),
    )

    result = ResponseOptimizationAgent(FailingRepository()).optimize_from_input(input_data, as_of=AS_OF)

    assert result.success is False
    assert result.response_plan_id is None
    assert result.plan_status is None
    assert result.error_message == "Response optimization failed."


@pytest.mark.parametrize(
    "config",
    [
        ResponseOptimizationConfig(population_size=2, tournament_size=2, generation_count=1),
        ResponseOptimizationConfig(population_size=4, generation_count=1),
        ResponseOptimizationConfig(population_size=4, crossover_rate=0.0),
        ResponseOptimizationConfig(population_size=4, mutation_rate=0.0),
        ResponseOptimizationConfig(population_size=4, elitism_count=0),
    ],
)
def test_ga_configuration_edge_cases_still_produce_valid_plan(repository, persisted_context, config):
    input_data = optimization_input(
        persisted_context,
        targets=(target(10, 0, 100.0),),
        resources=(resource("R1"),),
        routes=(route(100, "R1", 10, 100.0),),
    )

    plan = run_plan(repository, input_data, config)

    assert plan.status in set(ResponsePlanStatus)
    assert len(plan.actions) <= 1


def test_us_5_2_architecture_guards():
    pure_paths = list((Path(__file__).resolve().parents[3] / "backend/src/calculators/response_optimization").glob("*.py")) + [
        (Path(__file__).resolve().parents[3] / "backend/src/models/response_plan_chromosome.py"),
    ]
    pure_forbidden_modules = (
        "sqlalchemy",
        "src.repositories",
        "src.database",
        "src.agents",
        "src.simulation",
        "src.external",
        "road_network",
        "dijkstra",
    )
    agent_forbidden_names = (
        "FireDetectionAgent",
        "FireSeverityAssessmentAgent",
        "FireSpreadPredictionAgent",
        "ResponseTargetGenerationAgent",
        "OperationalRefreshOrchestrator",
        "Dijkstra",
        "RoadNetworkFetcher",
    )

    violations = []
    for path in pure_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            if any(fragment in module for fragment in pure_forbidden_modules):
                violations.append((str(path), module))

    agent_tree = ast.parse((Path(__file__).resolve().parents[3] / "backend/src/agents/analysis/response_optimization_agent.py").read_text(encoding="utf-8"))
    for node in ast.walk(agent_tree):
        if isinstance(node, ast.Name) and node.id in agent_forbidden_names:
            violations.append(("response_optimization_agent.py", node.id))
        module = ""
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
        elif isinstance(node, ast.Import):
            module = ",".join(alias.name for alias in node.names)
        if any(name in module for name in agent_forbidden_names):
            violations.append(("response_optimization_agent.py", module))

    status_mutation_paths = [
        (Path(__file__).resolve().parents[3] / "backend/src/agents/analysis/response_optimization_agent.py"),
        (Path(__file__).resolve().parents[3] / "backend/src/repositories/response_plan_repository.py"),
        *(Path(__file__).resolve().parents[3] / "backend/src/calculators/response_optimization").glob("*.py"),
    ]
    for path in status_mutation_paths:
        text = path.read_text(encoding="utf-8")
        if "ResourceStatus.ASSIGNED" in text or ".status = ResourceStatus" in text:
            violations.append((str(path), "resource status mutation"))

    assert violations == []
