"""Tests for ResponseOptimizationAgent orchestration."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.analysis.response_optimization_agent import ResponseOptimizationAgent
from src.agents.analysis.response_optimization_result import ResponseOptimizationStatus
from src.calculators.response_optimization import (
    GeneticResponsePlanOptimizer,
    ResponseOptimizationConfig,
    ResponsePlanScorer,
)
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
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

AS_OF = datetime(2026, 9, 16, 13, 0, tzinfo=timezone.utc)

# FND-05: response_plans.route_planning_run_id and response_actions.resource_id/
# response_target_id/route_result_id are now real FKs. optimization_input()
# persists a minimal real row for every id it references (idempotently).
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
        detected_at=AS_OF - timedelta(minutes=30),
        updated_at=AS_OF - timedelta(minutes=5),
        status="confirmed",
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    target_set = ResponseTargetSetDB(
        fire_event_id=event.id,
        generated_at=AS_OF - timedelta(minutes=1),
        methodology="TEST_TARGETS",
        methodology_version="1.0",
    )
    session.add(target_set)
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
    session.add(FireStationDB(id=FIXTURE_STATION_ID, name="Fixture Station", latitude=32.7, longitude=35.0))
    session.commit()
    context = {
        "fire_event_id": event.id,
        "response_target_set_id": target_set.id,
        "route_planning_run_id": ROUTE_PLANNING_RUN_ID,
        "session_factory": sqlite_session_factory,
    }
    session.close()
    return context


def target(target_id: int, order: int, priority: float = 100.0) -> OptimizationTarget:
    return OptimizationTarget(target_id, order, ResponseTargetType.ACTIVE_FIRE, priority)


def resource(resource_id: str) -> OptimizationResource:
    return OptimizationResource(resource_id)


def route(route_id: int, resource_id: str, target_id: int, eta: float | None, reachable: bool = True):
    return OptimizationRouteOption(
        route_result_id=route_id,
        resource_id=resource_id,
        response_target_id=target_id,
        is_reachable=reachable,
        travel_time_seconds=eta if reachable else None,
        distance_meters=1000.0 if reachable else None,
    )


def _persist_fk_prerequisites(context, targets, resources, routes) -> None:
    """Persist a real row for every target/resource/route-result id these
    inputs reference (FND-05 FK prerequisites). Idempotent."""
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


def optimization_input(context, *, partial: bool = False, unreachable: bool = False) -> ResponseOptimizationInput:
    targets = (target(10, 0, 100.0), target(20, 1, 50.0))
    resources = (resource("TRUCK-A"), resource("TRUCK-B"))
    routes = (
        route(100, "TRUCK-A", 10, 100.0, reachable=not unreachable),
        route(200, "TRUCK-B", 20, 200.0, reachable=not unreachable and not partial),
    )
    _persist_fk_prerequisites(context, targets, resources, routes)
    return ResponseOptimizationInput(
        fire_event_id=context["fire_event_id"],
        response_target_set_id=context["response_target_set_id"],
        route_planning_run_id=context["route_planning_run_id"],
        targets=targets,
        resources=resources,
        route_options=routes,
    )


def test_successful_complete_plan_persists_optimizer_result(repository, persisted_context):
    input_data = optimization_input(persisted_context)
    config = ResponseOptimizationConfig(population_size=4, generation_count=2, random_seed=12)
    agent = ResponseOptimizationAgent(repository)

    result = agent.optimize_from_input(input_data, as_of=AS_OF, config=config)
    stored = repository.get_by_id(result.response_plan_id)
    expected = GeneticResponsePlanOptimizer().optimize(input_data, config)

    assert result.success is True
    assert result.status is ResponseOptimizationStatus.OPTIMIZED
    assert result.plan_status is ResponsePlanStatus.COMPLETE
    assert stored.plan.generated_at == AS_OF
    assert stored.plan.random_seed == 12
    assert stored.plan.actions == expected.actions
    assert stored.plan.plan_score == pytest.approx(expected.score.total_score)
    assert stored.plan.coverage_score == pytest.approx(expected.score.coverage_score)
    assert stored.plan.average_eta_seconds == pytest.approx(expected.score.average_eta_seconds)


def test_partial_plan_persists_partial_status(repository, persisted_context):
    input_data = optimization_input(persisted_context, partial=True)

    result = ResponseOptimizationAgent(repository).optimize_from_input(
        input_data,
        as_of=AS_OF,
        config=ResponseOptimizationConfig(population_size=4, generation_count=1),
    )

    stored = repository.get_by_id(result.response_plan_id)
    assert result.success is True
    assert stored.plan.status is ResponsePlanStatus.PARTIAL
    assert stored.plan.uncovered_target_ids == (20,)


def test_no_feasible_assignments_is_successful_plan(repository, persisted_context):
    input_data = optimization_input(persisted_context, unreachable=True)

    result = ResponseOptimizationAgent(repository).optimize_from_input(
        input_data,
        as_of=AS_OF,
        config=ResponseOptimizationConfig(population_size=4, generation_count=1),
    )

    stored = repository.get_by_id(result.response_plan_id)
    assert result.success is True
    assert stored.plan.status is ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS
    assert stored.plan.actions == ()
    assert stored.plan.uncovered_target_ids == (10, 20)
    assert stored.plan.plan_score == pytest.approx(0.0)
    assert stored.plan.coverage_score == pytest.approx(0.0)
    assert stored.plan.average_eta_seconds is None


def test_naive_as_of_rejected(repository, persisted_context):
    with pytest.raises(ValueError):
        ResponseOptimizationAgent(repository).optimize_from_input(
            optimization_input(persisted_context),
            as_of=datetime(2026, 9, 16, 13, 0),
        )


def test_default_config_uses_documented_seed(repository, persisted_context):
    result = ResponseOptimizationAgent(repository).optimize_from_input(
        optimization_input(persisted_context),
        as_of=AS_OF,
    )

    assert repository.get_by_id(result.response_plan_id).plan.random_seed == ResponseOptimizationConfig().random_seed


def test_repository_failure_returns_failed_result(persisted_context):
    class FailingRepository:
        def save(self, plan):  # noqa: ANN001
            raise RuntimeError("database unavailable")

    result = ResponseOptimizationAgent(FailingRepository()).optimize_from_input(
        optimization_input(persisted_context),
        as_of=AS_OF,
    )

    assert result.success is False
    assert result.status is ResponseOptimizationStatus.FAILED
    assert result.response_plan_id is None
    assert result.error_message == "Response optimization failed."


def test_firefighting_resource_status_is_not_changed(repository, persisted_context, sqlite_session_factory):
    session = sqlite_session_factory()
    session.add(FireStationDB(id="S-1", name="Station 1", latitude=32.7, longitude=35.0))
    session.flush()
    session.add_all(
        [
            FirefightingResourceDB(id="TRUCK-A", station_id="S-1", status=ResourceStatus.AVAILABLE),
            FirefightingResourceDB(id="TRUCK-B", station_id="S-1", status=ResourceStatus.AVAILABLE),
        ]
    )
    session.commit()
    session.close()

    ResponseOptimizationAgent(repository).optimize_from_input(
        optimization_input(persisted_context),
        as_of=AS_OF,
        config=ResponseOptimizationConfig(population_size=4, generation_count=1),
    )

    session = sqlite_session_factory()
    statuses = {row.id: row.status for row in session.query(FirefightingResourceDB).all()}
    session.close()
    assert statuses == {"TRUCK-A": ResourceStatus.AVAILABLE, "TRUCK-B": ResourceStatus.AVAILABLE}


def test_running_same_optimization_twice_is_append_only(repository, persisted_context):
    agent = ResponseOptimizationAgent(repository)
    input_data = optimization_input(persisted_context)

    first = agent.optimize_from_input(input_data, as_of=AS_OF)
    second = agent.optimize_from_input(input_data, as_of=AS_OF)

    assert first.response_plan_id != second.response_plan_id
    history = repository.get_for_fire_event(persisted_context["fire_event_id"])
    assert {item.id for item in history} == {first.response_plan_id, second.response_plan_id}


# ---------------------------------------------------------------------------
# FND-04: exact GA configuration persistence and reproducibility
# ---------------------------------------------------------------------------


def test_non_default_config_is_persisted_exactly_not_defaults(repository, persisted_context):
    """Regression for FND-04's exact failure mode: if optimize_from_input ever
    silently reconstructed ResponseOptimizationConfig() during persistence
    instead of using the caller's exact object, every field below would come
    back at its default instead of these deliberately non-default values."""
    input_data = optimization_input(persisted_context)
    config = ResponseOptimizationConfig(
        population_size=18,
        generation_count=9,
        mutation_rate=0.37,
        crossover_rate=0.42,
        random_seed=777,
        eta_reference_seconds=555.0,
        initial_assignment_probability=0.33,
        tournament_size=4,
        elitism_count=2,
    )
    assert config != ResponseOptimizationConfig()  # sanity: genuinely non-default

    result = ResponseOptimizationAgent(repository).optimize_from_input(input_data, as_of=AS_OF, config=config)

    stored_config = repository.get_by_id(result.response_plan_id).plan.optimization_config
    assert stored_config == config
    assert stored_config.population_size == 18
    assert stored_config.generation_count == 9
    assert stored_config.mutation_rate == pytest.approx(0.37)
    assert stored_config.crossover_rate == pytest.approx(0.42)
    assert stored_config.random_seed == 777
    assert stored_config.eta_reference_seconds == pytest.approx(555.0)
    assert stored_config.initial_assignment_probability == pytest.approx(0.33)
    assert stored_config.tournament_size == 4
    assert stored_config.elitism_count == 2


def test_default_config_is_also_persisted_in_full(repository, persisted_context):
    """A caller relying on defaults still gets a complete, non-None config -
    optimization_config must not stay None just because the caller didn't
    override anything."""
    result = ResponseOptimizationAgent(repository).optimize_from_input(
        optimization_input(persisted_context), as_of=AS_OF
    )

    stored_config = repository.get_by_id(result.response_plan_id).plan.optimization_config
    assert stored_config == ResponseOptimizationConfig()


def test_reload_and_reconstructed_config_reproduces_identical_optimization_output(repository, persisted_context):
    """The actual FND-04 requirement: persisted data alone - not today's
    source-code defaults - must be enough to reproduce a historical run.

    Proven by actually running the optimizer a second time with only the
    reconstructed config and the same input snapshot, not by comparing
    config objects alone.
    """
    input_data = optimization_input(persisted_context)
    original_config = ResponseOptimizationConfig(
        population_size=16,
        generation_count=6,
        mutation_rate=0.22,
        crossover_rate=0.55,
        random_seed=999,
        eta_reference_seconds=650.0,
        initial_assignment_probability=0.4,
        tournament_size=3,
        elitism_count=1,
    )

    result = ResponseOptimizationAgent(repository).optimize_from_input(
        input_data, as_of=AS_OF, config=original_config
    )
    stored = repository.get_by_id(result.response_plan_id)

    reconstructed_config = stored.plan.optimization_config
    assert reconstructed_config is not None
    assert reconstructed_config == original_config

    reproduced = GeneticResponsePlanOptimizer().optimize(input_data, reconstructed_config)

    assert reproduced.actions == stored.plan.actions
    assert reproduced.score.total_score == pytest.approx(stored.plan.plan_score)
    assert reproduced.score.coverage_score == pytest.approx(stored.plan.coverage_score)
    assert reproduced.score.average_eta_seconds == pytest.approx(stored.plan.average_eta_seconds)


def test_persisted_scores_match_scorer(repository, persisted_context):
    input_data = optimization_input(persisted_context)
    result = ResponseOptimizationAgent(repository).optimize_from_input(
        input_data,
        as_of=AS_OF,
        config=ResponseOptimizationConfig(population_size=4, generation_count=1),
    )
    stored = repository.get_by_id(result.response_plan_id)
    expected = ResponsePlanScorer().evaluate(input_data, stored.plan.actions)

    assert stored.plan.plan_score == pytest.approx(expected.total_score)
    assert stored.plan.coverage_score == pytest.approx(expected.coverage_score)
    assert stored.plan.average_eta_seconds == pytest.approx(expected.average_eta_seconds)
