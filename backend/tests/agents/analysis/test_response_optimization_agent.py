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
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.models import (
    OptimizationResource,
    OptimizationRouteOption,
    OptimizationTarget,
    ResponseOptimizationInput,
    ResponsePlanStatus,
    ResponseTargetType,
)
from src.models.resource_status import ResourceStatus
from src.repositories.response_plan_repository import ResponsePlanRepository

AS_OF = datetime(2026, 9, 16, 13, 0, tzinfo=timezone.utc)


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
    session.commit()
    context = {"fire_event_id": event.id, "response_target_set_id": target_set.id}
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


def optimization_input(context, *, partial: bool = False, unreachable: bool = False) -> ResponseOptimizationInput:
    routes = (
        route(100, "TRUCK-A", 10, 100.0, reachable=not unreachable),
        route(200, "TRUCK-B", 20, 200.0, reachable=not unreachable and not partial),
    )
    return ResponseOptimizationInput(
        fire_event_id=context["fire_event_id"],
        response_target_set_id=context["response_target_set_id"],
        route_planning_run_id=77,
        targets=(target(10, 0, 100.0), target(20, 1, 50.0)),
        resources=(resource("TRUCK-A"), resource("TRUCK-B")),
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
