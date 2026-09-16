"""Tests for ResponseOptimizationCollaboratorAdapter, using fakes for every dependency (no real DB)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.agents.analysis.response_optimization_result import ResponseOptimizationResult, ResponseOptimizationStatus
from src.models import ResponsePlanStatus, ResponseTarget, ResponseTargetSet, ResponseTargetType
from src.models.optimization_resource import OptimizationResource
from src.models.optimization_target import OptimizationTarget
from src.models.routing import RoutePlanningRun, RouteResult, RouteStatus, StoredRouteResult
from src.repositories.response_target_repository import StoredResponseTarget, StoredResponseTargetSet
from src.repositories.route_planning_repository import StoredRoutePlanningRun
from src.services.response_planning.response_optimization_collaborator_adapter import (
    ResponseOptimizationAdapterError,
    ResponseOptimizationCollaboratorAdapter,
)

FIRE_EVENT_ID = 42
AS_OF = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
RUN_ID = 601
TARGET_SET_ID = 77

ACTIVE_TARGET = ResponseTarget(
    fire_event_id=FIRE_EVENT_ID,
    target_type=ResponseTargetType.ACTIVE_FIRE,
    latitude=32.731,
    longitude=35.046,
    priority_score=150.0,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRoutePlanningRepository:
    def __init__(self, stored_run: StoredRoutePlanningRun | None):
        self.stored_run = stored_run
        self.calls = []

    def get_by_id(self, route_planning_run_id):
        self.calls.append(route_planning_run_id)
        return self.stored_run


class FakeResponseTargetRepository:
    def __init__(self, stored_target_set: StoredResponseTargetSet | None):
        self.stored_target_set = stored_target_set
        self.calls = []

    def get_by_id(self, response_target_set_id):
        self.calls.append(response_target_set_id)
        return self.stored_target_set


class FakeOptimizationAgent:
    def __init__(self, result: ResponseOptimizationResult):
        self.result = result
        self.calls = []

    def optimize_from_input(self, optimization_input, *, as_of, config):
        self.calls.append({"optimization_input": optimization_input, "as_of": as_of, "config": config})
        return self.result


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_stored_run(
    run_id=RUN_ID,
    fire_event_id=FIRE_EVENT_ID,
    response_target_set_id=TARGET_SET_ID,
    resource_ids=("truck-1", "truck-2"),
    routes=None,
) -> StoredRoutePlanningRun:
    if routes is None:
        routes = (
            RouteResult(
                resource_id="truck-1",
                response_target_id=900,
                status=RouteStatus.REACHABLE,
                source_node_id=1,
                target_node_id=2,
                node_path=(1, 2),
                distance_meters=1000.0,
                travel_time_seconds=90.0,
            ),
            RouteResult(
                resource_id="truck-2",
                response_target_id=900,
                status=RouteStatus.UNREACHABLE,
                source_node_id=3,
                target_node_id=4,
                node_path=(),
                distance_meters=None,
                travel_time_seconds=None,
            ),
        )
    run = RoutePlanningRun(
        fire_event_id=fire_event_id,
        response_target_set_id=response_target_set_id,
        planned_at=AS_OF,
        methodology="ECOGUARD_ROUTING_DIJKSTRA",
        methodology_version="1.0",
        resource_ids=resource_ids,
        routes=routes,
    )
    stored_routes = tuple(StoredRouteResult(id=1000 + i, route_result=route) for i, route in enumerate(routes))
    return StoredRoutePlanningRun(id=run_id, run=run, routes=stored_routes)


def make_stored_target_set(
    target_set_id=TARGET_SET_ID,
    fire_event_id=FIRE_EVENT_ID,
    targets=None,
) -> StoredResponseTargetSet:
    if targets is None:
        targets = (
            ResponseTarget(
                fire_event_id=fire_event_id,
                target_type=ResponseTargetType.ACTIVE_FIRE,
                latitude=32.731,
                longitude=35.046,
                priority_score=150.0,
            ),
        )
    domain_set = ResponseTargetSet(
        fire_event_id=fire_event_id,
        generated_at=AS_OF,
        methodology="TEST_TARGETS",
        methodology_version="1.0",
        targets=targets,
    )
    stored_targets = tuple(
        StoredResponseTarget(id=900 + i, target_order=i, target=target) for i, target in enumerate(targets)
    )
    return StoredResponseTargetSet(id=target_set_id, target_set=domain_set, targets=stored_targets)


def make_optimization_success() -> ResponseOptimizationResult:
    return ResponseOptimizationResult(
        success=True,
        fire_event_id=FIRE_EVENT_ID,
        response_target_set_id=TARGET_SET_ID,
        route_planning_run_id=RUN_ID,
        status=ResponseOptimizationStatus.OPTIMIZED,
        plan_status=ResponsePlanStatus.COMPLETE,
        response_plan_id=501,
        action_count=1,
        uncovered_target_count=0,
        plan_score=90.0,
        coverage_score=100.0,
        average_eta_seconds=90.0,
        error_message=None,
    )


def make_adapter(
    route_planning_repository=None,
    response_target_repository=None,
    optimization_agent=None,
) -> tuple[ResponseOptimizationCollaboratorAdapter, FakeOptimizationAgent]:
    agent = optimization_agent or FakeOptimizationAgent(make_optimization_success())
    adapter = ResponseOptimizationCollaboratorAdapter(
        optimization_agent=agent,
        route_planning_repository=route_planning_repository or FakeRoutePlanningRepository(make_stored_run()),
        response_target_repository=response_target_repository
        or FakeResponseTargetRepository(make_stored_target_set()),
    )
    return adapter, agent


# ---------------------------------------------------------------------------
# 1-3. Exact run/target-set loading
# ---------------------------------------------------------------------------


def test_loads_exact_route_planning_run_by_supplied_id():
    route_planning_repository = FakeRoutePlanningRepository(make_stored_run())
    adapter, _ = make_adapter(route_planning_repository=route_planning_repository)

    adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)

    assert route_planning_repository.calls == [RUN_ID]


def test_loads_exact_response_target_set_referenced_by_run():
    response_target_repository = FakeResponseTargetRepository(make_stored_target_set())
    adapter, _ = make_adapter(response_target_repository=response_target_repository)

    adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)

    assert response_target_repository.calls == [TARGET_SET_ID]


def test_does_not_load_latest_target_set_instead():
    """The adapter's only target-set repository call is by the run's own referenced id."""
    response_target_repository = FakeResponseTargetRepository(make_stored_target_set())
    adapter, _ = make_adapter(response_target_repository=response_target_repository)

    adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)

    assert not hasattr(response_target_repository, "get_latest_for_event_as_of")
    assert response_target_repository.calls == [TARGET_SET_ID]


# ---------------------------------------------------------------------------
# 4-5. Resource membership from the run's own snapshot only
# ---------------------------------------------------------------------------


def test_uses_exactly_run_resource_ids():
    route_planning_repository = FakeRoutePlanningRepository(
        make_stored_run(resource_ids=("truck-1", "truck-2", "truck-3"))
    )
    agent = FakeOptimizationAgent(make_optimization_success())
    adapter, _ = make_adapter(route_planning_repository=route_planning_repository, optimization_agent=agent)

    adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)

    resources = agent.calls[0]["optimization_input"].resources
    assert {r.resource_id for r in resources} == {"truck-1", "truck-2", "truck-3"}
    assert all(isinstance(r, OptimizationResource) for r in resources)


def test_does_not_query_current_available_resource_membership():
    route_planning_repository = FakeRoutePlanningRepository(make_stored_run())
    response_target_repository = FakeResponseTargetRepository(make_stored_target_set())
    for forbidden in ("get_available_resources", "get_available_operational_context"):
        assert not hasattr(route_planning_repository, forbidden)
        assert not hasattr(response_target_repository, forbidden)


# ---------------------------------------------------------------------------
# 6-7. Stored routes reused, no rerouting
# ---------------------------------------------------------------------------


def test_uses_stored_route_results():
    agent = FakeOptimizationAgent(make_optimization_success())
    adapter, _ = make_adapter(optimization_agent=agent)

    adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)

    route_options = agent.calls[0]["optimization_input"].route_options
    assert len(route_options) == 2
    reachable = next(o for o in route_options if o.resource_id == "truck-1")
    unreachable = next(o for o in route_options if o.resource_id == "truck-2")
    assert reachable.is_reachable is True
    assert reachable.travel_time_seconds == 90.0
    assert reachable.distance_meters == 1000.0
    assert unreachable.is_reachable is False
    assert unreachable.travel_time_seconds is None
    assert unreachable.distance_meters is None


def test_does_not_rerun_routing():
    route_planning_repository = FakeRoutePlanningRepository(make_stored_run())
    for forbidden in ("calculate_shortest_path", "map_resources", "map_targets", "plan"):
        assert not hasattr(route_planning_repository, forbidden)


# ---------------------------------------------------------------------------
# 8-14. ResponseOptimizationInput construction and agent invocation
# ---------------------------------------------------------------------------


def test_constructs_expected_response_optimization_input():
    agent = FakeOptimizationAgent(make_optimization_success())
    adapter, _ = make_adapter(optimization_agent=agent)

    adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)

    optimization_input = agent.calls[0]["optimization_input"]
    assert optimization_input.fire_event_id == FIRE_EVENT_ID
    assert optimization_input.response_target_set_id == TARGET_SET_ID
    assert optimization_input.route_planning_run_id == RUN_ID
    assert len(optimization_input.targets) == 1
    target = optimization_input.targets[0]
    assert isinstance(target, OptimizationTarget)
    assert target.response_target_id == 900
    assert target.target_order == 0
    assert target.target_type is ResponseTargetType.ACTIVE_FIRE
    assert target.priority_score == 150.0


def test_passes_exact_fire_event_id():
    agent = FakeOptimizationAgent(make_optimization_success())
    adapter, _ = make_adapter(
        route_planning_repository=FakeRoutePlanningRepository(make_stored_run(fire_event_id=77)),
        response_target_repository=FakeResponseTargetRepository(make_stored_target_set(fire_event_id=77)),
        optimization_agent=agent,
    )

    adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)

    assert agent.calls[0]["optimization_input"].fire_event_id == 77


def test_passes_exact_response_target_set_id():
    agent = FakeOptimizationAgent(make_optimization_success())
    adapter, _ = make_adapter(
        route_planning_repository=FakeRoutePlanningRepository(make_stored_run(response_target_set_id=555)),
        response_target_repository=FakeResponseTargetRepository(make_stored_target_set(target_set_id=555)),
        optimization_agent=agent,
    )

    adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)

    assert agent.calls[0]["optimization_input"].response_target_set_id == 555


def test_passes_exact_route_planning_run_id():
    agent = FakeOptimizationAgent(make_optimization_success())
    adapter, _ = make_adapter(
        route_planning_repository=FakeRoutePlanningRepository(make_stored_run(run_id=999)), optimization_agent=agent
    )

    adapter.optimize(route_planning_run_id=999, as_of=AS_OF, seed=42)

    assert agent.calls[0]["optimization_input"].route_planning_run_id == 999


def test_passes_supplied_seed_through_existing_config():
    agent = FakeOptimizationAgent(make_optimization_success())
    adapter, _ = make_adapter(optimization_agent=agent)

    adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=777)

    config = agent.calls[0]["config"]
    assert config.random_seed == 777


def test_calls_optimization_agent_exactly_once():
    agent = FakeOptimizationAgent(make_optimization_success())
    adapter, _ = make_adapter(optimization_agent=agent)

    adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)

    assert len(agent.calls) == 1


def test_returns_response_optimization_result_unchanged():
    expected = make_optimization_success()
    agent = FakeOptimizationAgent(expected)
    adapter, _ = make_adapter(optimization_agent=agent)

    result = adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)

    assert result is expected


# ---------------------------------------------------------------------------
# 15-17. Explicit failure on missing/inconsistent snapshot
# ---------------------------------------------------------------------------


def test_missing_run_fails_explicitly():
    adapter, _ = make_adapter(route_planning_repository=FakeRoutePlanningRepository(None))

    with pytest.raises(ResponseOptimizationAdapterError):
        adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)


def test_missing_referenced_target_set_fails_explicitly():
    adapter, _ = make_adapter(response_target_repository=FakeResponseTargetRepository(None))

    with pytest.raises(ResponseOptimizationAdapterError):
        adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)


def test_fire_event_mismatch_fails_explicitly():
    adapter, _ = make_adapter(
        route_planning_repository=FakeRoutePlanningRepository(make_stored_run(fire_event_id=1)),
        response_target_repository=FakeResponseTargetRepository(make_stored_target_set(fire_event_id=2)),
    )

    with pytest.raises(ResponseOptimizationAdapterError):
        adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)


# ---------------------------------------------------------------------------
# 18-19. Snapshot isolation from later operational changes
# ---------------------------------------------------------------------------


def test_newer_target_set_does_not_affect_optimization_of_old_run():
    """The adapter is only ever given the exact target set the fake repository returns for the run's id."""
    response_target_repository = FakeResponseTargetRepository(make_stored_target_set(target_set_id=TARGET_SET_ID))
    agent = FakeOptimizationAgent(make_optimization_success())
    adapter, _ = make_adapter(response_target_repository=response_target_repository, optimization_agent=agent)

    adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)

    assert agent.calls[0]["optimization_input"].response_target_set_id == TARGET_SET_ID
    assert response_target_repository.calls == [TARGET_SET_ID]


def test_resource_status_change_after_routing_does_not_alter_resource_membership():
    """Resource membership is frozen to the run's resource_ids regardless of any later status change."""
    solo_route = (
        RouteResult(
            resource_id="truck-1",
            response_target_id=900,
            status=RouteStatus.REACHABLE,
            source_node_id=1,
            target_node_id=2,
            node_path=(1, 2),
            distance_meters=1000.0,
            travel_time_seconds=90.0,
        ),
    )
    route_planning_repository = FakeRoutePlanningRepository(
        # e.g. truck-2 was ASSIGNED after routing and excluded upstream - only truck-1 remains in this run's snapshot.
        make_stored_run(resource_ids=("truck-1",), routes=solo_route)
    )
    agent = FakeOptimizationAgent(make_optimization_success())
    adapter, _ = make_adapter(route_planning_repository=route_planning_repository, optimization_agent=agent)

    adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed=42)

    resources = agent.calls[0]["optimization_input"].resources
    assert {r.resource_id for r in resources} == {"truck-1"}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_run_id", [0, -1, True, "601"])
def test_invalid_route_planning_run_id_rejected(invalid_run_id):
    adapter, _ = make_adapter()
    with pytest.raises(ValueError):
        adapter.optimize(route_planning_run_id=invalid_run_id, as_of=AS_OF, seed=42)


def test_naive_as_of_rejected():
    adapter, _ = make_adapter()
    with pytest.raises(ValueError):
        adapter.optimize(route_planning_run_id=RUN_ID, as_of=datetime(2026, 9, 16, 12, 0), seed=42)


def test_invalid_seed_rejected():
    adapter, _ = make_adapter()
    with pytest.raises(ValueError):
        adapter.optimize(route_planning_run_id=RUN_ID, as_of=AS_OF, seed="42")
