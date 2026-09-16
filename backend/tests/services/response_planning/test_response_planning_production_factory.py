"""Tests for build_response_planning_refresh_orchestrator - construction/wiring smoke tests."""
from __future__ import annotations

from datetime import datetime, timezone

from src.services.baseline_comparison.response_plan_baseline_scorer_adapter import ResponsePlanBaselineScorerAdapter
from src.services.response_planning import PlanningRefreshStatus, ResponsePlanningRefreshOrchestrator
from src.services.response_planning.baseline_comparison_collaborator_adapter import (
    BaselineComparisonCollaboratorAdapter,
)
from src.services.response_planning.response_optimization_collaborator_adapter import (
    ResponseOptimizationCollaboratorAdapter,
)
from src.services.response_planning.response_planning_production_factory import (
    build_response_planning_refresh_orchestrator,
)

AS_OF = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


class FakeBaselinePlanScorer:
    def evaluate(self, context):
        raise NotImplementedError("not exercised in this smoke test")


def test_factory_returns_a_real_orchestrator(sqlite_session_factory):
    orchestrator = build_response_planning_refresh_orchestrator(session_factory=sqlite_session_factory)

    assert isinstance(orchestrator, ResponsePlanningRefreshOrchestrator)


def test_factory_wires_optimization_and_baseline_adapters(sqlite_session_factory):
    orchestrator = build_response_planning_refresh_orchestrator(session_factory=sqlite_session_factory)

    assert isinstance(orchestrator._optimization_collaborator, ResponseOptimizationCollaboratorAdapter)
    assert isinstance(orchestrator._baseline_collaborator, BaselineComparisonCollaboratorAdapter)


def test_factory_wired_orchestrator_is_fully_callable_against_empty_db(sqlite_session_factory):
    """A missing FireEvent produces a clean FAILED result rather than an AttributeError/NameError,
    proving every collaborator/repository resolved through the whole real dependency graph."""
    orchestrator = build_response_planning_refresh_orchestrator(session_factory=sqlite_session_factory)

    result = orchestrator.refresh(fire_event_id=1, as_of=AS_OF)

    assert result.status is PlanningRefreshStatus.FAILED
    assert result.error


# ---------------------------------------------------------------------------
# Task 6.1: real production baseline scorer wired by default
# ---------------------------------------------------------------------------


def test_factory_wires_real_baseline_scorer_by_default(sqlite_session_factory):
    """No caller-supplied baseline_plan_scorer is required for production wiring."""
    orchestrator = build_response_planning_refresh_orchestrator(session_factory=sqlite_session_factory)

    baseline_service = orchestrator._baseline_collaborator._baseline_comparison_service  # noqa: SLF001
    assert isinstance(baseline_service._scorer, ResponsePlanBaselineScorerAdapter)  # noqa: SLF001


def test_factory_still_accepts_explicit_scorer_override_for_customization(sqlite_session_factory):
    fake_scorer = FakeBaselinePlanScorer()

    orchestrator = build_response_planning_refresh_orchestrator(
        baseline_plan_scorer=fake_scorer,
        session_factory=sqlite_session_factory,
    )

    baseline_service = orchestrator._baseline_collaborator._baseline_comparison_service  # noqa: SLF001
    assert baseline_service._scorer is fake_scorer  # noqa: SLF001


# ---------------------------------------------------------------------------
# Regression: injected session_factory must reach the ENTIRE graph, including
# OperationalContextService/RoutePlanningAgent - a real end-to-end sanity run
# exposed that these two previously fell back to the process-wide default
# session factory even when a caller explicitly injected one, silently
# querying an unrelated database instead of the caller's.
# ---------------------------------------------------------------------------


def test_factory_threads_injected_session_factory_into_routing_agent_and_operational_context(
    sqlite_session_factory,
):
    orchestrator = build_response_planning_refresh_orchestrator(session_factory=sqlite_session_factory)

    routing_agent = orchestrator._routing_collaborator  # noqa: SLF001
    assert routing_agent._session_factory is sqlite_session_factory  # noqa: SLF001

    operational_context_service = routing_agent._operational_context_service  # noqa: SLF001
    assert operational_context_service._fire_station_repository._session_factory is sqlite_session_factory  # noqa: SLF001
    assert (
        operational_context_service._firefighting_resource_repository._session_factory  # noqa: SLF001
        is sqlite_session_factory
    )
