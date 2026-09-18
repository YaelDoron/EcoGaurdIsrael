"""Tests for build_operational_planning_refresh_coordinator - construction/wiring smoke tests.

Like tests/services/response_planning/test_response_planning_production_factory.py,
this file imports the real ResponsePlanningRefreshOrchestrator wiring
(transitively via OperationalContextService/RoadNetworkFetcher) and therefore
requires the optional osmnx dependency to even collect. See
src/services/operational_planning_refresh/__init__.py's docstring: the
coordinator itself does not require osmnx, only this production factory does.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.services.global_planning.global_planning_refresh_coordinator import GlobalPlanningRefreshCoordinator
from src.services.operational_planning_refresh.operational_planning_refresh_coordinator import (
    OperationalPlanningRefreshCoordinator,
)
from src.services.operational_planning_refresh.operational_planning_refresh_production_factory import (
    build_operational_planning_refresh_coordinator,
)
from src.services.operational_refresh.operational_refresh_orchestrator import OperationalRefreshOrchestrator

AS_OF = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def test_factory_returns_a_real_coordinator(sqlite_session_factory):
    coordinator = build_operational_planning_refresh_coordinator(session_factory=sqlite_session_factory)

    assert isinstance(coordinator, OperationalPlanningRefreshCoordinator)


def test_factory_wires_real_operational_and_global_planning_coordinators(sqlite_session_factory):
    """Stage 6, Task 47: production cutover - the global (not legacy
    per-event) planner is wired as the authoritative planning collaborator."""
    coordinator = build_operational_planning_refresh_coordinator(session_factory=sqlite_session_factory)

    assert isinstance(coordinator._operational_refresh_orchestrator, OperationalRefreshOrchestrator)  # noqa: SLF001
    assert isinstance(coordinator._global_planning_refresh, GlobalPlanningRefreshCoordinator)  # noqa: SLF001


def test_factory_wired_coordinator_is_fully_callable_against_empty_db(sqlite_session_factory):
    """A missing FireEvent produces a clean FAILED result rather than an
    AttributeError/NameError, proving every collaborator/repository resolved
    through the whole real dependency graph on both the US4.4 and Stage 6 sides."""
    coordinator = build_operational_planning_refresh_coordinator(session_factory=sqlite_session_factory)

    result = coordinator.refresh_fire_event(
        fire_event_id=1, trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE, as_of=AS_OF
    )

    assert result.operational_result.success is False
    assert result.global_planning_result is None
