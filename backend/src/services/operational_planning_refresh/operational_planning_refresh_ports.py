"""Structural interfaces OperationalPlanningRefreshCoordinator depends on.

Declared as typing.Protocol rather than a direct import of the concrete
orchestrator/coordinator classes, mirroring
src/services/response_planning/response_planning_refresh_ports.py's own
Protocol-based boundary for US 5.4's routing/optimization/baseline
collaborators. This keeps the coordinator (and its unit tests) importable
without pulling in Epic 5's routing package and its optional osmnx
dependency merely to type-check against the real implementations.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.services.global_planning.global_planning_refresh_coordinator import GlobalPlanningRefreshResult
    from src.services.response_planning.planning_refresh_result import PlanningRefreshResult


class PlanningRefreshPort(Protocol):
    """Whatever can run one US 5.4 planning-refresh cycle for a FireEvent.

    Matches ResponsePlanningRefreshOrchestrator.refresh's signature exactly.
    Retained for the LEGACY per-event planner (kept in source, no longer
    authoritative in production - Stage 6 Task 47) and for any test/history
    code that still exercises it directly. Production wiring now uses
    GlobalPlanningRefreshPort instead - see operational_planning_refresh_production_factory.py.
    """

    def refresh(self, *, fire_event_id: int, as_of: datetime) -> "PlanningRefreshResult": ...


class GlobalPlanningRefreshPort(Protocol):
    """Whatever can run one Stage 6 global planning-refresh cycle.

    Matches GlobalPlanningRefreshCoordinator.refresh's signature exactly;
    that class is the real, now-authoritative production implementation of
    this port (Stage 6, Task 47). Unlike PlanningRefreshPort, this covers
    EVERY currently active FireEvent in one call - there is no
    fire_event_id parameter, because there is one global optimization
    problem, not one per FireEvent.
    """

    def refresh(self, *, trigger: str, as_of: datetime) -> "GlobalPlanningRefreshResult": ...
