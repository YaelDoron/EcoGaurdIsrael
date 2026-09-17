"""Structural interface OperationalPlanningRefreshCoordinator depends on.

Declared as typing.Protocol rather than a direct import of
ResponsePlanningRefreshOrchestrator, mirroring
src/services/response_planning/response_planning_refresh_ports.py's own
Protocol-based boundary for US 5.4's routing/optimization/baseline
collaborators. This keeps the coordinator (and its unit tests) importable
without pulling in Epic 5's routing package and its optional osmnx
dependency merely to type-check against the real orchestrator.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.services.response_planning.planning_refresh_result import PlanningRefreshResult


class PlanningRefreshPort(Protocol):
    """Whatever can run one US 5.4 planning-refresh cycle for a FireEvent.

    Matches ResponsePlanningRefreshOrchestrator.refresh's signature exactly;
    that class is the real production implementation of this port.
    """

    def refresh(self, *, fire_event_id: int, as_of: datetime) -> "PlanningRefreshResult": ...
