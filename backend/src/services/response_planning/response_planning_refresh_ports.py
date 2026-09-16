"""Deferred-adapter collaborator ports for ResponsePlanningRefreshOrchestrator (Epic 5, US 5.4, Task 5).

Task 5 owns refresh POLICY only - it never invokes Dijkstra, GA, or the real
BaselineComparisonService directly. Company 1's RoutePlanningAgent and
Company 2's ResponseOptimizationAgent already expose stable result contracts
(RoutePlanningResult, ResponseOptimizationResult) that the Protocols below
reuse as-is; only the *call signatures* here are new, because the frozen
orchestrator contract needs `plan(fire_event_id, as_of)` and
`optimize(route_planning_run_id, as_of, seed)` shapes that do not match
Company 2's current `optimize_from_input(ResponseOptimizationInput, ...)`
entry point. Wiring real adapters that satisfy these Protocols against the
production agents is Task 6's job, not Task 5's - this module defines
dependency contracts, not replacement agents (mirrors
baseline_comparison_ports.py's own rationale).

BaselineComparisonCollaborator mirrors BaselineComparisonService.compare()'s
actual failure behavior (raises on failure) but returns StoredPlanComparison
(already present in our own US 5.3 code) instead of the unwrapped
PlanComparison compare() currently returns, since the orchestrator needs the
persisted comparison id for PlanningRefreshResult.comparison_id.
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from src.agents.analysis.response_optimization_result import ResponseOptimizationResult
from src.agents.routing.route_planning_result import RoutePlanningResult
from src.repositories.plan_comparison_repository import StoredPlanComparison


class RoutingCollaborator(Protocol):
    """Port for Company 1's routing step, as Task 5 needs to call it."""

    def plan(self, *, fire_event_id: int, as_of: datetime) -> RoutePlanningResult: ...


class OptimizationCollaborator(Protocol):
    """Port for Company 2's optimization step, as Task 5 needs to call it."""

    def optimize(
        self,
        *,
        route_planning_run_id: int,
        as_of: datetime,
        seed: int,
    ) -> ResponseOptimizationResult: ...


class BaselineComparisonCollaborator(Protocol):
    """Port for US 5.3's baseline comparison step, as Task 5 needs to call it."""

    def compare(self, *, response_plan_id: int) -> StoredPlanComparison: ...
