"""Operations Overview endpoint (Task A6).

A thin HTTP transport over OperationsOverviewQueryService
(src/services/operations/operations_overview_query_service.py): no business
logic, no repository SQL, and no agent/calculator/coordinator invocation
happens in this module. This is the last major backend aggregation task
before frontend data integration - it does NOT implement the Activity Feed
aggregate itself (that lives in the query service), does NOT start
simulation (`POST /api/v1/simulation/runs` remains the only way to do that),
and does NOT combine anything beyond what Task A6 specifies.

Fully read-only: works before a simulation ever starts, while one is
running, after it completes, and against a normal (non-demo) database -
never a 404/500 for an empty/just-reset DB.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_operations_overview_query_service
from src.api.schemas.operations_overview import OperationsOverviewResponse, to_operations_overview_response
from src.services.operations.operations_overview_query_service import (
    DEFAULT_ACTIVITY_LIMIT,
    MAX_ACTIVITY_LIMIT,
    MIN_ACTIVITY_LIMIT,
    OperationsOverviewQueryService,
)

operations_overview_router = APIRouter(prefix="/operations", tags=["operations-overview"])


@operations_overview_router.get(
    "/overview",
    response_model=OperationsOverviewResponse,
    summary="Get the current Operations Overview dashboard snapshot",
)
def get_operations_overview(
    activity_limit: int = Query(DEFAULT_ACTIVITY_LIMIT, ge=MIN_ACTIVITY_LIMIT, le=MAX_ACTIVITY_LIMIT),
    service: OperationsOverviewQueryService = Depends(get_operations_overview_query_service),
) -> OperationsOverviewResponse:
    """Return one coherent snapshot: Fire Danger areas, active fires with latest
    Severity, a bounded chronological Activity Feed, and the simulation-control
    summary. Never runs FireDangerAssessmentAgent/FFWICalculator, FireDetectionAgent,
    FireSeverityAssessmentAgent, a spread predictor, routing/Dijkstra, or the
    Global Optimizer/GA - only already-persisted/already-computed data.
    """
    snapshot = service.get_overview(activity_limit=activity_limit)
    return to_operations_overview_response(snapshot)
