"""Operations Activity detail endpoint (Task A5).

A thin HTTP transport over OperationsActivityQueryService
(src/services/operations/operations_activity_query_service.py): no business
logic, no repository SQL, and no agent/calculator/coordinator invocation
happens in this module. This is the ONLY endpoint A5 adds - it answers
exactly "I have an activity_type and entity_id; give me its persisted
details," nothing more. Feed aggregation/ordering and the Operations
Overview aggregate endpoint are explicitly out of scope (A6).

`activity_type` is parsed as a path enum (OperationsActivityType) - FastAPI
rejects any value outside the closed set with an automatic 422, so no
arbitrary table/model name is ever accepted from the URL. A known
activity_type with no matching entity_id returns a 404 with the shared
Epic 6 API error envelope (see src.api.routers.response_plans); this
endpoint never returns a null-body 200 for a specifically requested entity
that does not exist (contrast with A4's areas/latest, where "not in the
list" is the correct absence signal for a different kind of request).
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Path
from fastapi.responses import JSONResponse

from src.api.dependencies import get_operations_activity_query_service
from src.api.schemas.operations_activity import (
    OperationsActivityDetailResponse,
    to_operations_activity_detail_response,
)
from src.models.operations_activity import OperationsActivityType
from src.services.operations.operations_activity_query_service import OperationsActivityQueryService

operations_activity_router = APIRouter(prefix="/operations/activity", tags=["operations-activity"])

OPERATIONS_ACTIVITY_NOT_FOUND_CODE = "OPERATIONS_ACTIVITY_NOT_FOUND"


@operations_activity_router.get(
    "/{activity_type}/{entity_id}",
    response_model=OperationsActivityDetailResponse,
    summary="Get the persisted detail for one Operations Activity item",
)
def get_operations_activity_detail(
    activity_type: OperationsActivityType,
    entity_id: int = Path(..., gt=0),
    service: OperationsActivityQueryService = Depends(get_operations_activity_query_service),
) -> OperationsActivityDetailResponse | JSONResponse:
    """Return the persisted detail for one activity, or a 404 if entity_id does not exist.

    Read-only: never runs FireDangerAssessmentAgent/FFWICalculator,
    FireDetectionAgent, FireSeverityAssessmentAgent, a spread predictor,
    routing, or the Global Optimizer/GA - only already-persisted data,
    reshaped through OperationsActivityQueryService.
    """
    detail = service.get_activity_detail(activity_type, entity_id)
    if detail is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": OPERATIONS_ACTIVITY_NOT_FOUND_CODE,
                    "message": (
                        f"No {activity_type.value} activity exists with id={entity_id}."
                    ),
                }
            },
        )
    return to_operations_activity_detail_response(detail, now=datetime.now(timezone.utc))
