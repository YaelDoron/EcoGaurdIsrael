"""Fire Danger read endpoints (Task A4).

A thin HTTP transport over FireDangerQueryService
(src/services/fire_danger/fire_danger_query_service.py): no FFWI
calculation, no FireDangerAssessmentAgent invocation, no IMS access, and no
repository SQL happens in this module - strictly HTTP request -> query
service -> DTO, matching src.api.routers.response_plans's own thin-router
precedent. Every endpoint here is read-only: repeated GET requests have no
side effects (no assessment is inserted, no timestamp is updated, no
simulation/routing/Global Planning is triggered).

Fire Danger is PRE-FIRE area risk, not FireEvent severity/status, Fire
Detection confidence, or predicted spread - this router is deliberately
namespaced under `/fire-danger`, separate from `/fire-events`.

Unknown-area/unknown-assessment-id responses use the shared Epic 6 API error
envelope (`{"error": {"code": ..., "message": ...}}`, see
src.api.routers.response_plans._response_plan_not_found_response) rather
than FastAPI's default `{"detail": ...}` shape.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Path
from fastapi.responses import JSONResponse

from src.api.dependencies import get_fire_danger_query_service
from src.api.schemas.fire_danger import (
    FireDangerAreaLatestResponse,
    FireDangerAreasLatestResponse,
    FireDangerAssessmentDetailResponse,
    to_fire_danger_area_latest_response,
    to_fire_danger_areas_latest_response,
    to_fire_danger_assessment_detail_response,
)
from src.services.fire_danger.fire_danger_query_service import FireDangerQueryService

fire_danger_router = APIRouter(prefix="/fire-danger", tags=["fire-danger"])

FIRE_DANGER_AREA_NOT_FOUND_CODE = "FIRE_DANGER_AREA_NOT_FOUND"
FIRE_DANGER_ASSESSMENT_NOT_FOUND_CODE = "FIRE_DANGER_ASSESSMENT_NOT_FOUND"


def _not_found_response(code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": {"code": code, "message": message}})


@fire_danger_router.get(
    "/areas/latest",
    response_model=FireDangerAreasLatestResponse,
    summary="List every assessed area with its latest persisted Fire Danger assessment",
)
def get_fire_danger_areas_latest(
    service: FireDangerQueryService = Depends(get_fire_danger_query_service),
) -> FireDangerAreasLatestResponse:
    """Return every area that has ever been assessed, newest-per-area, sorted by area_name.

    An area with no persisted assessment at all simply is not in this list -
    see FireDangerQueryService's module docstring for why (no independent
    area registry exists in this architecture). Returns `areas: []` on an
    empty/just-reset database - never a 500, never a fabricated LOW entry.
    """
    result = service.get_latest_for_all_areas()
    return to_fire_danger_areas_latest_response(result)


@fire_danger_router.get(
    "/areas/{area_id}/latest",
    response_model=FireDangerAreaLatestResponse,
    summary="Get the latest persisted Fire Danger assessment for one area",
)
def get_fire_danger_area_latest(
    area_id: str = Path(..., min_length=1),
    service: FireDangerQueryService = Depends(get_fire_danger_query_service),
) -> FireDangerAreaLatestResponse | JSONResponse:
    """Return one area's latest persisted assessment, or a 404 if area_id has never been assessed.

    "Unknown area" here means exactly "no persisted assessment exists for
    this area_id" - this architecture cannot distinguish that from "this
    area_id was never configured to be assessed", since there is no
    independent area registry (see FireDangerQueryService's module
    docstring). A "known area with zero assessments" 200/null response is
    therefore not reachable today.
    """
    area = service.get_latest_for_area(area_id)
    if area is None:
        return _not_found_response(
            FIRE_DANGER_AREA_NOT_FOUND_CODE,
            f"No Fire Danger assessment has ever been recorded for area_id={area_id!r}.",
        )
    return to_fire_danger_area_latest_response(area, now=datetime.now(timezone.utc))


@fire_danger_router.get(
    "/assessments/{assessment_id}",
    response_model=FireDangerAssessmentDetailResponse,
    summary="Get one persisted Fire Danger assessment with its preserved weather-input trace",
)
def get_fire_danger_assessment_detail(
    assessment_id: int = Path(..., gt=0),
    service: FireDangerQueryService = Depends(get_fire_danger_query_service),
) -> FireDangerAssessmentDetailResponse | JSONResponse:
    """Return one persisted assessment by id, or a 404 if it does not exist.

    Describes the historical persisted assessment exactly as stored - never
    recalculates the score and never fetches current weather.
    """
    detail = service.get_assessment_detail(assessment_id)
    if detail is None:
        return _not_found_response(
            FIRE_DANGER_ASSESSMENT_NOT_FOUND_CODE,
            f"No Fire Danger assessment exists with id={assessment_id}.",
        )
    return to_fire_danger_assessment_detail_response(detail, now=datetime.now(timezone.utc))
