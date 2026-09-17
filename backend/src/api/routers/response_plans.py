"""ResponsePlan read endpoints (Epic 6, US 6.3, Tasks 2-3-4-5).

A thin HTTP transport over two read-only collaborators:

    ResponsePlanDetailsService (Epic 5, US 5.5) -> ResponsePlanDetails
    ResponsePlanPresenter (US 6.3, Task 4)      -> ResponsePlanDetailResponse

No routing, optimization, scoring, allocation, priority calculation,
baseline calculation, or resource-status mutation happens in this module,
and no repository/database access happens directly from a route handler -
`ResponsePlanDetailsService` and `ResponsePlanPresenter` already own all of
that between them. An unhandled exception from either collaborator is left
to propagate to FastAPI/Starlette's own default exception handling, which
returns a generic 500 without leaking internals (no stack trace, no SQL, no
DATABASE_URL) as long as the app is not run in debug mode - see
src.api.app, which never sets debug=True.

This router is deliberately not wired into `src.api.routers.v1_router` yet
(see that module's docstring/`__init__.py`) - registering it there is a
one-line follow-up left for whoever owns that shared file, so tests below
mount it onto a throwaway FastAPI app instead. `response_plans_router`
carries no router-level prefix (each route below spells out its own full
path) so that a single `v1_router.include_router(response_plans_router)`
produces both `/api/v1/fire-events/{fire_event_id}/response-plan` (Task 2)
and `/api/v1/response-plans/{plan_id}` (Task 3) without a nested prefix
mismatch.

`get_response_plan_details_service` and `get_response_plan_presenter` are
this module's own small dependency factories - mirroring
`get_active_fire_events_service` in `src.api.dependencies` - so that shared
file does not need to change for this US 6.3 task.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Path
from fastapi.responses import JSONResponse

from src.api.response_plan_presenter import ResponsePlanPresenter
from src.api.schemas.response_plans import ResponsePlanEnvelopeResponse
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.graph_node_read_repository import GraphNodeReadRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.response_planning.response_plan_details_service import ResponsePlanDetailsService

response_plans_router = APIRouter(tags=["response-plans"])

RESPONSE_PLAN_NOT_FOUND_CODE = "RESPONSE_PLAN_NOT_FOUND"
RESPONSE_PLAN_NOT_FOUND_MESSAGE = "The requested response plan was not found."


def _response_plan_not_found_response() -> JSONResponse:
    """The shared Epic 6 API error shape for a missing plan-by-id lookup.

    Returned as a plain `JSONResponse` (bypassing this endpoint's declared
    `response_model`) rather than `raise HTTPException(...)`, because
    FastAPI's default exception handler would otherwise wrap our body under
    a `"detail"` key (`{"detail": {"error": ...}}`) instead of producing the
    shared `{"error": {"code": ..., "message": ...}}` shape the frontend's
    `apiGet()` expects at the top level. This is a US 6.3-local transport
    detail only - no global exception handling is introduced.
    """
    return JSONResponse(
        status_code=404,
        content={"error": {"code": RESPONSE_PLAN_NOT_FOUND_CODE, "message": RESPONSE_PLAN_NOT_FOUND_MESSAGE}},
    )


def get_response_plan_details_service() -> ResponsePlanDetailsService:
    """FastAPI dependency providing a fully-wired ResponsePlanDetailsService.

    Matches `get_active_fire_events_service` (src/api/dependencies.py):
    `ResponsePlanDetailsService()` wires up its own repositories using each
    one's own default sessionmaker, so no request-scoped Session is passed
    in here.
    """
    return ResponsePlanDetailsService()


def get_response_plan_presenter() -> ResponsePlanPresenter:
    """FastAPI dependency providing a fully-wired ResponsePlanPresenter.

    Constructs the five read-only repositories the presenter enriches
    `ResponsePlanDetails` with - each one defaulting to the process-wide
    session factory the same way `ResponsePlanDetailsService` does, so no
    request-scoped Session is passed in here either. This is not a second
    business service: the presenter performs no routing/optimization/scoring
    of its own, only read/copy/enrich.
    """
    return ResponsePlanPresenter(
        response_plan_repository=ResponsePlanRepository(),
        response_target_repository=ResponseTargetRepository(),
        route_planning_repository=RoutePlanningRepository(),
        fire_station_repository=FireStationRepository(),
        graph_node_read_repository=GraphNodeReadRepository(),
    )


@response_plans_router.get(
    "/fire-events/{fire_event_id}/response-plan",
    response_model=ResponsePlanEnvelopeResponse,
    summary="Get the current recommended response plan for a wildfire event",
)
def get_current_response_plan(
    fire_event_id: int = Path(..., gt=0),
    service: ResponsePlanDetailsService = Depends(get_response_plan_details_service),
    presenter: ResponsePlanPresenter = Depends(get_response_plan_presenter),
) -> ResponsePlanEnvelopeResponse:
    """Return the FireEvent's current planning-safe plan, fully enriched, or `{"plan": null}` if none exists."""
    details = service.get_current_plan_details(fire_event_id)
    if details is None:
        return ResponsePlanEnvelopeResponse(plan=None)
    return ResponsePlanEnvelopeResponse(plan=presenter.present(details))


@response_plans_router.get(
    "/response-plans/{plan_id}",
    response_model=ResponsePlanEnvelopeResponse,
    summary="Get one specific persisted response plan by its plan ID",
)
def get_response_plan_by_id(
    plan_id: int = Path(..., gt=0),
    service: ResponsePlanDetailsService = Depends(get_response_plan_details_service),
    presenter: ResponsePlanPresenter = Depends(get_response_plan_presenter),
) -> ResponsePlanEnvelopeResponse | JSONResponse:
    """Return one persisted plan by id, fully enriched, current or superseded.

    `ResponsePlanDetailsService.get_plan_details_by_id` already owns the
    read-only current-vs-superseded resolution (it sets `is_current` by
    comparing against `CurrentResponsePlanResolver`'s result), and
    `ResponsePlanPresenter` enriches strictly from that exact plan's own
    persisted `response_target_set_id`/`route_planning_run_id` snapshot -
    this handler only forwards both untouched. A missing plan returns the
    shared Epic 6 API error shape - see `_response_plan_not_found_response`.
    """
    details = service.get_plan_details_by_id(plan_id)
    if details is None:
        return _response_plan_not_found_response()
    return ResponsePlanEnvelopeResponse(plan=presenter.present(details))
