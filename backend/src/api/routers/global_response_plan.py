"""GlobalResponsePlan read endpoint (Epic 6 UI/API Rework, Task B-BE-6).

A thin HTTP transport over one read-only collaborator:

    GlobalResponsePlanReadService (Tasks B-BE-4/B-BE-7) -> GlobalResponsePlanResponse

No routing, optimization, scoring, allocation, priority calculation, or
resource-status mutation happens in this module, and no repository/database
access happens directly from a route handler - `GlobalResponsePlanReadService`
already owns all of that (itself composing `ResponsePlanDetailsService` +
`ResponsePlanPresenter`, US 5.5/US 6.3, for per-event enrichment). An
unhandled exception from the service is left to propagate to FastAPI/
Starlette's own default exception handling, which returns a generic 500
without leaking internals (no stack trace, no SQL, no DATABASE_URL) as long
as the app is not run in debug mode - see src.api.app, which never sets
debug=True.

This router is not yet registered onto `src.api.routers.v1_router` (that
shared-file wiring - `src/api/routers/__init__.py` - is a follow-up left to
whoever owns that file, matching `response_plans_router`'s own precedent).
`global_response_plan_router` carries no router-level prefix so that
`v1_router.include_router(global_response_plan_router)` produces
`/api/v1/global-response-plan/current` without a nested prefix mismatch.

`get_global_response_plan_read_service` is this module's own small
dependency factory - mirroring `get_response_plan_details_service` in
`src.api.routers.response_plans` - so `src/api/dependencies.py` does not
need to change for this task.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.schemas.global_response_plan import GlobalResponsePlanResponse
from src.services.global_planning.global_response_plan_read_service import GlobalResponsePlanReadService

global_response_plan_router = APIRouter(tags=["global-response-plan"])


def get_global_response_plan_read_service() -> GlobalResponsePlanReadService:
    """FastAPI dependency providing a fully-wired GlobalResponsePlanReadService.

    Matches `get_response_plan_details_service` (src/api/routers/response_plans.py):
    `GlobalResponsePlanReadService()` wires up its own repositories/services
    using each one's own default sessionmaker, so no request-scoped Session
    is passed in here.
    """
    return GlobalResponsePlanReadService()


@global_response_plan_router.get(
    "/global-response-plan/current",
    response_model=GlobalResponsePlanResponse,
    summary="Get the current global (multi-incident) response plan for the Global Response Map",
)
def get_current_global_response_plan(
    service: GlobalResponsePlanReadService = Depends(get_global_response_plan_read_service),
) -> GlobalResponsePlanResponse:
    """Return the latest materialized GlobalPlanningRun, or `{"plan": null}` if none exists yet."""
    return service.get_current()
