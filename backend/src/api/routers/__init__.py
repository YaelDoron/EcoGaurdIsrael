"""API routers package.

`v1_router` is the single mount point for the `/api/v1` namespace. Epic 6+
endpoints attach to it via `v1_router.include_router(...)` rather than
being wired individually into the FastAPI app in `src.api.app`.
"""
from fastapi import APIRouter

from src.api.routers.fire_events import fire_events_router
from src.api.routers.global_response_plan import global_response_plan_router
from src.api.routers.response_plans import response_plans_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(fire_events_router)
v1_router.include_router(response_plans_router)
v1_router.include_router(global_response_plan_router)

__all__ = ["v1_router"]
