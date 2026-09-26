"""API routers package.

`v1_router` is the single mount point for the `/api/v1` namespace. Epic 6+
endpoints attach to it via `v1_router.include_router(...)` rather than
being wired individually into the FastAPI app in `src.api.app`.
"""
from fastapi import APIRouter, Depends

from src.api.read_only_request import read_only_request

from src.api.routers.chatbot import chatbot_router
from src.api.routers.fire_danger import fire_danger_router
from src.api.routers.fire_events import fire_events_router
from src.api.routers.global_response_plan import global_response_plan_router
from src.api.routers.operations_activity import operations_activity_router
from src.api.routers.operations_overview import operations_overview_router
from src.api.routers.response_plans import response_plans_router
from src.api.routers.simulation import simulation_router

v1_router = APIRouter(prefix="/api/v1")
# Pure GET read routers (their services never write) read on AUTOCOMMIT
# connections - see src/api/read_only_request.py.
_READ_ONLY = [Depends(read_only_request)]
v1_router.include_router(fire_events_router, dependencies=_READ_ONLY)
v1_router.include_router(fire_danger_router, dependencies=_READ_ONLY)
v1_router.include_router(operations_activity_router, dependencies=_READ_ONLY)
v1_router.include_router(operations_overview_router, dependencies=_READ_ONLY)
v1_router.include_router(response_plans_router, dependencies=_READ_ONLY)
v1_router.include_router(global_response_plan_router, dependencies=_READ_ONLY)
v1_router.include_router(simulation_router)
v1_router.include_router(chatbot_router)

__all__ = ["v1_router"]
