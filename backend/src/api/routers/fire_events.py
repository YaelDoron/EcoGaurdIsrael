"""FireEvent read endpoints (Epic 6, US 6.1, Task 3).

A thin HTTP transport over ActiveFireEventsService (Task 2): no business
logic, no direct repository/DB access, and no agent or external-provider
invocation happens in this module. An unhandled exception from the service
call below is left to propagate to FastAPI/Starlette's own default
exception handling, which returns a generic 500 without leaking internals
(no stack trace, no SQL, no DATABASE_URL) as long as the app is not run in
debug mode - see src.api.app, which never sets debug=True.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_active_fire_events_service
from src.api.schemas.active_fire_events import ActiveFireEventsResponse, to_active_fire_events_response
from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService

fire_events_router = APIRouter(prefix="/fire-events", tags=["fire-events"])


@fire_events_router.get(
    "/active",
    response_model=ActiveFireEventsResponse,
    summary="List currently active wildfire events",
)
def get_active_fire_events(
    service: ActiveFireEventsService = Depends(get_active_fire_events_service),
) -> ActiveFireEventsResponse:
    """Return every SUSPECTED/CONFIRMED FireEvent with its latest severity, if any."""
    result = service.get_active_events()
    return to_active_fire_events_response(result)
