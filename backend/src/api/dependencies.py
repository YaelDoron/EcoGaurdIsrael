"""Shared FastAPI dependencies for the API layer.

Reuses the existing process-wide SQLAlchemy session factory from
`src.database.connection` - no second engine or session system is created
here.
"""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from src.database.connection import get_session_factory
from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService
from src.services.fire_event_read.event_details_service import EventDetailsService


def get_db_session() -> Iterator[Session]:
    """FastAPI dependency yielding a database session, closed after the request.

    Usage: `session: Session = Depends(get_db_session)`. Callers are
    responsible for commit/rollback, matching `src.database.connection.get_session`.

    Not used by ActiveFireEventsService/its repositories below - see
    get_active_fire_events_service for why a single request-scoped Session
    doesn't fit their design. This dependency remains available for a
    future endpoint built directly around one Session.
    """
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def get_active_fire_events_service() -> ActiveFireEventsService:
    """FastAPI dependency providing a fully-wired ActiveFireEventsService.

    FireEventRepository/FireSeverityAssessmentRepository are each built
    around a `sessionmaker`, not a request-owned `Session`: every
    repository method opens its own session, commits/rolls back, and
    closes it internally (see `_session_scope` in
    src/repositories/fire_event_repository.py). That is why this
    dependency does not compose with get_db_session above (which hands out
    a single live Session) - a sessionmaker-per-call is exactly what each
    repository already defaults to when constructed with no arguments, via
    `get_session_factory()` (src/database/connection.py). No engine or
    session system is created here; this only reuses that existing
    process-wide default, matching this project's other production-wiring
    call sites (e.g. build_response_planning_refresh_orchestrator).
    """
    return ActiveFireEventsService()


def get_event_details_service() -> EventDetailsService:
    """FastAPI dependency providing a fully-wired EventDetailsService.

    Same sessionmaker-per-call rationale as get_active_fire_events_service
    above: every repository EventDetailsService uses (directly, or via the
    composed ResponsePlanDetailsService) defaults to the process-wide
    session factory when constructed with no arguments, so this dependency
    does not need - and does not compose with - get_db_session's single
    request-scoped Session.
    """
    return EventDetailsService()
