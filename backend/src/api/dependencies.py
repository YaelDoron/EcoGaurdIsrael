"""Shared FastAPI dependencies for the API layer.

Reuses the existing process-wide SQLAlchemy session factory from
`src.database.connection` - no second engine or session system is created
here.
"""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from src.agents.response.chatbot_agent import ChatbotAgent
from src.database.connection import get_session_factory
from src.external.gemini.gemini_client import GeminiClient
from src.services.fire_danger.fire_danger_query_service import FireDangerQueryService
from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService
from src.services.fire_event_read.event_details_service import EventDetailsService
from src.services.operations.operations_activity_query_service import OperationsActivityQueryService
from src.services.operations.operations_overview_query_service import OperationsOverviewQueryService
from src.services.simulation_control.simulation_run_manager import SimulationRunManager
from src.services.weather.weather_conditions_query_service import WeatherConditionsQueryService


# Worker threads for one Event Details snapshot's independent section reads.
EVENT_DETAILS_PARALLEL_LOADS = 4

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
    # Independent snapshot sections are read concurrently (see EventDetailsService).
    return EventDetailsService(max_parallel_loads=EVENT_DETAILS_PARALLEL_LOADS)


def get_chatbot_agent() -> ChatbotAgent:
    """FastAPI dependency providing a fully-wired ChatbotAgent (Task 5).

    Reuses this module's own get_active_fire_events_service/
    get_event_details_service factories above - the exact same
    ActiveFireEventsService/EventDetailsService construction every other
    fire-event-read endpoint already uses - rather than constructing them a
    second, independent way. `GeminiClient()` is a fresh, stateless HTTP
    client (no in-memory state to share across requests, unlike
    SimulationRunManager below) that already defaults to
    settings.GEMINI_API_KEY/GEMINI_MODEL/GEMINI_REQUEST_TIMEOUT (Task 3) -
    no new Gemini configuration is introduced here.
    """
    # Imported here, not at module level: src.api.routers' package __init__
    # imports the chatbot router, which imports this module (circular import).
    from src.api.routers.global_response_plan import get_global_response_plan_read_service

    return ChatbotAgent(
        active_fire_events_service=get_active_fire_events_service(),
        event_details_service=get_event_details_service(),
        weather_conditions_query_service=get_weather_conditions_query_service(),
        fire_danger_query_service=get_fire_danger_query_service(),
        # The Global Response Plan API's own factory: same coverage report and
        # process-wide plan-parts cache as GET /global-response-plan/current.
        global_response_plan_read_service=get_global_response_plan_read_service(),
        gemini_client=GeminiClient(),
    )


def get_weather_conditions_query_service() -> WeatherConditionsQueryService:
    """FastAPI dependency providing a fully-wired WeatherConditionsQueryService.

    Same fresh-instance-per-request rationale as get_fire_danger_query_service
    below: no in-memory state of its own, and its repositories default to the
    process-wide session factory.
    """
    return WeatherConditionsQueryService()


def get_fire_danger_query_service() -> FireDangerQueryService:
    """FastAPI dependency providing a fully-wired FireDangerQueryService (Task A4).

    Same sessionmaker-per-call rationale as get_active_fire_events_service
    above: FireDangerAssessmentRepository/WeatherRepository each default to
    the process-wide session factory when constructed with no arguments.
    Unlike get_simulation_run_manager below, this service holds no in-memory
    state of its own, so a fresh instance per request is correct - no
    singleton is needed here.
    """
    return FireDangerQueryService()


def get_operations_activity_query_service() -> OperationsActivityQueryService:
    """FastAPI dependency providing a fully-wired OperationsActivityQueryService (Task A5).

    Same fresh-instance-per-request rationale as get_fire_danger_query_service
    above: this service holds no in-memory state of its own, and each
    repository/collaborator it composes defaults to the process-wide session
    factory when constructed with no arguments.
    """
    return OperationsActivityQueryService()


_simulation_run_manager: SimulationRunManager | None = None


def get_simulation_run_manager() -> SimulationRunManager:
    """FastAPI dependency providing the process-local SimulationRunManager singleton (Task A3).

    Unlike the services above, this deliberately is NOT a fresh instance per
    request: SimulationRunManager owns in-memory run state (single-active-run
    enforcement, the current run's snapshot) that must be shared across every
    request in this process for that state to mean anything. The instance is
    created lazily on first use and then reused for the lifetime of the
    process - it is lost on restart, and running more than one Uvicorn worker
    would give each worker its own independent instance (see the module
    docstring of src/services/simulation_control/simulation_run_manager.py;
    this MVP requires `--workers 1`).

    Tests should call `set_simulation_run_manager_for_tests` rather than
    relying on this module-level singleton directly, so each test can start
    from a known-fresh manager (e.g. one built with a fake runner/executor)
    without leaking state into other tests.
    """
    global _simulation_run_manager
    if _simulation_run_manager is None:
        _simulation_run_manager = SimulationRunManager()
    return _simulation_run_manager


def set_simulation_run_manager_for_tests(manager: SimulationRunManager | None) -> None:
    """Test-only hook to replace or clear the process-local singleton above.

    Pass a purpose-built SimulationRunManager (e.g. with a fake runner_factory
    and a synchronous executor) to make it the one `get_simulation_run_manager`
    returns, or `None` to force the next call to build a fresh default
    instance. Never used by production code.
    """
    global _simulation_run_manager
    _simulation_run_manager = manager


def get_operations_overview_query_service() -> OperationsOverviewQueryService:
    """FastAPI dependency providing a fully-wired OperationsOverviewQueryService (Task A6).

    Same fresh-instance-per-request rationale as get_fire_danger_query_service
    for every collaborator except `simulation_run_manager`: that one MUST be
    the process-local singleton `get_simulation_run_manager()` returns, never
    a fresh `SimulationRunManager()` (which would always report IDLE,
    silently hiding a real running simulation) - see
    OperationsOverviewQueryService.__init__'s own docstring for why.
    """
    return OperationsOverviewQueryService(simulation_run_manager=get_simulation_run_manager())
