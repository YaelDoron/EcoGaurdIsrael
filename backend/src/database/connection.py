"""Database engine, session factory, and connection helpers.

Reads DATABASE_URL from configuration, normalizes it for the psycopg driver,
and creates a single process-wide SQLAlchemy engine/session factory (lazily,
on first use) rather than a new engine per call. Never logs the connection
string.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.exc import DisconnectionError
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import settings
from src.database.base import Base

logger = logging.getLogger(__name__)

_POSTGRES_SCHEME = "postgresql://"
_POSTGRES_PSYCOPG_SCHEME = "postgresql+psycopg://"


class DatabaseConfigurationError(Exception):
    """Raised when database functionality is used without a configured DATABASE_URL."""


def normalize_database_url(raw_url: str) -> str:
    """Normalize a PostgreSQL/Neon connection string for SQLAlchemy + psycopg.

    - "postgresql://..." becomes "postgresql+psycopg://..." so SQLAlchemy
      selects the psycopg (v3) driver.
    - "postgresql+psycopg://..." is left unchanged.
    - Anything else (host, credentials, query parameters, database name) is
      preserved untouched.

    Never logs the URL - it contains credentials.
    """
    if not raw_url:
        return raw_url
    if raw_url.startswith(_POSTGRES_PSYCOPG_SCHEME):
        return raw_url
    if raw_url.startswith(_POSTGRES_SCHEME):
        return _POSTGRES_PSYCOPG_SCHEME + raw_url[len(_POSTGRES_SCHEME):]
    return raw_url


_engine: Engine | None = None
_read_only_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None

# A pooled connection used within this many seconds is handed out without a
# liveness round trip; an older one is pinged first (see _ping_if_idle).
IDLE_PING_AFTER_SECONDS = 30.0
DATABASE_POOL_SIZE = 10

_read_only_access: ContextVar[bool] = ContextVar("ecoguard_read_only_database_access", default=False)


@contextmanager
def read_only_database_access() -> Iterator[None]:
    """Run the enclosed (read-only) work on AUTOCOMMIT connections.

    Every repository call opens its own short session, and against a remote
    Neon database each explicit BEGIN/COMMIT pair costs two extra network
    round trips per call - the dominant cost of the multi-repository read
    endpoints (Global Response Plan, Event Details, dashboard). A pure read
    does not need a transaction, so inside this scope sessions bind to an
    AUTOCOMMIT view of the same engine/pool. Never use it around writes.
    """
    token = _read_only_access.set(True)
    try:
        yield
    finally:
        _read_only_access.reset(token)


def is_read_only_database_access() -> bool:
    return _read_only_access.get()


class _ReadAwareSession(Session):
    """Session that binds to the AUTOCOMMIT engine view inside `read_only_database_access()`."""

    def get_bind(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if _read_only_access.get() and _read_only_engine is not None and self.bind is _engine:
            return _read_only_engine
        return super().get_bind(*args, **kwargs)


def _record_checkin(dbapi_connection, connection_record) -> None:  # type: ignore[no-untyped-def]
    connection_record.info["last_used"] = time.monotonic()


def _ping_if_idle(dbapi_connection, connection_record, connection_proxy) -> None:  # type: ignore[no-untyped-def]
    """Pessimistic disconnect handling for connections that sat idle.

    Replaces `pool_pre_ping=True`, which pings on EVERY checkout (one extra
    Neon round trip per repository call). A connection returned to the pool
    moments ago is known-good; only an idle one (Neon may have closed it) is
    checked. A failed ping raises DisconnectionError, which makes the pool
    discard it and transparently retry with a fresh connection.
    """
    last_used = connection_record.info.get("last_used")
    if last_used is None or time.monotonic() - last_used < IDLE_PING_AFTER_SECONDS:
        return
    try:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("SELECT 1")
        finally:
            cursor.close()
        if not dbapi_connection.autocommit:
            dbapi_connection.rollback()
    except Exception as exc:  # noqa: BLE001 - any failure means the connection is unusable
        raise DisconnectionError() from exc


def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine, creating it on first use."""
    global _engine, _read_only_engine
    if _engine is None:
        if not settings.DATABASE_URL:
            raise DatabaseConfigurationError("DATABASE_URL is not configured.")
        database_url = normalize_database_url(settings.DATABASE_URL)
        # pool_size 10 (default 5): the read endpoints load independent sections
        # concurrently, and a checkout beyond the persistent pool opens (and later
        # discards) a fresh remote connection each time.
        engine = create_engine(database_url, pool_size=DATABASE_POOL_SIZE)
        event.listen(engine, "checkin", _record_checkin)
        event.listen(engine, "checkout", _ping_if_idle)
        _read_only_engine = engine.execution_options(isolation_level="AUTOCOMMIT")
        _engine = engine
        logger.info("Database engine created")
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory, creating it on first use."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), class_=_ReadAwareSession, expire_on_commit=False)
    return _session_factory


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a SQLAlchemy session bound to the process-wide engine.

    Usage: `with get_session() as session: ...`. The caller is responsible
    for commit/rollback; the session is always closed on exit.
    """
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create all tables declared on Base.metadata if they do not already exist.

    TODO: introduce Alembic migrations once the schema needs to evolve
    beyond a simple create-all (e.g. altering or dropping existing columns).
    """
    # Import models so they are registered on Base.metadata before create_all.
    from src.database.models import (  # noqa: F401
        fire_danger_assessment_db,
        fire_danger_assessment_weather_input_db,
        fire_event_db,
        fire_event_news_evidence_db,
        fire_event_satellite_evidence_db,
        fire_severity_assessment_db,
        fire_severity_assessment_weather_input_db,
        fire_severity_assessment_satellite_input_db,
        fire_spread_prediction_cell_db,
        fire_spread_prediction_db,
        fire_spread_prediction_weather_input_db,
        fire_station_db,
        firefighting_resource_db,
        global_planning_run_db,
        global_planning_run_event_db,
        graph_edge_db,
        graph_node_db,
        plan_comparison_db,
        response_action_db,
        response_plan_db,
        response_plan_planning_state_db,
        response_plan_uncovered_target_db,
        response_target_db,
        response_target_set_db,
        resource_commitment_db,
        satellite_hotspot_db,
        weather_observation_db,
        weather_station_db,
        wildfire_report_db,
    )

    Base.metadata.create_all(bind=get_engine())
    logger.info("Database tables ensured (init_db)")
