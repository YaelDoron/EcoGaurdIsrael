"""Database engine, session factory, and connection helpers.

Reads DATABASE_URL from configuration, normalizes it for the psycopg driver,
and creates a single process-wide SQLAlchemy engine/session factory (lazily,
on first use) rather than a new engine per call. Never logs the connection
string.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
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
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine, creating it on first use."""
    global _engine
    if _engine is None:
        if not settings.DATABASE_URL:
            raise DatabaseConfigurationError("DATABASE_URL is not configured.")
        database_url = normalize_database_url(settings.DATABASE_URL)
        _engine = create_engine(database_url, pool_pre_ping=True)
        logger.info("Database engine created")
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory, creating it on first use."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
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
        graph_edge_db,
        graph_node_db,
        response_action_db,
        response_plan_db,
        response_plan_uncovered_target_db,
        response_target_db,
        response_target_set_db,
        satellite_hotspot_db,
        weather_observation_db,
        weather_station_db,
        wildfire_report_db,
    )

    Base.metadata.create_all(bind=get_engine())
    logger.info("Database tables ensured (init_db)")
