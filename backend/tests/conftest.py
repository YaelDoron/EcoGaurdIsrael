"""Shared pytest fixtures for database/repository tests.

Provides a fresh SQLite in-memory database per test - this is NOT the real
Neon/PostgreSQL database (see tests/integration/test_neon_database.py for
that). SQLite approximates PostgreSQL closely enough to unit-test basic
constraints (NOT NULL, UNIQUE, FOREIGN KEY), but its enforcement is not
identical to PostgreSQL's; the live Neon integration test additionally
covers real PostgreSQL behavior.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.base import Base
from src.database.models import weather_observation_db, weather_station_db  # noqa: F401


@pytest.fixture
def sqlite_engine() -> Engine:
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def sqlite_session_factory(sqlite_engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=sqlite_engine, expire_on_commit=False)
