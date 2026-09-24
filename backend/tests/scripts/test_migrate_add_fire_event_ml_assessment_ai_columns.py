"""Task 9B: the additive, idempotent migration for the AI audit columns (SQLite stand-in; never touches a live database)."""
from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

from scripts import migrate_add_fire_event_ml_assessment_ai_columns as migration
from src.database.base import Base

if tuple(int(part) for part in sqlite3.sqlite_version.split(".")[:2]) < (3, 35):
    pytest.skip("SQLite too old for DROP COLUMN", allow_module_level=True)


def unmigrated_engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        for name, _ in migration.COLUMNS:
            connection.execute(text(f"ALTER TABLE {migration.TABLE_NAME} DROP COLUMN {name}"))
    return engine


def columns(engine):
    return {column["name"] for column in inspect(engine).get_columns(migration.TABLE_NAME)}


def test_the_migration_adds_exactly_the_five_nullable_columns_and_is_idempotent():
    engine = unmigrated_engine()
    before = columns(engine)
    assert migration.missing_columns(engine) == tuple(name for name, _ in migration.COLUMNS)

    added = migration.migrate(engine)

    assert set(added) == {"policy_version", "policy_status", "history_available", "satellite_pass_count", "current_satellite_pixel_count"}
    assert columns(engine) == before | set(added)
    assert all(c["nullable"] for c in inspect(engine).get_columns(migration.TABLE_NAME) if c["name"] in added)
    assert migration.migrate(engine) == ()  # second run: no-op
    engine.dispose()


def test_the_migration_leaves_existing_rows_untouched():
    engine = unmigrated_engine()
    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys=OFF"))
        connection.execute(text(
            "INSERT INTO fire_event_ml_assessments "
            "(fire_event_id, decision_mode, rule_status, rule_confidence, ml_available, agreement, updated_at) "
            "VALUES (7, 'shadow', 'suspected', 0.6, 0, 'ml_unavailable', '2026-09-14 06:00:00')"))

    migration.migrate(engine)

    with engine.connect() as connection:
        rows = connection.execute(text(
            "SELECT fire_event_id, decision_mode, policy_version, satellite_pass_count FROM fire_event_ml_assessments")).all()
    assert [tuple(row) for row in rows] == [(7, "shadow", None, None)]
    assert migration.row_count(engine) == 1
    engine.dispose()


def test_the_migration_declares_columns_the_orm_model_has():
    from src.database.models.fire_event_ml_assessment_db import FireEventMLAssessmentDB

    orm_columns = {column.name for column in FireEventMLAssessmentDB.__table__.columns}
    assert {name for name, _ in migration.COLUMNS} <= orm_columns
