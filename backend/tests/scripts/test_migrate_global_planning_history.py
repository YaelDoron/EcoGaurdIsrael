"""Tests for scripts/migrate_global_planning_history.py (Stage 6 of the
Global Multi-Incident Optimizer refactor: demand/shortage historical
persistence).

Every step here is a plain ALTER TABLE ADD COLUMN, so - unlike
migrate_global_planning_run.py's response_plans FK step - all of it is
genuinely exercised against SQLite; no step is expected to fail here.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, text

import scripts.migrate_global_planning_history as migration_script
from src.database.base import Base
from src.database.models import (  # noqa: F401
    fire_event_db,
    fire_station_db,
    firefighting_resource_db,
    response_plan_db,
    response_target_db,
    response_target_set_db,
    route_planning_run_db,
    route_result_db,
)
from src.database.models.global_planning_run_db import GlobalPlanningRunDB
from src.database.models.global_planning_run_event_db import GlobalPlanningRunEventDB

RUN_TABLE_NAME = migration_script.RUN_TABLE_NAME
EVENT_TABLE_NAME = migration_script.EVENT_TABLE_NAME
RUN_COLUMNS = migration_script.RUN_COLUMNS
EVENT_COLUMNS = migration_script.EVENT_COLUMNS


@pytest.fixture
def pre_migration_engine():
    """Every table, but global_planning_runs/global_planning_run_events
    shaped like the pre-Stage-6-history-persistence schema (their original
    Stage 2 column set, before this migration's new columns existed)."""
    engine = create_engine("sqlite:///:memory:")
    tables_except_new = [
        table
        for table in Base.metadata.sorted_tables
        if table.name not in (RUN_TABLE_NAME, EVENT_TABLE_NAME)
    ]
    Base.metadata.create_all(bind=engine, tables=tables_except_new)
    with engine.begin() as connection:
        connection.execute(
            text(
                f"CREATE TABLE {RUN_TABLE_NAME} ("
                "id INTEGER PRIMARY KEY, "
                "started_at TIMESTAMP NOT NULL, "
                "completed_at TIMESTAMP, "
                "status TEXT NOT NULL, "
                "trigger TEXT NOT NULL, "
                "methodology TEXT NOT NULL, "
                "methodology_version TEXT NOT NULL, "
                "input_fingerprint VARCHAR(64), "
                "created_at TIMESTAMP NOT NULL"
                ")"
            )
        )
        connection.execute(
            text(
                f"CREATE TABLE {EVENT_TABLE_NAME} ("
                "id INTEGER PRIMARY KEY, "
                "global_planning_run_id INTEGER NOT NULL, "
                "fire_event_id INTEGER NOT NULL, "
                "event_order INTEGER NOT NULL, "
                "result_status TEXT, "
                "response_plan_id INTEGER, "
                "local_state_fingerprint VARCHAR(64), "
                "error_code TEXT, "
                "created_at TIMESTAMP NOT NULL"
                ")"
            )
        )
    yield engine
    engine.dispose()


def test_pre_migration_schema_is_missing_every_new_column(pre_migration_engine):
    with pre_migration_engine.connect() as connection:
        run_columns = {c["name"] for c in inspect(connection).get_columns(RUN_TABLE_NAME)}
        event_columns = {c["name"] for c in inspect(connection).get_columns(EVENT_TABLE_NAME)}
    for column_name, _ in RUN_COLUMNS:
        assert column_name not in run_columns
    for column_name, _ in EVENT_COLUMNS:
        assert column_name not in event_columns


def test_migrate_adds_every_run_column(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    results = migration_script.migrate()

    for column_name, _ in RUN_COLUMNS:
        assert results[f"{RUN_TABLE_NAME}.{column_name}"] == "added"
    with pre_migration_engine.connect() as connection:
        run_columns = {c["name"] for c in inspect(connection).get_columns(RUN_TABLE_NAME)}
    for column_name, _ in RUN_COLUMNS:
        assert column_name in run_columns


def test_migrate_adds_every_event_column(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    results = migration_script.migrate()

    for column_name, _ in EVENT_COLUMNS:
        assert results[f"{EVENT_TABLE_NAME}.{column_name}"] == "added"
    with pre_migration_engine.connect() as connection:
        event_columns = {c["name"] for c in inspect(connection).get_columns(EVENT_TABLE_NAME)}
    for column_name, _ in EVENT_COLUMNS:
        assert column_name in event_columns


def test_new_columns_are_nullable(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)
    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        run_columns = {c["name"]: c for c in inspect(connection).get_columns(RUN_TABLE_NAME)}
        event_columns = {c["name"]: c for c in inspect(connection).get_columns(EVENT_TABLE_NAME)}
    for column_name, _ in RUN_COLUMNS:
        assert run_columns[column_name]["nullable"] is True
    for column_name, _ in EVENT_COLUMNS:
        assert event_columns[column_name]["nullable"] is True


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_migrate_running_twice_in_a_row_is_safe_and_reports_already_present(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.migrate()
    second = migration_script.migrate()

    for column_name, _ in RUN_COLUMNS:
        assert second[f"{RUN_TABLE_NAME}.{column_name}"] == "already present"
    for column_name, _ in EVENT_COLUMNS:
        assert second[f"{EVENT_TABLE_NAME}.{column_name}"] == "already present"


def test_migrate_does_not_duplicate_columns_on_second_run(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.migrate()
    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        run_column_names = [c["name"] for c in inspect(connection).get_columns(RUN_TABLE_NAME)]
    for column_name, _ in RUN_COLUMNS:
        assert run_column_names.count(column_name) == 1


# ---------------------------------------------------------------------------
# No backfill / historical rows preserved
# ---------------------------------------------------------------------------


def _insert_pre_existing_run_and_member(engine) -> tuple[int, int]:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO fire_events (latitude, longitude, detected_at, updated_at, status, "
                "detection_confidence, methodology, methodology_version, created_at) "
                "VALUES (32.7, 35.0, :now, :now, 'confirmed', 0.9, 'm', '1.0', :now)"
            ),
            {"now": now},
        )
        run_result = connection.execute(
            text(
                f"INSERT INTO {RUN_TABLE_NAME} (started_at, status, trigger, methodology, methodology_version, "
                "created_at) VALUES (:now, 'completed', 'manual', 'm', '1.0', :now)"
            ),
            {"now": now},
        )
        run_id = run_result.lastrowid if run_result.lastrowid is not None else 1
        member_result = connection.execute(
            text(
                f"INSERT INTO {EVENT_TABLE_NAME} (global_planning_run_id, fire_event_id, event_order, created_at) "
                "VALUES (:run_id, 1, 0, :now)"
            ),
            {"run_id": run_id, "now": now},
        )
        member_id = member_result.lastrowid if member_result.lastrowid is not None else 1
    return run_id, member_id


def test_migrate_preserves_historical_rows_with_null_new_columns(monkeypatch, pre_migration_engine):
    run_id, member_id = _insert_pre_existing_run_and_member(pre_migration_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        run_row = connection.execute(
            text(f"SELECT fitness_score, shortage_total_required FROM {RUN_TABLE_NAME} WHERE id = :id"),
            {"id": run_id},
        ).one()
        member_row = connection.execute(
            text(f"SELECT severity_level, minimum_resources FROM {EVENT_TABLE_NAME} WHERE id = :id"),
            {"id": member_id},
        ).one()
    assert run_row[0] is None
    assert run_row[1] is None
    assert member_row[0] is None
    assert member_row[1] is None


# ---------------------------------------------------------------------------
# main() - status output
# ---------------------------------------------------------------------------


def test_main_prints_a_status_line_for_every_column_and_returns_zero(monkeypatch, pre_migration_engine, capsys):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    exit_code = migration_script.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    for column_name, _ in RUN_COLUMNS:
        assert f"{RUN_TABLE_NAME}.{column_name}: added" in output
    for column_name, _ in EVENT_COLUMNS:
        assert f"{EVENT_TABLE_NAME}.{column_name}: added" in output
