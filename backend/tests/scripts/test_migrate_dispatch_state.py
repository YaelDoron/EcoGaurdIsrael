"""Tests for scripts/migrate_dispatch_state.py.

Validated against a real SQLAlchemy engine (SQLite): `ALTER TABLE ... ADD
COLUMN ... NOT NULL DEFAULT '...'` with a constant default is supported by
SQLite, so the full add-and-backfill behavior is genuinely exercised, not
mocked.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

import scripts.migrate_dispatch_state as migration_script

TABLE_NAME = migration_script.TABLE_NAME
COLUMN_NAME = migration_script.COLUMN_NAME


@pytest.fixture
def sqlite_engine():
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


def create_legacy_table(engine) -> None:
    """A resource_commitments table shaped like the pre-Stage-6 (live Neon) schema."""
    with engine.begin() as connection:
        connection.execute(
            text(
                f"CREATE TABLE {TABLE_NAME} ("
                "resource_id TEXT PRIMARY KEY, "
                "fire_event_id INTEGER NOT NULL, "
                "response_plan_id INTEGER NOT NULL, "
                "committed_at TEXT NOT NULL"
                ")"
            )
        )
        connection.execute(
            text(
                f"INSERT INTO {TABLE_NAME} (resource_id, fire_event_id, response_plan_id, committed_at) "
                "VALUES ('R1', 1, 1, '2026-01-01T00:00:00+00:00')"
            )
        )


def test_migrate_adds_column_and_backfills_existing_rows_to_dispatched(sqlite_engine, monkeypatch):
    create_legacy_table(sqlite_engine)

    monkeypatch.setattr(migration_script, "get_engine", lambda: sqlite_engine)
    added = migration_script.migrate()

    assert added is True
    with sqlite_engine.connect() as connection:
        columns = {c["name"] for c in inspect(connection).get_columns(TABLE_NAME)}
        assert COLUMN_NAME in columns
        value = connection.execute(
            text(f"SELECT {COLUMN_NAME} FROM {TABLE_NAME} WHERE resource_id = 'R1'")
        ).scalar_one()
    assert value == "dispatched"


def test_migrate_is_a_no_op_when_column_already_exists(sqlite_engine, monkeypatch):
    create_legacy_table(sqlite_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: sqlite_engine)
    migration_script.migrate()

    second_added = migration_script.migrate()

    assert second_added is False


def test_migrate_running_twice_in_a_row_is_safe(sqlite_engine, monkeypatch):
    create_legacy_table(sqlite_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: sqlite_engine)

    migration_script.migrate()
    migration_script.migrate()  # must not raise (e.g. duplicate-column error)

    with sqlite_engine.connect() as connection:
        columns = [c["name"] for c in inspect(connection).get_columns(TABLE_NAME)]
    assert columns.count(COLUMN_NAME) == 1


def test_column_exists_reflects_reality(sqlite_engine, monkeypatch):
    create_legacy_table(sqlite_engine)
    with sqlite_engine.connect() as connection:
        assert migration_script.column_exists(connection) is False

    monkeypatch.setattr(migration_script, "get_engine", lambda: sqlite_engine)
    migration_script.migrate()

    with sqlite_engine.connect() as connection:
        assert migration_script.column_exists(connection) is True


def test_main_prints_a_status_line(monkeypatch, sqlite_engine, capsys):
    create_legacy_table(sqlite_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: sqlite_engine)

    exit_code = migration_script.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"{TABLE_NAME}.{COLUMN_NAME}: added" in output
