"""Tests for scripts/migrate_fire_spread_effective_state_fingerprint.py.

Validates the migration's idempotency CONTROL FLOW (introspect-then-add-if-
missing, for both the column and its index) against a real SQLAlchemy
engine/connection - not against the live Neon database. SQLAlchemy's
`inspect(connection).get_columns()`/`get_indexes()` reflection API and the
"begin a transaction, introspect, conditionally ALTER" logic under test are
backend-agnostic, so a SQLite engine genuinely exercises that control flow.

This does NOT validate PostgreSQL-specific DDL semantics (exact column type
representation, PostgreSQL's own ALTER TABLE/CREATE INDEX grammar) - SQLite
is not a stand-in for PostgreSQL DDL correctness. That comparison was done
separately and statically, by compiling the ORM model's own DDL against the
`postgresql` dialect (see the audit report) rather than by executing it
anywhere; there is no live/isolated PostgreSQL available in this test
environment to execute the actual migration against.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

import scripts.migrate_fire_spread_effective_state_fingerprint as migration_script

TABLE_NAME = migration_script.TABLE_NAME
COLUMN_NAME = migration_script.COLUMN_NAME
INDEX_NAME = migration_script.INDEX_NAME


@pytest.fixture
def sqlite_engine():
    # Plain in-memory SQLite: SQLAlchemy's default pooling for this URL keeps
    # one shared connection alive for the engine's lifetime, matching the
    # pattern the rest of this test suite already relies on (tests/conftest.py).
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


def create_legacy_table(engine) -> None:
    """A fire_spread_predictions table shaped like the pre-migration (live Neon) schema."""
    with engine.begin() as connection:
        connection.execute(
            text(
                f"CREATE TABLE {TABLE_NAME} ("
                "id INTEGER PRIMARY KEY, "
                "fire_event_id INTEGER NOT NULL, "
                "predicted_at TEXT NOT NULL, "
                "horizon_minutes INTEGER NOT NULL, "
                "status TEXT NOT NULL"
                ")"
            )
        )


def test_migrate_adds_missing_column_and_index(monkeypatch, sqlite_engine):
    create_legacy_table(sqlite_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: sqlite_engine)

    column_added, index_added = migration_script.migrate()

    assert (column_added, index_added) == (True, True)
    with sqlite_engine.connect() as connection:
        inspector = inspect(connection)
        columns = {c["name"] for c in inspector.get_columns(TABLE_NAME)}
        indexes = {i["name"] for i in inspector.get_indexes(TABLE_NAME)}
    assert COLUMN_NAME in columns
    assert INDEX_NAME in indexes


def test_migrate_is_a_no_op_when_column_and_index_already_exist(monkeypatch, sqlite_engine):
    create_legacy_table(sqlite_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: sqlite_engine)
    first_column_added, first_index_added = migration_script.migrate()
    assert (first_column_added, first_index_added) == (True, True)

    second_column_added, second_index_added = migration_script.migrate()

    assert (second_column_added, second_index_added) == (False, False)


def test_migrate_running_twice_in_a_row_is_safe(monkeypatch, sqlite_engine):
    """The exact scenario the migration may face in production: someone runs
    the script, it succeeds, and it is run again by mistake or by re-deploy."""
    create_legacy_table(sqlite_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: sqlite_engine)

    migration_script.migrate()
    # Must not raise (e.g. a duplicate-column/duplicate-index database error).
    migration_script.migrate()

    with sqlite_engine.connect() as connection:
        inspector = inspect(connection)
        columns = [c["name"] for c in inspector.get_columns(TABLE_NAME)]
        indexes = [i["name"] for i in inspector.get_indexes(TABLE_NAME)]
    # Exactly one column/index, not duplicated.
    assert columns.count(COLUMN_NAME) == 1
    assert indexes.count(INDEX_NAME) == 1


def test_migrate_adds_only_the_missing_piece_when_column_exists_but_index_does_not(monkeypatch, sqlite_engine):
    """Simulates a partially-applied prior state (e.g. someone ran only the
    old column-only version of this migration by hand)."""
    create_legacy_table(sqlite_engine)
    with sqlite_engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {COLUMN_NAME} VARCHAR(64)"))
    monkeypatch.setattr(migration_script, "get_engine", lambda: sqlite_engine)

    column_added, index_added = migration_script.migrate()

    assert (column_added, index_added) == (False, True)
    with sqlite_engine.connect() as connection:
        indexes = {i["name"] for i in inspect(connection).get_indexes(TABLE_NAME)}
    assert INDEX_NAME in indexes


def test_main_prints_a_clear_status_line_for_each_piece(monkeypatch, sqlite_engine, capsys):
    create_legacy_table(sqlite_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: sqlite_engine)

    exit_code = migration_script.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"{TABLE_NAME}.{COLUMN_NAME}: added" in output
    assert f"{INDEX_NAME}: added" in output
