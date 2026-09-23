"""Tests for scripts/migrate_add_news_wildfire_signal_strength.py.

Uses plain ADD COLUMN / cross-dialect SQL, so - unlike the raw
information_schema-based satellite_hotspots.created_at migration - this one
is fully exercisable against a real SQLite engine. No live Neon required.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

import scripts.migrate_add_news_wildfire_signal_strength as migration_script
from src.database.base import Base
from src.database.models.wildfire_report_db import WildfireReportDB

TABLE_NAME = migration_script.TABLE_NAME
COLUMN_NAME = migration_script.COLUMN_NAME

# Schema as it existed before this migration - no wildfire_signal_strength column.
_PRE_MIGRATION_DDL = """
CREATE TABLE wildfire_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url VARCHAR NOT NULL UNIQUE,
    source_feed VARCHAR,
    title VARCHAR NOT NULL,
    summary TEXT,
    location_name VARCHAR,
    latitude FLOAT,
    longitude FLOAT,
    published_at DATETIME,
    fetched_at DATETIME NOT NULL
)
"""


@pytest.fixture
def pre_migration_engine():
    """wildfire_reports WITHOUT the new column - the schema state before this migration runs."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text(_PRE_MIGRATION_DDL))
    yield engine
    engine.dispose()


@pytest.fixture
def post_migration_engine():
    """Full current ORM schema (column already present) - an already-migrated database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine, tables=[WildfireReportDB.__table__])
    yield engine
    engine.dispose()


def _insert_legacy_row(engine, source_url: str = "https://example.com/legacy") -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO wildfire_reports (source_url, source_feed, title, summary, fetched_at) "
                "VALUES (:source_url, 'Feed', 'Title', 'Summary', :fetched_at)"
            ),
            {"source_url": source_url, "fetched_at": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        )


# --- column_exists / inspection ---


def test_column_does_not_exist_before_migration(pre_migration_engine):
    assert migration_script.column_exists(pre_migration_engine) is False


def test_column_exists_when_already_migrated(post_migration_engine):
    assert migration_script.column_exists(post_migration_engine) is True


def test_column_exists_raises_if_table_missing():
    engine = create_engine("sqlite:///:memory:")
    with pytest.raises(RuntimeError):
        migration_script.column_exists(engine)
    engine.dispose()


# --- --inspect-only ---


def test_inspect_only_detects_missing_column(monkeypatch, pre_migration_engine, capsys):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    exit_code = migration_script.main(["--inspect-only"])

    assert exit_code == 0
    assert "MISSING" in capsys.readouterr().out
    assert migration_script.column_exists(pre_migration_engine) is False  # did not alter anything


def test_inspect_only_detects_existing_column(monkeypatch, post_migration_engine, capsys):
    monkeypatch.setattr(migration_script, "get_engine", lambda: post_migration_engine)

    exit_code = migration_script.main(["--inspect-only"])

    assert exit_code == 0
    assert "present" in capsys.readouterr().out


# --- --apply ---


def test_apply_adds_the_nullable_column(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    exit_code = migration_script.main(["--apply"])

    assert exit_code == 0
    assert migration_script.column_exists(pre_migration_engine) is True


def test_apply_is_idempotent_when_rerun(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    first_exit_code = migration_script.main(["--apply"])
    second_exit_code = migration_script.main(["--apply"])  # must not raise (e.g. duplicate-column error)

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert migration_script.column_exists(pre_migration_engine) is True


def test_apply_is_a_no_op_when_already_present(monkeypatch, post_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: post_migration_engine)

    created = migration_script.migrate()

    assert created is False


def test_apply_preserves_existing_row_count(monkeypatch, pre_migration_engine):
    _insert_legacy_row(pre_migration_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.main(["--apply"])

    assert migration_script.row_count(pre_migration_engine) == 1


def test_apply_does_not_fabricate_values_for_existing_rows(monkeypatch, pre_migration_engine):
    _insert_legacy_row(pre_migration_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.main(["--apply"])

    with pre_migration_engine.connect() as connection:
        value = connection.execute(
            text(f"SELECT {COLUMN_NAME} FROM {TABLE_NAME} WHERE source_url = 'https://example.com/legacy'")
        ).scalar_one()
    assert value is None


def test_apply_does_not_alter_other_column_data(monkeypatch, pre_migration_engine):
    _insert_legacy_row(pre_migration_engine, source_url="https://example.com/preserve-me")
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.main(["--apply"])

    with pre_migration_engine.connect() as connection:
        title = connection.execute(
            text(f"SELECT title FROM {TABLE_NAME} WHERE source_url = 'https://example.com/preserve-me'")
        ).scalar_one()
    assert title == "Title"
