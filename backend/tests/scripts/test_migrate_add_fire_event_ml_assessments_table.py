"""Tests for scripts/migrate_add_fire_event_ml_assessments_table.py.

A plain CREATE TABLE (with its FK) executes identically on SQLite and
PostgreSQL, so - like migrate_resource_commitments.py's own tests - this is
exercised fully against a real SQLite engine. No live Neon required.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, text

import scripts.migrate_add_fire_event_ml_assessments_table as migration_script
from src.database.base import Base
from src.database.models.fire_event_db import FireEventDB

TABLE_NAME = migration_script.TABLE_NAME
PARENT_TABLE_NAME = migration_script._PARENT_TABLE_NAME


@pytest.fixture
def pre_migration_engine():
    """Every table EXCEPT fire_event_ml_assessments - the schema state before this migration runs."""
    engine = create_engine("sqlite:///:memory:")
    tables_except_new = [table for table in Base.metadata.sorted_tables if table.name != TABLE_NAME]
    Base.metadata.create_all(bind=engine, tables=tables_except_new)
    yield engine
    engine.dispose()


def _insert_fire_event(engine, latitude=32.7, longitude=35.0) -> int:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with engine.begin() as connection:
        result = connection.execute(
            text(
                "INSERT INTO fire_events (latitude, longitude, detected_at, updated_at, status, "
                "detection_confidence, methodology, methodology_version, created_at) "
                "VALUES (:lat, :lon, :now, :now, 'confirmed', 0.9, 'm', '1.0', :now)"
            ),
            {"lat": latitude, "lon": longitude, "now": now},
        )
        return result.lastrowid


def test_table_does_not_exist_before_migration(pre_migration_engine):
    with pre_migration_engine.connect() as connection:
        assert not inspect(connection).has_table(TABLE_NAME)


def test_migrate_creates_the_table_when_absent(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    created = migration_script.migrate()

    assert created is True
    with pre_migration_engine.connect() as connection:
        assert inspect(connection).has_table(TABLE_NAME)


def test_migrate_creates_expected_columns_and_unique_fk(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)
    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        inspector = inspect(connection)
        columns = {column["name"] for column in inspector.get_columns(TABLE_NAME)}
        foreign_keys = inspector.get_foreign_keys(TABLE_NAME)

    assert columns == {
        "id",
        "fire_event_id",
        "decision_mode",
        "rule_status",
        "rule_confidence",
        "ml_available",
        "ml_probability",
        "ml_model_name",
        "ml_model_version",
        "ml_feature_schema_version",
        "ml_failure_reason",
        "agreement",
        "updated_at",
        # Task 9B: a freshly created table already has the (nullable) AI Hybrid V5 audit columns.
        "policy_version",
        "policy_status",
        "history_available",
        "satellite_pass_count",
        "current_satellite_pixel_count",
    }
    fk_pairs = {(tuple(fk["constrained_columns"]), fk["referred_table"]) for fk in foreign_keys}
    assert (("fire_event_id",), "fire_events") in fk_pairs


def test_migrate_is_a_no_op_when_already_present(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)
    first = migration_script.migrate()

    second = migration_script.migrate()

    assert first is True
    assert second is False


def test_migrate_running_twice_in_a_row_is_safe(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.migrate()
    migration_script.migrate()  # must not raise (e.g. table-already-exists error)

    with pre_migration_engine.connect() as connection:
        assert inspect(connection).has_table(TABLE_NAME)


def test_migrate_starts_empty_no_backfill(monkeypatch, pre_migration_engine):
    _insert_fire_event(pre_migration_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        count = connection.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
    assert count == 0


def test_migrate_preserves_existing_fire_event_rows(monkeypatch, pre_migration_engine):
    _insert_fire_event(pre_migration_engine)
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        count = connection.execute(text(f"SELECT COUNT(*) FROM {PARENT_TABLE_NAME}")).scalar()
    assert count == 1


def test_inspect_only_does_not_create_the_table(monkeypatch, pre_migration_engine, capsys):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    exit_code = migration_script.main(["--inspect-only"])

    assert exit_code == 0
    assert "MISSING" in capsys.readouterr().out
    with pre_migration_engine.connect() as connection:
        assert not inspect(connection).has_table(TABLE_NAME)


def test_apply_via_main_creates_the_table(monkeypatch, pre_migration_engine, capsys):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    exit_code = migration_script.main(["--apply"])

    assert exit_code == 0
    assert "created" in capsys.readouterr().out
    with pre_migration_engine.connect() as connection:
        assert inspect(connection).has_table(TABLE_NAME)


def test_main_prints_created_then_already_present(monkeypatch, pre_migration_engine, capsys):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.main(["--apply"])
    first_output = capsys.readouterr().out
    migration_script.main(["--apply"])
    second_output = capsys.readouterr().out

    assert f"{TABLE_NAME}: created" in first_output
    assert f"{TABLE_NAME}: already present" in second_output
