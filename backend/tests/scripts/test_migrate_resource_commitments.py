"""Tests for scripts/migrate_resource_commitments.py (Stage 1 of the Global
Multi-Incident Optimizer refactor).

Unlike scripts/migrate_response_plan_optimization_config.py's ALTER TABLE
ADD CONSTRAINT step, a plain CREATE TABLE (with its FKs/indexes/PK) executes
identically on SQLite and PostgreSQL, so this migration is exercised fully
against a real SQLite engine - no mocked connection/inspector needed.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, text

import scripts.migrate_resource_commitments as migration_script
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
from src.database.models.resource_commitment_db import ResourceCommitmentDB

TABLE_NAME = migration_script.TABLE_NAME


@pytest.fixture
def pre_migration_engine():
    """Every table EXCEPT resource_commitments - the schema state before this migration runs."""
    engine = create_engine("sqlite:///:memory:")
    tables_except_new = [table for table in Base.metadata.sorted_tables if table.name != TABLE_NAME]
    Base.metadata.create_all(bind=engine, tables=tables_except_new)
    yield engine
    engine.dispose()


def test_table_does_not_exist_before_migration(pre_migration_engine):
    with pre_migration_engine.connect() as connection:
        assert not inspect(connection).has_table(TABLE_NAME)


def test_migrate_creates_the_table_when_absent(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    created = migration_script.migrate()

    assert created is True
    with pre_migration_engine.connect() as connection:
        assert inspect(connection).has_table(TABLE_NAME)


def test_migrate_creates_expected_columns_pk_and_fks(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)
    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        inspector = inspect(connection)
        columns = {column["name"] for column in inspector.get_columns(TABLE_NAME)}
        pk = inspector.get_pk_constraint(TABLE_NAME)
        foreign_keys = inspector.get_foreign_keys(TABLE_NAME)

    # dispatch_state (Stage 6) is included because this script creates the
    # table from the CURRENT ORM metadata - a fresh deployment gets the full
    # up-to-date schema without needing to separately run migrate_dispatch_state.py.
    assert columns == {"resource_id", "fire_event_id", "response_plan_id", "committed_at", "dispatch_state"}
    assert pk["constrained_columns"] == ["resource_id"]
    fk_pairs = {(tuple(fk["constrained_columns"]), fk["referred_table"]) for fk in foreign_keys}
    assert (("resource_id",), "firefighting_resources") in fk_pairs
    assert (("fire_event_id",), "fire_events") in fk_pairs
    assert (("response_plan_id",), "response_plans") in fk_pairs


def test_migrate_creates_expected_indexes(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)
    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        indexed_columns = {
            column
            for index in inspect(connection).get_indexes(TABLE_NAME)
            for column in index["column_names"]
        }
    assert "fire_event_id" in indexed_columns
    assert "response_plan_id" in indexed_columns


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


def test_migrate_preserves_existing_rows_on_second_run(monkeypatch, pre_migration_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)
    migration_script.migrate()
    _insert_fixture_row(pre_migration_engine)

    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        count = connection.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
    assert count == 1


def test_migrate_does_not_fabricate_commitment_rows_for_preexisting_plans(monkeypatch, pre_migration_engine):
    """No backfill: even if FireEvents/ResponsePlans already existed before
    this migration ran, resource_commitments starts and stays empty."""
    with pre_migration_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO fire_events (latitude, longitude, detected_at, updated_at, status, "
                "detection_confidence, methodology, methodology_version, created_at) "
                "VALUES (32.7, 35.0, :now, :now, 'confirmed', 0.9, 'm', '1.0', :now)"
            ),
            {"now": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        )
        connection.execute(
            text("INSERT INTO fire_stations (id, name, latitude, longitude) VALUES ('S1', 'Station', 32.7, 35.0)")
        )
        connection.execute(
            text(
                "INSERT INTO firefighting_resources (id, station_id, status) VALUES ('R1', 'S1', 'available')"
            )
        )
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    migration_script.migrate()

    with pre_migration_engine.connect() as connection:
        count = connection.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
    assert count == 0


def _insert_fixture_row(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO fire_events (latitude, longitude, detected_at, updated_at, status, "
                "detection_confidence, methodology, methodology_version, created_at) "
                "VALUES (32.7, 35.0, :now, :now, 'confirmed', 0.9, 'm', '1.0', :now)"
            ),
            {"now": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        )
        connection.execute(
            text("INSERT INTO fire_stations (id, name, latitude, longitude) VALUES ('S1', 'Station', 32.7, 35.0)")
        )
        connection.execute(
            text("INSERT INTO firefighting_resources (id, station_id, status) VALUES ('R1', 'S1', 'available')")
        )
        connection.execute(
            text(
                "INSERT INTO response_target_sets "
                "(fire_event_id, generated_at, methodology, methodology_version, created_at) "
                "VALUES (1, :now, 'm', '1.0', :now)"
            ),
            {"now": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        )
        connection.execute(
            text(
                "INSERT INTO route_planning_runs "
                "(fire_event_id, response_target_set_id, planned_at, methodology, methodology_version, "
                "resource_ids, created_at) "
                "VALUES (1, 1, :now, 'm', '1.0', '[]', :now)"
            ),
            {"now": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        )
        connection.execute(
            text(
                "INSERT INTO response_plans (fire_event_id, response_target_set_id, route_planning_run_id, "
                "generated_at, status, methodology, methodology_version, random_seed, created_at) "
                "VALUES (1, 1, 1, :now, 'complete', 'm', '1.0', 1, :now)"
            ),
            {"now": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        )
        connection.execute(
            text(
                "INSERT INTO resource_commitments (resource_id, fire_event_id, response_plan_id, committed_at) "
                "VALUES ('R1', 1, 1, :now)"
            ),
            {"now": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        )


# ---------------------------------------------------------------------------
# main() - status output
# ---------------------------------------------------------------------------


def test_main_prints_created_then_already_present(monkeypatch, pre_migration_engine, capsys):
    monkeypatch.setattr(migration_script, "get_engine", lambda: pre_migration_engine)

    first_exit_code = migration_script.main()
    first_output = capsys.readouterr().out
    second_exit_code = migration_script.main()
    second_output = capsys.readouterr().out

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert f"{TABLE_NAME}: created" in first_output
    assert f"{TABLE_NAME}: already present" in second_output
