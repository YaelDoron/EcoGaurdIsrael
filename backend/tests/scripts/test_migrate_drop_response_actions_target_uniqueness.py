"""Tests for scripts/migrate_drop_response_actions_target_uniqueness.py
(Stage 6 of the Global Multi-Incident Optimizer refactor).

DROP CONSTRAINT is not supported by SQLite at all (same documented
limitation as migrate_global_planning_run.py's response_plans FK step) - so
this is verified directly against real Neon (see the Stage 6 completion
report's concurrency/history section) and, here, exercised only for its
introspection/idempotency logic against a SQLite table that never had the
constraint to begin with.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError

import scripts.migrate_drop_response_actions_target_uniqueness as migration_script

TABLE_NAME = migration_script.TABLE_NAME
CONSTRAINT_NAME = migration_script.CONSTRAINT_NAME


@pytest.fixture
def post_fix_engine():
    """A response_actions table shaped like the CURRENT (post-fix) ORM
    model - no uq_response_actions_plan_target constraint - matching what a
    fresh CREATE TABLE produces today."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                f"CREATE TABLE {TABLE_NAME} ("
                "id INTEGER PRIMARY KEY, "
                "response_plan_id INTEGER NOT NULL, "
                "action_order INTEGER NOT NULL, "
                "resource_id TEXT NOT NULL, "
                "response_target_id INTEGER NOT NULL, "
                "route_result_id INTEGER NOT NULL, "
                "UNIQUE (response_plan_id, action_order), "
                "UNIQUE (response_plan_id, resource_id)"
                ")"
            )
        )
    yield engine
    engine.dispose()


@pytest.fixture
def pre_fix_engine():
    """A response_actions table shaped like the OLD (pre-fix) schema - still
    carrying the now-stale uq_response_actions_plan_target constraint,
    matching live Neon before this migration runs."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                f"CREATE TABLE {TABLE_NAME} ("
                "id INTEGER PRIMARY KEY, "
                "response_plan_id INTEGER NOT NULL, "
                "action_order INTEGER NOT NULL, "
                "resource_id TEXT NOT NULL, "
                "response_target_id INTEGER NOT NULL, "
                "route_result_id INTEGER NOT NULL, "
                "UNIQUE (response_plan_id, action_order), "
                "UNIQUE (response_plan_id, resource_id), "
                f"CONSTRAINT {CONSTRAINT_NAME} UNIQUE (response_plan_id, response_target_id)"
                ")"
            )
        )
    yield engine
    engine.dispose()


def test_constraint_exists_detects_the_stale_constraint(pre_fix_engine):
    with pre_fix_engine.connect() as connection:
        assert migration_script.constraint_exists(connection) is True


def test_constraint_exists_is_false_once_already_absent(post_fix_engine):
    with post_fix_engine.connect() as connection:
        assert migration_script.constraint_exists(connection) is False


def test_migrate_is_a_no_op_when_the_constraint_is_already_absent(monkeypatch, post_fix_engine):
    monkeypatch.setattr(migration_script, "get_engine", lambda: post_fix_engine)

    dropped = migration_script.migrate()

    assert dropped is False


def test_main_reports_already_absent_and_returns_zero(monkeypatch, post_fix_engine, capsys):
    monkeypatch.setattr(migration_script, "get_engine", lambda: post_fix_engine)

    exit_code = migration_script.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"{TABLE_NAME}.{CONSTRAINT_NAME}: already absent" in output


def test_sqlite_does_not_support_alter_table_drop_constraint(pre_fix_engine):
    """Documents WHY this migration cannot be exercised end-to-end against
    SQLite - real removal is verified directly against Neon."""
    with pytest.raises(OperationalError):
        with pre_fix_engine.begin() as connection:
            connection.execute(text(f"ALTER TABLE {TABLE_NAME} DROP CONSTRAINT {CONSTRAINT_NAME}"))
